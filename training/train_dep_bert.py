#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ELECTRA + Biaffine Dependency Parser Training Script

Architecture: Dozat & Manning (2017) Deep Biaffine Attention
Encoder: dbmdz/electra-base-turkish-cased (256-dim)
Decoder: Biaffine arc scorer + Label classifier

Usage:
    python train_dep_bert.py                    # Full training
    python train_dep_bert.py --eval          # Evaluation only
    python train_dep_bert.py --checkpoint   # Resume from checkpoint
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.nn.utils import clip_grad_norm_
import numpy as np
from tqdm import tqdm

ENCODER_MODEL = "dbmdz/electra-base-turkish-cased"
MAX_LEN = 64
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
GRAD_ACCUM = 2
MAX_GRAD_NORM = 1.0

ARC_DIM = 500
LABEL_DIM = 500  # Use same dimension for both
DROPOUT = 0.33

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo kökü (script bir alt dizinde)
DATA_DIR = PROJECT_ROOT / "dep_data" / "bert"
OUTPUT_DIR = PROJECT_ROOT / "dep_data"
CHECKPOINT_PATH = OUTPUT_DIR / "dep_parser_latest.pt"

RELATIONS = [
    'dep', 'nsubj', 'nsubj:cop', 'obj', 'iobj', 'csubj', 'csubj:cop',
    'cc', 'conj', 'det', 'dislocated', 'advcl', 'advmod', 'advmod:emph',
    'amod', 'apostrophe', 'case', 'compound', 'compound:lvc', 'compound:redup',
    'cop', 'cop:neg', 'discourse', 'expl', 'flat', 'flat:name', 'fixed',
    'goeswith', 'list', 'mark', 'nmod', 'nmod:comp', 'nmod:part', 'nmod:poss',
    'nummod', 'obl', 'obl:agent', 'obl:tmod', 'orphan', 'parataxis',
    'punct', 'reparandum', 'root', 'vocative', 'xcomp'
]
NUM_LABELS = len(RELATIONS)
REL_TO_ID = {r: i for i, r in enumerate(RELATIONS)}
ID_TO_REL = {i: r for r, i in REL_TO_ID.items()}


class DepDataset(Dataset):
    def __init__(self, json_path: str, tokenizer, max_len: int = MAX_LEN):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        words = item["words"]
        heads = [int(h) for h in item["heads"]]  # Convert string to int
        deps = item["deprels"]  # Already strings

        num_words = len(words)
        if num_words > self.max_len - 1:
            words = words[:self.max_len - 1]
            heads = heads[:self.max_len - 1]
            deps = deps[:self.max_len - 1]
            num_words = self.max_len - 1

        encoded = self.tokenizer(
            words,
            return_tensors="pt",
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.max_len
        )

        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()
        word_ids = encoded.word_ids()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_ids": tuple(word_ids),
            "heads": heads,
            "deps": deps,
            "num_words": num_words,
            "words": words
        }


def collate_fn(batch: List[dict]) -> dict:
    max_len = max(item["input_ids"].size(0) for item in batch)
    max_words = max(item["num_words"] for item in batch)

    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    heads_tensor = torch.zeros(len(batch), max_words, dtype=torch.long)
    deps_tensor = torch.zeros(len(batch), max_words, dtype=torch.long)
    word_ids_list = []
    num_words_list = []

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = item["attention_mask"]

        nw = item["num_words"]
        # Convert string to int for heads
        heads_list = [int(h) for h in item["heads"]]
        heads_tensor[i, :nw] = torch.tensor(heads_list) + 1  # 1-indexed (BERT convention)
        deps_list = [REL_TO_ID.get(d, 0) for d in item["deps"]]  # Convert string rel to ID
        deps_tensor[i, :nw] = torch.tensor(deps_list)

        word_ids_list.append(item["word_ids"])
        num_words_list.append(nw)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "heads": heads_tensor,
        "deps": deps_tensor,
        "word_ids": word_ids_list,
        "num_words": num_words_list
    }


class BiaffineParser(nn.Module):
    def __init__(self, encoder_model: str, num_labels: int, arc_dim: int = ARC_DIM, 
                 label_dim: int = LABEL_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_model)
        hidden = self.encoder.config.hidden_size

        self.arc_mlp = nn.Sequential(
            nn.Linear(hidden, arc_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.label_mlp = nn.Sequential(
            nn.Linear(hidden, label_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.arc_biaffine = nn.Bilinear(arc_dim, arc_dim, 1)
        self.arc_dropout = nn.Dropout(dropout)

        self.label_classifier = nn.Linear(arc_dim + label_dim, num_labels)

    def get_word_representations(self, hidden, word_ids, num_words):
        word_hiddens = []
        for w in range(num_words):
            positions = [i for i, wid in enumerate(word_ids) if wid == w]
            if positions:
                word_hiddens.append(hidden[positions[0]])
            else:
                word_hiddens.append(hidden[0])
        return torch.stack(word_hiddens) if word_hiddens else hidden[:num_words]

    def forward(self, input_ids, attention_mask, word_ids, num_words):
        outputs = self.encoder(input_ids=input_ids.unsqueeze(0), 
                          attention_mask=attention_mask.unsqueeze(0))
        hidden = outputs.last_hidden_state[0]  # (seq, 256) for ELECTRA

        word_hidden = self.get_word_representations(hidden, word_ids, num_words)
        num_words = word_hidden.size(0)

        arc_h = self.arc_mlp(word_hidden)
        arc_d = self.arc_mlp(word_hidden)
        arc_h = self.arc_dropout(arc_h)
        arc_d = self.arc_dropout(arc_d)

        arc_scores = torch.zeros(num_words, num_words, device=arc_h.device)
        for i in range(num_words):
            for j in range(num_words):
                arc_scores[i, j] = self.arc_biaffine(arc_h[i], arc_d[j]).squeeze()

        label_h = self.label_mlp(word_hidden)
        label_d = self.label_mlp(word_hidden)
        label_features = torch.cat([label_h, label_d], dim=-1)
        label_logits = self.label_classifier(label_features)

        return arc_scores, label_logits


class MSTDecoder:
    @staticmethod
    def decode(arc_scores, num_words):
        """Decode: find best head for each token.
        arc_scores[head, dep] = score of head->dep
        """
        heads = [0] * num_words
        
        for dep in range(num_words):
            scores = arc_scores[:num_words, dep].cpu().numpy()
            best = scores.argmax()
            heads[dep] = best

        return heads


def train_epoch(model, dataloader, optimizer, scheduler, device, grad_accum):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    progress = tqdm(dataloader, desc="Training")

    for batch_idx, batch in enumerate(progress):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        heads = batch["heads"].to(device)
        deps = batch["deps"].to(device)
        word_ids_list = batch["word_ids"]
        num_words_list = batch["num_words"]

        loss = 0
        batch_size = len(batch["input_ids"])
        for i in range(batch_size):
            nw = num_words_list[i]
            word_ids = word_ids_list[i]
            if nw < 2:
                continue

            arc_scores, label_logits = model(
                input_ids[i], attention_mask[i], word_ids, nw
            )

            gold_heads = heads[i, :nw].cpu().numpy()
            gold_deps = deps[i, :nw].cpu().numpy()

            arc_loss = 0
            for idx in range(nw):
                gold_head = gold_heads[idx]
                if gold_head < nw:
                    arc_target = torch.zeros(nw, device=device)
                    arc_target[gold_head] = 1.0
                    arc_loss += F.binary_cross_entropy_with_logits(
                        arc_scores[idx, :nw], arc_target[:nw], reduction='sum'
                    ) / nw

            label_loss = F.cross_entropy(
                label_logits[:nw].view(-1, NUM_LABELS),
                deps[i, :nw].long(),
                ignore_index=0,
                reduction='sum'
            ) / max(nw, 1)

            batch_loss = (arc_loss + label_loss) / grad_accum
            batch_loss.backward()
            loss += batch_loss.item()

            if (batch_idx + 1) % grad_accum == 0:
                clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        total_loss += loss / batch_size

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()

    all_uas = []
    all_las = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            heads = batch["heads"]
            deps = batch["deps"]
            word_ids_list = batch["word_ids"]
            num_words_list = batch["num_words"]

            batch_size = len(batch["input_ids"])
            for i in range(batch_size):
                nw = num_words_list[i]
                word_ids = word_ids_list[i]

                arc_scores, label_logits = model(
                    input_ids[i], attention_mask[i], word_ids, nw
                )

                pred_heads = MSTDecoder.decode(arc_scores, nw)
                pred_deps = label_logits[:nw].argmax(dim=-1).cpu().numpy()

                gold_heads = heads[i, :nw].cpu().numpy().tolist()
                gold_deps = deps[i, :nw].cpu().numpy().tolist()

                for idx in range(nw):
                    pred_h = pred_heads[idx] if idx < len(pred_heads) else 0
                    gold_h = gold_heads[idx] - 1 if gold_heads[idx] > 0 else 0

                    all_uas.append(int(pred_h == gold_h))

                    pred_d = pred_deps[idx] if idx < len(pred_deps) else 0
                    gold_d = gold_deps[idx]

                    all_las.append(int(pred_h == gold_h and pred_d == gold_d))

    uas = np.mean(all_uas) * 100
    las = np.mean(all_las) * 100

    return {"UAS": uas, "LAS": las}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Evaluation only")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()

    # Debug: print args
    print(f"Args: checkpoint={args.checkpoint}, eval={args.eval}, epochs={args.epochs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL)

    train_base = DepDataset(str(DATA_DIR / "train.json"), tokenizer)
    dev_dataset = DepDataset(str(DATA_DIR / "dev.json"), tokenizer)

    # Augmented data from rule-based morph analyzer (correct analyses)
    augment_path = DATA_DIR / "augment_train.json"
    if augment_path.exists():
        aug_dataset = DepDataset(str(augment_path), tokenizer)
        train_dataset = torch.utils.data.ConcatDataset([train_base, aug_dataset])
        print(f"Train base: {len(train_base)} + augment: {len(aug_dataset)} = {len(train_dataset)}")
    else:
        train_dataset = train_base
        print(f"Train: {len(train_dataset)} (no augmented data)")

    # Synthetic data for known error patterns
    synthetic_path = DATA_DIR / "synthetic.json"
    if synthetic_path.exists():
        syn_dataset = DepDataset(str(synthetic_path), tokenizer)
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, syn_dataset])
        print(f"Train + synthetic: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        collate_fn=collate_fn, num_workers=0
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    print(f"Train: {len(train_dataset)}, Dev: {len(dev_dataset)}")

    model = BiaffineParser(ENCODER_MODEL, NUM_LABELS).to(device)
    start_epoch = 0

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
            start_epoch = checkpoint.get("epoch", 0)
            print(f"Resumed from epoch {start_epoch}")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded ELECTRA checkpoint")

    if args.eval:
        metrics = evaluate(model, dev_loader, device)
        print(f"Dev UAS: {metrics['UAS']:.2f}, LAS: {metrics['LAS']:.2f}")
        return

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY
    )

    total_steps = len(train_loader) * args.epochs // GRAD_ACCUM
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    best_uas = 0

    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")

        loss = train_epoch(model, train_loader, optimizer, scheduler, device, GRAD_ACCUM)
        print(f"Train loss: {loss:.4f}")

        metrics = evaluate(model, dev_loader, device)
        print(f"Dev UAS: {metrics['UAS']:.2f}, LAS: {metrics['LAS']:.2f}")

        if metrics["UAS"] > best_uas:
            best_uas = metrics["UAS"]
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "metrics": metrics
            }, OUTPUT_DIR / "best_dep_parser.pt")
            print("Saved best model!")

        torch.save({
            "epoch": epoch + 1,
            "model": model.state_dict()
        }, CHECKPOINT_PATH)

    print(f"\nBest UAS: {best_uas:.2f}")


if __name__ == "__main__":
    main()