"""BERT Dependency Parsing Data Preparation"""
import conllu
import os
from collections import defaultdict

def conllu_to_bert_format(sent_tokens):
    """Convert conllu to BERT format: word[SEP]pos[SEP] head deprel"""
    words = [tok['form'] for tok in sent_tokens]
    upos = [tok['upos'] for tok in sent_tokens]
    heads = [str(tok['head']) for tok in sent_tokens]
    deprels = [tok.get('deprel', 'dep') for tok in sent_tokens]
    
    return {
        'words': words,
        'upos': upos,
        'heads': heads,
        'deprels': deprels,
    }

def load_and_convert(conllu_path):
    """Load conllu and convert to BERT format"""
    with open(conllu_path, encoding='utf-8') as f:
        data = conllu.parse(f.read())
    
    sentences = []
    for sent in data:
        # Skip empty or very short sentences
        tokens = [tok for tok in sent if tok['upos'] not in ('PUNCT', 'X', '_')]
        if len(tokens) < 3:
            continue
        
        sentences.append(conllu_to_bert_format(tokens))
    
    return sentences

# Files to process
files = [
    ('benchmark/test.conllu', 'test'),
    ('benchmark/dev.conllu', 'dev'),
    ('ngram_pos/UD_Turkish-Kenet-master/tr_kenet-ud-train.conllu', 'train'),
    ('ngram_pos/UD_Turkish-Kenet-master/tr_kenet-ud-test.conllu', 'test'),
    ('ngram_pos/UD_Turkish-Kenet-master/tr_kenet-ud-dev.conllu', 'dev'),
]

print("=== BERT Dependency Parsing Veri Hazırlığı ===\n")

all_data = defaultdict(list)

for path, split in files:
    print(f"İşleniyor: {path}")
    try:
        sentences = load_and_convert(path)
        all_data[split].extend(sentences)
        print(f"  → {len(sentences)} cümle")
    except Exception as e:
        print(f"  Hata: {e}")

print(f"\n=== Özet ===")
for split, sents in all_data.items():
    print(f"{split}: {len(sents)} cümle")

# Save as JSON (for BERT)
import json

os.makedirs('dep_data/bert', exist_ok=True)

for split, sents in all_data.items():
    output_path = f'dep_data/bert/{split}.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sents, f, ensure_ascii=False, indent=2)
    print(f"Kaydedildi: {output_path}")

print("\n=== Örnek ===")
if all_data['train']:
    ex = all_data['train'][0]
    print(f"Cümle: {' '.join(ex['words'][:5])}...")
    print(f"POS:    {ex['upos'][:5]}")
    print(f"Head:   {ex['heads'][:5]}")
    print(f"Deprel: {ex['deprels'][:5]}")