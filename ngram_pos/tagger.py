"""
Character n-gram based POS tagger for Turkish.
"""

import conllu
from collections import Counter, defaultdict
import math


class CharNgramTagger:
    def __init__(self, min_n: int = 2, max_n: int = 5, alpha: float = 0.1):
        self.min_n = min_n
        self.max_n = max_n
        self.alpha = alpha  # Smoothing
        
        self.tag_counts = Counter()
        self.total_tokens = 0
        
        self.ngram_tag_counts = defaultdict(Counter)
        self.ngram_total = Counter()
        
        self.tag_list = []
    
    def _extract_ngrams(self, word: str) -> list[str]:
        word = word.lower()
        ngrams = []
        for n in range(self.min_n, self.max_n + 1):
            if n > len(word):
                continue
            for i in range(len(word) - n + 1):
                ngram = word[i:i+n]
                ngrams.append(ngram)
        return ngrams
    
    def train(self, conllu_path: str):
        with open(conllu_path, 'r', encoding='utf-8') as f:
            data = conllu.parse(f.read())
        
        for sent in data:
            for token in sent:
                if token['upos'] in ('PUNCT', 'X', '_'):
                    continue
                
                word = token['form']
                tag = token['upos']
                
                self.tag_counts[tag] += 1
                self.total_tokens += 1
                
                ngrams = self._extract_ngrams(word)
                for ng in ngrams:
                    self.ngram_tag_counts[ng][tag] += 1
                    self.ngram_total[ng] += 1
        
        self.tag_list = list(self.tag_counts.keys())
        
        print(f"Egitim tamamlandi:")
        print(f"  Toplam token: {self.total_tokens}")
        print(f"  Unique n-grams: {len(self.ngram_total)}")
        print(f"  Tag dagilimi: {dict(self.tag_counts)}")
    
    def train_multi(self, conllu_paths: list[str]):
        """Train on multiple conllu files."""
        total_words = 0
        for path in conllu_paths:
            with open(path, 'r', encoding='utf-8') as f:
                data = conllu.parse(f.read())
            
            for sent in data:
                for token in sent:
                    if token['upos'] in ('PUNCT', 'X', '_'):
                        continue
                    
                    word = token['form']
                    tag = token['upos']
                    
                    self.tag_counts[tag] += 1
                    self.total_tokens += 1
                    total_words += 1
                    
                    ngrams = self._extract_ngrams(word)
                    for ng in ngrams:
                        self.ngram_tag_counts[ng][tag] += 1
                        self.ngram_total[ng] += 1
        
        self.tag_list = list(self.tag_counts.keys())
        
        print(f"Egitim tamamlandi:")
        print(f"  Toplam token: {total_words}")
        print(f"  Unique n-grams: {len(self.ngram_total)}")
        print(f"  Tag dagilimi: {dict(self.tag_counts)}")
    
    def predict(self, word: str) -> str:
        ngrams = self._extract_ngrams(word.lower())
        
        if not ngrams:
            return 'NOUN'
        
        best_tag = None
        best_score = -float('inf')
        
        for tag in self.tag_list:
            score = math.log(self.tag_counts[tag] / self.total_tokens)
            
            for ng in ngrams:
                ng_count = self.ngram_tag_counts[ng].get(tag, 0)
                total_ng = self.ngram_total.get(ng, 0)
                
                prob = (ng_count + self.alpha) / (self.tag_counts[tag] + self.alpha * len(self.ngram_total))
                score += math.log(prob)
            
            if score > best_score:
                best_score = score
                best_tag = tag
        
        return best_tag if best_tag else 'NOUN'
    
    def predict_top_k(self, word: str, k: int = 3) -> list[tuple[str, float]]:
        ngrams = self._extract_ngrams(word.lower())
        
        scores = {}
        for tag in self.tag_list:
            score = math.log(self.tag_counts[tag] / self.total_tokens)
            
            for ng in ngrams:
                ng_count = self.ngram_tag_counts[ng].get(tag, 0)
                prob = (ng_count + self.alpha) / (self.tag_counts[tag] + self.alpha * len(self.ngram_total))
                score += math.log(prob)
            
            scores[tag] = score
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:k]
    
    def evaluate(self, conllu_path: str) -> dict:
        with open(conllu_path, 'r', encoding='utf-8') as f:
            data = conllu.parse(f.read())
        
        correct = 0
        total = 0
        by_tag = defaultdict(lambda: {'correct': 0, 'total': 0})
        errors = []
        
        for sent in data:
            for token in sent:
                if token['upos'] in ('PUNCT', 'X', '_'):
                    continue
                
                word = token['form']
                gold = token['upos']
                
                pred = self.predict(word)
                
                total += 1
                by_tag[gold]['total'] += 1
                if pred == gold:
                    correct += 1
                    by_tag[gold]['correct'] += 1
                else:
                    if len(errors) < 20:
                        errors.append((word, gold, pred))
        
        accuracy = correct / total if total > 0 else 0
        
        print(f"\nDegerlendirme: {correct}/{total} = {accuracy*100:.2f}%")
        print("\nPOS bazli:")
        for tag, stats in sorted(by_tag.items()):
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"  {tag}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
        
        print("\nBazi hatalar:")
        for w, g, p in errors[:10]:
            print(f"  {w} -> gold:{g} pred:{p}")
        
        return {'accuracy': accuracy, 'by_tag': dict(by_tag)}
    
    def save(self, path: str):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'min_n': self.min_n,
                'max_n': self.max_n,
                'alpha': self.alpha,
                'tag_counts': dict(self.tag_counts),
                'total_tokens': self.total_tokens,
                'ngram_tag_counts': {k: dict(v) for k, v in self.ngram_tag_counts.items()},
                'ngram_total': dict(self.ngram_total),
                'tag_list': self.tag_list
            }, f)
        print(f"Model kaydedildi: {path}")
    
    def load(self, path: str):
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.min_n = data['min_n']
        self.max_n = data['max_n']
        self.alpha = data['alpha']
        self.tag_counts = Counter(data['tag_counts'])
        self.total_tokens = data['total_tokens']
        self.ngram_tag_counts = defaultdict(Counter, {k: Counter(v) for k, v in data['ngram_tag_counts'].items()})
        self.ngram_total = Counter(data['ngram_total'])
        self.tag_list = data['tag_list']
        print(f"Model yuklendi: {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str)
    parser.add_argument('--test', type=str)
    parser.add_argument('--model', type=str, default='model.pkl')
    args = parser.parse_args()
    
    tagger = CharNgramTagger(min_n=2, max_n=4, alpha=0.1)
    
    if args.train:
        print(f"Egitim dosyasi: {args.train}")
        tagger.train(args.train)
    
    if args.test:
        print(f"Test dosyasi: {args.test}")
        tagger.evaluate(args.test)


if __name__ == '__main__':
    main()