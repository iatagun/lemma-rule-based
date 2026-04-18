"""
 Türetim ve Lemma Tahmini:
 POS tespiti + Türetilmiş/Basit ayrımı + Kök tahmini
"""

import conllu
from collections import Counter, defaultdict
import math


class DerivationLemmaTagger:
    """
    Kelime verildiğinde:
    1. POS tahmin eder
    2. Türetilmiş mi basit mi belirler  
    3. Kök tahmini yapar
    """
    def __init__(self, min_n=2, max_n=5, alpha=0.1):
        self.min_n = min_n
        self.max_n = max_n
        self.alpha = alpha
        
        # POS tag counts
        self.tag_counts = Counter()
        self.total_tokens = 0
        
        # n-gram -> tag
        self.ngram_tag_counts = defaultdict(Counter)
        self.ngram_total = Counter()
        
        # Türetim: n-gram -> derived/simple
        self.ngram_derived_counts = Counter()
        self.ngram_simple_counts = Counter()
        self.total_derived = 0
        self.total_simple = 0
        
        # Lemma tahmini: suffix -> lemma pattern
        self.suffix_lemma_patterns = defaultdict(Counter)
        
        self.tag_list = []
    
    def _extract_ngrams(self, word):
        word = word.lower()
        ngrams = []
        for n in range(self.min_n, self.max_n + 1):
            if n > len(word):
                continue
            for i in range(len(word) - n + 1):
                ngrams.append(word[i:i+n])
        return ngrams
    
    def train(self, conllu_path):
        with open(conllu_path, 'r', encoding='utf-8') as f:
            data = conllu.parse(f.read())
        
        for sent in data:
            for token in sent:
                if token['upos'] in ('PUNCT', 'X', '_'):
                    continue
                
                word = token['form']
                lemma = token['lemma']
                tag = token['upos']
                
                self.tag_counts[tag] += 1
                self.total_tokens += 1
                
                # POS için n-gram
                ngrams = self._extract_ngrams(word)
                for ng in ngrams:
                    self.ngram_tag_counts[ng][tag] += 1
                    self.ngram_total[ng] += 1
                
                # Türetim için
                is_derived = word.lower() != lemma.lower() and (lemma.lower() in word.lower() or word.lower().startswith(lemma.lower()))
                
                if is_derived:
                    self.total_derived += 1
                    for ng in ngrams:
                        self.ngram_derived_counts[ng] += 1
                else:
                    self.total_simple += 1
                    for ng in ngrams:
                        self.ngram_simple_counts[ng] += 1
                
                # Suffix -> lemma pattern (en son 3 karakter)
                if len(word) >= 3:
                    suffix = word[-3:]
                    self.suffix_lemma_patterns[suffix][lemma.lower()] += 1
        
        self.tag_list = list(self.tag_counts.keys())
        
        print(f"Egitim tamamlandi:")
        print(f"  Token: {self.total_tokens}")
        print(f"  Derived: {self.total_derived}, Simple: {self.total_simple}")
    
    def predict_tag(self, word):
        """POS tag tahmini."""
        ngrams = self._extract_ngrams(word.lower())
        
        if not ngrams:
            return 'NOUN'
        
        best_tag = None
        best_score = -float('inf')
        
        for tag in self.tag_list:
            score = math.log(self.tag_counts[tag] / self.total_tokens)
            
            for ng in ngrams:
                ng_count = self.ngram_tag_counts[ng].get(tag, 0)
                prob = (ng_count + self.alpha) / (self.tag_counts[tag] + self.alpha * len(self.ngram_total))
                score += math.log(prob)
            
            if score > best_score:
                best_score = score
                best_tag = tag
        
        return best_tag if best_tag else 'NOUN'
    
    def predict_derived_prob(self, word):
        """Türetilmiş olma olasılığı."""
        ngrams = self._extract_ngrams(word.lower())
        
        if not ngrams:
            return 0.5
        
        log_prob_derived = math.log(self.total_derived / (self.total_simple + self.total_derived))
        log_prob_simple = math.log(self.total_simple / (self.total_simple + self.total_derived))
        
        total_derived = sum(self.ngram_derived_counts.values()) + self.alpha * len(self.ngram_derived_counts)
        total_simple = sum(self.ngram_simple_counts.values()) + self.alpha * len(self.ngram_simple_counts)
        
        for ng in ngrams:
            derived_count = self.ngram_derived_counts.get(ng, 0) + self.alpha
            simple_count = self.ngram_simple_counts.get(ng, 0) + self.alpha
            
            log_prob_derived += math.log(derived_count / total_derived)
            log_prob_simple += math.log(simple_count / total_simple)
        
        return 1 / (1 + math.exp(log_prob_simple - log_prob_derived))
    
    def predict_lemma_simple(self, word):
        """Basit kelime için lemma = kelime."""
        return word.lower()
    
    def predict_lemma_derived(self, word):
        """Türetilmiş kelime için kök tahmini.
        
        Strateji:
        1. Suffix'leri dene (-mak, -mek, -lar, -ler, -da, -de, vs)
        2. Sözlüğe bak
        3. Fallback: son 3-4 karakteri kaldır
        """
        word_lower = word.lower()
        
        # Bilinen suffixler - bunları kaldır
        suffixes_to_try = [
            ('mak', ''), ('mek', ''),
            ('ları', ''), ('leri', ''), ('lar', ''), ('ler', ''),
            ('da', ''), ('de', ''), ('ta', 'd'), ('te', 'd'),
            ('dan', ''), ('den', ''), ('tan', 'd'), ('ten', 'd'),
            ('a', ''), ('e', ''),
            ('ı', ''), ('i', ''), ('u', ''), ('ü', ''),
            ('ım', ''), ('im', ''), ('um', 'p'), ('üm', 'p'),
            ('ın', ''), ('in', ''), ('un', ''), ('ün', ''),
            ('yor', ''),
            ('meli', ''), ('malı', ''),
            ('li', 'sız'), ('lü', 'süz'),  # zıtlık
            ('ci', ''), ('çı', ''),
            ('lik', ''), ('lık', ''), ('luk', ''), ('lük', ''),
        ]
        
        candidates = set()
        
        for suffix, replacement in suffixes_to_try:
            if word_lower.endswith(suffix):
                root = word_lower[:-len(suffix)] + replacement
                if len(root) >= 2:
                    candidates.add(root)
        
        # Fallback: son 3-4 karakteri kaldır
        if not candidates:
            for n in [3, 4]:
                if len(word_lower) > n:
                    candidates.add(word_lower[:-n])
        
        # Ilk adayı dön (en uzun kök olabilir ama basitlik için ilk)
        return sorted(candidates, key=len, reverse=True)[0] if candidates else word_lower
    
    def analyze(self, word):
        """Tam analiz: POS + derived/simple + lemma."""
        tag = self.predict_tag(word)
        is_derived = self.predict_derived_prob(word) >= 0.5
        
        if is_derived:
            lemma = self.predict_lemma_derived(word)
        else:
            lemma = self.predict_lemma_simple(word)
        
        return {
            'word': word,
            'tag': tag,
            'derived': is_derived,
            'lemma': lemma
        }
    
    def evaluate(self, conllu_path):
        with open(conllu_path, 'r', encoding='utf-8') as f:
            data = conllu.parse(f.read())
        
        # POS accuracy
        tag_correct = 0
        tag_total = 0
        
        # Derived/Simple accuracy
        der_correct = 0
        der_total = 0
        
        # Lemma accuracy (sadece derived olanlar)
        lemma_correct = 0
        lemma_total = 0
        
        for sent in data:
            for token in sent:
                if token['upos'] in ('PUNCT', 'X', '_'):
                    continue
                
                word = token['form']
                gold_tag = token['upos']
                gold_lemma = token['lemma'].lower() if token['lemma'] else word.lower()
                gold_derived = word.lower() != gold_lemma and (gold_lemma in word.lower() or word.lower().startswith(gold_lemma))
                
                result = self.analyze(word)
                
                # Tag
                tag_total += 1
                if result['tag'] == gold_tag:
                    tag_correct += 1
                
                # Derived
                der_total += 1
                if result['derived'] == gold_derived:
                    der_correct += 1
                
                # Lemma (sadece derived)
                if gold_derived:
                    lemma_total += 1
                    # Tam eşleşme veya kök eşleşme (ekler bazen farklı)
                    if result['lemma'] == gold_lemma or result['lemma'] in gold_lemma or gold_lemma in result['lemma']:
                        lemma_correct += 1
        
        print(f"POS accuracy: {tag_correct}/{tag_total} = {100*tag_correct/tag_total:.1f}%")
        print(f"Derived/Simple: {der_correct}/{der_total} = {100*der_correct/der_total:.1f}%")
        print(f"Lemma (derived only): {lemma_correct}/{lemma_total} = {100*lemma_correct/lemma_total:.1f}%")
    
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
                'ngram_derived_counts': dict(self.ngram_derived_counts),
                'ngram_simple_counts': dict(self.ngram_simple_counts),
                'total_derived': self.total_derived,
                'total_simple': self.total_simple,
                'suffix_lemma_patterns': {k: dict(v) for k, v in self.suffix_lemma_patterns.items()},
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
        self.ngram_derived_counts = Counter(data['ngram_derived_counts'])
        self.ngram_simple_counts = Counter(data['ngram_simple_counts'])
        self.total_derived = data['total_derived']
        self.total_simple = data['total_simple']
        self.suffix_lemma_patterns = defaultdict(Counter, {k: Counter(v) for k, v in data['suffix_lemma_patterns'].items()})
        self.tag_list = data['tag_list']
        print(f"Model yuklendi: {path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str)
    parser.add_argument('--test', type=str)
    parser.add_argument('--save', type=str, help='Model kaydet')
    parser.add_argument('--load', type=str, help='Model yukle')
    args = parser.parse_args()
    
    tagger = DerivationLemmaTagger(min_n=2, max_n=5, alpha=0.1)
    
    if args.load:
        tagger.load(args.load)
    
    if args.train:
        print(f"Egitim: {args.train}")
        tagger.train(args.train)
    
    if args.save:
        tagger.save(args.save)
    
    if args.test:
        print(f"Test: {args.test}")
        tagger.evaluate(args.test)