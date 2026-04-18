"""
 Türetim Belirsizliği Tespiti:
 - Simple: kelime = kök (degismez)
 - Derived: kelime = kök + ek (turemis)
 - Compound: bileşik kelime
"""

import conllu
from collections import Counter, defaultdict
import math


class DerivationDetector:
    """
    Kelimenin türetilmiş mi basit mi olduğunu tahmin eder.
    """
    def __init__(self, min_n=2, max_n=5, alpha=0.1):
        self.min_n = min_n
        self.max_n = max_n
        self.alpha = alpha
        
        self.ngram_simple_counts = Counter()
        self.ngram_derived_counts = Counter()
        
        self.total_simple = 0
        self.total_derived = 0
        
        # Bilinen türetim ekleri
        self.derivation_suffixes = {
            'cı', 'çı', 'ci', 'chi', 'cü', 'çü',
            'man', 'men',
            'nci', 'ncı', 'ncu', 'ncü',
            'lik', 'lık', 'luk', 'lük',
            'ciz', 'çiz',
            'la', 'le',
            'laş', 'leş',
            'ar', 'er',  # çoğul
            'yor',
            'meli', 'malı',
            'sı', 'si', 'su', 'sü',
            'ak', 'ek',
            'gan', 'gen', 'ğan', 'ğen',
            'il', 'il',
            'in', 'ın',
        }
        
        # Bilinen bileşik kelime başlangıçları
        self.compound_prefixes = {
            'ad', 'art', 'ay', 'bel', 'bir', 'bog', 'boy', 'buy',
            'ceza', 'dag', 'den', 'dok', 'el', 'gog', 'goz', 'gum',
            'hava', 'hay', 'is', 'kag', 'kar', 'kaz', 'kent', 'kir',
            'odun', 'okul', 'on', 'orta', 'oyun', 'sag', 'sant',
            'savas', 'sey', 'soz', 'sudas', 'sug', 'sus', 'tabak',
            'tavan', 'tek', 'yag', 'yaz', 'yol', 'yuz',
        }
        
    def _extract_ngrams(self, word):
        word = word.lower()
        ngrams = []
        for n in range(self.min_n, self.max_n + 1):
            if n > len(word):
                continue
            for i in range(len(word) - n + 1):
                ngrams.append(word[i:i+n])
        return ngrams
    
    def _is_derived(self, word, lemma):
        """Kelime türetilmiş mi?"""
        word = word.lower()
        lemma = lemma.lower() if lemma else word
        
        if word == lemma:
            return False
        
        # Kelime lemma ile başlıyor veya lemma kelimenin içinde
        if word.startswith(lemma) or lemma in word:
            return True
        
        # Ses değişimi olmuş olabilir (ünlü düşmesi, yumuşama)
        return False
    
    def train(self, conllu_path):
        with open(conllu_path, 'r', encoding='utf-8') as f:
            data = conllu.parse(f.read())
        
        for sent in data:
            for token in sent:
                if token['upos'] in ('PUNCT', 'X', '_'):
                    continue
                
                word = token['form']
                lemma = token['lemma']
                
                is_derived = self._is_derived(word, lemma)
                ngrams = self._extract_ngrams(word)
                
                if is_derived:
                    self.total_derived += 1
                    for ng in ngrams:
                        self.ngram_derived_counts[ng] += 1
                else:
                    self.total_simple += 1
                    for ng in ngrams:
                        self.ngram_simple_counts[ng] += 1
        
        print(f"Egitim tamamlandi:")
        print(f"  Simple: {self.total_simple}")
        print(f"  Derived: {self.total_derived}")
        print(f"  Derived orani: {100*self.total_derived/(self.total_simple+self.total_derived):.1f}%")
    
    def predict_derived_prob(self, word):
        """Kelimenin türetilmiş olma olasılığı."""
        ngrams = self._extract_ngrams(word)
        
        if not ngrams:
            return 0.5
        
        # Naive Bayes: P(derived|features) ∝ P(features|derived) * P(derived)
        log_prob_derived = math.log(self.total_derived / (self.total_simple + self.total_derived))
        log_prob_simple = math.log(self.total_simple / (self.total_simple + self.total_derived))
        
        for ng in ngrams:
            # Laplace smoothing
            derived_count = self.ngram_derived_counts.get(ng, 0) + self.alpha
            simple_count = self.ngram_simple_counts.get(ng, 0) + self.alpha
            
            # Toplam derived/simple n-gram sayısı
            total_derived = sum(self.ngram_derived_counts.values()) + self.alpha * len(self.ngram_derived_counts)
            total_simple = sum(self.ngram_simple_counts.values()) + self.alpha * len(self.ngram_simple_counts)
            
            log_prob_derived += math.log(derived_count / total_derived)
            log_prob_simple += math.log(simple_count / total_simple)
        
        # Convert to probability
        prob_derived = 1 / (1 + math.exp(log_prob_simple - log_prob_derived))
        return prob_derived
    
    def predict(self, word, threshold=0.5):
        """Tahmin: 'derived' veya 'simple'
        
        Hybrid approach: n-gram + suffix heuristics
        """
        word_lower = word.lower()
        
        # Heuristic 1: Bilinen derivation suffix
        for suff in self.derivation_suffixes:
            if word_lower.endswith(suff):
                # Ek var, ama kelimenin geri kalanı sözlükte var mı?
                # Basit kontrol: kök en az 2 karakter olsun
                root = word_lower[:-len(suff)]
                if len(root) >= 2:
                    return 'derived'
        
        # Heuristic 2: Compound prefix (bileşik kelime)
        for pref in self.compound_prefixes:
            if word_lower.startswith(pref) and len(word_lower) > len(pref) + 2:
                # Bileşik kelime - türetilmiş say
                return 'derived'
        
        # Fallback: n-gram model
        prob = self.predict_derived_prob(word)
        return 'derived' if prob >= threshold else 'simple'
    
    def evaluate(self, conllu_path):
        with open(conllu_path, 'r', encoding='utf-8') as f:
            data = conllu.parse(f.read())
        
        correct = 0
        total = 0
        errors = []
        
        for sent in data:
            for token in sent:
                if token['upos'] in ('PUNCT', 'X', '_'):
                    continue
                
                word = token['form']
                lemma = token['lemma']
                
                gold = 'derived' if self._is_derived(word, lemma) else 'simple'
                pred = self.predict(word)
                
                total += 1
                if pred == gold:
                    correct += 1
                else:
                    if len(errors) < 15:
                        errors.append((word, lemma, gold, pred))
        
        accuracy = correct / total if total > 0 else 0
        print(f"\nDegerlendirme: {correct}/{total} = {accuracy*100:.2f}%")
        print("\nBazi hatalar:")
        for w, l, g, p in errors[:10]:
            print(f"  {w} -> lemma:{l} gold:{g} pred:{p}")
        
        return accuracy


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str)
    parser.add_argument('--test', type=str)
    args = parser.parse_args()
    
    detector = DerivationDetector(min_n=2, max_n=5, alpha=0.1)
    
    if args.train:
        print(f"Egitim: {args.train}")
        detector.train(args.train)
    
    if args.test:
        print(f"Test: {args.test}")
        detector.evaluate(args.test)