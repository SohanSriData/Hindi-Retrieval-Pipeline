"""
Hindi Text Preprocessing Module
Implements tokenization, normalization, stopword removal, and stemming for Hindi
"""

import unicodedata
import re
from typing import List, Set, Dict
from pathlib import Path
from collections import Counter

class HindiPreprocessor:

    def __init__(self, stopwords_path: str = None):
        
        self.stopwords = self._load_stopwords(stopwords_path)
        
    def _load_stopwords(self, stopwords_path: str = None) -> Set[str]:
        
        if not stopwords_path:
            return set()

        p = Path(stopwords_path)
        
        if not p.exists():
            p = Path(__file__).parent / stopwords_path

        if p.exists():
            with p.open('r', encoding='utf-8') as f:
                return set(line.strip() for line in f if line.strip())

        return set()
    
    # Tokenization
    def tokenize(self, text: str) -> List[str]:
        
        # Add spaces around common punctuation
        text = re.sub(r'[।॥\!\?\,\.;:\"\'\(\)\[\]\-\—]', ' ', text)
        
        # Split on whitespace and filter empty tokens
        tokens = []
        for token in re.split(r'\s+', text):
            if token.strip():
                tokens.append(token.strip())
        
        return tokens
    
    # Normalization
    def normalize_unicode(self, text: str) -> str:
        
        text = unicodedata.normalize('NFC', text)
        return text
    
    def remove_zwj_zwnj(self, text: str) -> str:
        
        text = text.replace('\u200D', '')  # Remove ZWJ
        text = text.replace('\u200C', '')  # Remove ZWNJ
        return text
    
    def normalize_diacritics(self, text: str) -> str:
        
        text = unicodedata.normalize('NFC', text)
        
        replacements = {
            '\u0958': '\u0915',  # k with nukta -> without nukta
            '\u0959': '\u0916',  # kh with nukta -> without nukta
            '\u095A': '\u0917',  # g with nukta -> without nukta
            '\u095B': '\u091C',  # j with nukta -> without nukta
            '\u095E': '\u092B',  #ph with nukta -> without nukta
            '\u095F': '\u092F',  # ya with nukta -> without nukta
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        # Handle the decomposed equivalents of loanwords
        decomposed_pairs = {
            '\u0915\u093C': '\u0915',
            '\u0916\u093C': '\u0916',
            '\u0917\u093C': '\u0917',
            '\u091C\u093C': '\u091C',
            '\u092B\u093C': '\u092B',
            '\u092F\u093C': '\u092F',
        }
        for old, new in decomposed_pairs.items():
            text = text.replace(old, new)
        
        return text


    
    def normalize_nasalization(self, text: str) -> str:
        
        # Normalize chandrabindu (ँ U+0901) to anusvara (ं U+0902)
        # Chandrabindu can be replaced wihtout changing phonetics
        text = text.replace('\u0901', '\u0902')  # Devanagari chandrabindu to anusvara
        text = text.replace('\u0981', '\u0902') # Bengali chandrabindu to Devanagari anusvara 
        
        
        return text
    
    def preprocess_text(self, text: str, remove_zwj: bool = True) -> str:
       
        # 1: Unicode NFC normalization
        text = self.normalize_unicode(text)
        
        # 2: Remove ZWJ/ZWNJ if requested
        if remove_zwj:
            text = self.remove_zwj_zwnj(text)
        
        # 3: Normalize diacritics
        text = self.normalize_diacritics(text)
        
        # 4: Normalize nasalization marks
        text = self.normalize_nasalization(text)
        
        return text
    
    # Stopword Removal
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        new_tokens = []
        for token in tokens:
            if token not in self.stopwords:
                new_tokens.append(token)
            
        return new_tokens
    
    def get_stopword_count(self) -> int:
        return len(self.stopwords)
    
    # Stemming
    def stem_word(self, word: str) -> str:
        
        if len(word) <= 2:
            return word
        
        suffixes = [
            'ियां', 'ियों',
            'ेगा', 'ेगी', 'ेगे', 'एगा', 'एगी', 'एगे',
            'ाकर', 'िए', 'ाईं', 
            'कर', 'ना', 'ता', 'ती', 'ते', 'गा', 'गी', 'गे', 
            'ाए', 'ाई', 'ओं', 'एं', 'ें', 'ों', 
            'ा', 'े', 'ी', 'ो', 'ू', 'ं', 'ँ'
        ]

        for suffix in sorted(suffixes, key=len, reverse=True):
            if word.endswith(suffix):
                candidate = word[:-len(suffix)]
                if len(candidate) >= 2:
                    return candidate

        return word
    

    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        new_tokens = []
        for token in tokens:
            new_tokens.append(self.stem_word(token))
        return new_tokens
    
    # Full Pipeline
    def process(self, text: str, remove_stopwords: bool = True, 
                apply_stemming: bool = False) -> List[str]:
        # Normalization
        text = self.preprocess_text(text)
        
        # Tokenization
        tokens = self.tokenize(text)
        
        # Stopword removal
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Stemming
        if apply_stemming:
            tokens = self.stem_tokens(tokens)
        
        return tokens


def calculate_vocabulary_impact(raw_tokens: List[str], filtered_tokens: List[str]) -> Dict:
    raw_vocab = set(raw_tokens)
    filtered_vocab = set(filtered_tokens)
    removed_words = raw_vocab - filtered_vocab
    
    original_vocab_size = len(raw_vocab)
    filtered_vocab_size = len(filtered_vocab)
    vocab_reduction = original_vocab_size - filtered_vocab_size
    vocab_reduction_pct = (vocab_reduction / original_vocab_size * 100) if original_vocab_size > 0 else 0
    
    total_tokens_original = len(raw_tokens)
    total_tokens_filtered = len(filtered_tokens)
    tokens_removed = total_tokens_original - total_tokens_filtered
    
    return {
        'original_vocab_size': original_vocab_size,
        'filtered_vocab_size': filtered_vocab_size,
        'vocab_reduction': vocab_reduction,
        'vocab_reduction_pct': vocab_reduction_pct,
        'total_tokens_original': total_tokens_original,
        'total_tokens_filtered': total_tokens_filtered,
        'tokens_removed': tokens_removed,
        'removed_word_count': len(removed_words),
        'removed_words': removed_words
    }


if __name__ == "__main__":

    preprocessor = HindiPreprocessor()
    
    #Normalization demonstration

    print("NORMALIZATION EXAMPLES")
    
    #Diacritic variants test
    text1_precomposed = "आप कैसे हैं"  # precomposed
    text1_normalized = preprocessor.preprocess_text(text1_precomposed)
    print(f"Original: {text1_precomposed}")
    print(f"Normalized: {text1_normalized}")
    print(f"Same after NFC? {text1_precomposed == text1_normalized}")
    
    #ZWJ/ZWNJ removal test
    text_with_zwj = "राष्ट्र"  # may contain ZWJ for proper ligature
    text_without_zwj = preprocessor.remove_zwj_zwnj(text_with_zwj)
    print(f"\nWith ZWJ: {text_with_zwj}")
    print(f"Without ZWJ: {text_without_zwj}")
    
    #Nasalization normalization
    text_nasalized = "हँ"  # chandrabindu (archaic)
    text_nasalized_normalized = preprocessor.normalize_nasalization(text_nasalized)
    print(f"\nNasalized (chandrabindu): {text_nasalized}")
    print(f"Normalized (anusvara): {text_nasalized_normalized}")
    
    print("\n")
    print("TOKENIZATION EXAMPLE")
    
    sample_text = "नमस्ते! आप कैसे हैं? मैं ठीक हूँ।"
    tokens = preprocessor.tokenize(sample_text)
    print(f"Text: {sample_text}")
    print(f"Tokens: {tokens}")
    
    print("\n")
    print("STEMMING EXAMPLES")
    
    test_words = [
        'किताबों',     
        'बोलना',     
        'बोलती',     
        'सुंदर',     
        'सुंदरी',     
        'दिल',      
        'दिलों',      
        'जाना',      
        'जाता',      
        'भारत',     
    ]
    
    print("Word -> Stemmed Root")
    for word in test_words:
        stemmed = preprocessor.stem_word(word)
        print(f"{word:15} -> {stemmed}")




