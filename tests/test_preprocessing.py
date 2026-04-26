"""
Unit tests for Hindi preprocessing module
Tests tokenization, normalization, stopword removal, and stemming
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import unittest
from preprocessing import HindiPreprocessor

class TestHindiTokenization(unittest.TestCase):
    
    def setUp(self):
        self.preprocessor = HindiPreprocessor()
    
    def test_danda_and_punctuation_removal(self):
        # Testing if punctuation gets mixed with the tokens
        text = "मुझे किताब दी। (क्या यह सच है?) — हाँ!"
        tokens = self.preprocessor.tokenize(text)
        self.assertNotIn("दी।", tokens)
        self.assertIn("दी", tokens)
        self.assertNotIn("(क्या", tokens)
        self.assertIn("क्या", tokens)
        
    def test_hindi_numerals_preserved(self):
        text = "वर्ष २०२४ में ५०० पृष्ठ पढ़े।"
        tokens = self.preprocessor.tokenize(text)
        self.assertIn("२०२४", tokens)
        self.assertIn("५००", tokens)

    def test_empty_string(self):
        tokens = self.preprocessor.tokenize("")
        self.assertEqual(tokens, [])


class TestHindiNormalization(unittest.TestCase):
    
    def setUp(self):
        self.preprocessor = HindiPreprocessor()
    
    def test_zwj_zwnj_removal(self):
        text_with_zwj = "र\u200Dष्ट्र"  
        text_with_zwnj = "क\u200Cष" 
        self.assertNotIn('\u200D', self.preprocessor.remove_zwj_zwnj(text_with_zwj))
        self.assertNotIn('\u200C', self.preprocessor.remove_zwj_zwnj(text_with_zwnj))
    
    def test_diacritic_normalization_precomposed_nukta(self):
        # Testing replacements dict (e.g., U+0958 to U+0915)
        text_qa = "\u0958िताब" # क़िताब using precomposed character
        normalized = self.preprocessor.normalize_diacritics(text_qa)
        self.assertEqual(normalized, "\u0915िताब") # किताब with base Ka
        
        text_za = "\u095Bमीन" # ज़मीन 
        normalized = self.preprocessor.normalize_diacritics(text_za)
        self.assertEqual(normalized, "\u091Cमीन") # जमीन
    
    def test_diacritic_normalization_decomposed_nukta(self):
        # Testing your nukta_pairs dict (e.g., U+0915 + U+093C to U+0915)
        text_decomposed_qa = "\u0915\u093Cिताब" # क + ़ + िताब
        normalized = self.preprocessor.normalize_diacritics(text_decomposed_qa)
        self.assertEqual(normalized, "\u0915िताब") # किताब

    def test_nasalization_normalization(self):
        # Testing chandrabindu to anusvara mappings
        text_devanagari_chandra = "कहाँ" # U+0901
        self.assertEqual(self.preprocessor.normalize_nasalization(text_devanagari_chandra), "कहां") # U+0902
        
        # Testing Bengali chandrabindu (U+0981)
        text_bengali_chandra = "ह\u0981" 
        self.assertEqual(self.preprocessor.normalize_nasalization(text_bengali_chandra), "हं")


class TestStopwordRemoval(unittest.TestCase):
    
    def setUp(self):
        # Using a dummy path, mocking the behavior to test logic without needing the file
        self.preprocessor = HindiPreprocessor()
        self.preprocessor.stopwords = {'का', 'के', 'की', 'है', 'हैं', 'यह', 'वह'}
    
    def test_stopword_removal(self):
        tokens = ['यह', 'किताब', 'मेरी', 'है']
        filtered = self.preprocessor.remove_stopwords(tokens)
        self.assertEqual(filtered, ['किताब', 'मेरी'])
        self.assertNotIn('यह', filtered)
        self.assertNotIn('है', filtered)


class TestHindiStemming(unittest.TestCase):
    
    def setUp(self):
        self.preprocessor = HindiPreprocessor()

    def test_minimum_length_enforced(self):
        # Stems should not reduce a word to less than 2 characters
        self.assertEqual(self.preprocessor.stem_word('था'), 'था')
        self.assertEqual(self.preprocessor.stem_word('जा'), 'जा')
        
    def test_longest_match_first(self):
        # 'ियां' (4 chars) should be stripped before 'ं' (1 char)
        self.assertEqual(self.preprocessor.stem_word('लड़कियां'), 'लड़क')
        self.assertEqual(self.preprocessor.stem_word('कविताओं'), 'कविता')
        
    def test_verb_suffixes(self):
        # Testing the verb endings in your flat array
        self.assertEqual(self.preprocessor.stem_word('जाएगा'), 'जा')
        self.assertEqual(self.preprocessor.stem_word('जाकर'), 'जा')
        self.assertEqual(self.preprocessor.stem_word('करना'), 'कर')

    def test_50_sample_words_requirement(self):
        # 50 sample words
        test_words = {
            
            # Plurals and Gender
            'लड़कियों': 'लड़क', 'लड़कियां': 'लड़क', 'लड़की': 'लड़क', 'लड़का': 'लड़क', 'लड़के': 'लड़क',
            'किताबों': 'किताब', 'किताबें': 'किताब', 'किताब': 'किताब',
            'कविताओं': 'कविता', 'कविताएं': 'कविता', 'कविता': 'कविता', # <-- Fix is here
            'जातियों': 'जात', 'जातियां': 'जात', 'जाति': 'जाति',
            
            # Verbs (Root: कर)
            'करना': 'कर', 'करता': 'कर', 'करती': 'कर', 'करते': 'कर', 'करेगा': 'कर', 'करके': 'करक', 'करकर': 'कर',
            
            # Verbs (Root: जा)
            'जाना': 'जा', 'जाता': 'जा', 'जाती': 'जा', 'जाते': 'जा', 'जाएगा': 'जा', 'जाकर': 'जा',
            
            # Verbs (Root: खा)
            'खाना': 'खा', 'खाता': 'खा', 'खाती': 'खा', 'खाते': 'खा', 'खाएगा': 'खा', 'खाकर': 'खा',
            
            # Verbs (Root: पढ़)
            'पढ़ना': 'पढ़', 'पढ़ता': 'पढ़', 'पढ़ती': 'पढ़', 'पढ़ते': 'पढ़', 'पढ़ेगा': 'पढ़', 'पढ़ाई': 'पढ़', 'पढ़कर': 'पढ़',
            
            # Verbs (Root: लिख)
            'लिखना': 'लिख', 'लिखता': 'लिख', 'लिखती': 'लिख', 'लिखते': 'लिख', 'लिखेगा': 'लिख', 'लिखाई': 'लिख', 'लिखकर': 'लिख',
            
            # Adjectives
            'अच्छा': 'अच्छ', 'अच्छी': 'अच्छ', 'अच्छे': 'अच्छ'
        }
        
        success_count = 0
        errors = []
        for word, expected in test_words.items():
            stemmed = self.preprocessor.stem_word(word)
            if stemmed == expected:
                success_count += 1
            else:
                errors.append(f"Stemming Error: '{word}' -> expected '{expected}', got '{stemmed}'")
                
        # Measure and document stemming errors.
        print("\n--- Documented Stemming Errors ---")
        for error in errors:
            print(error)
            
        self.assertEqual(len(test_words), 50)
        # Verify the stemmer hits a reasonable accuracy  
        self.assertGreaterEqual(success_count, 35) 


class TestFullPipeline(unittest.TestCase):
    
    def setUp(self):
        self.preprocessor = HindiPreprocessor()
        self.preprocessor.stopwords = {'हैं', 'और'}
    
    def test_pipeline_integration(self):
        text = "लड़कियां अच्छी क़िताबें पढ़ती हैं।"
        tokens = self.preprocessor.process(text, remove_stopwords=True, apply_stemming=True)
        
        # Expected workflow: 
        # 1. 'क़िताबें' normalized to 'किताबें'
        # 2. 'हैं' removed by stopwords
        # 3. 'लड़कियां' -> 'लड़क', 'अच्छी' -> 'अच्छ', 'किताबें' -> 'किताब', 'पढ़ती' -> 'पढ़'
        self.assertIn('लड़क', tokens)
        self.assertIn('किताब', tokens)
        self.assertIn('पढ़', tokens)
        self.assertNotIn('हैं', tokens)

if __name__ == '__main__':
    unittest.main(verbosity=2)