import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import unittest
from transliterate import transliterate

class TestTransliteration(unittest.TestCase):

    def test_basic_consonants_and_matras(self):
        # In strict ITRANS, add 'a' to prevent trailing halants
        results = transliterate("kitaaba")
        self.assertIn("किताब", results)

        results = transliterate("dillii")
        self.assertIn("दिल्ली", results)

    def test_conjuncts_and_halants(self):
        results = transliterate("namaste")
        self.assertIn("नमस्ते", results)

        results = transliterate("shakti")
        self.assertIn("शक्ति", results)

    def test_independent_vowels(self):
        # Strict ITRANS requires 'a' to prevent halant on the 'm' and 'k'
        results = transliterate("aama")
        self.assertIn("आम", results)

        results = transliterate("eka")
        self.assertIn("एक", results)

    def test_itrans_vs_hk_schemes(self):
        # Appending 'a' prevents the halant 'त्'
        itrans_style = "bhaarata"
        results_itrans = transliterate(itrans_style)
        self.assertIn("भारत", results_itrans)

        hk_style = "bhArata"
        results_hk = transliterate(hk_style)
        self.assertIn("भारत", results_hk)

    def test_unknown_characters(self):
        # The library natively converts numbers to Devanagari numerals
        results = transliterate("namaste 123!")
        self.assertIn("नमस्ते १२३!", results)

    def test_deduplication(self):
        results = transliterate("namaste")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], "नमस्ते")

if __name__ == '__main__':
    unittest.main(verbosity=2)