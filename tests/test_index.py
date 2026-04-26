import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# ... your existing imports below ...
# import unittest
# from preprocessing import HindiPreprocessor
# etc.
import tempfile
from inverted_index import InvertedIndex
from preprocessing import HindiPreprocessor

class TestInvertedIndex(unittest.TestCase):

    def setUp(self):
        """Runs before every test to set up a fresh index with dummy data."""
        self.preprocessor = HindiPreprocessor()
        self.index = InvertedIndex(self.preprocessor)
        
        self.index.add_document(1, "भारत एक महान देश है।")
        self.index.add_document(5, "दिल्ली भारत की राजधानी है।") 
        self.index.add_document(12, "मुझे अपना देश बहुत पसंद है।")

    def test_df_and_postings_logic(self):
        """Tests if the index correctly calculates Document Frequency (DF) and stores positions."""
        bharat_token = self.preprocessor.process("भारत", apply_stemming=True)[0]
        desh_token = self.preprocessor.process("देश", apply_stemming=True)[0]

        self.assertEqual(self.index.terms[bharat_token]['df'], 2)
        
        self.assertEqual(self.index.terms[desh_token]['df'], 2)

        postings = self.index.get_postings(bharat_token)
        self.assertIn(1, postings)
        self.assertIn(5, postings)
        self.assertNotIn(12, postings) # Doc 12 doesn't have 'भारत'

    def test_compressed_save_and_load(self):
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            temp_path = tmp.name

        try:
            # Save the index to disk with compression ON
            self.index.save(temp_path, compress=True)
            
            # Load it back from the disk into a brand new instance
            loaded_index = InvertedIndex.load(temp_path, preprocessor=self.preprocessor)
            
            # Verify the metadata survived
            self.assertEqual(loaded_index.doc_count, 3)
            
            # Verify the VByte byte-streams unpacked perfectly into the exact same dictionary!
            bharat_token = self.preprocessor.process("भारत", apply_stemming=True)[0]
            
            original_postings = self.index.get_postings(bharat_token)
            loaded_postings = loaded_index.get_postings(bharat_token)
            
            self.assertEqual(original_postings, loaded_postings)

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == '__main__':
    unittest.main(verbosity=2)