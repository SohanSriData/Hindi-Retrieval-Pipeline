import pickle
from typing import Dict, List, Any
from pathlib import Path
from collections import defaultdict
from preprocessing import HindiPreprocessor
import compression

class InvertedIndex:

    def __init__(self, preprocessor: HindiPreprocessor = None):
        self.terms: Dict[str, Dict[str, Any]] = {}
        self.preprocessor = preprocessor
        self.doc_count = 0

    def add_document(self, doc_id: int, text: str):
        #Adds a document to the index
        doc_id = int(doc_id)
        self.doc_count += 1

        tokens = self.preprocessor.process(text, remove_stopwords=True, apply_stemming=True)

        # Track the positions of each token in doc
        positions = defaultdict(list)
        for position, token in enumerate(tokens):
            positions[token].append(position)

        # Update inverted index
        for token, position_list in positions.items():
            if token not in self.terms:
                self.terms[token] = {'df': 0, 'postings': {}}

            self.terms[token]['postings'][doc_id] = position_list
            self.terms[token]['df'] = len(self.terms[token]['postings'])

    def get_postings(self, term: str) -> Dict[int, List[int]]:
        return self.terms.get(term, {}).get('postings', {})

    def save(self, path: str, compress: bool = False):
        p = Path(path)
        payload = {
            'doc_count': self.doc_count,
            'compressed': compress,
            'terms': {}
        }

        for term, data in self.terms.items():
            if compress:
                sorted_doc_ids = sorted(data['postings'].keys())
                
                payload['terms'][term] = (
                    data['df'], 
                    compression.compress_docids(sorted_doc_ids), 
                    [data['postings'][d] for d in sorted_doc_ids]
                )
            else:
                payload['terms'][term] = data

        with p.open('wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str, preprocessor: HindiPreprocessor = None):
        """Loads the index from disk."""
        p = Path(path)
        
        # Open file in bin read mode
        with p.open('rb') as f:
            payload = pickle.load(f)

        inst = cls(preprocessor)
        inst.doc_count = payload.get('doc_count', 0)
        is_compressed = payload.get('compressed', False)

        for term, data in payload['terms'].items():
            if is_compressed:
                # Unpack 
                df, doc_bytes, pos_lists = data
                
                doc_ids = compression.decompress_docids(doc_bytes)
                postings = {d_id: pos for d_id, pos in zip(doc_ids, pos_lists)}
                
                inst.terms[term] = {'df': df, 'postings': postings}
            else:
                inst.terms[term] = data

        return inst


if __name__ == '__main__':
    print('InvertedIndex module optimized and ready.')