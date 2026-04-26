from typing import Dict, Set, Iterable, List, Optional
import re 

class KGramIndex:
    def __init__(self, k: int = 2):
        self.k = k
        self.gram_index: Dict[str, Set[str]] = {}
        self.terms: Set[str] = set()

    def _pad(self, term: str) -> str:
        return f"^{term}$" 

    def _generate_grams(self, text: str) -> Set[str]:
        #Generates set of k-grams for a given string
        k = self.k
        grams = set()
        if len(text) < k:
            return grams
        for i in range(len(text) - k + 1):
            grams.add(text[i:i + k])
        return grams

    def add_term(self, term: str):
        if term in self.terms:
            return
        self.terms.add(term)
        for g in self._generate_grams(self._pad(term)):
            self.gram_index.setdefault(g, set()).add(term)

    def add_terms(self, terms: Iterable[str]):
        for t in terms:
            self.add_term(t)

    def expand_wildcard(self, wildcard: str, max_candidates: Optional[int] = 5000) -> List[str]:
        wildcard = wildcard.strip()
        if not wildcard:
            return []

        padded_query = self._pad(wildcard)
        substrings = padded_query.split('*')
        candidate_terms: Optional[Set[str]] = None

        for sub in substrings:
            grams = self._generate_grams(sub)
            
            for g in grams:
                terms_for_g = self.gram_index.get(g, set())
                
                if candidate_terms is None:
                    candidate_terms = set(terms_for_g)
                else:
                    candidate_terms &= terms_for_g

                if not candidate_terms:
                    break
            
            if candidate_terms is not None and not candidate_terms:
                break

        if candidate_terms is None:
            candidate_terms = set()
            for t in self.terms:
                candidate_terms.add(t)
                if max_candidates and len(candidate_terms) > max_candidates:
                    break

        esc = re.escape(wildcard)
        pattern = '^' + esc.replace('\\*', '.*') + '$'
        prog = re.compile(pattern)
    
        results = [t for t in candidate_terms if prog.match(t)]
        results.sort()
        return results

    def stats(self) -> Dict[str, int]:
        return {
            'k': self.k,
            'vocab_size': len(self.terms),
            'num_grams': len(self.gram_index)
        }

if __name__ == '__main__':
    sample_terms = [
        'भारत', 'भारतीय', 'भोजन', 'भविष्य', 'किताब', 'किताबें', 'किताबों',
        'किताबघर', 'दिल्ली', 'देवनागरी', 'नमस्ते'
    ]

    idx = KGramIndex(k=2)
    idx.add_terms(sample_terms)
    print('Stats:', idx.stats())

    # *दी will correctly match ^दि and ली$ if k=2!
    queries = ['किताब*', '*ति*', 'भ*त', '*दी', 'भा*']
    for q in queries:
        print(f"Query: {q:10} -> {idx.expand_wildcard(q)}")