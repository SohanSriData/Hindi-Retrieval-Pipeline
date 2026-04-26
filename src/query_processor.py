from typing import List, Set
from inverted_index import InvertedIndex

class QueryProcessor:

    def __init__(self, index: InvertedIndex):
        self.index = index

    def preprocess_query(self, query_string: str) -> List[str]:    
        return self.index.preprocessor.process(
            query_string, 
            remove_stopwords=True, 
            apply_stemming=True
        )

    def _term_docs(self, term: str) -> Set[int]:
        return set(self.index.get_postings(term).keys())

    def boolean_query(self, query_string: str, operator: str = 'AND') -> Set[int]:
        """Processes AND, OR, NOT boolean queries."""
        terms = self.preprocess_query(query_string)
        if not terms:
            return set()

        op = operator.upper()
        
        # Process in order of increasing frequency (DF)
        sets = []
        for t in terms:
            sets.append(self._term_docs(t))
        sets.sort(key=len) 
        
        if op == 'AND':
            res = sets[0].copy()
            for s in sets[1:]:
                res &= s
                if not res: 
                    break
            return res
            
        elif op == 'OR':
            res = set()
            for s in sets:
                res |= s
            return res
            
        elif op == 'NOT':
            res = sets[0].copy()
            for s in sets[1:]:
                res -= s
            return res
            
        else:
            raise ValueError(f"Unknown operator: {operator}")

    def phrase_query(self, phrase_string: str) -> Set[int]:
        """Processes exact phrase queries using positional offsets."""
        terms = self.preprocess_query(phrase_string)
        if not terms:
            return set()
        if len(terms) == 1:
            return self._term_docs(terms[0])

        postings_lists = []
        for t in terms:
            postings_lists.append((t, self.index.get_postings(t)))
        postings_lists.sort(key=lambda x: len(x[1])) 

        # Intersect docs starting from the rarest term
        candidate_docs = set(postings_lists[0][1].keys())
        for _, p in postings_lists[1:]:
            candidate_docs &= set(p.keys())
            # Early exit if no documents contain all terms
            if not candidate_docs: 
                return set()

        # Positional Check for surviving docs
        # Use the original order of 'terms' to check offsets
        original_postings = []
        for t in terms:
            original_postings.append(self.index.get_postings(t))
        results = set()
        
        for doc in candidate_docs:
            positions_per_term = []
            for p in original_postings:
                positions_per_term.append(p[doc])
            first_positions = positions_per_term[0]
            
            posset_list = []
            for lst in positions_per_term[1:]:
                posset_list.append(set(lst))
            
            for p0 in first_positions:
                match = True
                for offset, posset in enumerate(posset_list, start=1):
                    if (p0 + offset) not in posset:
                        match = False
                        break
                
                # If we find the full phrase at this starting position, save doc
                if match:
                    results.add(doc)
                    break

        return results
    

    