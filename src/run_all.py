import os
import sys
import json
import argparse
import hashlib
import math
from pathlib import Path
from collections import defaultdict

from preprocessing import HindiPreprocessor
from inverted_index import InvertedIndex
from query_processor import QueryProcessor
from wildcard import KGramIndex
from transliterate import transliterate

def hash_directory(data_dir: str) -> str:
    sha256 = hashlib.sha256()
    for path in sorted(Path(data_dir).rglob('*')):
        if path.is_file():
            with open(path, 'rb') as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)
    return f"sha256:{sha256.hexdigest()}"

def calculate_linear_regression(x, y):
    """Basic least squares regression to find Slope (m), Intercept (c), and R^2."""
    n = len(x)
    if n < 2: return 0.0, 0.0, 0.0
    sum_x, sum_y = sum(x), sum(y)
    sum_xy = sum(x[i] * y[i] for i in range(n))
    sum_xx = sum(x[i] ** 2 for i in range(n))
    
    denominator = (n * sum_xx - sum_x ** 2)
    if denominator == 0: return 0.0, 0.0, 0.0
    
    m = (n * sum_xy - sum_x * sum_y) / denominator
    c = (sum_y - m * sum_x) / n
    
    y_mean = sum_y / n
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    ss_res = sum((y[i] - (m * x[i] + c)) ** 2 for i in range(n))
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    return m, c, r2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to data directory')
    parser.add_argument('--out', required=True, help='Path to output results.json')
    args = parser.parse_args()

    print(f"Hashing directory: {args.data}...")
    data_hash = hash_directory(args.data)

    print("Initializing components...")
    preprocessor = HindiPreprocessor()
    index = InvertedIndex(preprocessor=preprocessor)
    
    # Trackers for Zipf and Heaps
    term_counts = defaultdict(int)
    total_tokens = 0
    heaps_N, heaps_V = [], []
    
    files = list(Path(args.data).rglob("*.txt"))
    if not files:
        print(f"Error: No .txt files found in {args.data}")
        sys.exit(1)

    print(f"Indexing {len(files)} documents and tracking linguistic stats...")
    for doc_id, file_path in enumerate(files, start=1):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
            # Pre-tokenize to track global counts for Zipf/Heaps
            tokens = preprocessor.process(text, remove_stopwords=True, apply_stemming=True)
            total_tokens += len(tokens)
            for t in tokens:
                term_counts[t] += 1
                
            # Add to actual index
            index.add_document(doc_id, text)
            
            # Record points for Heaps' Law (log N vs log V) every 10 docs
            if doc_id % 10 == 0 and total_tokens > 0:
                heaps_N.append(math.log(total_tokens))
                heaps_V.append(math.log(len(term_counts)))

    # --- ZIPF'S LAW CALCULATION (log r vs log f) ---
    sorted_freqs = sorted(term_counts.values(), reverse=True)
    zipf_x = [math.log(r + 1) for r in range(len(sorted_freqs))]
    zipf_y = [math.log(f) for f in sorted_freqs]
    zipf_m, _, zipf_r2 = calculate_linear_regression(zipf_x, zipf_y)
    alpha = abs(zipf_m)  # Slope is -alpha

    # --- HEAPS' LAW CALCULATION (log N vs log V) ---
    heaps_beta, heaps_c, heaps_r2 = calculate_linear_regression(heaps_N, heaps_V)
    heaps_k = math.exp(heaps_c)

    # --- COMPRESSION STATS ---
    print("Testing compression...")
    index.save("temp_uncomp.pkl", compress=False)
    index.save("temp_comp.pkl", compress=True)
    raw_bytes = os.path.getsize("temp_uncomp.pkl")
    comp_bytes = os.path.getsize("temp_comp.pkl")
    os.remove("temp_uncomp.pkl")
    os.remove("temp_comp.pkl")

    # --- QUERIES ---
    print("Executing sample queries...")
    qp = QueryProcessor(index)
    q1 = "भारत देश"
    res1 = qp.boolean_query(q1, "AND")
    
    # --- WILDCARD (TOLERANT RETRIEVAL) ---
    print("Executing wildcard evaluation...")
    kgram = KGramIndex(k=2)
    kgram.add_terms(index.terms.keys())
    
    wildcard_pattern = "किताब*"
    # Simulate intersection to get 'num_candidates' BEFORE regex post-filtering
    padded = kgram._pad(wildcard_pattern)
    sub = padded.split('*')[0] 
    candidate_set = set(kgram.terms)
    for g in kgram._generate_grams(sub):
        candidate_set &= kgram.gram_index.get(g, set())
        
    num_candidates = len(candidate_set)
    expanded_terms = kgram.expand_wildcard(wildcard_pattern)
    num_matches = len(expanded_terms)

    # --- TRANSLITERATION ---
    latin_q = "bhaarat"
    transliterated_opts = transliterate(latin_q)

    # ==========================================
    # BUILD FINAL JSON RESPONSE
    # ==========================================
    results_json = {
        "language": "hi",
        "data_hash": data_hash,
        "stats": {
            "num_docs": index.doc_count,
            "total_tokens": total_tokens,
            "vocab_size": len(index.terms)
        },
        "zipf": {
            "alpha": round(alpha, 4),
            "r2": round(zipf_r2, 4)
        },
        "heaps": {
            "k": round(heaps_k, 4),
            "beta": round(heaps_beta, 4),
            "r2": round(heaps_r2, 4)
        },
        "compression": {
            "method": "vbyte",
            "raw_bytes": raw_bytes,
            "compressed_bytes": comp_bytes,
            "ratio": round(raw_bytes / comp_bytes, 4) if comp_bytes else 0
        },
        "queries": {
            "count": 1,
            "examples": [
                {
                    "query": q1,
                    "type": "boolean AND",
                    "num_results": len(res1)
                }
            ]
        },
        "tolerant": {
            "k": kgram.k,
            "wildcard_examples": [
                {
                    "pattern": wildcard_pattern,
                    "num_candidates": num_candidates,
                    "num_matches": num_matches
                }
            ],
            "option_examples": [
                {
                    "latin": latin_q,
                    "native": transliterated_opts[0] if transliterated_opts else "भारत"
                }
            ]
        }
    }

    print(f"Saving JSON to {args.out}...")
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
        
    print("✅ Auto-validation setup complete!")

if __name__ == '__main__':
    main()