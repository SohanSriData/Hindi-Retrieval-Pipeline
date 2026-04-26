import os
import time
from pathlib import Path
from preprocessing import HindiPreprocessor
from inverted_index import InvertedIndex
from query_processor import QueryProcessor
from wildcard import KGramIndex
from transliterate import transliterate

def generate_full_assignment_report(data_dir: str, num_docs: int = 500):
    print("\n" + "="*65)
    print(" 🚀 CS F469: INFORMATION RETRIEVAL - FINAL REPORT STATS")
    print("="*65)

    # --- SETUP ---
    preprocessor = HindiPreprocessor(stopwords_path='../data/stopwords_hi.txt')
    index = InvertedIndex(preprocessor=preprocessor)
    data_path = Path(data_dir)
    files = list(data_path.rglob("*.txt"))
    
    if not files:
        print(f"Error: No .txt files found in {data_dir}. Please check your path!")
        return

    # INDEXING STATISTICS
    print("\n[1] INDEXING STATISTICS")
    print("-" * 65)
    start_time = time.time()
    for doc_id, file_path in enumerate(files[:num_docs], start=1):
        with open(file_path, 'r', encoding='utf-8') as f:
            index.add_document(doc_id, f.read())
    idx_time = time.time() - start_time
    
    print(f"Documents Indexed:        {index.doc_count}")
    print(f"Total Unique Vocabulary:  {len(index.terms)} terms")
    print(f"Total Indexing Time:      {idx_time:.2f} seconds")

    # COMPRESSION METRICS (GAP + VBYTE)
    print("\n[2] COMPRESSION STATISTICS (Gap + VByte Encoding)")
    print("-" * 65)
    uncompressed_file = "report_uncompressed.pkl"
    compressed_file = "report_compressed.pkl"
    
    index.save(uncompressed_file, compress=False)
    index.save(compressed_file, compress=True)
    
    uncomp_size = os.path.getsize(uncompressed_file)
    comp_size = os.path.getsize(compressed_file)
    ratio = uncomp_size / comp_size
    savings = (1 - (comp_size / uncomp_size)) * 100
    
    print(f"Uncompressed Size:        {uncomp_size / 1024:.2f} KB")
    print(f"Compressed Size:          {comp_size / 1024:.2f} KB")
    print(f"Compression Ratio:        {ratio:.2f} : 1")
    print(f"Space Savings:            {savings:.2f}%")
    
    os.remove(uncompressed_file)
    os.remove(compressed_file)

    # QUERY PROCESSING & LATENCY
    print("\n[3] QUERY PROCESSING STATISTICS")
    print("-" * 65)
    qp = QueryProcessor(index)
    
    q_bool = "भारत AND देश"
    start_t = time.time()
    res_bool = qp.boolean_query(q_bool, "AND")
    t_bool = time.time() - start_t
    print(f"Boolean Query:            '{q_bool}'")
    print(f"Matches Found:            {len(res_bool)} documents")
    print(f"Execution Time (Latency): {t_bool*1000:.2f} ms")

    q_phrase = "भारतीय संस्कृति" # Change this to a phrase common in your dataset
    start_t = time.time()
    res_phrase = qp.phrase_query(q_phrase)
    t_phrase = time.time() - start_t
    print(f"Phrase Query:             '{q_phrase}'")
    print(f"Matches Found:            {len(res_phrase)} documents")
    print(f"Execution Time (Latency): {t_phrase*1000:.2f} ms")

    # K-GRAM & WILDCARD QUERY EXPANSION
    print("\n[4] K-GRAM WILDCARD STATISTICS")
    print("-" * 65)
    kgram = KGramIndex(k=2)
    kgram.add_terms(index.terms.keys())
    k_stats = kgram.stats()
    
    print(f"K-Gram Size (k):          {k_stats['k']} (Bigrams)")
    print(f"Total Grams Generated:    {k_stats['num_grams']}")
    
    wildcard_q = "किताब*"
    expanded = kgram.expand_wildcard(wildcard_q)
    print(f"Wildcard Query:           '{wildcard_q}'")
    print(f"Total Expansions Found:   {len(expanded)}")
    print(f"Expanded Terms (Sample):  {expanded[:5] if expanded else 'None'}")

    # TRANSLITERATION MAPPINGS (ITRANS vs HK)
    print("\n[5] TRANSLITERATION (LATIN -> DEVANAGARI)")
    print("-" * 65)
    latin_words = ["bharat", "dillii", "shakti", "namaste"]
    for word in latin_words:
        candidates = transliterate(word)
        print(f"Latin Input: '{word:8}' -> Candidates: {candidates}")
        
    print("\n" + "="*65)
    print(" ✅ All statistics generated successfully for the PDF report!")
    print("="*65 + "\n")

if __name__ == '__main__':
    generate_full_assignment_report(data_dir="data/hindi", num_docs=1500)