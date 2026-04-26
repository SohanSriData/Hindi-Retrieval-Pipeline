# Hindi Tolerant Retrieval (IR_asgn)

This repository contains a small information retrieval preprocessing and query toolkit focused on Hindi (Devanagari) text. It includes tokenization, Unicode normalization, zero-width character handling, diacritic normalization, stopword removal, a rule-based stemmer, a k-gram wildcard expansion, transliteration support, and analysis notebooks for corpus statistics.

**Quick Start**

- Activate the project's virtual environment (if present):

  ```bash
  source bin/activate
  ```

- Run unit tests:

  ```bash
  python -m unittest discover -s tests -v
  ```

- Open the corpus analysis notebook:

  ```bash
  jupyter notebook corpus_stats.ipynb
  ```

**Project Structure**

- `src/` : Core modules
  - `preprocessing.py` : `HindiPreprocessor` class (tokenize, normalize, remove stopwords, stem)
  - `inverted_index.py` : Simple in-memory inverted index with save/load
  - `transliterate.py` : Latin → Devanagari transliteration (uses `indic_transliteration`)
  - `wildcard.py`, `phonetic.py`, `query_processor.py`, etc. : Tolerant retrieval features
- `data/hindi/` : Hindi text corpus and `stopwords_hindi.txt`
- `tests/` : Unit tests covering preprocessing, transliteration, and index
- `corpus_stats.ipynb` : Notebook for corpus-level analysis (Zipf's and Heaps' laws)

**Notes & Requirements**

- `src/transliterate.py` expects the `indic_transliteration` package for full-featured transliteration. Install it with:

  ```bash
  pip install indic-transliteration
  ```

  Without this package some test setups may fail; ensure the environment has it if you need exact transliteration behavior.

- Stopwords file used by the notebook is `data/hindi/stopwords_hindi.txt`.

**Common Commands**

- Run a single test file:

  ```bash
  python -m unittest tests.test_preprocessing -v
  ```

- Run the transliteration script (quick check):

  ```bash
  python src/transliterate.py
  ```

**Examples (Python REPL)**

```python
from src.preprocessing import HindiPreprocessor
pre = HindiPreprocessor(stopwords_path='data/hindi/stopwords_hindi.txt')
text = "यह किताब है और यह बहुत अच्छी है।"
print(pre.process(text, remove_stopwords=False))
print(pre.process(text, remove_stopwords=True))

from src.transliterate import transliterate
print(transliterate('bharat'))
```
