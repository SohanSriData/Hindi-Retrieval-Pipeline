from typing import List, Dict
from indic_transliteration import sanscript

def transliterate(text: str) -> List[str]:
    """Returns candidate mappings using both ITRANS and HK schemes."""
    text = text.strip()
    if not text:
        return []

    candidates = []

    candidates.append(sanscript.transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI))
    candidates.append(sanscript.transliterate(text, sanscript.HK, sanscript.DEVANAGARI))
    
    return list(dict.fromkeys(candidates))

if __name__ == '__main__':
    # Test cases with difference between ITRANS and HK
    queries = ['bharat', 'bhaarat', 'bhArat', 'namaste', 'shakti', 'zakti']
    for q in queries:
        print(f"{q:10} -> {transliterate(q)}")