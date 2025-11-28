import re

def find_target_lines(ocr_results, pattern='_1_'):
    
    matches = []
    for entry in ocr_results:
        text = entry.get('text', '')
        # direct match
        if pattern in text:
            matches.append(entry)
            continue
        # fallback: find token with 1 and underscores/digits nearby
        tokens = re.split(r'\s+', text)
        for t in tokens:
            if '_1_' in t or re.search(r'\d+_1_\w*', t) or re.search(r'_1_\w*', t):
                matches.append(entry)
                break
        else:
            # last fallback: any token that has the substring '_1' or '1_'
            for t in tokens:
                if '_1' in t or '1_' in t:
                    matches.append(entry)
                    break
    return matches

def pick_best_match(matches):
    if not matches:
        return None
    # choose highest confidence
    matches_sorted = sorted(matches, key=lambda x: x.get('confidence', 0), reverse=True)
    return matches_sorted[0]
