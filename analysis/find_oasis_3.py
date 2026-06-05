import os
import glob
import re

log_dir = r"C:\Users\user\.gemini\antigravity\brain"
files = glob.glob(os.path.join(log_dir, '**', '*.txt'), recursive=True) + \
        glob.glob(os.path.join(log_dir, '**', '*.md'), recursive=True)

found_matches = []

for file in files:
    try:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Split into paragraphs or blocks to find co-occurrences of oasis and 3
            blocks = content.split('\n\n')
            for i, block in enumerate(blocks):
                lower_block = block.lower()
                if ('oasis' in lower_block or '오아시스' in lower_block) and '3' in lower_block:
                    found_matches.append(f"[{file}] Block {i}:\n{block}\n{'-'*40}")
    except Exception as e:
        pass

if not found_matches:
    print("NO MATCHES FOUND IN TEXT FILES.")
else:
    for match in found_matches:
        print(match)
