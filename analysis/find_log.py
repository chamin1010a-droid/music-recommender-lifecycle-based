import os
import glob

log_dir = r"C:\Users\user\.gemini\antigravity\brain"
files = glob.glob(os.path.join(log_dir, '**', 'overview.txt'), recursive=True)

for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if '오아시스' in line or 'Oasis' in line:
                # check surrounding lines for '3'
                start = max(0, i-5)
                end = min(len(lines), i+6)
                surrounding = "".join(lines[start:end])
                if '3' in surrounding:
                    print(f"[{file}] Line {i}")
                    print(surrounding)
                    print("-" * 40)
