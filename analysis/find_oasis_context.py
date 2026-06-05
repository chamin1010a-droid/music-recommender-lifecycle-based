import os
import glob

log_dir = r"C:\Users\user\.gemini\antigravity\brain\9f6d65ce-342d-4c61-a943-572f5bfa8d79\.system_generated\logs"
files = glob.glob(os.path.join(log_dir, 'overview.txt'))

for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if 'Oasis' in line or '오아시스' in line:
                start = max(0, i-10)
                end = min(len(lines), i+10)
                print(f"\n--- Context at line {i} ---")
                print("".join(lines[start:end]))
