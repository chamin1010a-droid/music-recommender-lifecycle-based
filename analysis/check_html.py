import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
html_path = os.path.join(BASE_DIR, 'Takeout', 'YouTube 및 YouTube Music', '시청 기록', '시청 기록.html')

oasis_count = 0
music_count = 0

with open(html_path, 'r', encoding='utf-8') as f:
    for line in f:
        lower_line = line.lower()
        if 'oasis' in lower_line or '오아시스' in lower_line:
            oasis_count += 1
        if 'youtube music' in lower_line:
            music_count += 1

print(f"Oasis mentions in raw HTML: {oasis_count}")
print(f"YouTube Music mentions: {music_count}")
