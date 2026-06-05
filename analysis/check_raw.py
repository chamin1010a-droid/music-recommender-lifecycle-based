import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
raw_csv_path = os.path.join(BASE_DIR, 'Takeout', 'YouTube 및 YouTube Music', '시청 기록', 'ytm_history.csv')

df_raw = pd.read_csv(raw_csv_path, encoding='utf-8-sig')
title_oasis = df_raw[df_raw['title'].str.contains('Oasis|오아시스', case=False, na=False)]
sub_oasis = df_raw[df_raw['subtitles'].str.contains('Oasis|오아시스', case=False, na=False)]

print(f"Raw CSV (ytm_history.csv) total rows: {len(df_raw)}")
print(f"Title has Oasis: {len(title_oasis)}")
print(f"Subtitles (artist) has Oasis: {len(sub_oasis)}")
