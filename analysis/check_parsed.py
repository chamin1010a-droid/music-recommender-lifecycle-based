import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
parsed_path = os.path.join(BASE_DIR, 'Takeout', 'YouTube 및 YouTube Music', '시청 기록', 'ytm_history_parsed.csv')

df_parsed = pd.read_csv(parsed_path, encoding='utf-8-sig')
print(f"Parsed rows count: {len(df_parsed)}")

oasis_parsed = df_parsed[df_parsed['artist'].str.contains('Oasis|오아시스', case=False, na=False)]
print(f"Oasis count in parsed: {len(oasis_parsed)}")
