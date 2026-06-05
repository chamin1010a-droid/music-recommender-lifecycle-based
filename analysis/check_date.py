import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
csv_path = os.path.join(BASE_DIR, 'Takeout', 'YouTube 및 YouTube Music', '시청 기록', 'ytm_history_features.csv')

df = pd.read_csv(csv_path, encoding='utf-8-sig')

oasis = df[df['artist'] == 'Oasis - Topic']
jannabi = df[df['artist'] == 'JANNABI - Topic']

print(f"Oasis: {len(oasis)} plays, from {oasis['timestamp'].min()} to {oasis['timestamp'].max()}")
print(f"Jannabi: {len(jannabi)} plays, from {jannabi['timestamp'].min()} to {jannabi['timestamp'].max()}")

