import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
csv_path = os.path.join(BASE_DIR, 'Takeout', 'YouTube 및 YouTube Music', '시청 기록', 'ytm_history_features.csv')

df = pd.read_csv(csv_path, encoding='utf-8-sig')

gallagher = df[df['artist'].str.contains('Gallagher', case=False, na=False)]
stats = gallagher['artist'].value_counts()

print("=" * 60)
print("🎸 Gallagher 관련 아티스트")
print("=" * 60)
for a, c in stats.items():
    print(f"  {a:<30}: {c}회")

