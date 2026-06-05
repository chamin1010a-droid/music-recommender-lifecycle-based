import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
csv_path = os.path.join(BASE_DIR, 'Takeout', 'YouTube 및 YouTube Music', '시청 기록', 'ytm_history_features.csv')

df = pd.read_csv(csv_path, encoding='utf-8-sig')

# 'Oasis'가 포함된 아티스트명 검색 (대소문자 무시)
oasis_df = df[df['artist'].str.contains('Oasis', case=False, na=False)]

print("=" * 60)
print("🎸 'Oasis' 관련 아티스트 이름별 재생 횟수")
print("=" * 60)

stats = oasis_df['artist'].value_counts()
for artist, count in stats.items():
    print(f"  {artist:<30}: {count}회")

print("\n--- 상위 5곡 (Oasis) ---")
print(oasis_df[oasis_df['artist'] == 'Oasis']['title'].value_counts().head(5))

print("\n--- 상위 5곡 (Oasis - Topic) ---")
print(oasis_df[oasis_df['artist'] == 'Oasis - Topic']['title'].value_counts().head(5))

