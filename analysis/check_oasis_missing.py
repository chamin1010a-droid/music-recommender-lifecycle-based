import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
csv_path = os.path.join(BASE_DIR, 'Takeout', 'YouTube 및 YouTube Music', '시청 기록', 'ytm_history_features.csv')

df = pd.read_csv(csv_path, encoding='utf-8-sig')

# 제목에 'Oasis'가 들어가지만, 아티스트 이름에 'Oasis'가 없는 경우 검색
# 단어 경계이거나 한글과 섞여있을 수 있음
title_oasis = df[df['title'].str.contains('Oasis|오아시스', case=False, na=False)]
missed_oasis = title_oasis[~title_oasis['artist'].str.contains('Oasis', case=False, na=False)]

print("=" * 60)
print(f"🎸 곡 제목에 'Oasis'나 '오아시스'가 있지만 아티스트가 다른 경우: {len(missed_oasis)}건")
print("=" * 60)

if len(missed_oasis) > 0:
    stats = missed_oasis['artist'].value_counts()
    print("--- 해당 곡들의 아티스트(채널명) 분포 ---")
    for artist, count in stats.items():
        print(f"  {artist:<30}: {count}회")
        
    print("\n--- 실제 곡명 예시 (TOP 10) ---")
    print(missed_oasis[['artist', 'title']].value_counts().head(10))
else:
    print("그런 곡이 없습니다.")

