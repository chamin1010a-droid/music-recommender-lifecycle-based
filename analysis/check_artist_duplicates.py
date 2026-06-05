import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

csv_path = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
df = pd.read_csv(csv_path, encoding='utf-8-sig')

# 한글명과 영문 Topic 명이 겹치는 아티스트 전수 조사
# 예: "잔나비" vs "JANNABI - Topic", "검정치마" vs "The Black Skirts - Topic"
topic_artists = [a for a in df['artist'].unique() if '- Topic' in str(a)]
non_topic = [a for a in df['artist'].unique() if '- Topic' not in str(a) and str(a) != 'nan']

# 한글명 아티스트가 영문 Topic에도 존재하는지 매핑 확인
print("=== 한글/영문명 중복 가능성이 있는 아티스트 ===\n")
known_mappings = {
    '잔나비': 'JANNABI - Topic',
    '검정치마': 'The Black Skirts - Topic',
    '한로로 HANRORO': 'HANRORO - Topic',
    '카더가든 (Car, the garden)': 'Car, the Garden - Topic',
    '장범준': 'Jang Beom June - Topic',
    '로이킴 Roy Kim': 'Roy Kim - Topic',
    '케이비': None,  # 확인 필요
}

for korean, english in known_mappings.items():
    k_count = len(df[df['artist'] == korean])
    if english:
        e_count = len(df[df['artist'] == english])
        print(f"  {korean} ({k_count}회) + {english} ({e_count}회) = 합계 {k_count+e_count}회")
    else:
        print(f"  {korean} ({k_count}회) — 매핑 미확인")

# Oasis 변형 전수
print("\n=== Oasis 변형 전수 ===")
oasis_variants = [a for a in df['artist'].unique() if 'oasis' in str(a).lower()]
for v in oasis_variants:
    cnt = len(df[df['artist'] == v])
    print(f"  {cnt:>4}회 | {v}")
print(f"  합계: {sum(len(df[df['artist']==v]) for v in oasis_variants)}회")

# 일본어 번역 채널 등 특수 케이스
print("\n=== 특수 채널 (뮤직 외) ===")
special = ['KBS Kpop', 'Beginagain 비긴어게인', '1theK (원더케이)', 'Stone Music Entertainment',
           'Mnet TV', 'BBC Music', 'danmooj1', 'Like It Music']
for s in special:
    cnt = len(df[df['artist'] == s])
    if cnt > 0:
        songs = df[df['artist'] == s]['title'].unique()[:3]
        print(f"  {cnt:>4}회 | {s}")
        for song in songs:
            print(f"         → {song[:50]}")
