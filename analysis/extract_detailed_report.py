import pandas as pd
import codecs, sys

sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

df = pd.read_csv(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv')

# Only consider valid plays (Not skipped within 30s)
valid_df = df[df['is_skipped'] == 0].copy()

# 1. 아티스트 100회 이상
artist_counts = valid_df['artist'].value_counts()
over_100_artists = artist_counts[artist_counts >= 100]

print("=== 1. 100회 이상 청취한 아티스트 랭킹 ===")
for i, (artist, count) in enumerate(over_100_artists.items(), 1):
    print(f"{i}위: {artist} ({count}회)")

# 2. 아티스트 10위 안의 경우, 가장 많이 들은 10곡 추출
top_10_artists = artist_counts.head(10).index

print("\n=== 2. Top 10 아티스트별 최다 청취 곡 Top 10 ===")
for artist in top_10_artists:
    print(f"\n[ {artist} ]")
    artist_df = valid_df[valid_df['artist'] == artist]
    # title 열만 추출해서 곡 카운트
    top_songs = artist_df['title'].value_counts().head(10)
    for j, (song, s_count) in enumerate(top_songs.items(), 1):
        print(f"  {j}. {song} ({s_count}회)")
