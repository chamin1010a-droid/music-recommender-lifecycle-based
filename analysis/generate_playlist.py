import pandas as pd
import numpy as np
import codecs, sys, os

sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

df = pd.read_csv(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# =============================================
# STEP 0: Seed Song
# =============================================
seed_keyword = "After School Activity"
seed_artist = "JANNABI"

seed_matches = df[df['song_id'].str.contains(seed_keyword, case=False, na=False)]
seed_song_id = seed_matches['song_id'].value_counts().index[0]
print(f"=== 시드곡: {seed_song_id} ===\n")

# =============================================
# STEP 1: 곡 분류 (1~4번 + 신곡 후보)
# =============================================
valid_df = df[df['is_skipped'] == 0].copy()

# 곡별 메타 계산
song_meta = []
song_counts_all = df['song_id'].value_counts()

for song_id in song_counts_all.index:
    song_df_full = df[df['song_id'] == song_id].sort_values('timestamp').reset_index(drop=True)
    total = len(song_df_full)
    
    if total < 3:
        # 1~2번만 들은 곡 = Tier 1 신곡 후보
        artist = song_df_full.iloc[0]['artist']
        song_meta.append({
            'song_id': song_id,
            'artist': artist,
            'total_plays': total,
            'category': 'new_candidate',
            'skip_rate': song_df_full['is_skipped'].mean()
        })
        continue
    
    half = total // 2
    first_half_skip = song_df_full.iloc[:half]['is_skipped'].mean()
    second_half_skip = song_df_full.iloc[half:]['is_skipped'].mean()
    first_3_skip = song_df_full.iloc[:min(3, total)]['is_skipped'].mean()
    overall_skip = song_df_full['is_skipped'].mean()
    artist = song_df_full.iloc[0]['artist']
    
    if first_3_skip <= 0.33 and overall_skip <= 0.3:
        cat = 'type1_loved'
    elif first_half_skip > second_half_skip + 0.15:
        cat = 'type2_grew'
    elif second_half_skip > first_half_skip + 0.15:
        cat = 'type3_tired'
    else:
        cat = 'type4_neutral'
    
    song_meta.append({
        'song_id': song_id,
        'artist': artist,
        'total_plays': total,
        'category': cat,
        'skip_rate': overall_skip
    })

meta_df = pd.DataFrame(song_meta)

# =============================================
# STEP 2: 시드곡 주변에서 자주 같이 재생된 아티스트 찾기 (Co-occurrence)
# =============================================
# 시드곡이 재생된 timestamp 주변 ±30분 이내에 재생된 다른 곡들
seed_timestamps = df[df['song_id'] == seed_song_id]['timestamp']

cooccurrence = {}
for ts in seed_timestamps:
    window = df[(df['timestamp'] >= ts - pd.Timedelta(minutes=30)) & 
                (df['timestamp'] <= ts + pd.Timedelta(minutes=30)) &
                (df['song_id'] != seed_song_id)]
    for _, row in window.iterrows():
        sid = row['song_id']
        if sid not in cooccurrence:
            cooccurrence[sid] = 0
        cooccurrence[sid] += 1

# 자주 같이 들은 곡 정렬
co_df = pd.DataFrame(list(cooccurrence.items()), columns=['song_id', 'co_count'])
co_df = co_df.sort_values('co_count', ascending=False)
co_df = co_df.merge(meta_df[['song_id', 'artist', 'category', 'total_plays']], on='song_id', how='left')

# =============================================
# STEP 3: 플레이리스트 생성
# =============================================
playlist = []
used = {seed_song_id}

def pick_song(pool, exclude_used=True):
    """pool에서 아직 안 쓴 곡 하나를 랜덤 추출"""
    candidates = pool[~pool['song_id'].isin(used)] if exclude_used else pool
    if len(candidates) == 0:
        return None
    chosen = candidates.sample(1).iloc[0]
    used.add(chosen['song_id'])
    return chosen

# 시드 아티스트 곡 풀
seed_artist_songs = meta_df[meta_df['artist'].str.contains(seed_artist, case=False, na=False)]
jannabi_type1 = seed_artist_songs[seed_artist_songs['category'] == 'type1_loved']
jannabi_type2 = seed_artist_songs[seed_artist_songs['category'] == 'type2_grew']
jannabi_type4 = seed_artist_songs[seed_artist_songs['category'] == 'type4_neutral']

# 다른 아티스트 중 co-occurrence 높은 type1 곡 (비슷한 무드)
other_artist_co = co_df[~co_df['artist'].str.contains(seed_artist, case=False, na=False)]
other_type1_co = other_artist_co[other_artist_co['category'] == 'type1_loved'].head(30)

# 시드 아티스트의 Tier1 신곡 (1~2번만 들은 곡)
jannabi_new = seed_artist_songs[seed_artist_songs['category'] == 'new_candidate']

# 다른 아티스트의 Tier1 신곡 (좋아하는 아티스트의 안 들은 곡)
top_artists = valid_df['artist'].value_counts().head(10).index
other_new = meta_df[(meta_df['category'] == 'new_candidate') & (meta_df['artist'].isin(top_artists))]

# 3번(질린) 곡
jannabi_type3 = seed_artist_songs[seed_artist_songs['category'] == 'type3_tired']

print("=== 🎧 Mode 3 프로토타입 플레이리스트 ===\n")
print(f"  0. 🎵 [시드곡] {seed_song_id}\n")

# 플레이리스트 구성 템플릿
template = [
    ('type1_same', '🟢 같은 아티스트 확신곡', jannabi_type1),
    ('type1_same', '🟢 같은 아티스트 확신곡', jannabi_type1),
    ('type2_same', '🟡 스며드는 중인 곡', jannabi_type2 if len(jannabi_type2) > 0 else jannabi_type4),
    ('type1_other', '🔵 비슷한 무드의 다른 아티스트', other_type1_co),
    ('type1_other', '🔵 비슷한 무드의 다른 아티스트', other_type1_co),
    ('type1_same', '🟢 같은 아티스트 확신곡', jannabi_type1),
    ('new_tier1',  '🆕 아는 아티스트의 안 들은 곡', pd.concat([jannabi_new, other_new])),
    ('type1_other', '🔵 비슷한 무드의 다른 아티스트', other_type1_co),
    ('type4_same', '⚪ 무난하게 듣는 곡', jannabi_type4 if len(jannabi_type4) > 0 else jannabi_type1),
    ('type1_other', '🔵 비슷한 무드의 다른 아티스트', other_type1_co),
    ('type2_other', '🟡 다른 아티스트 스며드는 곡', other_artist_co[other_artist_co['category'] == 'type2_grew'].head(20) if len(other_artist_co[other_artist_co['category'] == 'type2_grew']) > 0 else other_type1_co),
    ('type1_same', '🟢 같은 아티스트 확신곡', jannabi_type1),
    ('new_tier1',  '🆕 아는 아티스트의 안 들은 곡', pd.concat([jannabi_new, other_new])),
    ('type3_probe','🔴 취향 변화 감지 (한때 좋았던 곡)', jannabi_type3 if len(jannabi_type3) > 0 else jannabi_type4),
    ('type1_other', '🔵 비슷한 무드의 다른 아티스트', other_type1_co),
]

for i, (slot_type, label, pool) in enumerate(template, 1):
    song = pick_song(pool)
    if song is not None:
        print(f"  {i:2d}. {label}")
        print(f"      → {song['song_id']} ({song['total_plays']}회 재생)")
    else:
        print(f"  {i:2d}. {label}")
        print(f"      → (후보 소진)")
    print()
