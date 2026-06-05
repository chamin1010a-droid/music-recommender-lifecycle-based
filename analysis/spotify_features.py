import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
import json
import os
import time
import codecs, sys

sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

# =============================================
# ★ 여기에 Spotify Developer Dashboard에서
#   발급받은 키를 입력하세요
# =============================================
CLIENT_ID     = "여기에_Client_ID_입력"
CLIENT_SECRET = "여기에_Client_Secret_입력"

OUTPUT_DIR = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트'
FEATURES_CACHE = os.path.join(OUTPUT_DIR, 'spotify_features_cache.json')
HISTORY_CSV = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'

# =============================================
# Spotify 연결
# =============================================
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET
))

# =============================================
# STEP 1: 사용자 청취 곡 목록 로드
# 50회 이상 재생 + 완주(skip_type=0) 기준 상위 곡들
# =============================================
df = pd.read_csv(HISTORY_CSV)
if 'skip_type' not in df.columns:
    df['skip_type'] = df['is_skipped'].apply(lambda x: 2 if x == 1 else 0)

# 완주한 기록만
valid = df[df['skip_type'] == 0]
song_plays = valid.groupby(['title', 'artist']).size().reset_index(name='play_count')
song_plays = song_plays[song_plays['play_count'] >= 10].sort_values('play_count', ascending=False)

print(f"오디오 피처 수집 대상: {len(song_plays)}곡\n")

# =============================================
# STEP 2: Spotify에서 각 곡의 Track ID 검색
# =============================================
def search_track(title, artist):
    """제목 + 아티스트로 Spotify Track ID 검색"""
    artist_clean = artist.replace(' - Topic', '').replace(' Topic', '').strip()
    # 괄호 안 한글 제거
    import re
    title_clean = re.sub(r'\s*\([^)]*\)', '', title).strip()
    
    queries = [
        f"track:{title_clean} artist:{artist_clean}",
        f"{title_clean} {artist_clean}",
        title_clean,
    ]
    for q in queries:
        try:
            result = sp.search(q=q, type='track', limit=1, market='KR')
            items = result['tracks']['items']
            if items:
                return items[0]['id'], items[0]['name'], items[0]['artists'][0]['name']
        except Exception as e:
            pass
        time.sleep(0.05)
    return None, None, None

# =============================================
# STEP 3: Audio Features 수집
# =============================================
# 캐시 로드
if os.path.exists(FEATURES_CACHE):
    with open(FEATURES_CACHE, 'r', encoding='utf-8') as f:
        cache = json.load(f)
    print(f"기존 캐시 로드: {len(cache)}곡\n")
else:
    cache = {}

results = []
not_found = []

print("Spotify에서 오디오 피처 수집 중...\n")
for idx, row in song_plays.iterrows():
    title = row['title']
    artist = row['artist']
    key = f"{title}|||{artist}"
    
    if key in cache:
        results.append(cache[key])
        continue

    track_id, found_title, found_artist = search_track(title, artist)
    if not track_id:
        not_found.append(f"{title} - {artist}")
        continue

    try:
        features = sp.audio_features([track_id])[0]
        if not features:
            not_found.append(f"{title} - {artist}")
            continue

        entry = {
            'title': title,
            'artist': artist,
            'spotify_track_id': track_id,
            'spotify_title': found_title,
            'spotify_artist': found_artist,
            'play_count': row['play_count'],
            'energy': features['energy'],
            'valence': features['valence'],
            'danceability': features['danceability'],
            'acousticness': features['acousticness'],
            'tempo': features['tempo'],
            'loudness': features['loudness'],
            'instrumentalness': features['instrumentalness'],
            'speechiness': features['speechiness'],
        }
        cache[key] = entry
        results.append(entry)
        print(f"  ✅ {title[:30]} → energy:{features['energy']:.2f} valence:{features['valence']:.2f}")
    except Exception as e:
        not_found.append(f"{title} - {artist}")
    
    time.sleep(0.1)  # Rate limit 방지

# 캐시 저장
with open(FEATURES_CACHE, 'w', encoding='utf-8') as f:
    json.dump(cache, f, ensure_ascii=False, indent=2)

# =============================================
# STEP 4: 사용자 음악 취향 벡터 계산
# =============================================
features_df = pd.DataFrame(results)
features_df.to_csv(os.path.join(OUTPUT_DIR, 'spotify_features.csv'), index=False, encoding='utf-8-sig')

feature_cols = ['energy', 'valence', 'danceability', 'acousticness', 'tempo', 'loudness']

# 재생 횟수로 가중 평균 → "많이 들은 곡일수록 취향에 더 가깝다"
weights = features_df['play_count']
taste_vector = {}
for col in feature_cols:
    taste_vector[col] = np.average(features_df[col], weights=weights)

print(f"\n=== 사용자 음악 취향 벡터 ===")
print(f"  Energy (격렬함):      {taste_vector['energy']:.3f}  {'↑ 활기찬' if taste_vector['energy'] > 0.6 else '↓ 차분한'}")
print(f"  Valence (긍정/밝음):  {taste_vector['valence']:.3f}  {'↑ 밝은' if taste_vector['valence'] > 0.5 else '↓ 어두운'}")
print(f"  Danceability (리듬):  {taste_vector['danceability']:.3f}")
print(f"  Acousticness (어쿠스틱): {taste_vector['acousticness']:.3f}")
print(f"  Tempo (BPM):          {taste_vector['tempo']:.1f} BPM")
print(f"  Loudness:             {taste_vector['loudness']:.1f} dB")

print(f"\n✅ 피처 수집 완료: {len(results)}곡 성공, {len(not_found)}곡 미검색")
if not_found:
    print(f"  미검색 곡 (최대 10개):")
    for s in not_found[:10]:
        print(f"    - {s}")

print(f"\n취향 벡터 저장 완료")
print(f"다음 단계: recommend_new_songs.py 실행")

# taste vector 저장
with open(os.path.join(OUTPUT_DIR, 'user_taste_vector.json'), 'w', encoding='utf-8') as f:
    json.dump(taste_vector, f, ensure_ascii=False, indent=2)
