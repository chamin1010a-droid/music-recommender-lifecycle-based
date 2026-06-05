import pandas as pd
import numpy as np
import codecs, sys, os
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from title_alias import search_song, ENGLISH_TO_KOREAN_DISPLAY

sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

# =============================================
# 설정값 (사용자가 조정 가능)
# =============================================
SEED_KEYWORD    = "빨간 나를"
SEED_ARTIST_HINT = "Black Skirts"

RATIO = {
    'type1': 3,  # 확신곡
    'type2': 3,  # 스며드는 중
    'new':   2,  # 신곡
    'type4': 1,  # 무난곡
    'type3': 1,  # 취향 변화 테스트
}

# =============================================
# STEP 1: 데이터 로드
# =============================================
df = pd.read_csv(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# skip_type이 없으면 기존 is_skipped로 fallback
if 'skip_type' not in df.columns:
    df['skip_type'] = df['is_skipped'].apply(lambda x: 2 if x == 1 else 0)

# =============================================
# STEP 2: 아티스트 클러스터링
# =============================================
valid_df = df[df['skip_type'] == 0].copy().sort_values('timestamp').reset_index(drop=True)
valid_df['time_gap'] = valid_df['timestamp'].diff().dt.total_seconds()
valid_df['session_id'] = (valid_df['time_gap'] > 1800).cumsum()

artist_counts = df[df['skip_type'] == 0]['artist'].value_counts()
top_artists_list = artist_counts[artist_counts >= 50].index.tolist()

cooccur = defaultdict(lambda: defaultdict(int))
sessions = valid_df.groupby('session_id')['artist'].apply(set)
for sess in sessions:
    rel = [a for a in sess if a in top_artists_list]
    for i, a1 in enumerate(rel):
        for a2 in rel[i+1:]:
            cooccur[a1][a2] += 1
            cooccur[a2][a1] += 1

matrix = pd.DataFrame(0, index=top_artists_list, columns=top_artists_list)
for a1 in top_artists_list:
    for a2 in top_artists_list:
        if a1 != a2:
            matrix.loc[a1, a2] = cooccur[a1][a2]

max_val = matrix.max().max()
dist_vals = (max_val - matrix).values.copy().astype(float)
np.fill_diagonal(dist_vals, 0)
Z = linkage(squareform(dist_vals), method='ward')
clusters = fcluster(Z, t=4, criterion='maxclust')
artist_to_cluster = dict(zip(top_artists_list, clusters))

# =============================================
# STEP 3: 시드곡 찾기
# =============================================
seed_song_id = search_song(SEED_KEYWORD, artist_hint=SEED_ARTIST_HINT, df=df)
if seed_song_id is None:
    print(f"시드곡 '{SEED_KEYWORD}'을 찾을 수 없습니다.")
    sys.exit(1)

seed_artist  = df[df['song_id'] == seed_song_id].iloc[0]['artist']
seed_cluster = artist_to_cluster.get(seed_artist, 1)
seed_raw_title = df[df['song_id'] == seed_song_id].iloc[0]['title']
seed_display = ENGLISH_TO_KOREAN_DISPLAY.get(seed_raw_title, seed_raw_title)

print(f"\n=== 시드곡: {seed_display} - {seed_artist.replace(' - Topic','')} (그룹 {seed_cluster}) ===\n")

# =============================================
# STEP 4: 곡 분류 (skip_type 3단계 기반 고도화)
#
# 분류 기준:
#   type1_loved  : 처음부터 좋아한 곡
#     - 첫 3회의 즉시스킵(type2) 비율이 낮고 전체 완주율이 높음
#
#   type2_grew   : 듣다 보니 좋아진 곡 (골디락스 여정)
#     - 전반부에 즉시스킵/샘플스킵이 많았지만 후반부에 완주 비율이 올라간 곡
#
#   type3_tired  : 좋았다가 질린 곡
#     - 후반부에 즉시스킵(type2) 비율이 급증한 곡
#     - (핵심) 즉시스킵 증가는 "완전히 질렸다"는 강한 신호
#     - 샘플스킵(type1) 증가는 "기분/무드 불일치" 가능성 있음
#
#   type4_neutral: 무난하게 듣는 곡 (큰 변화 없음)
#
#   new_candidate: 3회 미만 재생 (신곡 후보)
# =============================================
song_meta = []
song_total_counts = df.groupby('song_id').size()

for sid, total in song_total_counts.items():
    sdf = df[df['song_id'] == sid].sort_values('timestamp').reset_index(drop=True)
    art = sdf.iloc[0]['artist']
    raw_title = sdf.iloc[0]['title']
    display = ENGLISH_TO_KOREAN_DISPLAY.get(raw_title, raw_title)

    if total < 3:
        song_meta.append({
            'song_id': sid, 'artist': art,
            'display_title': display, 'raw_title': raw_title,
            'total_plays': total, 'category': 'new_candidate',
            'skip_rate': sdf['is_skipped'].mean(),
            'instant_skip_rate': (sdf['skip_type'] == 2).mean(),
        })
        continue

    half = max(1, total // 2)
    first_half = sdf.iloc[:half]
    second_half = sdf.iloc[half:]

    # 지표 계산
    overall_complete    = (sdf['skip_type'] == 0).mean()
    first3_instant_skip = (sdf.iloc[:min(3,total)]['skip_type'] == 2).mean()
    first3_complete     = (sdf.iloc[:min(3,total)]['skip_type'] == 0).mean()

    # 전반/후반 즉시스킵률 vs 완주율
    f_instant  = (first_half['skip_type'] == 2).mean()
    s_instant  = (second_half['skip_type'] == 2).mean()
    f_complete = (first_half['skip_type'] == 0).mean()
    s_complete = (second_half['skip_type'] == 0).mean()

    # 전반/후반 샘플스킵률 (무드 불일치 신호)
    f_sample   = (first_half['skip_type'] == 1).mean()
    s_sample   = (second_half['skip_type'] == 1).mean()

    # 분류 로직
    # type3: 후반 즉시스킵이 전반 대비 +15%p 이상 증가 → "진짜 질린" 패턴
    tired_by_instant = (s_instant > f_instant + 0.15)

    # type2: 전반 즉시+샘플스킵 많음 → 후반 완주율 증가 패턴
    grew_on_me = (f_complete < s_complete - 0.15) and (f_instant + f_sample > 0.3)

    # type1: 처음 3회 완주 비율 높고 전체 완주율도 높음
    loved_from_start = (first3_complete >= 0.67) and (overall_complete >= 0.70)

    if loved_from_start:
        cat = 'type1_loved'
    elif grew_on_me:
        cat = 'type2_grew'
    elif tired_by_instant:
        cat = 'type3_tired'
    else:
        cat = 'type4_neutral'

    song_meta.append({
        'song_id': sid, 'artist': art,
        'display_title': display, 'raw_title': raw_title,
        'total_plays': total, 'category': cat,
        'skip_rate': 1 - overall_complete,
        'instant_skip_rate': s_instant,
        'sample_skip_rate_change': s_sample - f_sample,  # 양수면 무드 불일치 증가
    })

meta_df = pd.DataFrame(song_meta)

# =============================================
# STEP 5: 통계 출력 (분류 결과 확인)
# =============================================
print("=== 곡 분류 현황 (skip_type 고도화 기반) ===")
cat_labels = {
    'type1_loved':  '🟢 처음부터 좋아한 곡',
    'type2_grew':   '🟡 듣다 보니 좋아진 곡',
    'type3_tired':  '🔴 좋았다가 질린 곡',
    'type4_neutral':'⚪ 무난한 곡',
    'new_candidate':'🆕 신곡 후보 (3회 미만)',
}
total_songs = len(meta_df)
for cat, label in cat_labels.items():
    cnt = (meta_df['category'] == cat).sum()
    print(f"  {label}: {cnt}곡 ({cnt/total_songs*100:.1f}%)")

# =============================================
# STEP 6: 플레이리스트 생성 (같은 그룹 내)
# =============================================
allowed_artists = [a for a, c in artist_to_cluster.items() if c == seed_cluster]
filtered = meta_df[meta_df['artist'].isin(allowed_artists)]

type1 = filtered[filtered['category'] == 'type1_loved']
type2 = filtered[filtered['category'] == 'type2_grew']
type3 = filtered[filtered['category'] == 'type3_tired']
type4 = filtered[filtered['category'] == 'type4_neutral']
new_c = filtered[filtered['category'] == 'new_candidate']
seed_art_type1 = filtered[(filtered['artist'] == seed_artist) & (filtered['category'] == 'type1_loved')]

used = {seed_song_id}

def pick(pool):
    cands = pool[~pool['song_id'].isin(used)]
    if len(cands) == 0:
        return None
    chosen = cands.sample(1).iloc[0]
    used.add(chosen['song_id'])
    return chosen

def format_song(song):
    artist_clean = song['artist'].replace(' - Topic', '')
    # 즉시스킵률 힌트 표시
    skip_hint = ""
    if song['category'] == 'type3_tired':
        skip_hint = f" [최근 즉시스킵↑{song['instant_skip_rate']:.0%}]"
    elif song['category'] == 'type2_grew':
        skip_hint = " [골디락스 여정중]"
    return f"{song['display_title']} - {artist_clean} ({song['total_plays']}회){skip_hint}"

# RATIO 기반 교차 배치
slots = []
slots.append(('type1', '🟢 확신곡', type1))
slots.append(('type1_seed', '🟢 확신곡 (시드 아티스트)', seed_art_type1))

rem = {k: v for k, v in RATIO.items()}
rem['type1'] -= 2

while any(v > 0 for v in rem.values()):
    if rem['type2'] > 0:
        slots.append(('type2', '🟡 스며드는 중인 곡', type2))
        rem['type2'] -= 1
    if rem['type1'] > 0:
        slots.append(('type1', '🟢 확신곡', type1))
        rem['type1'] -= 1
    if rem['new'] > 0:
        slots.append(('new', '🆕 신곡 (안 들어본 곡)', new_c))
        rem['new'] -= 1
    if rem['type1'] > 0:
        slots.append(('type1', '🟢 확신곡', type1))
        rem['type1'] -= 1
    if rem['type4'] > 0:
        slots.append(('type4', '⚪ 무난곡', type4))
        rem['type4'] -= 1
    if rem['type3'] > 0:
        slots.append(('type3', '🔴 취향 감지 (질렸나 테스트)', type3))
        rem['type3'] -= 1

# =============================================
# STEP 7: 출력
# =============================================
group_artists = ', '.join([a.replace(' - Topic','') for a in allowed_artists[:5]])
print(f"\n[그룹 내 아티스트]: {group_artists} 등\n")
print(f"  0. 🎵 [시드곡] {seed_display} - {seed_artist.replace(' - Topic','')}\n")

for i, (st, label, pool) in enumerate(slots, 1):
    song = pick(pool)
    if song is None:
        song = pick(type1)
        label = label + " (대체: 확신곡)"
    if song is not None:
        print(f"  {i:2d}. {label}")
        print(f"      → {format_song(song)}")
    else:
        print(f"  {i:2d}. {label} — 후보 소진")
    print()

print(f"[총 {len(slots)+1}곡]  비율: 확신:{RATIO['type1']} / 스며드는:{RATIO['type2']} / 신곡:{RATIO['new']} / 무난:{RATIO['type4']} / 테스트:{RATIO['type3']}")
