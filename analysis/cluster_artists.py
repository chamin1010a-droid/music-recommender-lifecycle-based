import pandas as pd
import numpy as np
import codecs, sys
from collections import defaultdict

sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

df = pd.read_csv(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 유효 재생만
valid_df = df[df['is_skipped'] == 0].copy()

# 100회 이상 아티스트만 대상
artist_counts = valid_df['artist'].value_counts()
top_artists = artist_counts[artist_counts >= 50].index.tolist()

print(f"=== 50회 이상 재생된 아티스트: {len(top_artists)}명 ===\n")

# =============================================
# 세션 기반 Co-occurrence 매트릭스 구축
# "세션" = 연속 재생에서 30분 이상 gap이 나면 새 세션
# =============================================
valid_df = valid_df.sort_values('timestamp').reset_index(drop=True)
valid_df['time_gap'] = valid_df['timestamp'].diff().dt.total_seconds()
valid_df['session_id'] = (valid_df['time_gap'] > 1800).cumsum()  # 30분 gap = 새 세션

# 각 세션에서 등장한 아티스트 조합 카운트
cooccur = defaultdict(lambda: defaultdict(int))
sessions = valid_df.groupby('session_id')['artist'].apply(set)

for session_artists in sessions:
    # top_artists에 해당하는 것만
    relevant = [a for a in session_artists if a in top_artists]
    for i, a1 in enumerate(relevant):
        for a2 in relevant[i+1:]:
            cooccur[a1][a2] += 1
            cooccur[a2][a1] += 1

# Co-occurrence 매트릭스를 DataFrame으로
matrix = pd.DataFrame(0, index=top_artists, columns=top_artists)
for a1 in top_artists:
    for a2 in top_artists:
        if a1 != a2:
            matrix.loc[a1, a2] = cooccur[a1][a2]

# =============================================
# 간단한 클러스터링: 특정 아티스트와 가장 자주 같이 들은 아티스트 그룹
# =============================================
# - Topic 제거해서 읽기 쉽게
def clean(name):
    return name.replace(' - Topic', '')

# JANNABI 기준으로 가장 자주 같이 들은 아티스트
target = 'JANNABI - Topic'
print(f"=== [{clean(target)}]와 같은 세션에서 자주 들은 아티스트 ===")
jannabi_co = matrix.loc[target].sort_values(ascending=False).head(15)
for artist, count in jannabi_co.items():
    print(f"  {clean(artist)}: {count}세션")

# M.C the MAX 기준
target2 = 'M.C the MAX - Topic'
print(f"\n=== [{clean(target2)}]와 같은 세션에서 자주 들은 아티스트 ===")
mcmax_co = matrix.loc[target2].sort_values(ascending=False).head(15)
for artist, count in mcmax_co.items():
    print(f"  {clean(artist)}: {count}세션")

# =============================================
# 자동 클러스터링 (Hierarchical)
# =============================================
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# 유사도 → 거리 변환
max_val = matrix.max().max()
dist_matrix = (max_val - matrix).copy()
dist_vals = dist_matrix.values.copy()
np.fill_diagonal(dist_vals, 0)

# 클러스터링
condensed = squareform(dist_vals)
Z = linkage(condensed, method='ward')
clusters = fcluster(Z, t=4, criterion='maxclust')  # 4개 그룹으로

print("\n\n=== 🎵 아티스트 자동 클러스터링 결과 (4개 그룹) ===\n")
cluster_df = pd.DataFrame({'artist': top_artists, 'cluster': clusters})
for c in sorted(cluster_df['cluster'].unique()):
    members = cluster_df[cluster_df['cluster'] == c]['artist'].tolist()
    print(f"  [그룹 {c}]")
    for m in members:
        plays = artist_counts[m]
        print(f"    - {clean(m)} ({plays}회)")
    print()
