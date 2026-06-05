import os, sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'core'))

from lifecycle_recommender import run_pipeline

target_csv = os.path.join(BASE_DIR, 'Takeout', 'YouTube 및 YouTube Music', '시청 기록', 'ytm_history_features.csv')
metadata_path = os.path.join(BASE_DIR, 'data', 'caches', 'ytm_metadata_cache.csv')
if not os.path.exists(metadata_path):
    metadata_path = None

class HiddenPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

with HiddenPrint():
    result = run_pipeline(
        csv_path=target_csv,
        user_name='Kim_Score_Check',
        playlist_size=20,
        preset='default',
        metadata_path=metadata_path,
        user_birth_year=1998
    )

scorer = result['scorer']
scores = list(scorer.song_scores.values())

for s in scores:
    s['total'] = round(s.get('affinity', 0) * s.get('momentum', 0), 4)

df_scores = pd.DataFrame(scores)

major_artists = [
    'JANNABI - Topic', 
    'The Black Skirts - Topic', 
    'Charlie Puth - Topic', 
    'HANRORO - Topic'
]

output_lines = [
    "# v2 점수 체계 대규모 샘플링 (표 형식)",
    "요청하신 대로 기존보다 5배 많은 곡들의 점수를 표 형식으로 리스트업했습니다. 모멘텀 로직에 '7일 유예 기간'이 반영된 결과입니다.\n"
]

for artist in major_artists:
    output_lines.append(f"## 🎤 주요 가수: {artist}")
    output_lines.append("| 순위 | 곡명 | 호감도 | 모멘텀 | 총합 | 재생 | 스킵률 |")
    output_lines.append("|:---:|:---|:---:|:---:|:---:|:---:|:---:|")
    
    artist_df = df_scores[df_scores['artist'] == artist].sort_values('total', ascending=False)
    n = len(artist_df)
    
    if n <= 50:
        sampled = artist_df
    else:
        indices = [int(i * (n - 1) / 49) for i in range(50)]
        indices = sorted(list(set(indices)))
        sampled = artist_df.iloc[indices]
        
    for _, r in sampled.iterrows():
        title = r['title'][:40].replace('|', '\|')
        actual_rank = artist_df.index.get_loc(r.name) + 1
        output_lines.append(f"| {actual_rank}/{n} | {title} | **{r['affinity']:.2f}** | **{r['momentum']:.2f}** | {r['total']:.3f} | {r['total_plays']}회 | {r['skip_rate']*100:.0f}% |")
    output_lines.append("\n")

minor_df = df_scores[~df_scores['artist'].isin(major_artists)]
target_samples = [
    (minor_df[(minor_df['affinity']>0.6) & (minor_df['momentum']>0.8)].nlargest(15, 'total'), "🔥 [1] 호감 최상 + 모멘텀 상위 (푹 빠진 노래)"),
    (minor_df[(minor_df['affinity']>0.6) & (minor_df['momentum']<0.1)].nlargest(15, 'affinity'), "💖 [2] 호감 최상 + 모멘텀 바닥 (인생곡 / 현재 휴식 중)"),
    (minor_df[(minor_df['affinity']<0.5) & (minor_df['affinity']>0.35) & (minor_df['momentum']>0.8)].nlargest(15, 'total'), "⚡ [3] 호감 중간 + 모멘텀 최상 (요즘 갑자기 꽂힘 / 탐색 단계)"),
    (minor_df[(minor_df['affinity']>0.4) & (minor_df['affinity']<0.6) & (minor_df['momentum']>0.4) & (minor_df['momentum']<0.6)].sample(n=15, random_state=42), "🎵 [4] 호감 중간 + 모멘텀 중간 (플리에서 무난무난하게 재생)"),
    (minor_df[(minor_df['affinity']<0.25) & (minor_df['total_plays']>=3)].nsmallest(15, 'total'), "🧊 [5] 호감 바닥 (스킵 엄청 해서 정떨어진 곡)")
]

output_lines.append("## 🎸 기타 가수 (5가지 유형 스펙트럼)")

for df_group, label in target_samples:
    output_lines.append(f"### {label}")
    output_lines.append("| 아티스트 | 곡명 | 호감도 | 모멘텀 | 총합 | 재생 | 스킵률 |")
    output_lines.append("|:---|:---|:---:|:---:|:---:|:---:|:---:|")
    for _, r in df_group.iterrows():
        artist = str(r['artist'])[:20].replace('|', '\|')
        title = r['title'][:35].replace('|', '\|')
        output_lines.append(f"| {artist} | {title} | **{r['affinity']:.2f}** | **{r['momentum']:.2f}** | {r['total']:.3f} | {r['total_plays']}회 | {r['skip_rate']*100:.0f}% |")
    output_lines.append("\n")

artifact_path = r"C:\Users\user\.gemini\antigravity\brain\9f6d65ce-342d-4c61-a943-572f5bfa8d79\score_sampling_v2.md"
with open(artifact_path, "w", encoding='utf-8') as f:
    f.write("\n".join(output_lines))

print("Done")
