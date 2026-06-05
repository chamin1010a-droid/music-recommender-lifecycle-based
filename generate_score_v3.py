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
    'HANRORO - Topic',
    'DAY6 - Topic',
    'M.C the MAX - Topic',
    'Xdinary Heroes - Topic'
]

output_lines = [
    "# 주요 가수 전곡 리스트업",
    "요청하신 가수들의 모든 곡에 대한 점수입니다. 7일 모멘텀 유지 기간이 적용되었습니다.\n"
]

for artist in major_artists:
    output_lines.append(f"## 🎤 가수: {artist}")
    output_lines.append("| 순위 | 곡명 | 호감도 | 모멘텀 | 총합 | 재생 | 스킵률 |")
    output_lines.append("|:---:|:---|:---:|:---:|:---:|:---:|:---:|")
    
    artist_df = df_scores[df_scores['artist'] == artist].sort_values('total', ascending=False)
    n = len(artist_df)
    if artist_df.empty:
        output_lines.append(f"| - | (데이터 없음) | - | - | - | - | - |")
    
    for rank, (_, r) in enumerate(artist_df.iterrows(), start=1):
        title = r['title'].replace('|', '\|')
        output_lines.append(f"| {rank}/{n} | {title} | **{r['affinity']:.2f}** | **{r['momentum']:.2f}** | {r['total']:.3f} | {r['total_plays']}회 | {r['skip_rate']*100:.0f}% |")
    output_lines.append("\n")

artifact_path = r"C:\Users\user\Desktop\데이터분석\음악 프로젝트\score_sampling_v3_raw.md"
with open(artifact_path, "w", encoding='utf-8') as f:
    f.write("\n".join(output_lines))
print("DONE")
