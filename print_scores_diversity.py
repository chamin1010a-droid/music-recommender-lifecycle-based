import os, sys
import pandas as pd
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'core'))

from lifecycle_recommender import run_pipeline

target_csv = os.path.join(BASE_DIR, 'Takeout', 'YouTube 및 YouTube Music', '시청 기록', 'ytm_history_features.csv')
metadata_path = os.path.join(BASE_DIR, 'data', 'caches', 'ytm_metadata_cache.csv')
if not os.path.exists(metadata_path):
    metadata_path = None

# 무거운 분석을 덜기 위해 출력을 잠시 끕니다.
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

print("=== 🎤 주요 가수 (다양한 점수 분포 샘플) ===")
for artist in major_artists:
    print(f"\n[{artist}]")
    artist_df = df_scores[df_scores['artist'] == artist].sort_values('total', ascending=False)
    
    n = len(artist_df)
    if n == 0:
        continue
        
    # 상, 중, 하를 고르게 추출 (순위 기반)
    if n <= 12:
        indices = list(range(n))
    else:
        indices = [0, 1, 2, n//5, n//4, n//3, n//2, (n*2)//3, (n*3)//4, (n*4)//5, n-3, n-2, n-1]
        
    indices = sorted(list(set([i for i in indices if i < n])))
    sampled = artist_df.iloc[indices]
    
    print(f"{'순위':<4} | {'곡명':<32} | {'호감도':<5} | {'모멘텀':<5} | {'총합':<6} | {'재생(스킵률)'}")
    print("-" * 85)
    for idx_val, (_, r) in zip(indices, sampled.iterrows()):
        title = r['title'][:30]
        rank = f"{idx_val+1}/{n}"
        plays = f"{r['total_plays']}({r['skip_rate']*100:.0f}%)"
        print(f"{rank:<4} | {title:<32} | {r['affinity']:<6.2f} | {r['momentum']:<6.2f} | {r['total']:<6.3f} | {plays}")

print("\n\n=== 🎸 기타/비주류 가수 (다양한 점수 케이스 예시) ===")
minor_df = df_scores[~df_scores['artist'].isin(major_artists)]
# 점수대별 특징적 곡 추출
target_samples = [
    # 1. 호감도 극상 + 모멘텀 최고 (푹 빠져있는 인생곡)
    minor_df[(minor_df['affinity']>0.6) & (minor_df['momentum']>0.8)].nlargest(3, 'total'),
    
    # 2. 호감도 극상 + 모멘텀 바닥 (과거의 명곡 / 현재 휴식기)
    minor_df[(minor_df['affinity']>0.6) & (minor_df['momentum']<0.1)].nlargest(3, 'affinity'),
    
    # 3. 호감도 중간 + 모멘텀 높음 (최근에 갑자기 꽂혀서 듣는 곡)
    minor_df[(minor_df['affinity']<0.5) & (minor_df['affinity']>0.35) & (minor_df['momentum']>0.8)].nlargest(3, 'total'),
    
    # 4. 호감도 중간 + 모멘텀 중간 (그냥저냥 플리용 곡)
    minor_df[(minor_df['affinity']>0.4) & (minor_df['affinity']<0.6) & (minor_df['momentum']>0.4) & (minor_df['momentum']<0.6)].sample(n=3, random_state=42),
    
    # 5. 호감도 바닥 + 모멘텀 매우 낮음 (완전히 질리거나 싫어하는 곡)
    minor_df[(minor_df['affinity']<0.25) & (minor_df['momentum']<0.1) & (minor_df['total_plays']>=5)].nsmallest(3, 'total')
]

labels = [
    "🔥 [1] 호감 최상 + 모멘텀 최상 (지금 푹 빠진 노래)",
    "💖 [2] 호감 최상 + 모멘텀 바닥 (과거의 내 인생곡 / 현재 휴식 중)",
    "⚡ [3] 호감 중간 + 모멘텀 최상 (요즘 갑자기 자주 듣는 곡 / 탐색 중)",
    "🎵 [4] 호감 중간 + 모멘텀 중간 (플리에 섞여서 무난하게 나오는 곡)",
    "🧊 [5] 호감 바닥 + 모멘텀 바닥 (좋다가 완전히 질렸거나, 스킵 엄청 한 곡)"
]

print(f"{'유형':<45} | {'아티스트':<18} | {'곡명':<25} | {'호감도':<5} | {'모멘텀':<5} | {'총합':<6} | {'재생(스킵)'}")
print("-" * 135)

for label, df_group in zip(labels, target_samples):
    print(f"\n{label}")
    for _, r in df_group.iterrows():
        ans = r['artist'][:16]
        ti = r['title'][:23]
        plays = f"{r['total_plays']}({r['skip_rate']*100:.0f}%)"
        print(f"{'':<45} | {ans:<18} | {ti:<25} | {r['affinity']:<6.2f} | {r['momentum']:<6.2f} | {r['total']:<6.3f} | {plays}")
