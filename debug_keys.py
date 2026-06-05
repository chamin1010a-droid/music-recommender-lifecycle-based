import sys, os
sys.path.append(os.path.join(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트', 'core'))
sys.stdout.reconfigure(encoding='utf-8')
import io
from lifecycle_recommender import run_pipeline

csv_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
meta_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\data\caches\ytm_metadata_cache.csv'

old = sys.stdout
sys.stdout = io.TextIOWrapper(open(os.devnull, 'wb'), encoding='utf-8')
r = run_pipeline(csv_path=csv_p, user_name='d', playlist_size=20, preset='default', metadata_path=meta_p, user_birth_year=1998)
sys.stdout = old
sys.stdout.reconfigure(encoding='utf-8')

sc = r['scorer']

targets = [
    ('Charlie Puth - Topic', 'Hero'),
    ('Charlie Puth - Topic', 'Marvin Gaye'),
    ('Charlie Puth - Topic', 'Done for Me'),
    ('Charlie Puth - Topic', 'How Long'),
    ('Charlie Puth - Topic', 'The Way I Am'),
    ('Charlie Puth - Topic', 'Left and Right'),
    ('Xdinary Heroes - Topic', 'Enemy'),
    ('Xdinary Heroes - Topic', 'PLUTO'),
    ('JANNABI - Topic', 'Good Boy Twist'),
    ('JANNABI - Topic', 'Sunshine comedy club'),
    ('JANNABI - Topic', 'Pole Dance'),
]

for artist_q, title_q in targets:
    found = False
    for k, v in sc.song_scores.items():
        if k[0] == artist_q and title_q.lower() in k[1].lower():
            aff = v.get('affinity', 0)
            mom = v.get('momentum', 0)
            tp = v.get('total_plays', 0)
            sr = v.get('skip_rate', 0)
            print(f"[FOUND] {k[1][:50]}")
            print(f"   aff={aff:.3f}  mom={mom:.3f}  plays={tp}  skip={sr:.2f}")
            # momentum components
            for mk in ['recency_score', 'frequency_score', 'artist_form', 'trend_score']:
                if mk in v:
                    print(f"   {mk}={v[mk]:.3f}")
            found = True
            break
    if not found:
        print(f"[NOT FOUND] {artist_q} / {title_q}")

# Also check what momentum sub-components exist in any song score
print("\n--- Sample song_score keys ---")
sample = list(sc.song_scores.values())[0]
print(list(sample.keys()))
