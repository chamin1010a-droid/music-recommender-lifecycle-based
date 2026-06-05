import sys, os, io
sys.path.append(os.path.join(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트', 'core'))
sys.stdout.reconfigure(encoding='utf-8')
from lifecycle_recommender import run_pipeline

csv_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
meta_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\data\caches\ytm_metadata_cache.csv'

old = sys.stdout
sys.stdout = io.TextIOWrapper(open(os.devnull, 'wb'), encoding='utf-8')
r = run_pipeline(csv_path=csv_p, user_name='d', playlist_size=20, preset='default', metadata_path=meta_p, user_birth_year=1998)
sys.stdout = old
sys.stdout.reconfigure(encoding='utf-8')

sc = r['scorer']

# 비교 대상 곡들
targets = [
    # (label, artist, title_keyword)
    ('잔나비', 'JANNABI - Topic', 'Good Boy Twist'),
    ('잔나비', 'JANNABI - Topic', 'Sunshine comedy club'),
    ('잔나비', 'JANNABI - Topic', 'Pole Dance'),
    ('검정치마', 'The Black Skirts - Topic', 'The Music in Her'),
    ('찰리푸스(유저pick)', 'Charlie Puth - Topic', 'Hero'),
    ('찰리푸스(유저pick)', 'Charlie Puth - Topic', 'Done for Me'),
    ('찰리푸스(알고리즘)', 'Charlie Puth - Topic', 'Marvin Gaye (feat. Meghan'),
    ('찰리푸스(알고리즘)', 'Charlie Puth - Topic', 'Left and Right'),
    ('엑히', 'Xdinary Heroes - Topic', 'PLUTO'),
    ('엑히', 'Xdinary Heroes - Topic', 'Enemy'),
]

print("=" * 100)
print("📊 모멘텀 v2 (가속도 추가 + 완만한 감쇠) 적용 결과")
print("=" * 100)
print(f"{'분류':<18} {'곡명':<40} {'호감':>5} {'모멘':>5} {'총합':>6} {'last':>5} {'30d':>4} {'prev60d':>7}")
print("─" * 100)

for label, artist, title_q in targets:
    for k, v in sc.song_scores.items():
        if v.get('artist','') == artist and title_q.lower() in v.get('title','').lower():
            t = v['title'][:37]
            aff = v['affinity']
            mom = v['momentum']
            dsl = v['days_since_last']
            r30 = v['recent_30d_plays']
            print(f"  {label:<16} {t:<40} {aff:>5.2f} {mom:>5.2f} {aff*mom:>6.3f} {dsl:>4}일 {r30:>3}회")
            break

# PLUTO vs Enemy 리센시 비교
import numpy as np
print("\n")
print("🔬 리센시 감쇠 비교 (변경 전 vs 변경 후)")
print("─" * 60)
for days, name in [(23, 'PLUTO'), (36, 'Enemy')]:
    eff = max(0, days - 7)
    old_rec = np.exp(-eff / 30.0)
    new_rec = np.exp(-eff / 60.0)
    print(f"  {name:>8} ({days}일전) | 이전: {old_rec:.3f} | 변경후: {new_rec:.3f} | 차이: {abs(old_rec-new_rec):.3f}")

old_diff = np.exp(-max(0,23-7)/30) - np.exp(-max(0,36-7)/30)
new_diff = np.exp(-max(0,23-7)/60) - np.exp(-max(0,36-7)/60)
print(f"\n  23일 vs 36일 리센시 격차: 이전 {old_diff:.3f} → 변경후 {new_diff:.3f} (감소율 {(1-new_diff/old_diff)*100:.0f}%)")

print("\nDONE")
