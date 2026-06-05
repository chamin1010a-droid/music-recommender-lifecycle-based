import sys, os, io
import pandas as pd
import numpy as np
sys.path.append(os.path.join(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트', 'core'))
sys.stdout.reconfigure(encoding='utf-8')
from lifecycle_recommender import run_pipeline

csv_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
meta_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\data\caches\ytm_metadata_cache.csv'

# raw data
df = pd.read_csv(csv_p)
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp'])
if 'skipped' not in df.columns:
    df['skipped'] = False
df['skipped'] = df['skipped'].fillna(False).astype(bool)

# scorer
old = sys.stdout
sys.stdout = io.TextIOWrapper(open(os.devnull, 'wb'), encoding='utf-8')
r = run_pipeline(csv_path=csv_p, user_name='d', playlist_size=20, preset='default', metadata_path=meta_p, user_birth_year=1998)
sys.stdout = old
sys.stdout.reconfigure(encoding='utf-8')
sc = r['scorer']

now = pd.Timestamp.now()

# Helper: find scorer info by partial match
def find_score(artist_q, title_q):
    for k, v in sc.song_scores.items():
        if v.get('artist','') == artist_q and title_q.lower() in v.get('title','').lower():
            return v
    return {}

# Helper: raw listening data for a song
def analyze_song(artist, title_q):
    mask = (df['artist'] == artist) & (df['title'].str.contains(title_q, case=False, na=False))
    sdf = df[mask].sort_values('timestamp')
    if sdf.empty:
        return None
    
    total = len(sdf)
    skips = sdf['skipped'].sum()
    last = sdf['timestamp'].max()
    days_ago = (now - last).days
    p30 = len(sdf[sdf['timestamp'] > now - pd.Timedelta(days=30)])
    p60 = len(sdf[sdf['timestamp'] > now - pd.Timedelta(days=60)])
    p90 = len(sdf[sdf['timestamp'] > now - pd.Timedelta(days=90)])
    
    mid = total // 2
    fs = sdf.iloc[:mid]['skipped'].mean() if mid > 0 else 0
    ss = sdf.iloc[mid:]['skipped'].mean() if mid > 0 else 0
    
    recent_dates = sdf.tail(8)['timestamp'].dt.strftime('%m/%d').tolist()
    
    return {
        'total': total, 'skips': skips, 'skip_pct': skips/total*100,
        'last_play': last.strftime('%Y-%m-%d'), 'days_ago': days_ago,
        'p30': p30, 'p60': p60, 'p90': p90,
        'first_skip': fs*100, 'second_skip': ss*100,
        'recent': ', '.join(recent_dates)
    }

songs = [
    ('JANNABI - Topic', 'Good Boy Twist', '잔나비'),
    ('JANNABI - Topic', 'Sunshine comedy club', '잔나비'),
    ('JANNABI - Topic', 'Pole Dance', '잔나비'),
    ('The Black Skirts - Topic', 'The Music in Her', '검정치마'),
    ('Charlie Puth - Topic', 'Hero', '찰리푸스'),
    ('Charlie Puth - Topic', 'The Way I Am', '찰리푸스'),
    ('Charlie Puth - Topic', 'Done for Me', '찰리푸스'),
    ('Charlie Puth - Topic', 'How Long', '찰리푸스'),
    ('Charlie Puth - Topic', 'Marvin Gaye', '찰리푸스'),
    ('Charlie Puth - Topic', 'Left and Right', '찰리푸스'),
    ('Xdinary Heroes - Topic', 'PLUTO', '엑히'),
    ('Xdinary Heroes - Topic', 'Enemy', '엑히'),
]

print("=" * 100)
print("📊 곡별 상세 분석 (스코어 + raw 청취 데이터)")
print("=" * 100)

for artist, title_q, label in songs:
    sv = find_score(artist, title_q)
    rd = analyze_song(artist, title_q)
    
    aff = sv.get('affinity', 0)
    mom = sv.get('momentum', 0)
    tp = sv.get('total_plays', 0)
    sr = sv.get('skip_rate', 0)
    dsl = sv.get('days_since_last', '?')
    r30 = sv.get('recent_30d_plays', '?')
    
    print(f"\n{'─' * 90}")
    print(f"🎵 [{label}] {title_q}")
    print(f"   📈 스코어  | 호감도: {aff:.2f} | 모멘텀: {mom:.2f} | 총합: {aff*mom:.3f}")
    print(f"   📊 scorer  | 재생: {tp}회 | 스킵률: {sr*100:.0f}% | 마지막재생: {dsl}일전 | 최근30일: {r30}회")
    
    if rd:
        print(f"   📋 raw     | 재생: {rd['total']}회 | 스킵: {rd['skips']}회 ({rd['skip_pct']:.0f}%)")
        print(f"   📅 리센시  | 마지막: {rd['last_play']} ({rd['days_ago']}일전)")
        print(f"   📅 최근    | 30일={rd['p30']}회 | 60일={rd['p60']}회 | 90일={rd['p90']}회")
        print(f"   📉 스킵추이| 전반전: {rd['first_skip']:.0f}% → 후반전: {rd['second_skip']:.0f}%")
        print(f"   📆 최근재생| {rd['recent']}")

# 핵심 비교: 찰리푸스 "내가 찾아듣는 곡" vs "알고리즘 곡"
print(f"\n\n{'=' * 100}")
print("🔑 핵심 비교: 찰리푸스 '내가 찾아듣는 곡' vs '알고리즘 자동재생 곡'")
print("=" * 100)

my_picks = ['Hero', 'The Way I Am', 'Done for Me', 'How Long']
algo_picks = ['Marvin Gaye (feat. Meghan Trainor)', 'Left and Right']

print(f"\n{'곡명':<35} {'총재생':>5} {'30d':>4} {'60d':>4} {'90d':>4} {'last':>6} {'호감':>5} {'모멘':>5} {'총합':>6}")
print("─" * 85)

for title_q in my_picks + ['---'] + algo_picks:
    if title_q == '---':
        print(f"{'── 알고리즘 자동재생 추정 ──':─<85}")
        continue
    sv = find_score('Charlie Puth - Topic', title_q)
    rd = analyze_song('Charlie Puth - Topic', title_q)
    if sv and rd:
        print(f"  {title_q:<33} {rd['total']:>5} {rd['p30']:>4} {rd['p60']:>4} {rd['p90']:>4} {rd['days_ago']:>4}일 {sv['affinity']:>5.2f} {sv['momentum']:>5.2f} {sv['affinity']*sv['momentum']:>6.3f}")

# Enemy vs PLUTO 상세 비교
print(f"\n\n{'=' * 100}")
print("🔑 Enemy vs PLUTO 모멘텀 차이 분석")
print("=" * 100)

for title_q in ['Enemy', 'PLUTO']:
    sv = find_score('Xdinary Heroes - Topic', title_q)
    rd = analyze_song('Xdinary Heroes - Topic', title_q)
    if sv and rd:
        print(f"\n  🎵 {title_q}")
        print(f"     호감도: {sv['affinity']:.3f} | 모멘텀: {sv['momentum']:.3f}")
        print(f"     재생: {rd['total']}회 | 스킵: {rd['skip_pct']:.0f}%")
        print(f"     마지막재생: {rd['last_play']} ({rd['days_ago']}일전)")
        print(f"     30일={rd['p30']}회 | 60일={rd['p60']}회 | 90일={rd['p90']}회")
        print(f"     전반전 스킵: {rd['first_skip']:.0f}% → 후반전: {rd['second_skip']:.0f}%")
        print(f"     최근재생: {rd['recent']}")

print("\nDONE")
