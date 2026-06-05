"""December/SG Wannabe 플리가 장르 헤매는 원인 진단"""
import sys, os, io, json
sys.path.append(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\core')
sys.stdout.reconfigure(encoding='utf-8')
from lifecycle_recommender import run_pipeline

csv_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
meta_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\data\caches\ytm_metadata_cache.csv'

old = sys.stdout
sys.stdout = io.TextIOWrapper(open(os.devnull, 'wb'), encoding='utf-8')
r = run_pipeline(csv_path=csv_p, user_name='diag', playlist_size=5, preset='default',
                 metadata_path=meta_p, user_birth_year=1998, skip_external=True)
sys.stdout = old
sys.stdout.reconfigure(encoding='utf-8')

mixer = r['mixer']

# Last.fm 캐시에서 December, SG Wannabe의 유사 아티스트 확인
cache = json.load(open(r'data\caches\lastfm_artist_sim_cache.json', 'r', encoding='utf-8'))
print("=" * 80)
for artist_name in ['December', 'SG Wannabe']:
    key = f"similar_artists||{artist_name}"
    sims = cache.get(key, [])
    print(f"\n🎤 Last.fm: {artist_name}와 유사한 아티스트 TOP 10:")
    for s in sims[:10]:
        print(f"  {s['name']:30s} match: {s['match']:.3f}")
    if not sims:
        print("  (결과 없음!)")

# 핵심: December/SG Wannabe 장르 곡들 vs 검정치마/JANNABI 곡들의 호감도·모멘텀 비교
print("\n" + "=" * 80)
print("📊 아티스트별 호감도 · 모멘텀 평균 (시드가 아닌 곡 고유 속성)")
print("=" * 80)

from collections import defaultdict
artist_stats = defaultdict(lambda: {'aff': [], 'mom': [], 'plays': []})
for info in mixer.song_temps.values():
    a = info.get('artist', '').replace(' - Topic', '')
    artist_stats[a]['aff'].append(info.get('affinity', 0))
    artist_stats[a]['mom'].append(info.get('momentum', 0))
    artist_stats[a]['plays'].append(info.get('total_plays', 0))

target_artists = ['The Black Skirts', 'JANNABI', 'Car, the Garden', 'HYUKOH',
                  'December', 'SG Wannabe', 'M.C the MAX', 'Monday Kiz', 
                  'DAVICHI', 'SeeYa', 'CNBLUE', 'Realslow', 'Park Hyo Shin']

print(f"\n{'아티스트':20s} | {'호감도':>6s} | {'모멘텀':>6s} | {'평균재생':>6s} | {'곡수':>4s} | 장르")
print("-" * 80)
for a in target_artists:
    s = artist_stats.get(a, {'aff': [0], 'mom': [0], 'plays': [0]})
    if not s['aff']:
        continue
    avg_aff = sum(s['aff']) / len(s['aff'])
    avg_mom = sum(s['mom']) / len(s['mom'])
    avg_play = sum(s['plays']) / len(s['plays'])
    genre = "인디/록" if a in ['The Black Skirts', 'JANNABI', 'Car, the Garden', 'HYUKOH'] else "발라드/팝"
    print(f"{a:20s} | {avg_aff:6.3f} | {avg_mom:6.3f} | {avg_play:6.1f} | {len(s['aff']):4d} | {genre}")

# 시뮬레이션: 별이될께 시드로 fw 계산
print("\n" + "=" * 80)
print("📐 별이될께(December) 시드 기준 fw 시뮬레이션")
print("   fw = 0.25×호감도 + 0.25×모멘텀 + 0.50×맥락")
print("=" * 80)

seed = None
for info in mixer.song_temps.values():
    if '별이될께' in info.get('title', '') and 'December' in info.get('artist', ''):
        seed = info
        break

if seed:
    # 주요 곡 비교
    compare_songs = []
    for info in mixer.song_temps.values():
        artist = info.get('artist', '').replace(' - Topic', '')
        if artist in ['The Black Skirts', 'JANNABI', 'December', 'SG Wannabe', 'M.C the MAX', 'CNBLUE']:
            sim = mixer._calculate_similarity(info, seed_song=seed)
            ctx = max(0.01, sim / 100.0)
            aff = info.get('affinity', 0.5)
            mom = info.get('momentum', 0.5)
            fw = 0.25 * aff + 0.25 * mom + 0.50 * ctx
            compare_songs.append({
                'title': info.get('title', '')[:25],
                'artist': artist,
                'aff': aff, 'mom': mom, 'ctx': ctx, 'fw': fw,
                'plays': info.get('total_plays', 0)
            })
    
    compare_songs.sort(key=lambda x: x['fw'], reverse=True)
    
    print(f"\n{'곡명':25s} | {'아티스트':15s} | {'호감':>5s} | {'모멘':>5s} | {'맥락':>5s} | {'FW':>6s} | {'재생':>4s}")
    print("-" * 95)
    for s in compare_songs[:30]:
        genre_mark = "🟢" if s['artist'] in ['December', 'SG Wannabe', 'M.C the MAX', 'DAVICHI'] else "🔴"
        print(f"{genre_mark} {s['title']:24s} | {s['artist']:15s} | {s['aff']:5.3f} | {s['mom']:5.3f} | {s['ctx']:5.3f} | {s['fw']:6.4f} | {s['plays']:4d}")
