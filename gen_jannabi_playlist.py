"""잔나비 Summer 시드 단일 플레이리스트 생성"""
import sys, os, io, random, urllib.parse
sys.path.append(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\core')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from collections import defaultdict
from lifecycle_recommender import run_pipeline

csv_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
meta_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\data\caches\ytm_metadata_cache.csv'

print("파이프라인 로딩...", flush=True)
old = sys.stdout
sys.stdout = io.TextIOWrapper(open(os.devnull, 'wb'), encoding='utf-8')
r = run_pipeline(csv_path=csv_p, user_name='single', playlist_size=20, preset='default',
                 metadata_path=meta_p, user_birth_year=1998, skip_external=True)
sys.stdout = old
sys.stdout.reconfigure(encoding='utf-8')
print("파이프라인 완료!", flush=True)

mixer = r['mixer']

# 시드: Xdinary Heroes - Strawberry Cake
seed = None
for info in mixer.song_temps.values():
    t = info.get('title', '').lower()
    a = info.get('artist', '').lower()
    if 'strawberry' in t and ('xdinary' in a or 'hero' in a):
        seed = info
        break

if not seed:
    print("❌ 시드 못 찾음!")
    sys.exit(1)

seed_title = seed.get('title', '')[:40]
seed_artist = seed.get('artist', '').replace(' - Topic', '')
print(f"시드: {seed_title} ({seed_artist})")

# 기본 프리셋
W_AFF, W_MOM, W_CTX = 0.5, 1.5, 1.5
n_discovery = 4

mixer._seed_artist_sim_cache = {}
scored_pool = []
seed_id = seed.get('song_id', '')
for sid, info in mixer.song_temps.items():
    if sid == seed_id:
        continue  # ★ 시드곡 자체는 추천에서 제외
    aff = info.get('affinity', 0.5)
    mom = info.get('momentum', 0.5)
    sim = mixer._calculate_similarity(info, seed_song=seed)
    ctx = max(0.01, sim / 100.0)
    fw = (aff ** W_AFF) * (mom ** W_MOM) * (ctx ** W_CTX) + np.random.uniform(0, 0.005)
    sc = info.copy()
    sc['similarity_score'] = round(ctx * 100, 1)
    sc['final_weight'] = round(fw, 4)
    scored_pool.append(sc)
scored_pool.sort(key=lambda x: x['final_weight'], reverse=True)

# Discovery
low_play = [s for s in mixer.song_temps.values() if s['total_plays'] <= 3 and s.get('affinity', 0) >= 0.3]
int_scored = []
for s in low_play:
    sim = mixer._calculate_similarity(s, seed_song=seed)
    if sim > 15:
        int_scored.append((s, sim))
int_scored.sort(key=lambda x: x[1], reverse=True)

artist_counts = defaultdict(int)
chosen_discovery = []
for s, sim in int_scored[:n_discovery]:
    sc = s.copy()
    sc['reason'] = 'Discovery·내부 (새 발견)'
    sc['similarity_score'] = round(sim, 1)
    if artist_counts[sc['artist']] < 4:
        chosen_discovery.append(sc)
        artist_counts[sc['artist']] += 1

# Main
n_main = 20 - len(chosen_discovery)
main_songs = []
for s in scored_pool:
    if artist_counts[s['artist']] < 4:
        s['reason'] = 'Main'
        main_songs.append(s)
        artist_counts[s['artist']] += 1
    if len(main_songs) >= n_main:
        break

playlist = main_songs + chosen_discovery
playlist = mixer._arrange_playlist_order(playlist)

def get_pos(song):
    reason = song.get('reason', '')
    if 'Discovery' in reason: return '💎발견'
    mom = song.get('momentum', 0.5)
    aff = song.get('affinity', 0.5)
    if mom > 0.8: return '🔥상승'
    elif aff >= 0.7: return '💖고호감'
    elif mom < 0.3: return '🕰️추억'
    else: return '🎵메인'

# 아티파트 MD 출력
lines = []
lines.append(f"# 🎵 잔나비 Summer 플레이리스트 (기본 모드)\n")
lines.append(f"**시드**: {seed_title} ({seed_artist})")
lines.append(f"**수식**: `fw = aff^{W_AFF} × mom^{W_MOM} × ctx^{W_CTX}`\n")
lines.append(f"| # | 곡명 | 아티스트 | 포지션 | 유사% | 호감 | 모멘 | 재생 |")
lines.append(f"|--:|:---|:---|:---:|---:|---:|---:|---:|")
for i, s in enumerate(playlist):
    t = s.get('title', '')[:40]
    a = s.get('artist', '').replace(' - Topic', '')
    pos = get_pos(s)
    ctx = s.get('similarity_score', 0)
    aff = s.get('affinity', 0)
    mom = s.get('momentum', 0)
    plays = s.get('total_plays', 0)
    lines.append(f"| {i+1} | {t} | {a} | {pos} | {ctx} | {aff:.2f} | {mom:.2f} | {plays} |")

md = '\n'.join(lines)
out_path = r'C:\Users\user\.gemini\antigravity\brain\9f6d65ce-342d-4c61-a943-572f5bfa8d79\jannabi_summer_playlist.md'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(md)

print(f"\n✅ 완료! {out_path}")
print(md)
