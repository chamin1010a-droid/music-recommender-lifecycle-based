"""잔나비 Summer 시드 + 추억 강화 모드 테스트"""
import sys, os, io, json
sys.path.append(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\core')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from collections import defaultdict
from lifecycle_recommender import run_pipeline

csv_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
meta_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\data\caches\ytm_metadata_cache.csv'

old = sys.stdout
sys.stdout = io.TextIOWrapper(open(os.devnull, 'wb'), encoding='utf-8')
r = run_pipeline(csv_path=csv_p, user_name='nostalgia', playlist_size=20, preset='default',
                 metadata_path=meta_p, user_birth_year=1998, skip_external=True)
sys.stdout = old
sys.stdout.reconfigure(encoding='utf-8')

mixer = r['mixer']

# 시드: 잔나비 Summer
seed = None
for info in mixer.song_temps.values():
    t = info.get('title', '')
    a = info.get('artist', '')
    if '뜨거운' in t and 'JANNABI' in a:
        seed = info
        break

if not seed:
    print("❌ 시드를 못 찾음!")
    sys.exit(1)

print(f"🎵 시드: {seed['title'][:40]} ({seed['artist'].replace(' - Topic','')})")
print(f"   호감: {seed.get('affinity',0):.2f} | 모멘: {seed.get('momentum',0):.2f} | 재생: {seed.get('total_plays',0)}\n")

# ===== 두 가지 모드 비교 =====
modes = {
    'default': {'W_AFF': 0.5, 'W_MOM': 0.5, 'W_CTX': 2.0, 'label': '기본 (모멘텀 루트)'},
    'nostalgia': {'W_AFF': 0.8, 'W_MOM': 0.1, 'W_CTX': 2.0, 'label': '추억 (모멘텀 거의 무시)'},
}

def get_position(aff, mom):
    if mom > 0.8: return '🔥상승'
    elif aff >= 0.7: return '💖고호감'
    elif mom < 0.3: return '🕰️추억'
    else: return '🎵메인'

for mode_name, params in modes.items():
    W_AFF = params['W_AFF']
    W_MOM = params['W_MOM']
    W_CTX = params['W_CTX']
    
    mixer._seed_artist_sim_cache = {}
    scored = []
    
    for sid, info in mixer.song_temps.items():
        aff = info.get('affinity', 0.5)
        mom = info.get('momentum', 0.5)
        sim = mixer._calculate_similarity(info, seed_song=seed)
        ctx = max(0.01, sim / 100.0)
        fw = (aff ** W_AFF) * (mom ** W_MOM) * (ctx ** W_CTX)
        scored.append({**info, 'ctx': ctx, 'fw': fw})
    
    scored.sort(key=lambda x: x['fw'], reverse=True)
    
    # 아티스트당 4곡 제한
    artist_cnt = defaultdict(int)
    playlist = []
    for s in scored:
        a = s.get('artist', '')
        if artist_cnt[a] >= 4:
            continue
        artist_cnt[a] += 1
        playlist.append(s)
        if len(playlist) >= 20:
            break
    
    # 포지션 카운트
    pos_cnt = defaultdict(int)
    for s in playlist:
        pos = get_position(s.get('affinity', 0), s.get('momentum', 0))
        pos_cnt[pos] += 1
    
    print(f"\n{'='*85}")
    print(f"📋 [{params['label']}]  aff^{W_AFF} × mom^{W_MOM} × ctx^{W_CTX}")
    print(f"   포지션: {dict(pos_cnt)}")
    print(f"{'='*85}")
    print(f"{'#':>3s} | {'곡명':25s} | {'아티스트':15s} | {'포지션':6s} | {'맥락':>5s} | {'호감':>5s} | {'모멘':>5s} | {'재생':>4s}")
    print("-" * 85)
    for i, s in enumerate(playlist):
        pos = get_position(s.get('affinity', 0), s.get('momentum', 0))
        artist = s.get('artist', '').replace(' - Topic', '')
        title = s.get('title', '')[:25]
        print(f"{i+1:3d} | {title:25s} | {artist:15s} | {pos:6s} | {s['ctx']:5.2f} | {s.get('affinity',0):5.2f} | {s.get('momentum',0):5.2f} | {s.get('total_plays',0):4d}")
