import sys, os, io, random, urllib.parse
sys.path.append(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\core')
sys.stdout.reconfigure(encoding='utf-8')
from lifecycle_recommender import run_pipeline

csv_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
meta_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\data\caches\ytm_metadata_cache.csv'

# ═══════════════════════════════════════════════
# ★ 프리셋 설정 (여기만 바꾸면 모드 전환)
# ═══════════════════════════════════════════════
PLAYLIST_PRESETS = {
    'default': {
        'label': '🎵 기본',
        'desc': '요즘 자주 듣는 곡 위주 + 장르 일치',
        'W_AFF': 0.5,   # 호감도 루트 (압축)
        'W_MOM': 0.5,   # 모멘텀 루트 (압축)
        'W_CTX': 2.0,   # 맥락 제곱 (장르 veto)
        'discovery_ratio': 0.20,  # 20% = 4곡
    },
    'nostalgia': {
        'label': '🕰️ 추억',
        'desc': '옛날에 좋아했던 곡 리마인드',
        'W_AFF': 0.8,   # 호감도 강조 (좋아했던 곡)
        'W_MOM': 0.1,   # 모멘텀 거의 무시 (오래된 곡 OK)
        'W_CTX': 2.0,   # 장르 veto 유지
        'discovery_ratio': 0.15,  # 발견 줄임
    },
    'explore': {
        'label': '🔭 탐험',
        'desc': '안 들어본 곡 위주 발굴',
        'W_AFF': 0.3,   # 호감도 약화 (새 곡 우선)
        'W_MOM': 0.3,   # 모멘텀 약화
        'W_CTX': 2.5,   # 맥락 더 강화 (장르 엄격)
        'discovery_ratio': 0.40,  # 40% = 8곡 발견
    },
}

ACTIVE_PRESET = 'default'  # ← 여기를 'nostalgia' 또는 'explore'로 변경
# ═══════════════════════════════════════════════

preset = PLAYLIST_PRESETS[ACTIVE_PRESET]
print(f"프리셋: {preset['label']} — {preset['desc']}", flush=True)
print("파이프라인 로딩...", flush=True)
old = sys.stdout
sys.stdout = io.TextIOWrapper(open(os.devnull, 'wb'), encoding='utf-8')
r = run_pipeline(csv_path=csv_p, user_name='multi', playlist_size=20, preset='default', metadata_path=meta_p, user_birth_year=1998, skip_external=True)
sys.stdout = old
sys.stdout.reconfigure(encoding='utf-8')
print("파이프라인 완료!", flush=True)

mixer = r['mixer']
all_songs = list(mixer.song_temps.values())
scored = sorted(all_songs, key=lambda x: x.get('affinity', 0) * x.get('momentum', 0), reverse=True)
seeds = []
seen_artists = set()
for s in scored:
    a = s.get('artist', '')
    if a not in seen_artists:
        seeds.append(s)
        seen_artists.add(a)
    if len(seeds) >= 5:
        break

import types, numpy as np
from collections import defaultdict

# ★ 외부 Discovery 비활성화 (2800곡 내부에서만 추천)
print("외부 Discovery 1회 수집...", flush=True)
ext_external = []
print("라이브러리 내부(2800곡)에서만 추천", flush=True)

# 포지션 레이블
def get_position_label(song):
    reason = song.get('reason', '')
    if 'Discovery' in reason and '외부' in reason:
        return '🆕외부'
    elif 'Discovery' in reason:
        return '💎발견'
    else:
        mom = song.get('momentum', 0.5)
        aff = song.get('affinity', 0.5)
        if mom > 0.8:
            return '🔥상승'
        elif aff >= 0.7:
            return '💖고호감'
        elif mom < 0.3:
            return '🕰️추억'
        else:
            return '🎵메인'

playlists_data = []
used_ext_ids = set()

W_AFF = preset['W_AFF']
W_MOM = preset['W_MOM']
W_CTX = preset['W_CTX']
n_discovery = int(round(20 * preset['discovery_ratio']))

for i, seed in enumerate(seeds):
    seed_title = seed.get('title', '?')[:35]
    seed_artist = seed.get('artist', '').replace(' - Topic', '')
    print(f"  #{i+1} {seed_title} ({seed_artist}) 생성중...", flush=True)
    
    max_per_artist = max(2, int(20 * 0.2))
    
    print(f"    아티스트 유사도 캐시 빌드중...", flush=True)
    mixer._seed_artist_sim_cache = {}
    artist_counts = defaultdict(int)
    scored_pool = []
    
    for song_id, info in mixer.song_temps.items():
        aff, mom = info.get('affinity', 0.5), info.get('momentum', 0.5)
        sim = mixer._calculate_similarity(info, seed_song=seed)
        ctx = max(0.01, sim / 100.0)
        fw = (aff ** W_AFF) * (mom ** W_MOM) * (ctx ** W_CTX) + np.random.uniform(0, 0.005)
        sc = info.copy()
        sc['similarity_score'] = round(ctx * 100, 1)
        sc['final_weight'] = round(fw, 4)
        scored_pool.append(sc)
    scored_pool.sort(key=lambda x: x['final_weight'], reverse=True)
    
    # Discovery: 외부 풀에서 아직 안 쓴 곡 배분 + 내부는 시드와 유사도 재계산
    chosen_discovery = []
    
    # 외부곡: 안 쓴 것 중에서 랜덤 셔플 후 배분
    available_ext = [s for s in ext_external if s.get('song_id', '') not in used_ext_ids]
    random.shuffle(available_ext)
    for s in available_ext[:max(1, n_discovery // 2)]:
        if artist_counts[s['artist']] < max_per_artist:
            chosen_discovery.append(s)
            artist_counts[s['artist']] += 1
            used_ext_ids.add(s.get('song_id', ''))
    
    # 내부곡: 시드 기반으로 유사도 높은 저재생곡
    low_play = [s for s in mixer.song_temps.values() if s['total_plays'] <= 3 and s.get('affinity', 0) >= 0.3]
    int_scored = []
    for s in low_play:
        sim = mixer._calculate_similarity(s, seed_song=seed)
        if sim > 15:
            int_scored.append((s, sim))
    int_scored.sort(key=lambda x: x[1], reverse=True)
    
    for s, sim in int_scored[:n_discovery - len(chosen_discovery)]:
        sc = s.copy()
        sc['reason'] = 'Discovery·내부 (새 발견)'
        sc['similarity_score'] = round(sim, 1)
        sc['discovery_source'] = 'internal'
        if artist_counts[sc['artist']] < max_per_artist:
            chosen_discovery.append(sc)
            artist_counts[sc['artist']] += 1
    
    # Main 곡
    n_main = 20 - len(chosen_discovery)
    main_songs = []
    for s in scored_pool:
        if artist_counts[s['artist']] < max_per_artist:
            s['reason'] = 'Main'
            main_songs.append(s)
            artist_counts[s['artist']] += 1
        if len(main_songs) >= n_main:
            break
    
    playlist = main_songs + chosen_discovery
    playlist = mixer._arrange_playlist_order(playlist)
    
    playlists_data.append({
        'seed_title': seed_title,
        'seed_artist': seed_artist,
        'tracks': playlist,
    })

print("HTML 생성...", flush=True)

# === HTML 생성 ===
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
seed_emojis = ['🎸', '💿', '🎹', '🎺', '🥁']

def ytm_link(title, artist):
    q = f"{title} {artist}".replace(' - Topic', '')
    return f"https://music.youtube.com/search?q={urllib.parse.quote(q)}"

html = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Multi-Signal 플레이리스트 v2</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Inter', sans-serif; background: #0a0a0f; color: #e0e0e0; padding: 20px; }
h1 { text-align: center; font-size: 26px; margin: 20px 0 6px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.subtitle { text-align: center; color: #888; font-size: 13px; margin-bottom: 24px; }
.preset-badge { text-align: center; margin-bottom: 16px; }
.preset-badge span { background: linear-gradient(135deg, #667eea33, #764ba233); border: 1px solid #667eea55;
    padding: 6px 18px; border-radius: 20px; font-size: 13px; color: #a8b4ff; }
.playlists { display: flex; gap: 14px; overflow-x: auto; padding-bottom: 20px; }
.playlist-card { flex: 0 0 400px; background: #14141f; border-radius: 14px; border: 1px solid #222; overflow: hidden; }
.playlist-header { padding: 16px 20px; text-align: center; }
.playlist-header h2 { font-size: 16px; margin-bottom: 2px; }
.playlist-header .artist { font-size: 12px; opacity: 0.5; }
.playlist-header .emoji { font-size: 30px; display: block; margin-bottom: 6px; }
.track-list { padding: 0 10px 14px; }
.track { display: flex; align-items: center; padding: 6px 8px; border-radius: 6px;
    transition: background 0.2s; text-decoration: none; color: inherit; gap: 8px; }
.track:hover { background: rgba(255,255,255,0.05); }
.track-num { font-size: 11px; color: #555; width: 18px; text-align: right; flex-shrink: 0; }
.track-info { flex: 1; min-width: 0; }
.track-title { font-size: 12px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.track-artist { font-size: 10px; color: #777; }
.track-meta { display: flex; gap: 4px; flex-shrink: 0; align-items: center; font-size: 10px; color: #666; }
.badge { display: inline-block; font-size: 9px; padding: 1px 5px; border-radius: 4px; font-weight: 600; white-space: nowrap; }
.badge-main { background: #1a1a2e; color: #667eea; }
.badge-hot { background: #2d1a1a; color: #e74c3c; }
.badge-fav { background: #2d1a2d; color: #e84393; }
.badge-nostalgia { background: #1a2d1a; color: #00b894; }
.badge-disc-int { background: #1a2d2d; color: #00cec9; }
.badge-disc-ext { background: #2d2d1a; color: #fdcb6e; }
.sim-bar { width: 30px; height: 4px; border-radius: 2px; background: #222; overflow: hidden; display: inline-block; vertical-align: middle; }
.sim-fill { height: 100%; border-radius: 2px; }
.play-btn { width: 22px; height: 22px; border-radius: 50%; display: flex; align-items: center;
    justify-content: center; flex-shrink: 0; opacity: 0; transition: opacity 0.2s; font-size: 10px; }
.track:hover .play-btn { opacity: 1; }
</style>
</head>
<body>
<h1>🎯 Multi-Signal 추천 플레이리스트 v2</h1>
<p class="subtitle">fw = aff^{W_AFF} × mom^{W_MOM} × ctx^{W_CTX} • Last.fm 아티스트 유사도 • StandardScaler 오디오</p>
<div class="preset-badge"><span>{preset['label']} {preset['desc']}</span></div>
<div class="playlists">
"""

for idx, pl in enumerate(playlists_data):
    color = colors[idx % len(colors)]
    emoji = seed_emojis[idx % len(seed_emojis)]
    html += f"""
<div class="playlist-card">
    <div class="playlist-header" style="background: linear-gradient(135deg, {color}22, {color}08);">
        <span class="emoji">{emoji}</span>
        <h2 style="color: {color};">{pl['seed_title']}</h2>
        <div class="artist">{pl['seed_artist']}</div>
    </div>
    <div class="track-list">
"""
    for j, s in enumerate(pl['tracks'], 1):
        title = s.get('title', '?')[:30]
        artist = s.get('artist', '?').replace(' - Topic', '')[:16]
        sim = s.get('similarity_score', 0)
        pos = get_position_label(s)
        link = ytm_link(s.get('title', ''), s.get('artist', ''))
        badge_cls = 'badge-main'
        if '외부' in pos: badge_cls = 'badge-disc-ext'
        elif '발견' in pos: badge_cls = 'badge-disc-int'
        elif '상승' in pos: badge_cls = 'badge-hot'
        elif '고호감' in pos: badge_cls = 'badge-fav'
        elif '추억' in pos: badge_cls = 'badge-nostalgia'
        sim_pct = min(sim, 100)
        html += f"""        <a class="track" href="{link}" target="_blank">
            <span class="track-num">{j}</span>
            <div class="track-info">
                <div class="track-title">{title}</div>
                <div class="track-artist">{artist}</div>
            </div>
            <div class="track-meta">
                <span class="badge {badge_cls}">{pos}</span>
                <span class="sim-bar"><span class="sim-fill" style="width:{sim_pct}%; background:{color};"></span></span>
                <span>{sim:.0f}%</span>
            </div>
            <div class="play-btn" style="background: {color}33; color: {color};">▶</div>
        </a>
"""
    html += "    </div>\n</div>\n"

html += "</div>\n</body>\n</html>"

with open(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\playlist_viewer.html', 'w', encoding='utf-8') as f:
    f.write(html)

# MD 아티팩트
lines = []
for i, pl in enumerate(playlists_data):
    lines.append(f"## 🎲 #{i+1} — 시드: {pl['seed_title']} ({pl['seed_artist']})\n")
    lines.append("| # | 곡명 | 아티스트 | 포지션 | 유사% | 호감 | 모멘 | 재생 |")
    lines.append("|--:|:---|:---|:---:|---:|---:|---:|---:|")
    for j, s in enumerate(pl['tracks'], 1):
        t = s.get('title','?')[:40]
        a = s.get('artist','?').replace(' - Topic','')[:18]
        lines.append(f"| {j} | {t} | {a} | {get_position_label(s)} | {s.get('similarity_score',0):.1f} | {s.get('affinity',0):.2f} | {s.get('momentum',0):.2f} | {s.get('total_plays',0)} |")
    lines.append("\n---\n")

out_md = r'C:\Users\user\.gemini\antigravity\brain\9f6d65ce-342d-4c61-a943-572f5bfa8d79\playlist_comparison.md'
with open(out_md, 'w', encoding='utf-8') as f:
    f.write("# 🎲 시드별 플레이리스트 v2 (Multi-Signal + 시드별 Discovery)\n\n")
    f.write("가사(30%) + 오디오(30%) + 메타(15%) + 태그(25%). 포지션 표시 포함.\n\n---\n\n")
    for line in lines:
        f.write(line + "\n")

print("✅ 완료!")
for i, pl in enumerate(playlists_data):
    ext = [s for s in pl['tracks'] if '외부' in get_position_label(s)]
    disc = [s for s in pl['tracks'] if '발견' in get_position_label(s)]
    print(f"  #{i+1} {pl['seed_title']:25s} | 외부: {[s.get('title','?')[:15] for s in ext]} | 내부발견: {[s.get('title','?')[:15] for s in disc]}")
