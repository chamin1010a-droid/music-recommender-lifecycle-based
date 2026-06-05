import os
"""
🎶 최종 통합 플레이리스트 v6 — 시작곡 리스트 중심 파이프라인
"""
import sys, random, re
sys.stdout.reconfigure(encoding='utf-8')

from lifecycle_recommender import run_pipeline
from external_discovery import ExternalDiscoveryEngine
from nostalgia_engine import NostalgiaEngine

CSV_PATH = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
META_PATH = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\ytm_metadata_cache.csv'
API_KEY = os.environ.get("LASTFM_API_KEY", "")

# ═══════════════════════════════════════════════════════
# 🎯 시작곡 (Seed Tracks) 설정
# 여러 곡을 함께 넣으면 이들의 평균 분위기로 추천이 작동합니다.
# 가중치(weight)를 높게 주면 해당 곡의 분위기가 더 반영됩니다.
# ═══════════════════════════════════════════════════════
START_SONGS = [
    {'artist': 'JANNABI', 'title': '애프터스쿨 액티비티', 'weight': 1.0},
]

# ── 이미 추천했던 곡 블랙리스트 (같은 곡 계속 나오는 걸 방지) ──
BLOCKLIST = {
    'cnblue||black flower', 'cnblue||can\'t stop',
    'oasis||all around the world',
    'hanroro||mirror', 'hanroro||the last stop of our pain',
    'nerd connection||if i have you only',
    'nerd connection||i\'ll just walk',
    'shin in ryu||for us, summer is always short',
    'shin in ryu||fairy castor',
}
# 시작곡들도 블랙리스트에 추가해서 추천목록에 또 나오지 않게 함
for st in START_SONGS:
    artist_clean = st['artist'].lower().replace(' - topic', '').strip()
    title_clean = st['title'].lower()
    BLOCKLIST.add(f"{artist_clean}||{title_clean}")

# ── Step 1: 기본 파이프라인 (시작곡 전달) ──
print(f"🔄 파이프라인 실행 중... (시작곡 중심 매칭)")
result = run_pipeline(
    csv_path=CSV_PATH, 
    user_name='user', 
    playlist_size=10, 
    preset='default', 
    metadata_path=META_PATH, 
    user_birth_year=1998,
    seed_tracks=START_SONGS  # 핵심 변경점: 파이프라인에 시드 전달!
)
song_temps = result['temp_tracker'].song_temps
main_playlist = result['playlist']
similarity_engine = result.get('similarity_engine')

# 시작곡의 평균 태그 벡터 가져오기
seed_vector = None
if similarity_engine:
    seed_vector = similarity_engine.build_seed_vector(START_SONGS)

# ── 메인 플레이리스트 블랙리스트 제거 및 보충 ──
cleaned_main = []
for s in main_playlist:
    a = str(s.get('artist', '')).replace(' - Topic', '').strip().lower()
    t = str(s.get('title', '')).lower()
    t_clean = re.sub(r'\s*\([^)]*\)', '', t).strip()
    key1 = f"{a}||{t}"
    key2 = f"{a}||{t_clean}"
    
    if key1 in BLOCKLIST or key2 in BLOCKLIST:
        continue  # 블랙리스트 곡 스킵
    cleaned_main.append(s)

if len(cleaned_main) < 15:
    good_artists = set()
    for s in cleaned_main:
        sim = s.get('similarity_score', 0)
        if sim >= 50:
            a = str(s.get('artist', '')).replace(' - Topic', '').strip().lower()
            good_artists.add(a)
    
    discovery_pool = []
    for sid, info in song_temps.items():
        if 1 <= info.get('total_plays', 0) <= 3:
            a = str(info.get('artist', '')).replace(' - Topic', '').strip().lower()
            t = str(info.get('title', '')).lower()
            t_clean = re.sub(r'\s*\([^)]*\)', '', t).strip()
            if f"{a}||{t}" not in BLOCKLIST and f"{a}||{t_clean}" not in BLOCKLIST:
                if not any(s.get('song_id') == sid for s in cleaned_main):
                    if a in good_artists:
                        discovery_pool.append(info)
    
    random.shuffle(discovery_pool)
    for dp in discovery_pool:
        if len(cleaned_main) >= 15:
            break
        dp_copy = dp.copy()
        dp_copy['reason'] = 'Discovery (새 발견)'
        dp_copy['similarity_score'] = 0
        if seed_vector is not None and similarity_engine:
            dp_copy['similarity_score'] = similarity_engine.calculate_similarity_to_vector(sid, seed_vector) * 100
        cleaned_main.append(dp_copy)

main_playlist = cleaned_main[:10]

# 중복 방지 키
used_keys = set()
for s in main_playlist:
    a = str(s.get('artist', '')).replace(' - Topic', '').strip().lower()
    t = str(s.get('title', '')).lower()
    used_keys.add(f"{a}||{t}")
used_keys.update(BLOCKLIST)

# ── Step 2: 외부 신곡 (시작곡 전달) ──
discovery = ExternalDiscoveryEngine(API_KEY, song_temps)
ext_all = discovery.discover_new_songs(
    n=15, 
    discovery_preset='default',
    seed_tracks=START_SONGS,
    seed_vector=seed_vector,
    similarity_engine=similarity_engine
)

ext_clean = []
for p in ext_all:
    a = p['artist'].lower()
    t = p['track'].lower()
    t_clean = re.sub(r'\s*\([^)]*\)', '', t).strip()
    key1 = f"{a}||{t}"
    key2 = f"{a}||{t_clean}"
    if key1 not in used_keys and key2 not in used_keys:
        ext_clean.append(p)
        used_keys.add(key1)

random.shuffle(ext_clean)
ext_picks = ext_clean[:2]

# ── Step 3: Nostalgia (시작곡 전달) ──
nostalgia = NostalgiaEngine(API_KEY, song_temps, CSV_PATH, 1998)
nostalgia_all = nostalgia.select_nostalgia_for_playlist(
    main_playlist, 
    n=10,
    seed_tracks=START_SONGS
)

nostalgia_clean = []
for p in nostalgia_all:
    key = f"{p['artist'].lower()}||{p['title'].lower()}"
    if key not in used_keys:
        nostalgia_clean.append(p)
        used_keys.add(key)

nostalgia_pick = [random.choice(nostalgia_clean[:5])] if nostalgia_clean else []

# ═══════════════════════════════════════════════════════
# 통합 플레이리스트 조립
# ═══════════════════════════════════════════════════════
final = []
# 사용자가 시작곡으로 넣은 곡들을 플레이리스트 최상단에 배치
for st in START_SONGS:
    final.append({
        'artist': st['artist'],
        'title': st['title'],
        'slot_type': 'start',
    })

main_idx = 0
ext_idx = 0

nostalgia_pos = random.randint(5 + len(START_SONGS), 10 + len(START_SONGS)) if nostalgia_pick else -1
ext_positions = [4 + len(START_SONGS), 12 + len(START_SONGS)]

total = len(main_playlist) + len(ext_picks) + len(nostalgia_pick)

for pos in range(1 + len(START_SONGS), total + len(START_SONGS) + 1):
    if pos == nostalgia_pos and nostalgia_pick:
        pick = nostalgia_pick[0]
        final.append({
            'artist': pick['artist'],
            'title': pick['title'],
            'slot_type': 'nostalgia',
            'reason': pick.get('reason', ''),
            'source': '내 기록' if pick.get('source') == 'internal' else '추측',
        })
        continue
    
    if pos in ext_positions and ext_idx < len(ext_picks):
        pick = ext_picks[ext_idx]
        ext_idx += 1
        tier_icons = {'deep_dive': '🔵', 'expand': '🟢', 'explore': '🌱'}
        tier_labels = {'deep_dive': '꽤 아는 가수', 'expand': '조금 아는 가수', 'explore': '새 가수'}
        fam = pick.get('fam_type', 'explore')
        final.append({
            'artist': pick['artist'],
            'title': pick['track'],
            'slot_type': 'ext_discovery',
            'fam_label': tier_labels.get(fam, ''),
            'fam_icon': tier_icons.get(fam, '🌱'),
            'fam_count': pick.get('fam_count', 0),
            'tag_sim': pick.get('tag_similarity', 0),
        })
        continue
    
    if main_idx < len(main_playlist):
        song = main_playlist[main_idx]
        main_idx += 1
        temp = song.get('temperature', '')
        reason = song.get('reason', temp)
        
        temp_icons = {
            'Rising': '🔥📈', 'Steady': '🔥', 'Warm': '🟡', 
            'Cool': '🧊', 'Frozen': '❄️',
        }
        icon = '🆕' if 'Discovery' in str(reason) else temp_icons.get(temp, '🎵')
        
        final.append({
            'artist': str(song.get('artist', '')).replace(' - Topic', '').strip(),
            'title': song.get('title', ''),
            'slot_type': 'main',
            'temperature': temp,
            'icon': icon,
            'reason': reason,
            'plays': song.get('total_plays', 0),
            'tag_sim': song.get('similarity_score', 0),
        })

# ═══════════════════════════════════════════════════════
# 출력
# ═══════════════════════════════════════════════════════
print()
print("╔" + "═" * 62 + "╗")
print("║  🎶  최종 통합 플레이리스트  —  user                      ║")
print("╚" + "═" * 62 + "╝")
print()

for i, s in enumerate(final, 1):
    slot = s['slot_type']
    artist = s['artist'][:20]
    title = s['title'][:32]
    
    if slot == 'start':
        print(f"  {i:>2}. 🎵 {artist:<20} — {title}")
        print(f"      [시작곡 (Seed)]")
    
    elif slot == 'main':
        icon = s.get('icon', '🎵')
        temp = s.get('temperature', '')
        plays = s.get('plays', 0)
        sim = s.get('tag_sim', 0)
        label = 'Discovery' if 'Discovery' in str(s.get('reason', '')) else temp
        print(f"  {i:>2}. {icon} {artist:<20} — {title}")
        print(f"      [{label}] 재생 {plays}회 | 시드유사도 {sim:.0f}%")
    
    elif slot == 'nostalgia':
        src = s.get('source', '')
        reason = s.get('reason', '')
        print(f"  {i:>2}. 🕰️  {artist:<20} — {title}")
        print(f"      [Nostalgia — {src}] {reason}")
    
    elif slot == 'ext_discovery':
        fam_icon = s.get('fam_icon', '🌱')
        fam_label = s.get('fam_label', '')
        fam_count = s.get('fam_count', 0)
        sim = s.get('tag_sim', 0)
        fam_str = f"보유 {fam_count}곡" if fam_count > 0 else "신규"
        print(f"  {i:>2}. {fam_icon} {artist:<20} — {title}")
        print(f"      [외부 신곡 — {fam_label}] {fam_str} | 시드유사도 {sim*100:.0f}%")
    
    print()

print("─" * 64)
counts = {}
for s in final:
    counts[s['slot_type']] = counts.get(s['slot_type'], 0) + 1
labels = {'start': '🎵 시작곡', 'main': '🎵 메인', 'nostalgia': '🕰️ Nostalgia', 'ext_discovery': '🆕 외부 신곡'}
for k, label in labels.items():
    if counts.get(k, 0) > 0:
        print(f"  {label}: {counts[k]}곡")
print(f"  총: {len(final)}곡")
print("─" * 64)
