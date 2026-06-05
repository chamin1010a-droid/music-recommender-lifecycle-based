"""검정치마 곡들이 잔나비 Summer와 유사도가 왜 낮은지 분해 분석"""
import sys, os, io
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
sim_engine = mixer.similarity_engine

# 시드: 잔나비 Summer
seed = None
for info in mixer.song_temps.values():
    if '뜨거운' in info.get('title', '') and 'JANNABI' in info.get('artist', ''):
        seed = info
        break

seed_id = seed['song_id']
seed_artist = seed['artist'].replace(' - Topic', '')
print(f"시드: {seed['title'][:40]} ({seed_artist})")
print(f"시드 ID: {seed_id}\n")

# 검정치마 상위 곡들과의 유사도 분해
target_artists = ['The Black Skirts', 'HANRORO', 'JANNABI', 'Car, the Garden', 'DAY6']
print(f"{'곡명':30s} | {'아티스트':15s} | {'곡유사':>6s} | {'아티유사':>6s} | {'맥락':>6s} | {'가사':>6s} | {'오디오':>6s} | {'메타':>6s}")
print("-" * 115)

for artist in target_artists:
    songs = sorted(
        [s for s in mixer.song_temps.values() if s.get('artist','').replace(' - Topic','') == artist],
        key=lambda x: x.get('total_plays', 0), reverse=True
    )[:5]  # 상위 5곡
    
    for s in songs:
        sid = s['song_id']
        title = s.get('title', '')[:30]
        plays = s.get('total_plays', 0)
        
        # 1. Multi-Signal 곡 유사도 분해
        song_sim = 0
        lyrics_sim = 0
        audio_sim = 0
        meta_sim = 0
        
        if sim_engine:
            signals = {}
            
            # 가사
            if hasattr(sim_engine, 'lyrics_engine') and sim_engine.lyrics_engine:
                try:
                    ls = sim_engine.lyrics_engine.calculate_similarity(seed_id, sid)
                    if ls is not None:
                        lyrics_sim = ls
                except:
                    pass
            
            # 오디오
            if hasattr(sim_engine, 'audio_engine') and sim_engine.audio_engine:
                try:
                    aus = sim_engine.audio_engine.calculate_similarity(seed_id, sid)
                    if aus is not None:
                        audio_sim = max(0.0, min(float(aus), 1.0))
                except:
                    pass
            
            # 메타
            if hasattr(sim_engine, '_metadata_similarity'):
                try:
                    ms = sim_engine._metadata_similarity(seed_id, sid)
                    if ms is not None:
                        meta_sim = ms
                except:
                    pass
            
            # 전체 곡 유사도 (가사40 + 오디오25 + 메타35)
            song_sim = lyrics_sim * 0.40 + audio_sim * 0.25 + meta_sim * 0.35
        
        # 2. 아티스트 유사도
        artist_sim = mixer._get_artist_similarity(artist, seed_artist)
        
        # 3. 맥락 = 곡유사도*0.6 + 아티유사도*0.4
        context = song_sim * 0.6 + artist_sim * 0.4
        
        print(f"{title:30s} | {artist:15s} | {song_sim:6.3f} | {artist_sim:6.3f} | {context:6.3f} | {lyrics_sim:6.3f} | {audio_sim:6.3f} | {meta_sim:6.3f}")
    print()
