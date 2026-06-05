"""
장르 파괴 원인 진단:
Multi-Signal 유사도가 장르를 구분하지 못하는 이유를 수치로 보여줌
"""
import sys, os, io
sys.path.append(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\core')
sys.stdout.reconfigure(encoding='utf-8')
from lifecycle_recommender import run_pipeline

csv_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
meta_p = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\data\caches\ytm_metadata_cache.csv'

old = sys.stdout
sys.stdout = io.TextIOWrapper(open(os.devnull, 'wb'), encoding='utf-8')
r = run_pipeline(csv_path=csv_p, user_name='diag', playlist_size=5, preset='default', metadata_path=meta_p, user_birth_year=1998, skip_external=True)
sys.stdout = old
sys.stdout.reconfigure(encoding='utf-8')

mixer = r['mixer']
ms = mixer.similarity_engine

# 시드: Love Is All (검정치마)
seed_id = 'Love Is All - The Black Skirts - Topic'

# 같은 장르 vs 다른 장르 비교
targets = [
    ('같은 아티스트', 'Antifreeze - The Black Skirts - Topic'),
    ('같은 장르(인디)', 'TOMBOY - HYUKOH - Topic'),
    ('같은 장르(인디)', 'for lovers who hesitate (주저하는 연인들을 위해) - JANNABI - Topic'),
    ('다른 장르(발라드)', '페이지원 - SG Wannabe - Topic'),
    ('다른 장르(발라드)', '세글자 - SG Wannabe - Topic'),
    ('다른 장르(아이돌)', 'Shoot Me - DAY6 - Topic'),
    ('다른 장르(아이돌)', 'Ring Ding Dong - SHINee - Topic'),
]

print("=" * 90)
print("🔍 장르 파괴 원인 진단: Love Is All(검정치마) 기준")
print("=" * 90)

from multi_signal_engine import MultiSignalSimilarityEngine

print(f"\n{'장르':10s} | {'대상곡':35s} | {'종합':5s} | {'가사':5s} | {'오디오':5s} | {'메타':5s} | {'태그':5s}")
print("-" * 90)

for label, tid in targets:
    total = ms.calculate_similarity(seed_id, tid)
    bd = ms.get_signal_breakdown(seed_id, tid)
    lyrics = bd.get('lyrics')
    audio = bd.get('audio')
    meta = bd.get('metadata')
    tag = bd.get('tag')
    
    ly_s = f"{lyrics:.2f}" if lyrics is not None else 'N/A'
    au_s = f"{audio:.2f}" if audio is not None else 'N/A'
    me_s = f"{meta:.2f}" if meta is not None else 'N/A'
    ta_s = f"{tag:.2f}" if tag is not None else 'N/A'
    
    print(f"{label:10s} | {tid[:35]:35s} | {total:.2f} | {ly_s:5s} | {au_s:5s} | {me_s:5s} | {ta_s:5s}")

# 최종 가중치 합산
print(f"\n\n📊 가중치 구성:")
print(f"  태그: 25% ← 현재 없음! → 랜덤 0.25~0.35 (장르 무관 동일값)")
print(f"  가사: 30% ← 유의미하지만 다른장르도 0.3~0.5")
print(f"  오디오: 30% ← ⚠️ 모든 곡이 0.90~0.99 (장르 무관!)")
print(f"  메타: 15% ← 유의미하지만 비중 너무 작음")
print(f"\n  → 결론: 총 55%(태그25%+오디오30%)가 장르 구분 불가!")

# 맥락 점수 시뮬레이션
print(f"\n\n📐 맥락 점수 시뮬레이션 (곡유사도×0.6 + 아티스트유사도×0.4):")
for label, tid in targets:
    song_sim = ms.calculate_similarity(seed_id, tid) if ms else 0
    artist = tid.split(' - ')[1] if ' - ' in tid else '?'
    art_sim = mixer._get_artist_similarity(tid.rsplit(' - ', 2)[0] + ' - ' + artist + ' - Topic', 'Love Is All - The Black Skirts - Topic'.rsplit(' - ', 2)[0] + ' - The Black Skirts - Topic')
    context = song_sim * 0.6 + art_sim * 0.4
    print(f"  {label:10s} {artist[:15]:15s} → 곡:{song_sim:.2f} × 0.6 + 아티:{art_sim:.2f} × 0.4 = 맥락:{context:.2f}")
