"""Frozen에서 구출된 곡 분석 — proactive_score의 Frozen 기준일 연장 효과"""
import os, sys
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(os.path.join(BASE_DIR, 'core'))

from lifecycle_recommender import *

# 데이터 로드
csv_path = os.path.join(BASE_DIR, 'Takeout', 'YouTube 및 YouTube Music', '시청 기록', 'ytm_history_features.csv')
metadata_path = os.path.join(BASE_DIR, 'data', 'caches', 'ytm_metadata_cache.csv')
df = pd.read_csv(csv_path, encoding='utf-8-sig')

normalizer = ArtistNameNormalizer()
df = normalizer.normalize_dataframe(df)

tier_classifier = ArtistTierClassifier(df)
tier_map = tier_classifier.classify_tiers()

# --- A: proactive_score 반영 (현재 방식) ---
temp_tracker_new = SongTemperatureTracker(df, tier_map, metadata_path=metadata_path, user_birth_year=1998)
new_temps = temp_tracker_new.classify_all_songs()

# --- B: proactive_score 무시 (원래 방식 시뮬레이션) ---
# avg_proactive를 0으로 강제해서 90일 기준 적용
df_no_proactive = df.copy()
if 'proactive_score' in df_no_proactive.columns:
    df_no_proactive['proactive_score'] = 0.0
    df_no_proactive['is_proactive'] = 0

temp_tracker_old = SongTemperatureTracker(df_no_proactive, tier_map, metadata_path=metadata_path, user_birth_year=1998)
old_temps = temp_tracker_old.classify_all_songs()

# --- 비교: 원래 Frozen인데 현재 Frozen이 아닌 곡 찾기 ---
rescued = []
for song_id in old_temps:
    old_temp = old_temps[song_id]['temperature']
    new_temp = new_temps[song_id]['temperature']
    if old_temp == 'Frozen' and new_temp != 'Frozen':
        info = new_temps[song_id]
        # 곡별 검색 기록 집계
        song_plays = df[df['song_id'] == song_id]
        search_count = song_plays['is_proactive'].sum() if 'is_proactive' in song_plays.columns else 0
        avg_ps = song_plays['proactive_score'].mean() if 'proactive_score' in song_plays.columns else 0
        session_starts = song_plays['is_session_start'].sum() if 'is_session_start' in song_plays.columns else 0
        deep_dives = song_plays['is_artist_deep_dive'].sum() if 'is_artist_deep_dive' in song_plays.columns else 0
        
        rescued.append({
            'title': info['title'][:35],
            'artist': info['artist'][:25],
            'old_temp': old_temp,
            'new_temp': new_temp,
            'total_plays': info['total_plays'],
            'days_since_last': info['days_since_last'],
            'avg_proactive': round(avg_ps, 3),
            'search_plays': int(search_count),
            'session_starts': int(session_starts),
            'deep_dives': int(deep_dives),
            'skip_rate': info['skip_rate'],
        })

rescued.sort(key=lambda x: x['avg_proactive'], reverse=True)

print(f"\n{'='*80}")
print(f"🔓 Frozen에서 구출된 곡 — {len(rescued)}곡")
print(f"{'='*80}")
print(f"  이 곡들은 원래 90일+ 미청취로 Frozen이었지만,")
print(f"  과거에 능동적으로 자주 듣던 곡이라 120~150일로 연장되어 Warm으로 살아남음.\n")

print(f"  {'곡명':<37} {'가수':<22} {'→':>3} {'재생':>4} {'마지막':>6} {'능동':>5} {'검색':>4} {'세션':>4} {'연속':>5} {'스킵':>5}")
print(f"  {'-'*95}")
for r in rescued:
    print(f"  {r['title']:<37} {r['artist']:<22} {r['new_temp']:>5} "
          f"{r['total_plays']:>4} {r['days_since_last']:>5}일 "
          f"{r['avg_proactive']:>5.2f} {r['search_plays']:>4} {r['session_starts']:>4} {r['deep_dives']:>5} "
          f"{r['skip_rate']:>5.0%}")

# 구출된 곡 중 의미있는 패턴 분석
if rescued:
    avg_proactive_rescued = sum(r['avg_proactive'] for r in rescued) / len(rescued)
    avg_days = sum(r['days_since_last'] for r in rescued) / len(rescued)
    avg_plays = sum(r['total_plays'] for r in rescued) / len(rescued)
    
    print(f"\n--- 구출된 곡들의 요약 통계 ---")
    print(f"  평균 능동성 점수: {avg_proactive_rescued:.3f}")
    print(f"  평균 마지막 재생: {avg_days:.0f}일 전")
    print(f"  평균 재생 횟수: {avg_plays:.1f}회")
    print(f"  Frozen 기준 90일을 넘지만 150일 이내에 있는 곡들 = '아직 돌아올 수 있는 곡'")
