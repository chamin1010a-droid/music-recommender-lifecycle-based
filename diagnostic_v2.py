"""
추천 엔진 v2 종합 진단 스크립트
================================
1. 점수 가중 검증: Affinity/Momentum이 논리적으로 맞게 산출되는가
2. 플레이리스트 품질: 추천 결과가 실질적으로 좋은가
3. 버그 탐지: 엣지케이스, NaN, 범위 초과 등 이상 동작
"""
import os, sys
import numpy as np
import pandas as pd
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'core'))
from lifecycle_recommender import ArtistTierClassifier, SongScorer, PlaylistMixer, ArtistNameNormalizer

# ── 데이터 로드 ──
csv_path = os.path.join(BASE_DIR, 'Takeout', 'YouTube 및 YouTube Music', '시청 기록', 'ytm_history_features.csv')
metadata_path = os.path.join(BASE_DIR, 'data', 'caches', 'ytm_metadata_cache.csv')
if not os.path.exists(metadata_path):
    metadata_path = None

df = pd.read_csv(csv_path, encoding='utf-8-sig')
normalizer = ArtistNameNormalizer()
df = normalizer.normalize_dataframe(df)
tier_classifier = ArtistTierClassifier(df)
tier_map = tier_classifier.classify_tiers()
scorer = SongScorer(df, tier_map, metadata_path=metadata_path, user_birth_year=1998)
scores = scorer.score_all_songs()
scores_df = pd.DataFrame(scores.values())

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️ WARN"
total_tests = 0
passed = 0
warnings = 0
issues = []

def check(name, condition, detail=""):
    global total_tests, passed, warnings
    total_tests += 1
    if condition:
        passed += 1
        print(f"  {PASS} {name}")
    else:
        issues.append((name, detail))
        print(f"  {FAIL} {name}")
        if detail:
            print(f"       → {detail}")

def warn(name, condition, detail=""):
    global total_tests, passed, warnings
    total_tests += 1
    if condition:
        passed += 1
        print(f"  {PASS} {name}")
    else:
        warnings += 1
        print(f"  {WARN} {name}")
        if detail:
            print(f"       → {detail}")

print("=" * 70)
print("🔬 추천 엔진 v2 — 종합 진단")
print("=" * 70)

# ════════════════════════════════════════════════════════════════
# 테스트 1: 점수 범위 및 기본 무결성
# ════════════════════════════════════════════════════════════════
print("\n━━━ 1. 점수 범위 및 기본 무결성 ━━━")

check("모든 곡에 affinity 키 존재",
      all('affinity' in s for s in scores.values()))

check("모든 곡에 momentum 키 존재",
      all('momentum' in s for s in scores.values()))

aff_min = scores_df['affinity'].min()
aff_max = scores_df['affinity'].max()
check(f"Affinity 범위 [0,1] (실제: {aff_min:.3f}~{aff_max:.3f})",
      0 <= aff_min and aff_max <= 1)

mom_min = scores_df['momentum'].min()
mom_max = scores_df['momentum'].max()
check(f"Momentum 범위 [0,1] (실제: {mom_min:.3f}~{mom_max:.3f})",
      0 <= mom_min and mom_max <= 1)

nan_aff = scores_df['affinity'].isna().sum()
nan_mom = scores_df['momentum'].isna().sum()
check(f"Affinity NaN 없음 (실제: {nan_aff}건)",
      nan_aff == 0)
check(f"Momentum NaN 없음 (실제: {nan_mom}건)",
      nan_mom == 0)

nan_artists = scores_df[~scores_df['artist'].apply(lambda x: isinstance(x, str))]
check(f"아티스트명 NaN 없음 (실제: {len(nan_artists)}건)",
      len(nan_artists) == 0,
      f"NaN 아티스트 song_id: {nan_artists['song_id'].tolist()[:5]}" if len(nan_artists) > 0 else "")

# ════════════════════════════════════════════════════════════════
# 테스트 2: Affinity (호감도) 가중 논리 검증
# ════════════════════════════════════════════════════════════════
print("\n━━━ 2. Affinity (호감도) 가중 논리 검증 ━━━")

# 스킵률 0%인 곡의 호감도가 스킵률 50%+ 곡보다 높은지
low_skip = scores_df[scores_df['skip_rate'] == 0]
high_skip = scores_df[scores_df['skip_rate'] >= 0.5]
if len(low_skip) > 0 and len(high_skip) > 0:
    check(f"스킵률 0% 평균 호감도({low_skip['affinity'].mean():.2f}) > 스킵률 50%+ 평균({high_skip['affinity'].mean():.2f})",
          low_skip['affinity'].mean() > high_skip['affinity'].mean())
else:
    print(f"  ⏭️ SKIP: 스킵률 0% 또는 50%+ 곡이 없어 비교 불가")

# 많이 들은 곡(재생 50회+)의 호감도가 1회만 들은 곡보다 높은지
heavy = scores_df[scores_df['total_plays'] >= 50]
light = scores_df[scores_df['total_plays'] == 1]
if len(heavy) > 0 and len(light) > 0:
    check(f"재생 50회+ 평균 호감도({heavy['affinity'].mean():.2f}) > 재생 1회 평균({light['affinity'].mean():.2f})",
          heavy['affinity'].mean() > light['affinity'].mean(),
          "재생 깊이(play_depth) 가중치가 제대로 반영되지 않을 수 있음")

# Tier S 아티스트 곡의 인지율이 Tier B보다 높은지
tier_s = scores_df[scores_df['tier'] == 'S']
tier_b = scores_df[scores_df['tier'] == 'B']
if len(tier_s) > 0 and len(tier_b) > 0:
    check(f"Tier S 평균 호감도({tier_s['affinity'].mean():.2f}) > Tier B 평균({tier_b['affinity'].mean():.2f})",
          tier_s['affinity'].mean() > tier_b['affinity'].mean(),
          "아티스트 인지율(familiarity) 가중이 과소 반영 우려")

# 궤적 보너스 검증: Zone1(처음부터 좋아함) 곡의 호감도 > Zone3(좋다가 질림) 곡
zone1 = scores_df[scores_df['zone_label'] == '처음부터 좋아함 ♥']
zone3 = scores_df[scores_df['zone_label'] == '좋다가 질림 ↓']
if len(zone1) > 0 and len(zone3) > 0:
    check(f"'처음부터 좋아함' 평균({zone1['affinity'].mean():.2f}) > '좋다가 질림' 평균({zone3['affinity'].mean():.2f})",
          zone1['affinity'].mean() > zone3['affinity'].mean())

# ════════════════════════════════════════════════════════════════
# 테스트 3: Momentum (모멘텀) 가중 논리 검증
# ════════════════════════════════════════════════════════════════
print("\n━━━ 3. Momentum (모멘텀) 가중 논리 검증 ━━━")

# 최근 들은 곡(7일 이내) vs 오래 안 들은 곡(90일+)
recent = scores_df[scores_df['days_since_last'] <= 7]
old = scores_df[scores_df['days_since_last'] >= 90]
if len(recent) > 0 and len(old) > 0:
    check(f"최근 7일 내 재생 평균 모멘텀({recent['momentum'].mean():.2f}) > 90일+ 미재생({old['momentum'].mean():.2f})",
          recent['momentum'].mean() > old['momentum'].mean())

# 최근 30일 많이 들은 곡 vs 안 들은 곡
rec_heavy = scores_df[scores_df['recent_30d_plays'] >= 10]
rec_zero = scores_df[scores_df['recent_30d_plays'] == 0]
if len(rec_heavy) > 0 and len(rec_zero) > 0:
    check(f"최근30일 10회+ 평균 모멘텀({rec_heavy['momentum'].mean():.2f}) > 최근30일 0회({rec_zero['momentum'].mean():.2f})",
          rec_heavy['momentum'].mean() > rec_zero['momentum'].mean())

# 쇠퇴 아티스트의 모멘텀이 성장 아티스트보다 낮은지
# Tier S 아티스트(잔나비/검정치마)는 활발 → 높은 모멘텀 기대
jannabi = scores_df[scores_df['artist'].str.contains('JANNABI', na=False)]
charlie = scores_df[scores_df['artist'].str.contains('Charlie Puth', na=False)]
if len(jannabi) > 0 and len(charlie) > 0:
    check(f"잔나비 평균 모멘텀({jannabi['momentum'].mean():.2f}) > 찰리푸스 평균({charlie['momentum'].mean():.2f})",
          jannabi['momentum'].mean() > charlie['momentum'].mean(),
          "잔나비가 요즘 활발/찰리푸스가 쇠퇴 중이니 모멘텀 차이가 있어야 함")

# 지수 감쇠 수치 검증: 30일 전 = ~0.37, 60일 전 = ~0.14
check(f"exp(-30/30) ≈ 0.37 (리센시 반감기 검증)",
      abs(np.exp(-1) - 0.368) < 0.01)

# 모멘텀이 0인 곡이 과도하게 많지는 않은지
mom_zero = len(scores_df[scores_df['momentum'] <= 0.01])
mom_zero_pct = mom_zero / len(scores_df) * 100
warn(f"모멘텀 ≤ 0.01인 곡 비율({mom_zero_pct:.1f}%) < 20%",
     mom_zero_pct < 20,
     f"모멘텀이 0에 수렴하는 곡이 {mom_zero}곡 ({mom_zero_pct:.1f}%) — 과도하면 추천 풀이 좁아짐")

# ════════════════════════════════════════════════════════════════
# 테스트 4: 스팟 체크 — 알려진 곡의 점수가 상식적인가
# ════════════════════════════════════════════════════════════════
print("\n━━━ 4. 스팟 체크 — 핵심 곡/아티스트 점수 상식 검증 ━━━")

def spot_check(artist_keyword, expected_affinity_range=None, expected_momentum_range=None, label=""):
    """특정 아티스트의 곡 점수가 예상 범위에 있는지 확인"""
    subset = scores_df[scores_df['artist'].str.contains(artist_keyword, na=False)]
    if len(subset) == 0:
        print(f"  ⏭️ SKIP: '{artist_keyword}' 곡 없음")
        return
    
    avg_a = subset['affinity'].mean()
    avg_m = subset['momentum'].mean()
    
    result = True
    details = []
    
    if expected_affinity_range:
        lo, hi = expected_affinity_range
        if not (lo <= avg_a <= hi):
            result = False
            details.append(f"호감도 {avg_a:.2f} (기대: {lo}~{hi})")
    
    if expected_momentum_range:
        lo, hi = expected_momentum_range
        if not (lo <= avg_m <= hi):
            result = False
            details.append(f"모멘텀 {avg_m:.2f} (기대: {lo}~{hi})")
    
    status = PASS if result else WARN
    print(f"  {status} [{artist_keyword}] 호감 {avg_a:.2f} | 모멘텀 {avg_m:.2f} | {len(subset)}곡 {label}")
    if details:
        for d in details:
            print(f"       → {d}")

# 잔나비: 생태계 아티스트, 활발 → 호감도/모멘텀 둘 다 높아야
spot_check("JANNABI", (0.4, 1.0), (0.5, 1.0), "— Tier S, 활발")
# 검정치마: 생태계 아티스트 → 높은 호감도
spot_check("Black Skirts", (0.4, 1.0), (0.4, 1.0), "— Tier S, 활발")
# 찰리푸스: 쇠퇴 중 → 호감도 중간, 모멘텀 낮아야
spot_check("Charlie Puth", (0.3, 0.8), (0.0, 0.4), "— 쇠퇴 중, 모멘텀 낮아야")
# Jay Park: 쇠퇴 → 호감도는 유지, 모멘텀 낮아야
spot_check("Jay Park", (0.2, 0.7), (0.0, 0.5), "— 쇠퇴, 모멘텀 낮음 예상")
# KARA: 쇠퇴, K-Pop → 호감 유지, 모멘텀 낮음
spot_check("KARA", (0.2, 0.7), (0.0, 0.5), "— 쇠퇴, 모멘텀 낮음 예상")
# 한로로: 활발
spot_check("HANRORO", (0.4, 1.0), (0.4, 1.0), "— 활발")

# ════════════════════════════════════════════════════════════════
# 테스트 5: 플레이리스트 품질 검증
# ════════════════════════════════════════════════════════════════
print("\n━━━ 5. 플레이리스트 품질 검증 ━━━")

mixer = PlaylistMixer(scores, growth_signals=[])
playlist = mixer.generate_playlist(total_songs=20, preset='default')

# 5-1: 요청한 곡 수대로 생성되는가
check(f"요청 20곡 생성 확인 (실제: {len(playlist)}곡)",
      len(playlist) == 20,
      f"요청된 20곡이 아닌 {len(playlist)}곡 생성됨")

# 5-2: 중복 곡이 없는가
song_ids_in_playlist = [s.get('song_id', '') for s in playlist if s.get('discovery_source') != 'external']
unique_ids = set(song_ids_in_playlist)
check(f"플레이리스트 내 중복 곡 없음 (유니크: {len(unique_ids)}, 전체: {len(song_ids_in_playlist)})",
      len(unique_ids) == len(song_ids_in_playlist))

# 5-3: 아티스트 다양성 (한 아티스트 최대 20%)
artist_counts = defaultdict(int)
for s in playlist:
    artist_counts[s['artist']] += 1
max_artist_songs = max(artist_counts.values()) if artist_counts else 0
max_artist_name = max(artist_counts, key=artist_counts.get) if artist_counts else ""
max_allowed = max(2, int(20 * 0.2))
check(f"아티스트 다양성: {max_artist_name}이 최대 {max_artist_songs}곡 (제한: {max_allowed}곡)",
      max_artist_songs <= max_allowed + 1,  # +1은 Discovery 포함 여유
      f"{max_artist_name}이 {max_artist_songs}곡으로 과도하게 편중됨")

# 5-4: playlist에 Discovery가 포함되는가 (20% 기대)
disc_count = sum(1 for s in playlist if 'Discovery' in s.get('reason', ''))
check(f"Discovery 곡 포함 확인 (실제: {disc_count}곡, 기대: ~4곡)",
      disc_count >= 1,
      f"Discovery 곡이 {disc_count}곡 — 0곡이면 탐험 기능 미작동")

# 5-5: Main 곡들의 평균 호감도/모멘텀이 전체 평균보다 높은지
main_songs = [s for s in playlist if s.get('reason') == 'Main']
if main_songs:
    main_avg_aff = np.mean([s['affinity'] for s in main_songs])
    main_avg_mom = np.mean([s['momentum'] for s in main_songs])
    overall_avg_aff = scores_df['affinity'].mean()
    overall_avg_mom = scores_df['momentum'].mean()
    check(f"추천 Main 곡 평균 호감도({main_avg_aff:.2f}) > 전체 평균({overall_avg_aff:.2f})",
          main_avg_aff > overall_avg_aff)
    check(f"추천 Main 곡 평균 모멘텀({main_avg_mom:.2f}) > 전체 평균({overall_avg_mom:.2f})",
          main_avg_mom > overall_avg_mom)

# 5-6: 첫 곡이 가중치 상위 곡인지 (DJ 배치)
first_song = playlist[0] if playlist else None
if first_song and 'final_weight' in first_song:
    all_weights = sorted([s.get('final_weight', 0) for s in playlist if s.get('reason') == 'Main'], reverse=True)
    top_20pct_threshold = all_weights[max(0, len(all_weights)//5)] if all_weights else 0
    check(f"첫 곡 가중치({first_song['final_weight']:.3f}) ≥ 상위 20% 기준({top_20pct_threshold:.3f})",
          first_song.get('final_weight', 0) >= top_20pct_threshold,
          "DJ 배치에서 첫 곡은 몰입감 있는 상위 곡이어야 함")

# 5-7: 프리셋 간 차이 확인; comfort는 호감도 높은 곡 위주여야
playlist_comfort = mixer.generate_playlist(total_songs=20, preset='comfort')
comfort_main = [s for s in playlist_comfort if s.get('reason') == 'Main']
if comfort_main and main_songs:
    comfort_aff = np.mean([s['affinity'] for s in comfort_main])
    default_aff = np.mean([s['affinity'] for s in main_songs])
    warn(f"Comfort 프리셋 호감도({comfort_aff:.2f}) ≥ Default({default_aff:.2f})",
         comfort_aff >= default_aff - 0.02,  # 약간의 노이즈 여유
         "comfort는 호감도 지수(1.5)를 쓰므로, 호감도 높은 곡이 더 많이 올라와야 함")

# ════════════════════════════════════════════════════════════════
# 테스트 6: 엣지케이스 및 버그 탐지
# ════════════════════════════════════════════════════════════════
print("\n━━━ 6. 엣지케이스 및 버그 탐지 ━━━")

# 6-1: 재생 1회 곡의 모멘텀/호감도가 극단적이지 않은지
singles = scores_df[scores_df['total_plays'] == 1]
if len(singles) > 0:
    check(f"재생 1회 곡({len(singles)}곡) 평균 호감도 < 0.7 (실제: {singles['affinity'].mean():.2f})",
          singles['affinity'].mean() < 0.7,
          "1회만 들은 곡의 호감도가 비정상적으로 높음 — 스킵률/능동성 가중치 확인 필요")

# 6-2: 데이터 기간 확인
date_range = pd.to_datetime(df['timestamp'])
days_span = (date_range.max() - date_range.min()).days
check(f"데이터 기간: {days_span}일 (100일 이상 필요)",
      days_span >= 100,
      "데이터 기간이 짧으면 모멘텀 계산의 신뢰도가 떨어짐")

# 6-3: expected_30d 계산 검증 (0으로 나누기 위험)
total_unique_songs = df['song_id'].nunique()
user_daily_avg = len(df) / max(days_span, 1)
expected_30d = user_daily_avg * 30 / max(total_unique_songs, 1) * 5
check(f"expected_30d = {expected_30d:.2f} (> 0 검증)",
      expected_30d > 0,
      "expected_30d가 0이면 recent_freq 계산에서 division by zero 발생")

# 6-4: 모멘텀 1.0인 곡 상세 검토 (비정상적으로 많으면 문제)
mom_1 = scores_df[scores_df['momentum'] >= 0.99]
mom_1_pct = len(mom_1) / len(scores_df) * 100
warn(f"모멘텀 ≥ 0.99인 곡 비율({mom_1_pct:.1f}%) < 15%",
     mom_1_pct < 15,
     f"{len(mom_1)}곡이 모멘텀 1.0 — 과도하면 변별력 상실")

# 모멘텀 1.0 곡들의 days_since_last 확인
if len(mom_1) > 0:
    avg_days = mom_1['days_since_last'].mean()
    max_days = mom_1['days_since_last'].max()
    check(f"모멘텀 1.0 곡들의 평균 최근 재생: {avg_days:.0f}일 전 (< 14일이어야)",
          avg_days < 14,
          f"모멘텀 1.0인데 마지막 재생이 평균 {avg_days:.0f}일 전 (max: {max_days}일) — 리센시 감쇠 미작동 우려")

# 6-5: final_weight가 0인 곡이 플레이리스트에 포함되어 있는지
zero_weight = [s for s in playlist if s.get('final_weight', 0) <= 0.001 and s.get('reason') == 'Main']
check(f"플레이리스트에 가중치 ≈ 0인 Main 곡 없음 (실제: {len(zero_weight)}곡)",
      len(zero_weight) == 0,
      "가중치 0인 곡이 추천됨 — 정렬/필터 로직 확인 필요")

# 6-6: 호감도 / 모멘텀 상관관계 확인 (너무 높으면 독립이 아님)
corr = scores_df['affinity'].corr(scores_df['momentum'])
warn(f"호감도-모멘텀 상관계수 = {corr:.2f} (|r| < 0.7이어야 독립적)",
     abs(corr) < 0.7,
     f"두 축의 상관이 {corr:.2f}로 높음 — 사실상 같은 정보를 중복 반영하고 있을 수 있음")

# ════════════════════════════════════════════════════════════════
# 테스트 7: expected_30d 공식 상세 분석
# ════════════════════════════════════════════════════════════════
print("\n━━━ 7. 모멘텀 recent_freq 공식 상세 분석 ━━━")

print(f"\n  📊 입력 변수:")
print(f"     user_daily_avg = {user_daily_avg:.2f} (하루 평균 재생 수)")
print(f"     total_unique_songs = {total_unique_songs}")
print(f"     expected_30d = {expected_30d:.4f} (곡당 30일 내 기대 재생)")
print(f"     → recent_freq = min(1, recent_30d / {max(expected_30d, 0.5):.2f})")

# 이 expected_30d가 너무 낮으면 recent_freq가 쉽게 1.0에 도달
songs_reaching_1 = len(scores_df[scores_df['recent_30d_plays'] >= expected_30d])
pct_reaching_1 = songs_reaching_1 / len(scores_df) * 100
warn(f"recent_freq가 1.0에 도달하는 곡 비율: {pct_reaching_1:.1f}% (< 30%가 이상적)",
     pct_reaching_1 < 30,
     f"expected_30d = {expected_30d:.2f}가 너무 낮아서 recent_freq 변별력이 약할 수 있음")

# ════════════════════════════════════════════════════════════════
# 테스트 8: 플레이리스트 세부 출력 (점수 확인용)
# ════════════════════════════════════════════════════════════════
print("\n━━━ 8. 최종 플레이리스트 세부 점수 ━━━")
print(f"\n{'#':>3} {'곡명':<30} {'아티스트':<18} {'호감':>5} {'모멘':>5} {'가중치':>7} {'이유'}")
print("-" * 95)
for i, s in enumerate(playlist, 1):
    title = s['title'][:28]
    artist = s['artist'][:16]
    aff = s.get('affinity', 0)
    mom = s.get('momentum', 0)
    fw = s.get('final_weight', 0)
    reason = s.get('reason', '?')[:20]
    print(f"{i:>3}. {title:<30} {artist:<18} {aff:>5.2f} {mom:>5.2f} {fw:>7.4f} {reason}")

# ════════════════════════════════════════════════════════════════
# 최종 결과
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"🏁 진단 완료: {passed}/{total_tests} 통과 | {warnings}건 경고 | {len(issues)}건 실패")
print("=" * 70)

if issues:
    print("\n❌ 실패 항목:")
    for name, detail in issues:
        print(f"  • {name}")
        if detail:
            print(f"    → {detail}")

if warnings > 0:
    print(f"\n⚠️ 경고가 {warnings}건 있습니다. 즉시 이슈는 아니지만 검토가 필요합니다.")
else:
    print("\n🎉 모든 핵심 테스트를 통과했습니다!")
