# -*- coding: utf-8 -*-
"""
생애주기 가설 검증 (클린 재현 스크립트)
======================================
README의 핵심 통계를 개인 청취 로그로 처음부터 다시, 재현 가능하게 계산한다.

데이터
- 김차민_features.csv          : 재생 로그(시간순). familiarity=곡별 재생순번, is_skipped, title/artist
- 호감도_라벨링.csv / 재교정    : 수동 호감도 라벨(1~5)
- _new_takeout_parsed.csv      : 약 2달 뒤(6월) 신규 재생 → 홀드아웃

핵심 조작화
- '노출 기반 간격(exposure interval)': 전체 재생을 시간순 정렬했을 때
  같은 곡의 연속 재생 사이에 '낀 다른 재생 수'. 날짜가 아니라 '곡 수'로 재므로
  한동안 음악을 안 들은 드라우트 구간이 상쇄된다.
- '질림 기울기(cooling slope)': 곡별 (재생순번 ↔ 노출간격) 스피어만 ρ.
  양수가 크면 들을수록 간격이 벌어진다 = 가파르게 질린다.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding='utf-8')

BASE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.join(BASE, '유튜브 뮤직 로그들', '김차민')
FEATURES = os.path.join(LOGDIR, '김차민_features.csv')
LABEL1 = os.path.join(LOGDIR, '호감도_라벨링.csv')
LABEL2 = os.path.join(LOGDIR, '호감도_재교정.csv')
NEW_TAKEOUT = os.path.join(LOGDIR, '_new_takeout_parsed.csv')

MIN_PLAYS = 8   # 곡별 추세를 보려면 최소 재생 수

def hr(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


# ----------------------------------------------------------------------
# 0. 로드 & 노출 기반 간격 계산
# ----------------------------------------------------------------------
hr("[0] 데이터 로드 & 노출 기반 간격 계산")
df = pd.read_csv(FEATURES, encoding='utf-8-sig')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
df['global_order'] = np.arange(len(df))   # 전체 재생 순번(시간순)
print(f"  재생 {len(df):,}건 · 곡 {df['song_id'].nunique():,}개 · "
      f"{df['timestamp'].min().date()} ~ {df['timestamp'].max().date()}")

# 곡별로: 같은 곡 연속 재생 사이의 '낀 다른 재생 수' = 노출 간격
df['exposure_interval'] = np.nan
for sid, g in df.groupby('song_id'):
    idx = g.index
    order = g['global_order'].values
    # 두 번째 재생부터 직전 재생과의 전체-순번 차이(곡 수 간격)
    gaps = np.diff(order)
    df.loc[idx[1:], 'exposure_interval'] = gaps

# 곡별 재생순번(familiarity가 결측이면 등장순으로 부여)
if df['familiarity'].isna().any():
    df['familiarity'] = df.groupby('song_id').cumcount()


# ----------------------------------------------------------------------
# 1. 곡은 정말 질려가는가  (재생순번 ↔ 노출간격, 곡별 스피어만)
# ----------------------------------------------------------------------
hr("[1] 곡은 들을수록 재생 간격이 벌어지는가? (질림 존재)")
cooling = {}   # song_id -> cooling slope rho
for sid, g in df.groupby('song_id'):
    sub = g.dropna(subset=['exposure_interval'])
    if len(sub) < MIN_PLAYS:
        continue
    rho, _ = stats.spearmanr(sub['familiarity'], sub['exposure_interval'])
    if not np.isnan(rho):
        cooling[sid] = rho

cool_vals = np.array(list(cooling.values()))
print(f"  대상 곡(재생 {MIN_PLAYS}회 이상): {len(cool_vals)}곡")
print(f"  곡별 (재생순번 ↔ 노출간격) 스피어만 ρ")
print(f"    양(+)의 비율 : {np.mean(cool_vals > 0)*100:.0f}%   "
      f"(양수 = 들을수록 간격 벌어짐 = 질려감)")
print(f"    중앙값 ρ     : {np.median(cool_vals):+.2f}")
print(f"    평균 ρ       : {np.mean(cool_vals):+.2f}")
# 일표본: 분포가 0보다 큰가
t, p = stats.wilcoxon(cool_vals)
print(f"    Wilcoxon (ρ분포 ≠ 0): p = {p:.2e}")


# ----------------------------------------------------------------------
# 2. 질림은 '덜 즐김(완주)'인가 '덜 들음(빈도)'인가
#    곡 내부 (재생순번 ↔ is_skipped) 스피어만
# ----------------------------------------------------------------------
hr("[2] 질림은 완주율 저하로 나타나는가? (질 vs 빈도)")
comp_rhos = []
for sid, g in df.groupby('song_id'):
    if len(g) < MIN_PLAYS:
        continue
    if g['is_skipped'].nunique() < 2:   # 전부 0 또는 1이면 상관 정의 안 됨
        continue
    rho, _ = stats.spearmanr(g['familiarity'], g['is_skipped'])
    if not np.isnan(rho):
        comp_rhos.append(rho)
comp_rhos = np.array(comp_rhos)
print(f"  대상 곡: {len(comp_rhos)}곡")
print(f"  곡 내부 (재생순번 ↔ 스킵여부) 스피어만 ρ 중앙값: {np.median(comp_rhos):+.3f}")
print(f"    → 0 근처면 '질려도 들을 때의 완주율은 그대로' = 질림은 '덜 들음(빈도)' 쪽")


# ----------------------------------------------------------------------
# 3. ★ 무엇이 질림을 결정하는가  (질림기울기 ↔ 호감도 / 총재생수)
# ----------------------------------------------------------------------
hr("[3] ★ 무엇이 질림의 기울기를 결정하는가")
# 라벨 로드 & (title, artist) -> label
def norm(s):
    return str(s).strip().lower()

lab = pd.read_csv(LABEL1, encoding='utf-8-sig')
lab = lab.rename(columns={'곡_이름': 'title', '아티스트': 'artist', '호감도(1-5)': 'label'})
label_map = {}
for _, r in lab.iterrows():
    if pd.notna(r.get('label')):
        label_map[(norm(r['title']), norm(r['artist']))] = float(r['label'])

# 재교정(최종) 라벨로 덮어쓰기 — 마지막 숫자 열이 최종 라벨
re = pd.read_csv(LABEL2, encoding='utf-8-sig')
val_col = None
for c in reversed(re.columns):
    if re[c].apply(lambda x: isinstance(x, (int, float)) and pd.notna(x)).any():
        coerced = pd.to_numeric(re[c], errors='coerce')
        if coerced.notna().sum() > 5 and coerced.max() <= 5.5:
            val_col = c
            break
if val_col is not None:
    for _, r in re.iterrows():
        v = pd.to_numeric(pd.Series([r[val_col]]), errors='coerce').iloc[0]
        if pd.notna(v):
            label_map[(norm(r['title']), norm(r['artist']))] = float(v)
print(f"  수동 호감도 라벨: {len(label_map)}곡")

# 곡별 메타(첫 등장 title/artist, 총재생수) + cooling slope + label
song_meta = df.groupby('song_id').agg(
    title=('title', 'first'),
    artist=('artist', 'first'),
    total_plays=('song_id', 'size'),
).reset_index()
song_meta['cooling'] = song_meta['song_id'].map(cooling)
song_meta['label'] = song_meta.apply(
    lambda r: label_map.get((norm(r['title']), norm(r['artist'])), np.nan), axis=1)

valid = song_meta.dropna(subset=['cooling', 'label'])
print(f"  검정 대상(질림기울기+라벨 둘 다 있는 곡): n = {len(valid)}")

r1, p1 = stats.spearmanr(valid['cooling'], valid['label'])
r2, p2 = stats.spearmanr(valid['cooling'], valid['total_plays'])
r3, p3 = stats.spearmanr(valid['label'], valid['total_plays'])
print(f"\n  질림기울기 ↔ 호감도라벨 : ρ = {r1:+.2f}  (p = {p1:.2e}, n={len(valid)})")
print(f"     → 음수면 '사랑하는 곡일수록 완만하게 질린다'")
print(f"  질림기울기 ↔ 총재생수   : ρ = {r2:+.2f}  (p = {p2:.2e})")
print(f"     → 0 근처면 '몇 번 들었나'는 질림과 무관")
print(f"  호감도라벨 ↔ 총재생수   : ρ = {r3:+.2f}  (p = {p3:.2e})")
print(f"     → 사랑하는 곡 ≠ 많이 들은 곡")


# ----------------------------------------------------------------------
# 4. 두 유형인가 연속 스펙트럼인가  (생애곡선 KMeans 실루엣)
# ----------------------------------------------------------------------
hr("[4] '질리는 곡 / 안 질리는 곡' 두 덩어리인가? (군집)")
# 각 곡 생애를 5구간으로 나눠 평균 노출간격 곡선 → 표준화 → KMeans k=2
curves, ids = [], []
N_BIN = 5
for sid, g in df.groupby('song_id'):
    sub = g.dropna(subset=['exposure_interval']).sort_values('familiarity')
    if len(sub) < MIN_PLAYS:
        continue
    life = np.linspace(0, 1, len(sub))
    binned = []
    ok = True
    for b in range(N_BIN):
        mask = (life >= b / N_BIN) & (life <= (b + 1) / N_BIN)
        if mask.sum() == 0:
            ok = False
            break
        binned.append(sub['exposure_interval'].values[mask].mean())
    if ok:
        curves.append(binned)
        ids.append(sid)
curves = np.array(curves)
if len(curves) >= 10:
    Xc = StandardScaler().fit_transform(curves)
    km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(Xc)
    sil = silhouette_score(Xc, km.labels_)
    print(f"  대상 곡: {len(curves)}곡, 생애 {N_BIN}구간 곡선")
    print(f"  KMeans(k=2) 실루엣 계수: {sil:.3f}")
    print(f"    → 0.25 미만이면 깔끔한 두 덩어리가 아니라 '연속 스펙트럼'")
else:
    print("  곡 수 부족으로 군집 생략")


# ----------------------------------------------------------------------
# 5. 홀드아웃: 4월 '식는 중' 판정이 6월 신규데이터에서 맞았나
# ----------------------------------------------------------------------
hr("[5] 홀드아웃 — 4월 '식는 중' 판정 vs 6월 신규 재생")
if os.path.exists(NEW_TAKEOUT):
    new = pd.read_csv(NEW_TAKEOUT, encoding='utf-8-sig')
    new['timestamp'] = pd.to_datetime(new['timestamp'], errors='coerce')
    last_train = df['timestamp'].max()
    new_fut = new[new['timestamp'] > last_train].copy()
    print(f"  6월 신규 재생(4월 이후): {len(new_fut):,}건  "
          f"({new_fut['timestamp'].min()} ~ {new_fut['timestamp'].max()})")
    # 신규기간 곡별 재생수 (title+artist 기준)
    new_fut['key'] = new_fut['title'].map(norm) + ' :: ' + new_fut['artist'].map(norm)
    fut_plays = new_fut.groupby('key').size()

    song_meta['key'] = song_meta['title'].map(norm) + ' :: ' + song_meta['artist'].map(norm)
    song_meta['future_plays'] = song_meta['key'].map(fut_plays).fillna(0)

    sub = song_meta.dropna(subset=['cooling'])
    # 식는 중(상위 25% cooling) vs 안정(하위 25%)
    hi = sub['cooling'] >= sub['cooling'].quantile(0.75)
    lo = sub['cooling'] <= sub['cooling'].quantile(0.25)
    print(f"  '식는 중' 곡(질림기울기 상위25%) 6월 평균 재생수 : "
          f"{sub.loc[hi, 'future_plays'].mean():.2f}")
    print(f"  '안정'   곡(질림기울기 하위25%) 6월 평균 재생수 : "
          f"{sub.loc[lo, 'future_plays'].mean():.2f}")
    u, pu = stats.mannwhitneyu(sub.loc[hi, 'future_plays'], sub.loc[lo, 'future_plays'])
    print(f"  Mann-Whitney U: p = {pu:.2e}")
    print(f"    → 식는 곡이 실제로 6월에 덜 재생됐으면 위치 신호가 검증된 것")
else:
    print("  신규 takeout 파일 없음 — 홀드아웃 생략")

hr("완료")
print("위 수치를 README에 그대로 반영하면 재현 가능한 검증 섹션이 됩니다.")
