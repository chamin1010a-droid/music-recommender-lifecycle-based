# -*- coding: utf-8 -*-
"""
냉각 상태 탐지기 (approach 2) — '지금 식는 중인가'를 연속 강도로 예측
=====================================================================
[5] 식별성 결과상 '정확히 언제 질리나(τ)'는 조기 예측이 거의 불가능했다.
대신 '지금 식는 중인가'는 최근 궤적으로 직접 관측되므로 식별이 잘 된다.

타깃   : 다음 28일 완주수 (연속 강도). 누수 0 (T 이전 기록만으로 피처).
피처   : 최근 강도/궤적(c1..c4·기울기·가속)·마지막재생후경과·누적완주·나이
         + 호감도 A (song_scores.json 연속 점수) ← 곡 간 신호
핵심검정:
  (1) 호감도 편기여  : 궤적만 vs 궤적+A 교차검증 R² 차이 = '1번이 따로 필요한가'의 답
  (2) 부활 vs 사망   : 지금 조용한 곡 중 미래 재생되는 곡을 A로 구분 (AUC)
"""
import os, sys, json, re, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score, roc_auc_score
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')

BASE = os.path.dirname(os.path.abspath(__file__))
KCM = os.path.join(BASE, '유튜브 뮤직 로그들', '김차민')

def hr(t): print("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)
def norm(s): return re.sub(r'\s+', ' ', str(s).strip().lower())

# ----------------------------------------------------------------------
# 로드 + 호감도 조인
# ----------------------------------------------------------------------
df = pd.read_csv(os.path.join(KCM, '김차민_features.csv'), encoding='utf-8-sig')
df['t'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('t').reset_index(drop=True)
comp = df[df['skip_type'] == 0].copy()      # 완주 = 소비 사건

# 호감도 점수: 키 = "제목 - 아티스트"(아티스트에 ' - Topic' 포함). features와 동일 표기 → 직접 조인
scores = json.load(open(os.path.join(BASE, 'data', 'caches', 'song_scores.json'), encoding='utf-8'))
A_raw = {k: float(v['A']) for k, v in scores.items()}
A_norm = {norm(k): float(v['A']) for k, v in scores.items()}
# 곡별 A (features의 title/artist로 매칭)
sid_ta = df.groupby('song_id').agg(title=('title', 'first'), artist=('artist', 'first'))
def lookup_A(title, artist):
    key = f"{title} - {artist}"
    if key in A_raw: return A_raw[key]
    return A_norm.get(norm(key), np.nan)
sid_A = {sid: lookup_A(r.title, r.artist) for sid, r in sid_ta.iterrows()}
cov = np.mean([not np.isnan(a) for a in sid_A.values()])
print(f"완주 사건 {len(comp):,}건 · 곡 {comp['song_id'].nunique():,}개")
print(f"호감도 A 매칭 커버리지(전체 곡 기준): {cov*100:.0f}%")

# 곡별 완주 타임스탬프(초)
DAY = 86400
play = {s: np.sort(g['t'].values.astype('datetime64[s]').astype(np.int64))
        for s, g in comp.groupby('song_id') if len(g) >= 5}
t0 = comp['t'].min().value // 10**9
tN = comp['t'].max().value // 10**9
W, FUT, WARM = 7*DAY, 28*DAY, 90*DAY

# ----------------------------------------------------------------------
# 스냅샷 생성 (T 이전만으로 피처, T 이후 28일이 타깃)
# ----------------------------------------------------------------------
rows = []
for s, ts in play.items():
    A = sid_A.get(s, np.nan)
    first = ts[0]
    for T in np.arange(t0 + WARM, tN - FUT, 14*DAY):
        past = ts[ts < T]
        if len(past) < 3 or first >= T:        # 최소 이력
            continue
        c = [int(((past >= T-(k+1)*W) & (past < T-k*W)).sum()) for k in range(3, -1, -1)]  # c1..c4(오래된→최근)
        level = sum(c)
        fut = int(((ts >= T) & (ts < T + FUT)).sum())
        rows.append(dict(
            sid=s, A=A,
            c1=c[0], c2=c[1], c3=c[2], c4=c[3],
            level=level, slope=c[3]+c[2]-c[1]-c[0], accel=(c[3]-c[2])-(c[1]-c[0]),
            days_since_last=(T - past[-1]) / DAY,
            familiarity=len(past),
            age=(T - first) / DAY,
            fut=fut,
        ))
P = pd.DataFrame(rows)
print(f"스냅샷 {len(P):,}개 · 곡 {P['sid'].nunique()}개 · 미래28일 완주 중앙값 {P['fut'].median():.0f}")

TRAJ = ['level', 'c1', 'c2', 'c3', 'c4', 'slope', 'accel', 'days_since_last', 'familiarity', 'age']

# ----------------------------------------------------------------------
# (1) 호감도 편기여 — 궤적만 vs 궤적+A  (A 있는 스냅샷으로 공정 비교)
# ----------------------------------------------------------------------
hr("[1] 호감도 A가 최근 궤적 너머로 미래를 더 설명하는가 (= 1번 필요성)")
Q = P.dropna(subset=['A']).reset_index(drop=True)
g = Q['sid'].values; y = Q['fut'].values.astype(float)
print(f"  A 있는 스냅샷 {len(Q):,}개 · 곡 {Q['sid'].nunique()}개")

def cvr2(cols, model):
    X = Q[cols].values.astype(float)
    pred = cross_val_predict(model, X, y, groups=g, cv=GroupKFold(5))
    return r2_score(y, pred)

for name, mk in [("선형회귀", lambda: LinearRegression()),
                 ("GBM", lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                              max_depth=3, min_samples_leaf=30, random_state=0))]:
    r_traj = cvr2(TRAJ, mk())
    r_full = cvr2(TRAJ + ['A'], mk())
    r_Aonly = cvr2(['A'], mk())
    print(f"  [{name}]  궤적만 R²={r_traj:.4f}   궤적+A R²={r_full:.4f}   (ΔA = {r_full-r_traj:+.4f})   A단독 R²={r_Aonly:.4f}")

# 편상관: A ↔ 미래 | 최근강도(level) 통제
def partial(x, y, z):
    rxy = stats.spearmanr(x, y).correlation; rxz = stats.spearmanr(x, z).correlation; ryz = stats.spearmanr(y, z).correlation
    d = np.sqrt((1-rxz**2)*(1-ryz**2)); return (rxy-rxz*ryz)/d if d > 0 else np.nan
print(f"\n  A ↔ 미래 단순 상관          : {stats.spearmanr(Q['A'], y).correlation:+.3f}")
print(f"  A ↔ 미래 | 최근강도(level) 통제: {partial(Q['A'], y, Q['level']):+.3f}")
print("  → ΔA가 뚜렷한 +이고 편상관도 +면: 호감도는 궤적 너머의 신호 = 1번 가치(이미 흡수됨)")

# ----------------------------------------------------------------------
# (2) 부활 vs 사망 — 지금 조용한 곡 중 미래 재생을 A로 구분
# ----------------------------------------------------------------------
hr("[2] 부활 vs 사망 — '지금 조용한(level=0)' 곡의 미래 재생을 A가 가르나")
quiet = Q[Q['level'] == 0].reset_index(drop=True)
quiet['revived'] = (quiet['fut'] > 0).astype(int)
print(f"  조용한 스냅샷 {len(quiet):,}개 · 부활률(미래 재생>0) {quiet['revived'].mean()*100:.0f}%")
if quiet['revived'].nunique() == 2 and len(quiet) > 50:
    gq = quiet['sid'].values; yq = quiet['revived'].values
    QF = ['days_since_last', 'familiarity', 'age']     # 조용한 곡엔 c1..c4가 다 0 → 의미 피처만
    def cvauc(cols):
        X = quiet[cols].values.astype(float)
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3,
                                             min_samples_leaf=20, random_state=0)
        proba = cross_val_predict(clf, X, yq, groups=gq, cv=GroupKFold(5), method='predict_proba')[:, 1]
        return roc_auc_score(yq, proba)
    auc_base = cvauc(QF)
    auc_A = cvauc(QF + ['A'])
    print(f"  부활 예측 AUC  궤적만={auc_base:.3f}   +호감도A={auc_A:.3f}   (ΔA = {auc_A-auc_base:+.3f})")
    r = stats.spearmanr(quiet['A'], quiet['revived']).correlation
    print(f"  A ↔ 부활여부 상관: {r:+.3f}  → +면 '고호감=휴면(부활), 저호감=사망' 가설 지지")
else:
    print("  조용한 스냅샷 부족 또는 클래스 단일 — 생략")

# ----------------------------------------------------------------------
# (3) 상태 라벨 분포(해석용) — 회귀로 만든 강도에서 상태 파생
# ----------------------------------------------------------------------
hr("[3] 참고 — 현재강도 대비 미래강도로 본 상태 분포")
def state(r):
    cur, fut = r['level'], r['fut']
    if cur == 0 and fut == 0: return '사망/휴면'
    if cur == 0 and fut > 0:  return '부활'
    if fut > 1.5*cur:         return '가열'
    if fut < 0.5*cur:         return '냉각'
    return '안정'
P['state'] = P.apply(state, axis=1)
vc = P['state'].value_counts()
for k in ['가열', '안정', '냉각', '사망/휴면', '부활']:
    print(f"  {k:8s}: {vc.get(k,0):5d}  ({vc.get(k,0)/len(P)*100:4.1f}%)")

hr("끝")
