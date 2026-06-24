# -*- coding: utf-8 -*-
"""
질림(satiation) 재검정 — 날짜 축을 [0,1]로 정규화하지 않고 절대 시간으로
=========================================================================
배경
----
기존 fPCA(C_validate_b_oof.py:53)는 곡 생애를 u=(t-t0)/(t_last-t0) 로 [0,1] 정규화했다.
분모에 t_last(마지막 재생)가 들어간다 = '언제 질리는가'의 정답을 이미 써서 x축을 그린 것
(look-ahead/라벨 누수). 그래서 PC1=83% '보편 모양'은 참이지만 *예측 불가능한* 참이다.

이 스크립트가 하는 일
--------------------
[1] 누수 진단: 기존 정규화 fPCA(≈83%)를 재현하고, 정규화가 곡 고유 '타임스케일 τ'를
    통째로 지웠음을 보인다.
[2] 절대시간 재검정: 곡마다 누적재생 곡선을 NHPP(비동질 포아송) Weibull 강도로 적합.
       λ_i(t) = A_i·(k/τ_i)(t/τ_i)^{k-1} exp[-(t/τ_i)^k]
    - 모양 k_i 분포가 좁고 τ_i 분포가 넓은가? (= 모양 보편 / 수명 곡고유)
    - M1(k 공유) vs M2(k 곡별) BIC → 누수 없는 '보편 모양' 검정
    - 두 시계: A=캘린더(일), B=활동(전역 누적재생수). 안 들은 휴지기 상쇄.
[3] 누수 증명: 적합 τ_i 와 관측수명(t_last-t0)의 상관 → '수명으로 정규화 = 정답으로 나누기'
[4] 예측 시험: 초기 데이터만으로 τ̂ 추정 → 미래 재생 예측 vs 단순 베이스라인
[5] 식별성: 사건의 앞 일부만으로 τ̂가 얼마나 흔들리나 → '원리적으로 언제부터 예측 가능한가'
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, minimize_scalar
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

BASE = os.path.dirname(os.path.abspath(__file__))
KCM = os.path.join(BASE, '유튜브 뮤직 로그들', '김차민')
FEATURES = os.path.join(KCM, '김차민_features.csv')

MIN_PLAYS = 16      # 기존 fPCA와 동일 기준
MIN_SPAN_D = 7      # 생애 7일 미만 제외(기존과 동일)
C0_DAY = 1.0        # 캘린더 시계 offset (t=0 특이점 회피)
C0_ACT = 1.0        # 활동 시계 offset

def hr(t):
    print("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)

# ----------------------------------------------------------------------
# 로드
# ----------------------------------------------------------------------
df = pd.read_csv(FEATURES, encoding='utf-8-sig')
df['t'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('t').reset_index(drop=True)
df['gidx'] = np.arange(len(df))            # 전역 재생 순번(시간순) = 활동 시계
LOG_END = df['t'].max()
LOG_END_G = len(df) - 1
print(f"재생 {len(df):,}건 · 곡 {df['song_id'].nunique():,}개 · "
      f"{df['t'].min().date()} ~ {LOG_END.date()}")

# 곡별 사건 리스트 (전 재생; 기존 fPCA와 동일 정의)
songs = {}
for sid, g in df.groupby('song_id'):
    ts = g['t'].values.astype('datetime64[s]').astype(np.int64)   # 초
    gi = g['gidx'].values
    n = len(ts)
    span_d = (ts[-1] - ts[0]) / 86400.0
    if n < MIN_PLAYS or span_d < MIN_SPAN_D:
        continue
    songs[sid] = dict(
        ts=ts, gi=gi, n=n,
        day=(ts - ts[0]) / 86400.0,                      # 첫재생 후 경과일
        act=(gi - gi[0]).astype(float),                  # 활동 시계(전역 순번 차)
        Tobs_day=(LOG_END.value // 10**9 - ts[0]) / 86400.0,   # 로그끝까지(검열 지평)
        Tobs_act=float(LOG_END_G - gi[0]),
        span_d=span_d,
    )
print(f"분석 대상 곡(재생≥{MIN_PLAYS}, 생애≥{MIN_SPAN_D}일): {len(songs)}개")

# ======================================================================
# [1] 기존 정규화 fPCA 재현 (누수판)
# ======================================================================
hr("[1] 기존 방식 재현 — 생애를 [0,1] 정규화한 fPCA (누수판)")
grid = np.linspace(0, 1, 21)
cur = []
for sid, s in songs.items():
    ts = s['ts'].astype(float)
    u = (ts - ts[0]) / (ts[-1] - ts[0])            # ← t_last 사용(누수)
    cur.append(np.searchsorted(u, grid, side='right') / s['n'])
M = np.array(cur)
Mc = M - M.mean(0)
U, Sv, Vt = np.linalg.svd(Mc, full_matrices=False)
pc1 = Sv[0]**2 / (Sv**2).sum()
# 평균 곡선이 생애 첫 25%에 도달한 누적비율
i25 = np.argmin(np.abs(grid - 0.25))
print(f"  PC1 설명력 : {pc1*100:.1f}%   (README의 83% 재현 확인용)")
print(f"  평균곡선이 생애 첫 25%에서 도달한 누적재생 비율: {M.mean(0)[i25]*100:.0f}%  (무작위면 25%)")
print("  → 모양은 보편적으로 보이지만, 이 x축은 t_last를 알아야 그릴 수 있다(예측 불가).")

# ======================================================================
# Weibull-NHPP 적합기 (프로파일 우도: A는 닫힌형 A=n/F(Tobs))
# ======================================================================
def _negll(params, t, Tobs):
    lk, ltau = params
    k = np.exp(lk); tau = np.exp(ltau)
    logf = np.log(k) - np.log(tau) + (k - 1)*(np.log(t) - np.log(tau)) - (t/tau)**k
    F_T = 1 - np.exp(-(Tobs/tau)**k)
    if F_T <= 1e-12 or not np.isfinite(F_T):
        return 1e18
    ll = logf.sum() - len(t)*np.log(F_T)
    return -ll if np.isfinite(ll) else 1e18

def fit_weibull(t, Tobs):
    """t: 사건시각(>0, offset 포함), Tobs: 검열지평. return k,tau,A,LL"""
    best = None
    med = max(np.median(t), 1e-3)
    for k0 in (0.5, 0.8, 1.0, 1.5, 2.5):
        for tau0 in (med, Tobs, Tobs*2, np.percentile(t, 75)+1e-3):
            try:
                r = minimize(_negll, [np.log(k0), np.log(max(tau0, 1e-3))],
                             args=(t, Tobs), method='Nelder-Mead',
                             options={'xatol':1e-4, 'fatol':1e-4, 'maxiter':3000})
            except Exception:
                continue
            if best is None or r.fun < best.fun:
                best = r
    k, tau = np.exp(best.x)
    F_T = 1 - np.exp(-(Tobs/tau)**k)
    A = len(t) / max(F_T, 1e-12)
    return k, tau, A, -best.fun

def fit_tau_given_k(t, Tobs, k):
    def nll(ltau):
        tau = np.exp(ltau)
        logf = np.log(k) - np.log(tau) + (k-1)*(np.log(t)-np.log(tau)) - (t/tau)**k
        F_T = 1 - np.exp(-(Tobs/tau)**k)
        if F_T <= 1e-12: return 1e18
        return -(logf.sum() - len(t)*np.log(F_T))
    r = minimize_scalar(nll, bounds=(np.log(1e-2), np.log(Tobs*200)), method='bounded')
    return np.exp(r.x), -r.fun

# ======================================================================
# [2] 절대시간 재검정 (두 시계)
# ======================================================================
def run_clock(clock_name, key_t, key_T, c0):
    hr(f"[2-{clock_name}] 절대시간 Weibull-NHPP 적합 — {clock_name} 시계")
    K, TAU, A_, LL2 = {}, {}, {}, 0.0
    ev = {}
    for sid, s in songs.items():
        t = s[key_t] + c0
        Tobs = s[key_T] + c0
        t = t[t > 0]
        if len(t) < MIN_PLAYS:
            continue
        k, tau, A, ll = fit_weibull(t, Tobs)
        K[sid], TAU[sid], A_[sid], ev[sid] = k, tau, A, (t, Tobs)
        LL2 += ll
    ks = np.array(list(K.values()))
    taus = np.array(list(TAU.values()))
    n_song = len(K)
    n_ev = sum(len(v[0]) for v in ev.values())
    # 검열(아직 안 식음): τ가 관측지평의 5배를 넘으면 사건창 안에서 식지 않은 곡
    Tmed = np.median([ev[s][1] for s in K])
    censored = np.array([TAU[s] > 5*ev[s][1] for s in K])
    res = taus[~censored]   # 식음이 식별된 곡들만

    # M1: k 공유
    def m1_negsumll(k):
        tot = 0.0
        for sid in K:
            t, Tobs = ev[sid]
            _, ll = fit_tau_given_k(t, Tobs, k)
            tot += ll
        return -tot
    rk = minimize_scalar(m1_negsumll, bounds=(0.2, 5.0), method='bounded')
    k_shared = rk.x
    LL1 = -rk.fun

    # BIC = -2LL + p·ln(N).  M2: 2*n_song 모수(k,tau)+n_song(A);  M1: 1+ n_song(tau)+n_song(A)
    p2 = 2*n_song + n_song
    p1 = 1 + n_song + n_song
    bic2 = -2*LL2 + p2*np.log(n_ev)
    bic1 = -2*LL1 + p1*np.log(n_ev)

    print(f"  곡 {n_song}개 · 사건 {n_ev:,}개")
    print(f"  모양 k_i 분포  : 중앙값 {np.median(ks):.2f}  IQR[{np.percentile(ks,25):.2f},{np.percentile(ks,75):.2f}]  (k<1=초반몰입형 감쇠)")
    print(f"  ▶ 관측창 안에서 '아직 안 식은'(검열) 곡: {censored.sum()}/{n_song} ({censored.mean()*100:.0f}%)  ← τ 식별 불가")
    if len(res) > 5:
        print(f"  스케일 τ_i 분포(식음 식별된 곡만): 중앙값 {np.median(res):.1f}  IQR[{np.percentile(res,25):.1f},{np.percentile(res,75):.1f}]  p90/p10={np.percentile(res,90)/max(np.percentile(res,10),1e-9):.1f}배")
        print(f"     ('수명 스케일'이 곡마다 {np.percentile(res,90)/max(np.percentile(res,10),1e-9):.0f}배 차이 — 정규화가 통째로 지운 변수)")
    print(f"  보편 모양 검정 — 공유 k = {k_shared:.2f}")
    print(f"     M2(k 곡별)  : LL={LL2:.0f}  BIC={bic2:.0f}")
    print(f"     M1(k 공유)  : LL={LL1:.0f}  BIC={bic1:.0f}")
    verdict = "M1 승 → 모양은 보편(곡마다 k 따로 둘 필요 없음)" if bic1 < bic2 else "M2 승 → 모양도 곡마다 다름(보편성 약함)"
    print(f"     ΔBIC(M1-M2) = {bic1-bic2:+.0f}  →  {verdict}")
    return K, TAU, A_, ev

K_d, TAU_d, A_d, ev_d = run_clock("캘린더", 'day', 'Tobs_day', C0_DAY)
K_a, TAU_a, A_a, ev_a = run_clock("활동",  'act', 'Tobs_act', C0_ACT)

# ======================================================================
# [3] 누수 증명: 적합 τ ↔ 관측수명
# ======================================================================
hr("[3] 누수 증명 — 적합 τ_i 가 곧 관측수명인가? (정규화=정답으로 나누기)")
sids = [s for s in songs if s in TAU_d]
# 식음 식별된 곡만(검열 τ 제외)
res_sids = [s for s in sids if TAU_d[s] <= 5*(songs[s]['Tobs_day']+C0_DAY)]
tau_d = np.array([TAU_d[s] for s in res_sids])
lifespan = np.array([songs[s]['span_d'] for s in res_sids])
r = stats.spearmanr(tau_d, lifespan)
print(f"  캘린더 τ_i ↔ 관측수명(t_last-t0) : ρ = {r.correlation:+.2f}  (p={r.pvalue:.1e}, n={len(res_sids)} 식음식별곡)")
print(f"  → ρ가 양(+)이면: 관측수명이 곧 스케일 정보를 담음. 기존 정규화는 이 τ로 나눠 변수를 소거")
print(f"  ※ 더 근본적 누수는 통계가 아니라 정의: x=(t-t0)/(t_last-t0) 의 분모 t_last가 곧 '질림시점'")

# ======================================================================
# [4] 예측 시험 — 초기 데이터로 미래 재생 예측 (누수 없음)
# ======================================================================
hr("[4] 예측 시험 — 캘린더 시계, 생애 전반부로 후반부 재생수 예측")
# 결정시점 d = 첫재생 + Tobs*FRAC. d 이전 사건만으로 적합 → (d, Tobs] 재생수 예측.
FRAC = 0.5
def F_weib(t, k, tau):
    return 1 - np.exp(-(np.maximum(t, 0)/tau)**k)

pred_w, pred_persist, pred_pop, actual = [], [], [], []
# 모집단 베이스라인용: 전반/후반 비율
ratios = []
for sid in sids:
    s = songs[sid]
    t = s['day'] + C0_DAY
    Tobs = s['Tobs_day'] + C0_DAY
    d = C0_DAY + FRAC * s['Tobs_day']
    tr = t[t <= d]
    fut_actual = int(((t > d) & (t <= Tobs)).sum())
    if len(tr) < 8:        # 전반부 사건 충분해야
        continue
    # 전반부만으로 Weibull 적합 (검열지평 = d)
    k, tau, A, _ = fit_weibull(tr, d)
    pred = A * (F_weib(Tobs, k, tau) - F_weib(d, k, tau))   # 후반 기대 재생
    pred = min(pred, 5*s['n'])                              # 검열 곡 폭주 방지(상한)
    pred_w.append(pred)
    pred_persist.append(len(tr))                            # 지속 베이스라인
    actual.append(fut_actual)
    ratios.append((fut_actual, len(tr)))
actual = np.array(actual, float)
pred_w = np.array(pred_w, float)
pred_persist = np.array(pred_persist, float)
pop_ratio = np.median([f/max(p,1) for f, p in ratios])
pred_pop = pred_persist * pop_ratio

def report(name, pred):
    mae = np.mean(np.abs(pred - actual))
    mdae = np.median(np.abs(pred - actual))
    rho = stats.spearmanr(pred, actual).correlation
    print(f"  {name:24s} MAE={mae:6.2f}  중앙오차={mdae:5.2f}  Spearman ρ={rho:+.3f}")

print(f"  대상 곡 {len(actual)}개 (전반부≥8재생) · 후반 실제재생 중앙값 {np.median(actual):.0f}")
report("① 지속(전반부 개수)", pred_persist)
report(f"② 모집단비율(×{pop_ratio:.2f})", pred_pop)
report("③ Weibull τ̂(곡별 모양)", pred_w)
print("  → ③이 ①②를 뚜렷이 못 이기면: 초기 데이터로 '질림 시점' 개별 예측은 사실상 실패")

# ======================================================================
# [5] 식별성 — 사건 앞 일부로 τ̂ 가 얼마나 흔들리나
# ======================================================================
hr("[5] 식별성 — 앞 f%% 사건만으로 추정한 τ̂ 가 최종 τ̂ 와 얼마나 다른가")
fracs = [0.3, 0.5, 0.7, 0.9]
rows = []
big = [s for s in sids if songs[s]['n'] >= 30]
for sid in big:
    s = songs[sid]
    t = s['day'] + C0_DAY
    Tobs = s['Tobs_day'] + C0_DAY
    _, tau_full, _, _ = fit_weibull(t, Tobs)
    rec = []
    for f in fracs:
        m = max(int(len(t)*f), 8)
        tt = t[:m]
        d = tt[-1]                       # 그 시점까지만 관측했다고 보고 검열지평=마지막관측
        try:
            _, tau_f, _, _ = fit_weibull(tt, max(d, tt[-1]+1e-6))
            rec.append(abs(tau_f - tau_full)/max(tau_full, 1e-9))
        except Exception:
            rec.append(np.nan)
    rows.append(rec)
R = np.array(rows, float)
print(f"  대상 곡(재생≥30): {len(big)}개")
print("  앞 f%% 사건으로 추정한 τ̂의 |상대오차| 중앙값 (최종 τ̂ 대비)")
for j, f in enumerate(fracs):
    print(f"    f={int(f*100)}% :  {np.nanmedian(R[:,j])*100:5.0f}%")
print("  → 후반(f=70~90%)까지도 오차가 크면, τ는 '곡이 거의 식고 나서야' 식별됨 = 조기예측 한계")

hr("끝 — 해석은 위 5개 블록 순서대로")
