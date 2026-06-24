# -*- coding: utf-8 -*-
"""
D_recommender.py — 추천 점수 = sim × A^α × R^β  (지속 취향 + 현재 행동 블렌드)
================================================================================
설계 근거 (검증된 두 신호를 손잡이로 섞음):
  A = 진폭(호감도) : 곡을 *원래* 얼마나 사랑하나. 생애 누적 신호(song_scores.json).
                    recency는 A에 안 들어감 → 아래 R과 이중계산 아님.
  R = 현재 강도    : 지금 이 곡을 얼마나 듣고 있나. 시간감쇠 최근 완주수(반감기 30일).
                    '다음 28일 재생수'를 R²≈0.39로 설명함이 D_cooling_detector.py에서 검증됨
                    (그 설명력의 대부분이 이 recency라서, R을 넣을 근거가 된다).

왜 둘 다 겨냥하나:
  내가 매긴 호감도(A)는 진심의 *추정치*라 불완전하다. 호감도는 낮게 적었지만 실제로 자주
  트는 곡이 있다면, 그 행동(R)이 진짜 마음을 더 정직하게 드러낸다(revealed preference).
  단 R만 세게 쓰면 '최근 듣던 것 되먹임'(에코)이 되므로, A와 손잡이로 섞는다.

점수:  score = sim × A^α × (1+R)^β
  - α↑, β=0  → 순수 A      (편안/사랑 모드)
  - α=β      → 균형
  - β↑       → 지금 기분(현재 자주 듣는 곡) 부각
  (sim=시드 유사도. 이 독립 시연은 전역 평가라 sim=1; app.py 이식 시 곱함.)

참고: '오랜만에 다시 들을 곡'(휴면 고호감)은 R이 낮아 위 식에선 안 뜬다 — 의도된 분리.
      그 변주가 필요하면 dormant 슬롯(A×P부활)을 따로 둔다(주석 하단).
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')

BASE = os.path.dirname(os.path.abspath(__file__))
KCM = os.path.join(BASE, '유튜브 뮤직 로그들', '김차민')
DAY = 86400
HALF_LIFE = 30          # R 시간감쇠 반감기(일)
TOTAL = 20              # ① 플레이리스트 곡 수 (사용자 설정)
N_NEW = 4               # ② 그 중 '새로 듣는(거의 안 들은) 곡' 수 (사용자 설정)
# ③ 모드는 아래 MODES에서 선택

def disp(title, artist):
    a = artist[:-len(' - Topic')] if str(artist).endswith(' - Topic') else artist
    return f"{title} — {a}"

# ----------------------------------------------------------------------
# 로드 + 호감도 A
# ----------------------------------------------------------------------
df = pd.read_csv(os.path.join(KCM, '김차민_features.csv'), encoding='utf-8-sig')
df['t'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('t').reset_index(drop=True)
comp = df[df['skip_type'] == 0].copy()

scores = json.load(open(os.path.join(BASE, 'data', 'caches', 'song_scores.json'), encoding='utf-8'))
A_raw = {k: float(v['A']) for k, v in scores.items()}
meta = df.groupby('song_id').agg(title=('title', 'first'), artist=('artist', 'first'))
def get_A(sid):
    r = meta.loc[sid]
    return A_raw.get(f"{r.title} - {r.artist}", np.nan)

play = {s: np.sort(g['t'].values.astype('datetime64[s]').astype(np.int64))
        for s, g in comp.groupby('song_id') if len(g) >= 5}
now = comp['t'].max().value // 10**9
print(f"완주 {len(comp):,}건 · 곡(≥5재생) {len(play)}개 · 기준시각 now = {pd.to_datetime(now, unit='s').date()}")
print("R = 시간감쇠 최근 완주수(반감기 30일) · '다음28일 재생수'를 R²≈0.39로 설명(D_cooling_detector.py 검증)\n")

# ----------------------------------------------------------------------
# now 시점에서 곡별 A, R 계산
# ----------------------------------------------------------------------
rows = []
for s, ts in play.items():
    A = get_A(s)
    if np.isnan(A):
        continue
    age_d = (now - ts) / DAY                       # 각 완주의 경과일
    R = float(np.sum(0.5 ** (age_d / HALF_LIFE)))  # 시간감쇠 최근 강도
    dsl = float((now - ts[-1]) / DAY)
    level28 = int((age_d <= 28).sum())
    rows.append(dict(sid=s, title=meta.loc[s].title, artist=meta.loc[s].artist,
                     A=A, R=R, dsl=dsl, level28=level28, familiarity=len(ts)))
C = pd.DataFrame(rows)
print(f"평가 곡(A 보유): {len(C)}개 · R 중앙값 {C['R'].median():.2f} · 최근28일 1회+ {int((C['level28']>0).sum())}곡")

# ----------------------------------------------------------------------
# 플레이리스트 생성: 본편(아는 곡, 모드별 랭킹) + 신곡 슬롯(거의 안 들은 곡)
# ----------------------------------------------------------------------
MODES = {'편안': (1.0, 0.0), '균형': (1.0, 1.0), '지금 기분': (1.0, 2.0)}

# '거의 안 들은 곡' = 완주수 하위 30% → 신곡 슬롯 후보 / 나머지 = 본편 후보
fam_lo = C['familiarity'].quantile(0.30)
known = C[C['familiarity'] > fam_lo]
newcand = C[C['familiarity'] <= fam_lo]

def make_playlist(mode, total=TOTAL, n_new=N_NEW):
    a, b = MODES[mode]
    k = known.copy()
    k['s'] = k['A']**a * (1 + k['R'])**b                      # 본편: 모드별 블렌드 랭킹
    main = k.sort_values('s', ascending=False).head(total - n_new)
    nc = newcand.sort_values('A', ascending=False)            # 신곡: 예측 호감도(A) 순
    nc = nc[~nc['sid'].isin(main['sid'])].head(n_new)
    return main.assign(slot='본편'), nc.assign(slot='신곡')

def show_playlist(mode):
    main, nc = make_playlist(mode)
    a, b = MODES[mode]
    print("\n" + "="*80)
    print(f"[{mode} 모드]  곡 수 {TOTAL} · 신곡 {N_NEW}곡  (α={a}, β={b})")
    print("="*80)
    for i, (_, r) in enumerate(pd.concat([main, nc]).iterrows(), 1):
        if r.slot == '신곡':
            tag = '★신곡(거의 안 들음)'
        elif r.level28 >= 3:
            tag = '지금 자주'
        elif r.dsl >= 30:
            tag = '휴면'
        else:
            tag = ''
        print(f"  {i:2d}. {disp(r.title, r.artist):44s}  A={r.A:.2f}  R={r.R:5.2f}  들은수={int(r.familiarity):3d}  {tag}")

print(f"\n'거의 안 들은 곡' 기준: 완주수 ≤ {fam_lo:.0f}회 (하위 30%) · 후보 {len(newcand)}곡")
show_playlist('편안')
show_playlist('지금 기분')

print("\n" + "-"*80)
print("점수 = sim × A^α × (1+R)^β.  sim은 이 전역 시연에선 1 (app.py 이식 시 시드 유사도를 곱함).")
print(f"설정: 플레이리스트 {TOTAL}곡 = 본편 {TOTAL-N_NEW}곡(모드별 랭킹) + 신곡 {N_NEW}곡(거의 안 들은 곡 중 예측 호감도 순).")
