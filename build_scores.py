"""
build_scores.py — app.py가 쓸 곡별 점수 사전계산 (진폭 v6)
============================================================
출력: data/caches/song_scores.json = { song_id: {A} }
  A = 진폭(정리된 행동 피처 + 오디오 + 준지도 아티스트 인코딩 LightGBM)

* 부활(rb)·신선도(dsl)는 제거했다 — 검증 결과 부활/위치/이탈도/망각보정 등
  모든 '생애주기' 신호가 진폭(A)으로 환원돼, 추천은 sim×A 하나로 충분하다.
"""
import pandas as pd, numpy as np, os, json, warnings, sys
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')
import lightgbm as lgb
from sklearn.decomposition import PCA
BASE = r'c:\Users\김차민\Desktop\데이터분석\음악 프로젝트'; KCM = os.path.join(BASE, '유튜브 뮤직 로그들', '김차민')
df = pd.read_csv(os.path.join(KCM, '김차민_features.csv'), encoding='utf-8-sig')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# ---------- 정리된 피처 (완주/스킵을 구분해 집계 — '들었다'=완주) ----------
rows = []
for sid, g in df.groupby('song_id'):
    if len(g) < 5: continue
    st = g['skip_type']
    rows.append({'song_id': sid, 'artist': g['artist'].iloc[0],
        'completed_plays': int((st == 0).sum()), 'completion_rate': (st == 0).mean(),
        'immediate_skip_rate': (st == 2).mean(), 'sample_skip_rate': (st == 1).mean(),
        'proactive_rate': g['is_proactive'].mean(), 'session_start_rate': g['is_session_start'].mean()})
S = pd.DataFrame(rows)
araw = json.load(open(os.path.join(BASE, 'data', 'caches', 'audio_features_cache.json'), encoding='utf-8'))
adim = len(next(iter(araw.values())))
A = np.array([araw.get(sid, [np.nan]*adim) for sid in S['song_id']], float)
A = np.where(np.isnan(A), np.nanmean(A, 0), A)
for i, col in enumerate(PCA(8, random_state=42).fit_transform((A-A.mean(0))/(A.std(0)+1e-9)).T):
    S[f'audio_{i}'] = col

ra = pd.read_csv(os.path.join(KCM, '호감도_재교정.csv'), encoding='utf-8-sig'); ra['v'] = pd.to_numeric(ra['Unnamed: 13'], errors='coerce')
sid_of = df.groupby(['title', 'artist'])['song_id'].first().to_dict()
lab_map = {sid_of.get((r['title'], r['artist'])): r['v'] for _, r in ra.dropna(subset=['v']).iterrows() if sid_of.get((r['title'], r['artist']))}
S['label'] = S['song_id'].map(lab_map)

CORE = ['completed_plays', 'completion_rate', 'immediate_skip_rate', 'sample_skip_rate', 'proactive_rate', 'session_start_rate']
FEATS = CORE + [f'audio_{i}' for i in range(8)]
P = dict(objective='regression', verbosity=-1, n_estimators=400, learning_rate=0.03, num_leaves=15,
         max_depth=4, min_child_samples=5, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0)

# ---------- 진폭: 준지도 2-pass 아티스트 인코딩 ----------
lab = S[S['label'].notna()]; y = lab['label'].values; art = lab['artist'].values; gmean = y.mean()
m1 = pd.Series(y).groupby(art).mean().to_dict()                                   # pass1: 라벨만
enc1 = lambda arts: np.array([m1.get(a, gmean) for a in arts]).reshape(-1, 1)
mod1 = lgb.LGBMRegressor(**P, random_state=1).fit(np.column_stack([lab[FEATS].values, enc1(art)]), y)
allpred = mod1.predict(np.column_stack([S[FEATS].values, enc1(S['artist'].values)]))
unlab = S['label'].isna().values                                                 # 미라벨 곡 예측을 평균에 합침
merged_art = list(art) + list(S['artist'][unlab]); merged_val = list(y) + list(allpred[unlab])
m2 = pd.Series(merged_val).groupby(pd.Series(merged_art)).mean().to_dict()
enc2 = lambda arts: np.array([m2.get(a, gmean) for a in arts]).reshape(-1, 1)
mod2 = lgb.LGBMRegressor(**P, random_state=0).fit(np.column_stack([lab[FEATS].values, enc2(art)]), y)
S['A'] = mod2.predict(np.column_stack([S[FEATS].values, enc2(S['artist'].values)]))

# ---------- 출력: 곡별 사랑(A)만 ----------
scores = {sid: {'A': round(float(a), 4)} for sid, a in zip(S['song_id'], S['A'])}
out = os.path.join(BASE, 'data', 'caches', 'song_scores.json')
json.dump(scores, open(out, 'w', encoding='utf-8'), ensure_ascii=False)
print(f"저장: {len(scores)}곡 → song_scores.json (A only)")
print("샘플:", dict(list(scores.items())[:1]))
