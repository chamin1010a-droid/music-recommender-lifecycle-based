"""
진폭 모델 v6 — 피처 정리 + 아티스트 인코딩 3방식 비교
=======================================================
v5 대비 수정(사용자 피드백):
 · 총 재생수 → 완주 재생수(스킵 제외)
 · avg_proactive_score 제거 (검색·세션시작과 이중계산; 안 쓴 deep-dive만 별도 테스트)
 · avg_satisfaction 제거 (스킵점수=완주율과 중복 + 7일내 재청취=빼기로 한 frequency 신호)
아티스트 인코딩: ① 라벨만 LOO  ② shrinkage(희소 보완)  ③ 준지도 2-pass(미라벨곡 예측 포함)
타깃: 4월 재교정 라벨. 정확도: 반복 5-겹 CV의 피어슨 r.
"""
import pandas as pd, numpy as np, os, json, warnings, sys
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
BASE = r'c:\Users\김차민\Desktop\데이터분석\음악 프로젝트'; KCM = os.path.join(BASE, '유튜브 뮤직 로그들', '김차민')
df = pd.read_csv(os.path.join(KCM, '김차민_features.csv'), encoding='utf-8-sig')

# ---------- 정리된 피처 ----------
rows = []
for sid, g in df.groupby('song_id'):
    if len(g) < 5: continue
    st = g['skip_type']
    rows.append({
        'song_id': sid, 'artist': g['artist'].iloc[0], 'title': g['title'].iloc[0],
        'completed_plays': int((st == 0).sum()),          # 완주 재생수 (스킵 제외)
        'completion_rate': (st == 0).mean(),
        'immediate_skip_rate': (st == 2).mean(),
        'sample_skip_rate': (st == 1).mean(),
        'proactive_rate': g['is_proactive'].mean(),
        'session_start_rate': g['is_session_start'].mean(),
        'deep_dive_rate': g['is_artist_deep_dive'].mean(),   # 안 쓰던 신호 (옵션)
    })
S = pd.DataFrame(rows)

# 오디오 PCA 8 (앞 실험서 4월 라벨에 도움)
araw = json.load(open(os.path.join(BASE, 'data', 'caches', 'audio_features_cache.json'), encoding='utf-8'))
adim = len(next(iter(araw.values())))
A = np.array([araw.get(sid, [np.nan]*adim) for sid in S['song_id']], float)
A = np.where(np.isnan(A), np.nanmean(A, 0), A)
apca = PCA(8, random_state=42).fit_transform((A - A.mean(0))/(A.std(0)+1e-9))
for i in range(8): S[f'audio_{i}'] = apca[:, i]

# 4월 라벨
ra = pd.read_csv(os.path.join(KCM, '호감도_재교정.csv'), encoding='utf-8-sig'); ra['v'] = pd.to_numeric(ra['Unnamed: 13'], errors='coerce')
sid_of = df.groupby(['title', 'artist'])['song_id'].first().to_dict()
lab_map = {sid_of.get((r['title'], r['artist'])): r['v'] for _, r in ra.dropna(subset=['v']).iterrows() if sid_of.get((r['title'], r['artist']))}
S['label'] = S['song_id'].map(lab_map)

CORE = ['completed_plays', 'completion_rate', 'immediate_skip_rate', 'sample_skip_rate', 'proactive_rate', 'session_start_rate']
AUDIO = [f'audio_{i}' for i in range(8)]
P = dict(objective='regression', verbosity=-1, n_estimators=400, learning_rate=0.03, num_leaves=15,
         max_depth=4, min_child_samples=5, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0)

def art_enc(method, tr_art, tr_y, want_art, k=3, S_all=None, feat=None, tr_feat=None):
    g = tr_y.mean()
    if method == 'labeled':
        m = pd.Series(tr_y).groupby(tr_art).mean().to_dict()
        return np.array([m.get(a, g) for a in want_art])
    if method == 'shrink':
        grp = pd.Series(tr_y).groupby(tr_art)
        mean, cnt = grp.mean().to_dict(), grp.count().to_dict()
        return np.array([(cnt.get(a,0)*mean.get(a,g)+k*g)/(cnt.get(a,0)+k) for a in want_art])
    if method == 'semi':
        # pass1: 라벨LOO 인코딩으로 학습 → 전체곡(S_all) 예측
        m1 = pd.Series(tr_y).groupby(tr_art).mean().to_dict()
        Xtr1 = np.column_stack([tr_feat, np.array([m1.get(a, g) for a in tr_art]).reshape(-1, 1)])
        mod1 = lgb.LGBMRegressor(**P, random_state=1).fit(Xtr1, tr_y)
        allpred = mod1.predict(np.column_stack([S_all[feat].values, np.array([m1.get(a, g) for a in S_all['artist']]).reshape(-1, 1)]))
        # 아티스트 평균 = {tr 라벨} ∪ {미라벨곡 pass1 예측}
        unlab = S_all['label'].isna().values
        merged_art = list(tr_art) + list(S_all['artist'][unlab])
        merged_val = list(tr_y) + list(allpred[unlab])
        m2 = pd.Series(merged_val).groupby(pd.Series(merged_art)).mean().to_dict()
        return np.array([m2.get(a, g) for a in want_art]), m1  # m1 재사용 위해
    return None

def cv(feat, method, nrep=10):
    L = S[S['label'].notna()].reset_index(drop=True)
    y = L['label'].values; art = L['artist'].values
    rs = []
    for rep in range(nrep):
        oof = np.zeros(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=rep).split(y):
            if method == 'semi':
                te_enc, m1 = art_enc('semi', art[tr], y[tr], art[te], S_all=S, feat=feat, tr_feat=L.iloc[tr][feat].values)
                tr_enc, _ = art_enc('semi', art[tr], y[tr], art[tr], S_all=S, feat=feat, tr_feat=L.iloc[tr][feat].values)
            else:
                te_enc = art_enc(method, art[tr], y[tr], art[te])
                tr_enc = art_enc(method, art[tr], y[tr], art[tr])
            m = lgb.LGBMRegressor(**P, random_state=rep)
            m.fit(np.column_stack([L.iloc[tr][feat].values, tr_enc]), y[tr])
            oof[te] = m.predict(np.column_stack([L.iloc[te][feat].values, te_enc]))
        rs.append(pearsonr(oof, y)[0])
    return np.mean(rs), np.std(rs)

print(f"라벨곡 {S['label'].notna().sum()} / 전체곡 {len(S)}\n")
print(f"{'피처셋':<22}{'라벨LOO':>10}{'shrink':>9}{'준지도':>9}")
for name, feat in [('cleaned core', CORE), ('  +오디오', CORE+AUDIO), ('  +오디오+deepdive', CORE+AUDIO+['deep_dive_rate'])]:
    r1 = cv(feat, 'labeled'); r2 = cv(feat, 'shrink'); r3 = cv(feat, 'semi')
    print(f"{name:<22}{r1[0]:>9.3f}{r2[0]:>9.3f}{r3[0]:>9.3f}")
print("\n(참고: 옛 v5 = 총재생수+능동점수+만족도 포함, 라벨LOO 기준 r≈0.56)")
