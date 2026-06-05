"""오디오 유사도가 왜 0.93~0.98에 몰리는지 증명"""
import sys, json, numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
sys.stdout.reconfigure(encoding='utf-8')

data = json.load(open(r'data\caches\audio_features_cache.json', 'r', encoding='utf-8'))
ids = list(data.keys())
raw = np.array([data[sid] for sid in ids])  # (2573, 37)

print(f"곡 수: {raw.shape[0]}, 피쳐: {raw.shape[1]}차원")
print(f"원본 데이터 범위: {raw.min():.1f} ~ {raw.max():.1f}")
print(f"원본 음수 비율: {(raw < 0).sum() / raw.size * 100:.1f}%")

# 현재 방식: MinMaxScaler
mm = MinMaxScaler().fit_transform(raw)
sims_mm = cosine_similarity(mm)
np.fill_diagonal(sims_mm, np.nan)
flat_mm = sims_mm[~np.isnan(sims_mm)]

# 대안: StandardScaler (z-score)
ss = StandardScaler().fit_transform(raw)
sims_ss = cosine_similarity(ss)
np.fill_diagonal(sims_ss, np.nan)
flat_ss = sims_ss[~np.isnan(sims_ss)]

# 대안: 원본 (정규화 없음)
sims_raw = cosine_similarity(raw)
np.fill_diagonal(sims_raw, np.nan)
flat_raw = sims_raw[~np.isnan(sims_raw)]

print(f"\n{'방식':15s} | {'최소':>6s} | {'25%':>6s} | {'중앙':>6s} | {'75%':>6s} | {'최대':>6s} | {'0.9이상':>6s}")
print("-" * 80)
for name, flat in [('MinMaxScaler', flat_mm), ('StandardScaler', flat_ss), ('원본(no scale)', flat_raw)]:
    print(f"{name:15s} | {flat.min():6.3f} | {np.percentile(flat,25):6.3f} | {np.median(flat):6.3f} | {np.percentile(flat,75):6.3f} | {flat.max():6.3f} | {(flat>0.9).sum()/len(flat)*100:5.1f}%")

print(f"\n★ MinMaxScaler가 모든 값을 양수(0~1)로 만들어서 코사인 유사도가 자연적으로 높아짐")
print(f"★ StandardScaler(z-score)는 음수도 포함 → 각도 차이가 커져서 차별력 상승")
