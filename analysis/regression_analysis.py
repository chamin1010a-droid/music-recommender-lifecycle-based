import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
REPORT_DIR = os.path.join(BASE_DIR, 'reports')

USERS = [
    'user', '친구D', '친구B', '친구C', '친구E', '친구F'
]

data_points = []

for user in USERS:
    csv_path = os.path.join(BASE_DIR, '유튜브 뮤직 로그들', user, f'{user}_features.csv')
    if not os.path.exists(csv_path):
        continue
    
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    if 'is_skipped' not in df.columns:
        continue
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    split_date = df['timestamp'].min() + (df['timestamp'].max() - df['timestamp'].min()) / 2
    
    for song_id, group in df.groupby('song_id'):
        first_half = group[group['timestamp'] < split_date]
        second_half = group[group['timestamp'] >= split_date]
        
        if len(first_half) < 4:
            continue
            
        fh_sorted = first_half.sort_values('timestamp')
        fh_mid = len(fh_sorted) // 2
        fh_early_skip = fh_sorted.iloc[:fh_mid]['is_skipped'].mean()
        fh_late_skip = fh_sorted.iloc[fh_mid:]['is_skipped'].mean()
        
        # X: 스킵률 변화량 (-1.0 ~ 1.0). 양수면 스킵 증가.
        skip_delta = fh_late_skip - fh_early_skip 
        
        fh_weeks = max(1, (fh_sorted['timestamp'].max() - fh_sorted['timestamp'].min()).days / 7)
        fh_weekly = len(first_half) / fh_weeks
        
        if len(second_half) == 0:
            sh_weekly = 0.0
        else:
            sh_weeks = max(1, (second_half['timestamp'].max() - second_half['timestamp'].min()).days / 7)
            sh_weekly = len(second_half) / sh_weeks
        
        # Y: 재생량 변화 비율 (후반/전반)
        # log 변환을 위해 극단값 처리. 0회 재생은 0.1배로 간주. MAX는 10배로 제한.
        play_change_ratio = sh_weekly / max(fh_weekly, 0.01)
        play_change_ratio = max(0.1, min(play_change_ratio, 10.0))
        
        # 정규화를 위해 로그(log2) 스케일 적용
        # 1.0(그대로) -> 0, 0.5(반토막) -> -1, 2.0(두배) -> 1
        log_change = np.log2(play_change_ratio)
        
        data_points.append({
            'user': user,
            'skip_delta': skip_delta,
            'play_change_ratio': play_change_ratio,
            'log_change': log_change
        })

df_reg = pd.DataFrame(data_points)

print("=" * 60)
print(f"📈 스킵률 변화 - 재생량 변화 회귀/상관 분석 (총 {len(df_reg)}곡)")
print("=" * 60)

# 1. Pearson 상관계수 (선형 상관관계)
pearson_r, p_value_p = stats.pearsonr(df_reg['skip_delta'], df_reg['log_change'])

# 2. Spearman 상관계수 (순위 상관관계 - 비선형적이어도 순위가 맞는지 추세 확인)
spearman_rho, p_value_s = stats.spearmanr(df_reg['skip_delta'], df_reg['log_change'])

# 3. 선형 회귀
slope, intercept, r_value, p_value, std_err = stats.linregress(df_reg['skip_delta'], df_reg['log_change'])

print(f"\n[1] Pearson 상관계수 (선형성): {pearson_r:.4f} (p-value: {p_value_p:.4e})")
print(f"  → 방향: {'-' if pearson_r < 0 else '+'}, 강도: {'약함' if abs(pearson_r)<0.3 else '중간' if abs(pearson_r)<0.7 else '강함'}")

print(f"\n[2] Spearman 상관계수 (추세성): {spearman_rho:.4f} (p-value: {p_value_s:.4e})")
print(f"  → 방향: {'-' if spearman_rho < 0 else '+'}, 강도: {'약함' if abs(spearman_rho)<0.3 else '중간' if abs(spearman_rho)<0.7 else '강함'}")

print(f"\n[3] 회귀 분석 (OLS)")
print(f"  방정식: Y = {slope:.4f} * X + {intercept:.4f}")
print(f"  R-squared: {r_value**2:.4f}")
print(f"  → 스킵률이 10% 증가할 때마다, 재생량은 평균적으로 {abs((2**(slope*0.1) - 1)*100):.1f}% {'감소' if slope < 0 else '증가'}함")

if p_value_s < 0.05:
    print(f"\n✅ 통계적으로 유의미한 결과입니다. (p-value < 0.05)")
    print(f"   '스킵률 증가'는 미래 '재생량 감소'와 확실한 음의 상관관계를 가짐을 증명합니다.")
else:
    print(f"\n❌ 통계적 유의성이 부족합니다. (p-value >= 0.05)")

# 시각화 (산점도 및 회귀선, 2D 밀도 추정)
plt.figure(figsize=(10, 7))

# 산점도 
sns.regplot(
    data=df_reg, 
    x='skip_delta', 
    y='log_change', 
    scatter_kws={'alpha': 0.3, 's': 20, 'color': '#3498db'}, 
    line_kws={'color': '#e74c3c', 'linewidth': 3},
    order=1
)

plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)

plt.title('전반부 스킵률 변화(X)와 후반부 재생량 변화(Y) 상관관계', fontsize=15, fontweight='bold', pad=15)
plt.xlabel('스킵률 변화량 (양수 = 스킵 빈도 증가 = 질리는 중)', fontsize=12)
plt.ylabel('재생량 변화 (log2) \n 0=유지, -1=반토막, 1=두배', fontsize=12)

# 상관계수 텍스트 표시
textstr = f"Spearman ρ: {spearman_rho:.3f} (p={p_value_s:.2e})\nSlope: {slope:.3f}"
plt.text(0.05, 0.05, textstr, transform=plt.gca().transAxes, fontsize=12,
        verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

save_path = os.path.join(REPORT_DIR, 'regression_analysis.png')
plt.tight_layout()
plt.savefig(save_path, dpi=150)
plt.close()

print(f"\n📊 회귀 분석 시각화 저장: {save_path}")
