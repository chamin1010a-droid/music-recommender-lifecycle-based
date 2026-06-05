"""
가설 1 검증: 골디락스 존 (Goldilocks Zone)
"누적 N번 들은 곡의 재청취 확률은 역-U 형태"

Step A: 전체 풀링 2차 회귀
Step B: 사용자별 개별 2차 회귀 → β₂<0 비율 이항검정
Robustness: 7일/3일/30일 재생수 3가지 지표
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from numpy.polynomial import polynomial as P

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")
FEATURES_FILE = os.path.join(DATA_DIR, "lastfm_1k_features.parquet")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_features():
    print("Loading features...")
    df = pd.read_parquet(FEATURES_FILE)
    print(f"  {len(df):,} rows loaded")
    return df


def step_a_pooled_analysis(df):
    """Step A: 전체 풀링 — familiarity bin별 재청취 확률"""
    print("\n" + "="*60)
    print("STEP A: 전체 풀링 분석 (Pooled)")
    print("="*60)
    
    # familiarity cap (100 이상은 극소수 → 노이즈)
    MAX_FAM = 100
    df_capped = df[df['familiarity'] <= MAX_FAM].copy()
    
    # Bin 생성 (0, 1, 2, 3-5, 6-10, 11-20, 21-30, 31-50, 51-100)
    bins = [0, 1, 2, 3, 6, 11, 21, 31, 51, 101]
    labels = ['0', '1', '2', '3-5', '6-10', '11-20', '21-30', '31-50', '51-100']
    df_capped['fam_bin'] = pd.cut(df_capped['familiarity'], bins=bins, labels=labels, right=False)
    
    results = {}
    for proxy_name, col in [('Relisten_7d', 'relisten_7d'), ('Relisten_3d', 'relisten_3d')]:
        grouped = df_capped.groupby('fam_bin', observed=True)[col].agg(['mean', 'count'])
        grouped.columns = ['avg_relisten', 'n_events']
        results[proxy_name] = grouped
        
        print(f"\n--- {proxy_name} ---")
        print(grouped.to_string())
    
    # 2차 회귀 (연속 변수로)
    print("\n--- 2차 회귀 (풀링) ---")
    for proxy_name, col in [('Relisten_7d', 'relisten_7d'), ('Relisten_3d', 'relisten_3d')]:
        x = df_capped['familiarity'].values.astype(float)
        y = df_capped[col].values.astype(float)
        
        # OLS: y = β₀ + β₁x + β₂x²
        X = np.column_stack([np.ones_like(x), x, x**2])
        beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        
        # t-검정 for β₂
        n = len(x)
        p = 3  # 파라미터 수
        y_hat = X @ beta
        mse = np.sum((y - y_hat)**2) / (n - p)
        var_beta = mse * np.linalg.inv(X.T @ X)
        se_beta2 = np.sqrt(var_beta[2, 2])
        t_stat = beta[2] / se_beta2
        p_value = 2 * stats.t.sf(abs(t_stat), n - p)
        
        peak_f = -beta[1] / (2 * beta[2]) if beta[2] != 0 else float('inf')
        
        print(f"\n  [{proxy_name}]")
        print(f"    beta_0 = {beta[0]:.6f}")
        print(f"    beta_1 = {beta[1]:.6f}")
        print(f"    beta_2 = {beta[2]:.8f} (key!)")
        print(f"    t(beta_2) = {t_stat:.2f}, p = {p_value:.2e}")
        print(f"    beta_2 < 0? {'YES' if beta[2] < 0 else 'NO'}")
        print(f"    p < 0.05?   {'YES' if p_value < 0.05 else 'NO'}")
        print(f"    Peak position (f*) = {peak_f:.1f}")
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (proxy_name, col) in enumerate([('Relisten 7일', 'relisten_7d'), ('Relisten 3일', 'relisten_3d')]):
        ax = axes[idx]
        grouped = df_capped.groupby('fam_bin', observed=True)[col].agg(['mean', 'count'])
        
        x_pos = range(len(grouped))
        bars = ax.bar(x_pos, grouped['mean'], color='#4ecdc4', alpha=0.8, edgecolor='white')
        
        # 데이터 포인트 수 표시
        for i, (val, cnt) in enumerate(zip(grouped['mean'], grouped['count'])):
            ax.text(i, val + 0.005, f'n={cnt:,}', ha='center', va='bottom', fontsize=7, color='gray')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(grouped.index, rotation=45)
        ax.set_xlabel('누적 청취 횟수 (Familiarity)')
        ax.set_ylabel(f'평균 재청취 확률')
        ax.set_title(f'가설 1: 골디락스 존 — {proxy_name}')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'h1_goldilocks_pooled.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Chart saved: h1_goldilocks_pooled.png")
    
    return results


def step_b_per_user_analysis(df):
    """Step B: 사용자별 개별 2차 회귀 → β₂<0 비율"""
    print("\n" + "="*60)
    print("STEP B: 사용자별 개별 분석")
    print("="*60)
    
    # 충분한 데이터가 있는 사용자만 (재생 500회+, 고유 곡 50곡+)
    user_stats = df.groupby('user_id').agg(
        n_plays=('song_key', 'count'),
        n_unique_songs=('song_key', 'nunique'),
        max_fam=('familiarity', 'max')
    )
    qualified_users = user_stats[
        (user_stats['n_plays'] >= 500) & 
        (user_stats['n_unique_songs'] >= 50) &
        (user_stats['max_fam'] >= 10)  # 최소 10번 이상 들은 곡이 있어야
    ].index
    
    print(f"  분석 대상 사용자: {len(qualified_users)} / {df['user_id'].nunique()}")
    
    results_per_user = []
    
    for uid in qualified_users:
        user_df = df[df['user_id'] == uid]
        x = user_df['familiarity'].values.astype(float)
        y = user_df['relisten_7d'].values.astype(float)
        
        if len(x) < 30:  # 최소 데이터 포인트
            continue
            
        # 2차 회귀
        try:
            X = np.column_stack([np.ones_like(x), x, x**2])
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            
            # β₂ 유의성 (간략)
            n = len(x)
            y_hat = X @ beta
            mse = np.sum((y - y_hat)**2) / (n - 3)
            if mse <= 0:
                continue
            var_beta = mse * np.linalg.inv(X.T @ X)
            se_beta2 = np.sqrt(max(var_beta[2, 2], 1e-30))
            t_stat = beta[2] / se_beta2
            p_value = 2 * stats.t.sf(abs(t_stat), n - 3)
            
            peak_f = -beta[1] / (2 * beta[2]) if beta[2] != 0 else float('inf')
            
            results_per_user.append({
                'user_id': uid,
                'beta2': beta[2],
                'beta2_significant': p_value < 0.05,
                'inverted_u': beta[2] < 0 and p_value < 0.05,
                'peak_f': peak_f,
                'n_plays': len(x)
            })
        except:
            continue
    
    results_df = pd.DataFrame(results_per_user)
    
    n_total = len(results_df)
    n_negative_beta2 = (results_df['beta2'] < 0).sum()
    n_inverted_u = results_df['inverted_u'].sum()
    
    print(f"\n  분석 완료 사용자: {n_total}")
    print(f"  β₂ < 0 (역-U 경향): {n_negative_beta2} ({n_negative_beta2/n_total:.1%})")
    print(f"  β₂ < 0 + 유의 (p<0.05): {n_inverted_u} ({n_inverted_u/n_total:.1%})")
    
    # 이항 검정: β₂ < 0인 비율이 50%보다 유의하게 큰가?
    binom_result = stats.binomtest(n_negative_beta2, n_total, p=0.5, alternative='greater')
    print(f"\n  이항 검정 (β₂<0 비율 > 50%):")
    print(f"    p-value = {binom_result.pvalue:.2e}")
    print(f"    결론: {'PASS - 과반이 역-U를 보임' if binom_result.pvalue < 0.05 else 'FAIL'}")
    
    # 피크 위치 분포 시각화
    valid_peaks = results_df[results_df['inverted_u'] & (results_df['peak_f'] > 0) & (results_df['peak_f'] < 200)]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # β₂ 분포
    ax1 = axes[0]
    ax1.hist(results_df['beta2'], bins=50, color='#ff6b6b', alpha=0.7, edgecolor='white')
    ax1.axvline(0, color='black', linestyle='--', linewidth=2, label='β₂ = 0')
    ax1.set_xlabel('β₂ (2차항 계수)')
    ax1.set_ylabel('사용자 수')
    ax1.set_title(f'사용자별 β₂ 분포\n(β₂<0: {n_negative_beta2}/{n_total} = {n_negative_beta2/n_total:.1%})')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 피크 위치 분포
    ax2 = axes[1]
    if len(valid_peaks) > 0:
        ax2.hist(valid_peaks['peak_f'], bins=30, color='#4ecdc4', alpha=0.7, edgecolor='white')
        median_peak = valid_peaks['peak_f'].median()
        ax2.axvline(median_peak, color='red', linestyle='--', linewidth=2, 
                     label=f'중앙값 = {median_peak:.1f}회')
        ax2.legend()
    ax2.set_xlabel('피크 위치 (f* = 최적 청취 횟수)')
    ax2.set_ylabel('사용자 수')
    ax2.set_title(f'사용자별 골디락스 피크 위치 분포\n(유의한 역-U 사용자 {len(valid_peaks)}명)')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'h1_goldilocks_per_user.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Chart saved: h1_goldilocks_per_user.png")
    
    return results_df


if __name__ == "__main__":
    df = load_features()
    
    # Step A
    pooled_results = step_a_pooled_analysis(df)
    
    # Step B
    per_user_results = step_b_per_user_analysis(df)
    
    print("\n" + "="*60)
    print("가설 1 검증 완료!")
    print("="*60)
