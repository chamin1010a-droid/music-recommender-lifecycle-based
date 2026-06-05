"""
가설 3 (최종): 개인별 골디락스 피크 일관성
"한 사람은 곡을 가장 좋아하게 되는 고유한 재생 횟수가 있다"

지표: 청취 간격 (다음 재생까지의 일수) — 짧을수록 그 시점에 곡을 좋아함
검증: 한 사용자의 곡을 5그룹으로 나누고, 각 그룹의 피크가 일치하는지 확인
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")
FEATURES_FILE = os.path.join(DATA_DIR, "lastfm_1k_features.parquet")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_features():
    print("Loading features...")
    df = pd.read_parquet(FEATURES_FILE, columns=['user_id', 'song_key', 'familiarity', 'days_to_next'])
    print(f"  {len(df):,} rows loaded")
    return df


def find_peaks_by_interval(df, min_max_fam=20):
    """
    곡별 피크 = 청취 간격이 가장 짧은 familiarity 구간
    
    days_to_next: 현재 재생 → 같은 곡의 다음 재생까지 일수
    이 값이 작을수록 "지금 이 곡이 좋다"는 뜻
    """
    print("\n[1/4] 청취 간격 기반 피크 탐색...")
    
    # NaN 제거 (마지막 재생은 다음이 없으므로)
    df = df.dropna(subset=['days_to_next'])

    # 충분히 많이 들은 곡만
    max_fam = df.groupby(['user_id', 'song_key'])['familiarity'].max().reset_index()
    max_fam.columns = ['user_id', 'song_key', 'max_fam']
    qualified = max_fam[max_fam['max_fam'] >= min_max_fam]
    print(f"  최대 재생 {min_max_fam}회 이상 사용자-곡: {len(qualified):,}")

    df_q = df.merge(qualified[['user_id', 'song_key']], on=['user_id', 'song_key'])

    # 초반 "발견 효과" 제거: familiarity 3 이상만 사용
    df_q = df_q[df_q['familiarity'] >= 3]

    # 사용자-곡-familiarity별 평균 간격
    interval_by_fam = df_q.groupby(['user_id', 'song_key', 'familiarity'])['days_to_next'].mean().reset_index()

    song_peaks = []
    pairs = interval_by_fam.groupby(['user_id', 'song_key'])
    total = len(pairs)

    for i, ((uid, skey), grp) in enumerate(pairs):
        if i % 10000 == 0:
            print(f"  Processing {i:,}/{total:,}...", end='\r')

        grp = grp.sort_values('familiarity')
        fam_vals = grp['familiarity'].values
        interval_vals = grp['days_to_next'].values

        if len(fam_vals) < 5:
            continue

        # 이동평균 스무딩 (window=3)
        if len(interval_vals) >= 3:
            smoothed = np.convolve(interval_vals, np.ones(3)/3, mode='valid')
            fam_smoothed = fam_vals[1:1+len(smoothed)]
        else:
            smoothed = interval_vals
            fam_smoothed = fam_vals

        # 간격이 가장 짧은 지점 = 피크 (가장 좋아하던 시점)
        peak_idx = np.argmin(smoothed)
        peak_fam = fam_smoothed[peak_idx]
        peak_interval = smoothed[peak_idx]

        song_peaks.append({
            'user_id': uid,
            'song_key': skey,
            'peak_fam': int(peak_fam),
            'peak_interval': peak_interval,
            'max_fam': fam_vals.max(),
        })

    peaks_df = pd.DataFrame(song_peaks)
    print(f"\n  피크 탐색 완료: {len(peaks_df):,}곡")
    print(f"  피크 재생 횟수 분포:")
    print(f"    중앙값: {peaks_df['peak_fam'].median():.0f}회")
    print(f"    평균: {peaks_df['peak_fam'].mean():.1f}회")
    print(f"    25%: {peaks_df['peak_fam'].quantile(0.25):.0f}회")
    print(f"    75%: {peaks_df['peak_fam'].quantile(0.75):.0f}회")

    return peaks_df


def kfold_consistency(peaks_df, k=5, min_songs=25, n_bootstrap=100):
    """
    K-fold 일관성 검증
    
    한 사용자의 곡을 K개 그룹으로 나누고,
    각 그룹의 평균 피크를 구한 뒤,
    K개 피크의 편차가 작으면 → 개인 고유 피크 존재
    """
    print(f"\n[2/4] {k}-fold 일관성 검증 (곡 {min_songs}개+ 사용자, {n_bootstrap}회 반복)...")

    # 곡이 충분한 사용자만
    user_counts = peaks_df.groupby('user_id').size()
    qualified_users = user_counts[user_counts >= min_songs].index
    peaks_q = peaks_df[peaks_df['user_id'].isin(qualified_users)]

    n_users = len(qualified_users)
    print(f"  분석 대상: {n_users}명 (곡 {min_songs}개+)")

    user_results = []

    for uid in qualified_users:
        user_peaks = peaks_q[peaks_q['user_id'] == uid]['peak_fam'].values.copy()

        # n_bootstrap번 랜덤 셔플 → K-fold 분할 → 각 fold 평균 피크
        fold_stds = []
        for _ in range(n_bootstrap):
            np.random.shuffle(user_peaks)
            folds = np.array_split(user_peaks, k)
            fold_means = [f.mean() for f in folds if len(f) > 0]
            fold_stds.append(np.std(fold_means))

        avg_fold_std = np.mean(fold_stds)
        user_mean_peak = user_peaks.mean()

        user_results.append({
            'user_id': uid,
            'n_songs': len(user_peaks),
            'user_mean_peak': user_mean_peak,
            'user_overall_std': user_peaks.std(),
            'kfold_std': avg_fold_std,  # 그룹간 편차 (작으면 일관)
        })

    results_df = pd.DataFrame(user_results)

    # 비교: k-fold 평균 편차 vs 셔플된 가짜 데이터
    # 귀무가설: 피크에 개인 패턴이 없다면, 곡들을 무작위로 섞어 배정해도 편차가 같아야
    print(f"\n  --- 실제 vs 랜덤 비교 ---")

    real_kfold_stds = results_df['kfold_std'].values

    # 랜덤 baseline: 모든 사용자의 피크를 한 풀에 넣고 랜덤으로 재배정
    all_peaks = peaks_q['peak_fam'].values
    random_stds = []
    for _ in range(n_bootstrap):
        for uid in qualified_users:
            n = len(peaks_q[peaks_q['user_id'] == uid])
            fake_peaks = np.random.choice(all_peaks, n, replace=True)
            folds = np.array_split(fake_peaks, k)
            fold_means = [f.mean() for f in folds if len(f) > 0]
            random_stds.append(np.std(fold_means))

    random_stds = np.array(random_stds).reshape(n_bootstrap, n_users).mean(axis=0)

    real_mean_std = real_kfold_stds.mean()
    random_mean_std = random_stds.mean()

    print(f"  실제 K-fold 편차 평균: {real_mean_std:.2f}")
    print(f"  랜덤 K-fold 편차 평균: {random_mean_std:.2f}")
    print(f"  비율: {real_mean_std / random_mean_std:.2f}x (1보다 작으면 실제가 더 일관)")

    # Wilcoxon 검정: 실제 편차 < 랜덤 편차?
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(real_kfold_stds, random_stds, alternative='less')
    print(f"\n  Wilcoxon 검정 (실제 편차 < 랜덤 편차?):")
    print(f"    statistic = {wilcoxon_stat:,.0f}")
    print(f"    p-value = {wilcoxon_p:.2e}")
    print(f"    결론: {'PASS — 같은 사람의 곡들은 랜덤보다 피크가 일관됨' if wilcoxon_p < 0.05 else 'FAIL'}")

    # ICC 계산
    within_var = (results_df['user_overall_std'] ** 2).mean()
    between_var = results_df['user_mean_peak'].var()
    icc = between_var / (between_var + within_var) if (between_var + within_var) > 0 else 0

    print(f"\n  ICC = {icc:.4f}")
    if icc >= 0.4:
        print(f"  (Good — 개인 고유 피크가 존재함)")
    elif icc >= 0.2:
        print(f"  (Fair — 어느 정도 일관성)")
    else:
        print(f"  (Weak — 약한 일관성)")

    # ANOVA
    groups = [g['peak_fam'].values for _, g in peaks_q.groupby('user_id')]
    f_stat, f_p = stats.f_oneway(*groups)
    print(f"\n  ANOVA: F = {f_stat:.2f}, p = {f_p:.2e}")

    return results_df, real_kfold_stds, random_stds, icc


def visualize(peaks_df, results_df, real_stds, random_stds, icc, peaks_q_users):
    """시각화"""
    print("\n[3/4] 시각화...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('가설 3 (최종): 개인별 골디락스 피크의 일관성\n'
                 '"한 사람은 곡을 가장 좋아하게 되는 고유한 재생 횟수가 있다"\n'
                 '(지표: 청취 간격이 가장 짧은 시점 = 가장 좋아하던 시점)',
                 fontsize=12, fontweight='bold')

    # 1) 전체 피크 분포
    ax1 = axes[0, 0]
    peaks_capped = peaks_df['peak_fam'].clip(upper=60)
    ax1.hist(peaks_capped, bins=40, color='#4ecdc4', alpha=0.7, edgecolor='white')
    median_val = peaks_df['peak_fam'].median()
    ax1.axvline(median_val, color='red', linestyle='--', linewidth=2,
                label=f'중앙값 = {median_val:.0f}회')
    ax1.set_xlabel('피크 재생 횟수')
    ax1.set_ylabel('곡 수')
    ax1.set_title('전체 곡의 피크 분포\n(청취 간격이 가장 짧았던 재생 횟수)', fontsize=11)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2) 실제 vs 랜덤 K-fold 편차 비교 (핵심!)
    ax2 = axes[0, 1]
    ax2.hist(real_stds, bins=30, alpha=0.6, color='#4ecdc4', label=f'실제 (평균={real_stds.mean():.2f})', density=True)
    ax2.hist(random_stds, bins=30, alpha=0.6, color='#ff6b6b', label=f'랜덤 (평균={random_stds.mean():.2f})', density=True)
    ax2.set_xlabel('5-fold 피크 편차')
    ax2.set_ylabel('밀도')
    ax2.set_title('핵심: 실제 vs 랜덤 편차 비교\n(실제가 왼쪽이면 개인 고유 피크 존재)', fontsize=11)
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3) 사용자별 평균 피크 분포
    ax3 = axes[1, 0]
    ax3.hist(results_df['user_mean_peak'].clip(upper=30), bins=30, color='#45b7d1', alpha=0.7, edgecolor='white')
    grand_mean = results_df['user_mean_peak'].mean()
    ax3.axvline(grand_mean, color='red', linestyle='--', linewidth=2,
                label=f'전체 평균 = {grand_mean:.1f}회')
    ax3.set_xlabel('사용자별 평균 피크 재생 횟수')
    ax3.set_ylabel('사용자 수')
    ax3.set_title(f'사용자별 고유 피크 분포 (ICC = {icc:.3f})', fontsize=11)
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4) 예시 사용자: 곡들의 피크 분포
    ax4 = axes[1, 1]
    # 곡이 많고, 편차가 다양한 사용자 6명 선택
    example_users = results_df.nlargest(6, 'n_songs')
    colors_ex = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#ffd93d', '#95a5a6', '#ff9ff3']

    qualified_user_ids = example_users['user_id'].values
    peaks_q = peaks_df[peaks_df['user_id'].isin(qualified_user_ids)]

    for idx, uid in enumerate(qualified_user_ids):
        user_data = peaks_q[peaks_q['user_id'] == uid]['peak_fam'].clip(upper=40)
        user_mean = user_data.mean()
        ax4.hist(user_data, bins=20, alpha=0.4, color=colors_ex[idx],
                 label=f'User {idx+1} (n={len(user_data)}, avg={user_mean:.0f})', density=True)

    ax4.set_xlabel('피크 재생 횟수')
    ax4.set_ylabel('밀도')
    ax4.set_title('예시: 6명의 곡별 피크 분포\n(사람마다 다른 위치에 몰리는가?)', fontsize=11)
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'h3_final_personal_peak.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved: h3_final_personal_peak.png")


if __name__ == "__main__":
    df = load_features()
    peaks_df = find_peaks_by_interval(df)
    results_df, real_stds, random_stds, icc = kfold_consistency(peaks_df)
    visualize(peaks_df, results_df, real_stds, random_stds, icc, peaks_df)

    print("\n" + "="*60)
    print("가설 3 (최종) 검증 완료!")
    print("="*60)
