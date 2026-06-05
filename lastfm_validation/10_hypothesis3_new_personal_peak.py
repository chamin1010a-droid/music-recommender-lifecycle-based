"""
새 가설 3: 개인별 골디락스 피크 일관성
"한 사람은 곡을 가장 좋아하게 되는 고유한 재생 횟수(N)가 있다"

증명 방법:
1. 사용자-곡별로 재청취 확률이 가장 높은 재생 횟수(피크)를 찾는다
2. 한 사용자의 곡들 간에 피크가 일관되는지 본다 (within-user std)
3. 사용자 간 피크 차이가 큰지 본다 (between-user std)
4. ICC(급내상관계수)로 "개인 고유 피크"의 존재를 검정
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
    df = pd.read_parquet(FEATURES_FILE, columns=['user_id', 'song_key', 'familiarity', 'relisten_7d'])
    print(f"  {len(df):,} rows loaded")
    return df


def find_song_peaks(df, min_max_fam=15, smoothing_window=3):
    """
    각 사용자-곡별로 재청취 확률이 가장 높은 재생 횟수(피크)를 찾는다

    방법:
    - 사용자-곡별로 familiarity 값마다 relisten_7d 평균을 구함
    - 노이즈 제거를 위해 이동평균(window=3)으로 스무딩
    - 스무딩된 곡선에서 재청취 확률이 최대인 familiarity = 피크
    """
    print("\n[1/4] 곡별 피크 재생 횟수 탐색...")

    # max familiarity가 min_max_fam 이상인 곡만 (충분히 많이 들은 곡)
    max_fam = df.groupby(['user_id', 'song_key'])['familiarity'].max().reset_index()
    max_fam.columns = ['user_id', 'song_key', 'max_fam']
    qualified = max_fam[max_fam['max_fam'] >= min_max_fam]
    print(f"  최대 재생 {min_max_fam}회 이상 사용자-곡: {len(qualified):,}")

    df_q = df.merge(qualified[['user_id', 'song_key']], on=['user_id', 'song_key'])

    # 사용자-곡-familiarity별 재청취 확률
    grouped = df_q.groupby(['user_id', 'song_key', 'familiarity'])['relisten_7d'].mean().reset_index()

    song_peaks = []
    pairs = grouped.groupby(['user_id', 'song_key'])
    total = len(pairs)

    for i, ((uid, skey), grp) in enumerate(pairs):
        if i % 10000 == 0:
            print(f"  Processing {i:,}/{total:,}...", end='\r')

        grp = grp.sort_values('familiarity')
        fam_vals = grp['familiarity'].values
        rel_vals = grp['relisten_7d'].values

        # 데이터가 너무 적으면 스킵
        if len(fam_vals) < 5:
            continue

        # 이동평균 스무딩
        if len(rel_vals) >= smoothing_window:
            smoothed = np.convolve(rel_vals, np.ones(smoothing_window)/smoothing_window, mode='valid')
            fam_smoothed = fam_vals[(smoothing_window-1)//2 : (smoothing_window-1)//2 + len(smoothed)]
        else:
            smoothed = rel_vals
            fam_smoothed = fam_vals

        # 피크 위치
        peak_idx = np.argmax(smoothed)
        peak_fam = fam_smoothed[peak_idx]
        peak_relisten = smoothed[peak_idx]

        song_peaks.append({
            'user_id': uid,
            'song_key': skey,
            'peak_fam': peak_fam,
            'peak_relisten': peak_relisten,
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


def compute_icc(peaks_df, min_songs=10):
    """
    ICC(급내상관계수) 계산: 한 사용자의 곡들 간 피크 일관성

    ICC가 높으면 → 같은 사람의 피크는 비슷하고, 다른 사람의 피크는 다르다
    = "개인별 고유 피크가 존재한다"
    """
    print("\n[2/4] ICC 계산...")

    # 곡이 min_songs개 이상인 사용자만
    user_song_counts = peaks_df.groupby('user_id').size().reset_index(name='n_songs')
    qualified_users = user_song_counts[user_song_counts['n_songs'] >= min_songs]['user_id']
    peaks_q = peaks_df[peaks_df['user_id'].isin(qualified_users)]

    n_users = peaks_q['user_id'].nunique()
    print(f"  분석 대상 사용자: {n_users} (곡 {min_songs}개+)")
    print(f"  분석 대상 곡: {len(peaks_q):,}")

    # 사용자별 피크 통계
    user_stats = peaks_q.groupby('user_id')['peak_fam'].agg(['mean', 'std', 'count']).reset_index()
    user_stats.columns = ['user_id', 'user_mean_peak', 'user_std_peak', 'n_songs']

    # Within-user variance (사용자 안의 편차)
    within_var = (user_stats['user_std_peak'] ** 2).mean()

    # Between-user variance (사용자 간 편차)
    grand_mean = user_stats['user_mean_peak'].mean()
    between_var = user_stats['user_mean_peak'].var()

    # ICC(1) = between / (between + within)
    icc = between_var / (between_var + within_var) if (between_var + within_var) > 0 else 0

    print(f"\n  전체 평균 피크: {grand_mean:.1f}회")
    print(f"  사용자 간 분산 (Between): {between_var:.1f}")
    print(f"  사용자 내 분산 (Within):  {within_var:.1f}")
    print(f"  ICC = {icc:.4f}")

    # ICC 해석
    if icc >= 0.75:
        interp = "Excellent — 개인 고유 피크가 매우 뚜렷함"
    elif icc >= 0.50:
        interp = "Good — 개인 고유 피크가 존재함"
    elif icc >= 0.25:
        interp = "Fair — 어느 정도 일관성 있음"
    else:
        interp = "Poor — 개인 고유 피크가 뚜렷하지 않음"

    print(f"  해석: {interp}")

    # F-검정 (one-way ANOVA: 사용자별 그룹)
    groups = [g['peak_fam'].values for _, g in peaks_q.groupby('user_id')]
    f_stat, f_pvalue = stats.f_oneway(*groups)
    print(f"\n  One-way ANOVA:")
    print(f"    F = {f_stat:.2f}, p = {f_pvalue:.2e}")
    print(f"    결론: {'PASS — 사용자별 피크가 유의미하게 다름' if f_pvalue < 0.05 else 'FAIL'}")

    return user_stats, icc, f_stat, f_pvalue, peaks_q


def visualize(peaks_df, user_stats, peaks_q, icc):
    """시각화"""
    print("\n[3/4] 시각화...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('가설 3 (수정): 개인별 골디락스 피크의 일관성\n'
                 '"한 사람은 곡을 가장 좋아하게 되는 고유한 재생 횟수가 있다"',
                 fontsize=13, fontweight='bold')

    # 1) 전체 피크 분포
    ax1 = axes[0, 0]
    peaks_capped = peaks_df['peak_fam'].clip(upper=100)
    ax1.hist(peaks_capped, bins=50, color='#4ecdc4', alpha=0.7, edgecolor='white')
    median_peak = peaks_df['peak_fam'].median()
    ax1.axvline(median_peak, color='red', linestyle='--', linewidth=2,
                label=f'중앙값 = {median_peak:.0f}회')
    ax1.set_xlabel('피크 재생 횟수')
    ax1.set_ylabel('곡 수')
    ax1.set_title('전체 곡의 피크 위치 분포', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2) 사용자별 평균 피크 분포
    ax2 = axes[0, 1]
    user_peaks_capped = user_stats['user_mean_peak'].clip(upper=80)
    ax2.hist(user_peaks_capped, bins=40, color='#45b7d1', alpha=0.7, edgecolor='white')
    grand_mean = user_stats['user_mean_peak'].mean()
    ax2.axvline(grand_mean, color='red', linestyle='--', linewidth=2,
                label=f'전체 평균 = {grand_mean:.1f}회')
    ax2.set_xlabel('사용자별 평균 피크 재생 횟수')
    ax2.set_ylabel('사용자 수')
    ax2.set_title(f'사용자별 "고유 피크" 분포\nICC = {icc:.3f}', fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3) Within vs Between 비교 (핵심)
    ax3 = axes[1, 0]
    within_std = user_stats['user_std_peak'].mean()
    between_std = user_stats['user_mean_peak'].std()
    bars = ax3.bar(['사용자 내 편차\n(Within)', '사용자 간 편차\n(Between)'],
                    [within_std, between_std],
                    color=['#ff6b6b', '#4ecdc4'], edgecolor='white', width=0.5)
    for bar, val in zip(bars, [within_std, between_std]):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.1f}', ha='center', fontsize=12, fontweight='bold')
    ax3.set_ylabel('표준편차')
    ax3.set_title('핵심 비교: 같은 사람의 곡들은 비슷한 피크를 갖는가?\n'
                   '(Between > Within이면 개인 고유 피크 존재)', fontsize=11)
    ax3.grid(axis='y', alpha=0.3)

    # 4) 예시 사용자들의 피크 분포 비교
    ax4 = axes[1, 1]
    # 곡이 많은 상위 6명 선택
    top_users = user_stats.nlargest(6, 'n_songs')['user_id'].values
    colors_ex = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#ffd93d', '#95a5a6', '#ff9ff3']

    for idx, uid in enumerate(top_users):
        user_peaks = peaks_q[peaks_q['user_id'] == uid]['peak_fam'].clip(upper=80)
        ax4.hist(user_peaks, bins=20, alpha=0.4, color=colors_ex[idx],
                 label=f'User {idx+1} (n={len(user_peaks)}, '
                       f'avg={user_peaks.mean():.0f})', density=True)

    ax4.set_xlabel('피크 재생 횟수')
    ax4.set_ylabel('밀도')
    ax4.set_title('예시: 6명의 피크 분포 비교\n(사람마다 다른 위치에 몰리는가?)', fontsize=11)
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'h3_new_personal_peak.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved: h3_new_personal_peak.png")


def additional_analysis(peaks_q, user_stats):
    """추가 분석: 피크 일관성과 청취 성향의 관계"""
    print("\n[4/4] 추가 분석...")

    # 사용자별 피크 일관성 (std가 낮을수록 일관)
    consistent = user_stats[user_stats['user_std_peak'] <= user_stats['user_std_peak'].median()]
    inconsistent = user_stats[user_stats['user_std_peak'] > user_stats['user_std_peak'].median()]

    print(f"\n  일관된 사용자 (std ≤ 중앙값):")
    print(f"    평균 피크: {consistent['user_mean_peak'].mean():.1f}회")
    print(f"    평균 편차: {consistent['user_std_peak'].mean():.1f}")
    print(f"  비일관 사용자 (std > 중앙값):")
    print(f"    평균 피크: {inconsistent['user_mean_peak'].mean():.1f}회")
    print(f"    평균 편차: {inconsistent['user_std_peak'].mean():.1f}")


if __name__ == "__main__":
    df = load_features()
    peaks_df = find_song_peaks(df)
    user_stats, icc, f_stat, f_pvalue, peaks_q = compute_icc(peaks_df)
    visualize(peaks_df, user_stats, peaks_q, icc)
    additional_analysis(peaks_q, user_stats)

    print("\n" + "="*60)
    print("가설 3 (수정) 검증 완료!")
    print("="*60)
