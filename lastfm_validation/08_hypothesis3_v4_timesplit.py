"""
가설 3 v4: 시간 기준 분리 (Time-based Split)

핵심:
- 각 사용자의 전체 활동 기간을 시간 기준으로 70/30 분리
- 관찰 기간(70%): Zone 분류
- 검증 기간(30%): 해당 곡이 실제로 재생되었나? (0번일 수 있음 = 은퇴)
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
    df = pd.read_parquet(FEATURES_FILE, columns=['user_id', 'timestamp', 'song_key'])
    print(f"  {len(df):,} rows loaded")
    return df


def classify_zones_time_split(df, min_plays=15, train_ratio=0.7):
    """
    시간 기준 분리

    1) 각 사용자의 전체 활동 기간에서 70% 시점을 cutoff로 잡음
    2) cutoff 이전 재생만으로 Zone 분류
    3) cutoff 이후 재생이 있는지로 생존 측정
    """
    print(f"\n[1/3] 시간 기준 분리 (관찰 {train_ratio:.0%} / 검증 {1-train_ratio:.0%})...")

    # 사용자별 활동 기간 & cutoff
    user_periods = df.groupby('user_id')['timestamp'].agg(['min', 'max'])
    user_periods.columns = ['first_ts', 'last_ts']
    user_periods['total_days'] = (user_periods['last_ts'] - user_periods['first_ts']).dt.days
    # 활동 기간 60일 미만인 사용자 제외
    user_periods = user_periods[user_periods['total_days'] >= 60]
    user_periods['cutoff'] = user_periods['first_ts'] + pd.to_timedelta(
        user_periods['total_days'] * train_ratio, unit='D'
    )

    print(f"  분석 대상 사용자: {len(user_periods):,}")

    # cutoff 정보를 메인 df에 결합
    df = df.merge(user_periods[['cutoff']], left_on='user_id', right_index=True)

    # 관찰/검증 분리
    df_train = df[df['timestamp'] <= df['cutoff']]
    df_test = df[df['timestamp'] > df['cutoff']]

    # 관찰 기간에서 min_plays 이상 들은 사용자-곡 쌍만
    train_counts = df_train.groupby(['user_id', 'song_key']).size().reset_index(name='train_plays')
    qualified = train_counts[train_counts['train_plays'] >= min_plays]
    print(f"  관찰 기간 {min_plays}회+ 사용자-곡 쌍: {len(qualified):,}")

    # 검증 기간 재생 횟수
    test_counts = df_test.groupby(['user_id', 'song_key']).size().reset_index(name='test_plays')

    # Zone 분류 (관찰 기간 데이터만 사용)
    df_train_q = df_train.merge(qualified[['user_id', 'song_key']], on=['user_id', 'song_key'])

    zone_results = []
    grouped = df_train_q.groupby(['user_id', 'song_key'])
    total = len(grouped)

    for i, ((uid, skey), grp) in enumerate(grouped):
        if i % 10000 == 0:
            print(f"  Processing {i:,}/{total:,}...", end='\r')

        grp = grp.sort_values('timestamp')
        n = len(grp)
        half = n // 2

        first_half = grp.iloc[:half]
        second_half = grp.iloc[half:]

        fh_days = max((first_half['timestamp'].max() - first_half['timestamp'].min()).days, 1)
        sh_days = max((second_half['timestamp'].max() - second_half['timestamp'].min()).days, 1)

        fh_intensity = len(first_half) / fh_days
        sh_intensity = len(second_half) / sh_days

        if fh_intensity > 0:
            change_ratio = (sh_intensity - fh_intensity) / fh_intensity
        else:
            change_ratio = 0

        THRESHOLD = 0.3
        if change_ratio > THRESHOLD:
            zone = 'Zone 2'
        elif change_ratio < -THRESHOLD:
            zone = 'Zone 3'
        elif fh_intensity > 0.1 and sh_intensity > 0.1:
            zone = 'Zone 1'
        else:
            zone = 'Zone 4'

        zone_results.append({
            'user_id': uid,
            'song_key': skey,
            'train_plays': n,
            'fh_intensity': fh_intensity,
            'sh_intensity': sh_intensity,
            'change_ratio': change_ratio,
            'zone': zone,
        })

    zone_df = pd.DataFrame(zone_results)

    # 검증 기간 재생 횟수 결합 (없으면 0)
    zone_df = zone_df.merge(test_counts, on=['user_id', 'song_key'], how='left')
    zone_df['test_plays'] = zone_df['test_plays'].fillna(0).astype(int)
    zone_df['survived'] = zone_df['test_plays'] > 0

    print(f"\n  분류 완료: {len(zone_df):,}곡")
    print(f"\n  Zone 분포:")
    for z, cnt in zone_df['zone'].value_counts().items():
        print(f"    {z}: {cnt:,} ({cnt/len(zone_df):.1%})")

    print(f"\n  전체 생존율: {zone_df['survived'].mean():.1%}")

    return zone_df


def analyze(zone_df):
    """분석"""
    print("\n[2/3] 예측력 분석...")

    survival = zone_df.groupby('zone').agg(
        survival_rate=('survived', 'mean'),
        n_songs=('survived', 'count'),
        avg_test_plays=('test_plays', 'mean'),
        median_test_plays=('test_plays', 'median'),
    )

    print(f"\n  Zone별 결과:")
    print(survival.to_string())

    # 카이제곱 (생존 여부)
    contingency = pd.crosstab(zone_df['zone'], zone_df['survived'])
    if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
        chi2, p_chi2, dof, _ = stats.chi2_contingency(contingency)
        print(f"\n  카이제곱 검정 (Zone별 생존 여부):")
        print(f"    chi2 = {chi2:.2f}, df = {dof}, p = {p_chi2:.2e}")
        print(f"    결론: {'PASS' if p_chi2 < 0.05 else 'FAIL'}")

    # Kruskal-Wallis (향후 재생 횟수)
    groups = [g['test_plays'].values for _, g in zone_df.groupby('zone')]
    if len(groups) >= 2:
        kw_stat, kw_p = stats.kruskal(*groups)
        print(f"\n  Kruskal-Wallis 검정 (향후 재생 횟수):")
        print(f"    H = {kw_stat:.2f}, p = {kw_p:.2e}")

    return survival


def visualize(zone_df, survival):
    """시각화"""
    print("\n[3/3] 시각화...")

    zone_order = ['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4']
    colors = {'Zone 1': '#4ecdc4', 'Zone 2': '#45b7d1', 'Zone 3': '#ff6b6b', 'Zone 4': '#95a5a6'}

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('가설 3 v4: 시간 기준 분리 — Zone의 미래 예측력\n'
                 '(사용자 활동 기간의 앞 70%로 Zone 판별 → 뒤 30%에서 재생 유무/횟수 측정)',
                 fontsize=13, fontweight='bold')

    # 1) Zone 분포
    ax1 = axes[0, 0]
    zone_counts = zone_df['zone'].value_counts().reindex([z for z in zone_order if z in zone_df['zone'].values])
    zc = [colors.get(z, '#999') for z in zone_counts.index]
    bars = ax1.bar(zone_counts.index, zone_counts.values, color=zc, edgecolor='white')
    for bar, val in zip(bars, zone_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f'{val:,}\n({val/len(zone_df)*100:.1f}%)', ha='center', fontsize=9)
    ax1.set_title('Zone 분포 (관찰 기간 기준)', fontsize=12)
    ax1.set_ylabel('곡 수')
    ax1.grid(axis='y', alpha=0.3)

    # 2) Zone별 생존율 (핵심! 0%~100% 범위가 나와야 정상)
    ax2 = axes[0, 1]
    s = survival.reindex([z for z in zone_order if z in survival.index])
    zc2 = [colors.get(z, '#999') for z in s.index]
    bars = ax2.bar(s.index, s['survival_rate'], color=zc2, edgecolor='white')
    for bar, (idx, row) in zip(bars, s.iterrows()):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{row["survival_rate"]:.1%}\n(n={int(row["n_songs"]):,})',
                 ha='center', fontsize=9)
    ax2.set_title('Zone별 생존율\n(검증 기간 30%에 1번이라도 재생됐는가?)', fontsize=12)
    ax2.set_ylabel('생존율')
    ax2.set_ylim(0, 1.15)
    ax2.grid(axis='y', alpha=0.3)

    # 3) Zone별 향후 평균 재생 횟수
    ax3 = axes[1, 0]
    tp = zone_df.groupby('zone')['test_plays'].mean().reindex(
        [z for z in zone_order if z in zone_df['zone'].values])
    zc3 = [colors.get(z, '#999') for z in tp.index]
    bars = ax3.bar(tp.index, tp.values, color=zc3, edgecolor='white')
    for bar, val in zip(bars, tp.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.1f}회', ha='center', fontsize=9)
    ax3.set_title('Zone별 향후 평균 재생 횟수\n(검증 기간 30%)', fontsize=12)
    ax3.set_ylabel('평균 재생 횟수')
    ax3.grid(axis='y', alpha=0.3)

    # 4) Zone별 향후 재생 횟수 박스플롯
    ax4 = axes[1, 1]
    zone_data = []
    zone_labels = []
    for z in zone_order:
        data = zone_df[zone_df['zone'] == z]['test_plays']
        if len(data) > 0:
            zone_data.append(data.values)
            zone_labels.append(z)
    bp = ax4.boxplot(zone_data, labels=zone_labels, patch_artist=True, showfliers=False)
    for patch, z in zip(bp['boxes'], zone_labels):
        patch.set_facecolor(colors.get(z, '#999'))
        patch.set_alpha(0.7)
    ax4.set_title('Zone별 향후 재생 횟수 분포', fontsize=12)
    ax4.set_ylabel('재생 횟수')
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'h3_zones_v4_timesplit.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved: h3_zones_v4_timesplit.png")


if __name__ == "__main__":
    df = load_features()
    zone_df = classify_zones_time_split(df)
    survival = analyze(zone_df)
    visualize(zone_df, survival)

    print("\n" + "="*60)
    print("가설 3 v4 (시간 기준 분리) 완료!")
    print("="*60)
