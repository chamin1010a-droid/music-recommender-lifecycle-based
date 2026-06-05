"""
가설 3 v3: 재생 밀도 기반 Zone 분류 — 시간 분리(Train/Test Split)

핵심 수정:
- 이전: 곡의 전체 이력으로 Zone 분류 + 같은 데이터로 은퇴 판정 (순환 논증)
- 수정: 곡의 이력 앞쪽 70%만으로 Zone 분류 → 나머지 30% 기간에서 생존 여부 검증

이렇게 해야 "Zone 분류가 미래 행동을 예측할 수 있는가?"를 진짜로 검증
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


def classify_zones_temporal_split(df, min_plays=15, train_ratio=0.7):
    """
    시간 분리된 Zone 분류
    
    1) 곡의 재생 이력에서 앞쪽 70%를 "관찰 구간"으로 씀
    2) 관찰 구간을 다시 전반/후반으로 나눠 밀도 변화(Zone) 판별
    3) 나머지 30%는 "검증 구간"으로, 실제로 곡이 살아남았는지 측정
    """
    print(f"\n[1/3] 시간 분리 Zone 분류 (관찰 {train_ratio:.0%} / 검증 {1-train_ratio:.0%})...")
    
    # 재생 15회 이상인 사용자-곡 쌍
    pair_counts = df.groupby(['user_id', 'song_key']).size().reset_index(name='total_plays')
    qualified = pair_counts[pair_counts['total_plays'] >= min_plays]
    print(f"  재생 {min_plays}회 이상 사용자-곡 쌍: {len(qualified):,}")
    
    df_q = df.merge(qualified[['user_id', 'song_key']], on=['user_id', 'song_key'])
    
    zone_results = []
    grouped = df_q.groupby(['user_id', 'song_key'])
    total = len(grouped)
    
    for i, ((uid, skey), grp) in enumerate(grouped):
        if i % 10000 == 0:
            print(f"  Processing {i:,}/{total:,}...", end='\r')
        
        grp = grp.sort_values('timestamp')
        n = len(grp)
        
        # ===== 시간 분리: 앞쪽 70%만 관찰 =====
        train_n = int(n * train_ratio)
        if train_n < 8:  # 관찰 구간이 너무 작으면 스킵
            continue
        
        train_plays = grp.iloc[:train_n]   # Zone 분류용
        test_plays = grp.iloc[train_n:]    # 검증용
        
        # ===== 관찰 구간을 전반/후반으로 나눔 =====
        train_half = train_n // 2
        first_half = train_plays.iloc[:train_half]
        second_half = train_plays.iloc[train_half:]
        
        # 전반전/후반전 기간 (일) — 최소 1일
        fh_days = max((first_half['timestamp'].max() - first_half['timestamp'].min()).days, 1)
        sh_days = max((second_half['timestamp'].max() - second_half['timestamp'].min()).days, 1)
        
        # 재생 밀도 (plays/day)
        fh_intensity = len(first_half) / fh_days
        sh_intensity = len(second_half) / sh_days
        
        # 변화율
        if fh_intensity > 0:
            change_ratio = (sh_intensity - fh_intensity) / fh_intensity
        else:
            change_ratio = 0
        
        # Zone 분류 (우선순위: Z2 → Z3 → Z1 → Z4)
        THRESHOLD = 0.3
        if change_ratio > THRESHOLD:
            zone = 'Zone 2'
        elif change_ratio < -THRESHOLD:
            zone = 'Zone 3'
        elif fh_intensity > 0.1 and sh_intensity > 0.1:
            zone = 'Zone 1'
        else:
            zone = 'Zone 4'
        
        # ===== 검증 구간에서 생존 측정 =====
        # 관찰 구간 마지막 재생 시점
        train_end = train_plays['timestamp'].max()
        
        # 검증 구간에 재생이 있는가?
        test_count = len(test_plays)
        
        # 검증 구간의 기간
        if test_count > 0:
            test_first = test_plays['timestamp'].min()
            test_last = test_plays['timestamp'].max()
            test_span_days = max((test_last - train_end).days, 1)
            test_intensity = test_count / test_span_days
            survived = True
        else:
            test_span_days = 0
            test_intensity = 0
            survived = False
        
        zone_results.append({
            'user_id': uid,
            'song_key': skey,
            'total_plays': n,
            'train_plays': train_n,
            'test_plays': test_count,
            'fh_intensity': fh_intensity,
            'sh_intensity': sh_intensity,
            'change_ratio': change_ratio,
            'zone': zone,
            'test_intensity': test_intensity,
            'survived': survived,
        })
    
    zone_df = pd.DataFrame(zone_results)
    
    print(f"\n  분류 완료: {len(zone_df):,}곡")
    print(f"\n  Zone 분포:")
    zone_counts = zone_df['zone'].value_counts()
    for z, cnt in zone_counts.items():
        print(f"    {z}: {cnt:,} ({cnt/len(zone_df):.1%})")
    
    return zone_df


def analyze_predictive_power(zone_df):
    """Zone 분류의 예측력 분석"""
    print("\n[2/3] 예측력 분석...")
    
    # Zone별 생존율
    survival = zone_df.groupby('zone').agg(
        survival_rate=('survived', 'mean'),
        n_songs=('survived', 'count'),
        avg_test_intensity=('test_intensity', 'mean'),
        median_test_plays=('test_plays', 'median'),
        avg_test_plays=('test_plays', 'mean')
    )
    survival['retirement_rate'] = 1 - survival['survival_rate']
    
    print(f"\n  Zone별 결과 (검증 구간 30% 기준):")
    print(survival.to_string())
    
    # 카이제곱 검정
    contingency = pd.crosstab(zone_df['zone'], zone_df['survived'])
    if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
        chi2, p_chi2, dof, _ = stats.chi2_contingency(contingency)
        print(f"\n  카이제곱 검정 (Zone별 생존율 차이):")
        print(f"    chi2 = {chi2:.2f}, df = {dof}, p = {p_chi2:.2e}")
        print(f"    결론: {'PASS' if p_chi2 < 0.05 else 'FAIL'}")
    
    # Kruskal-Wallis (향후 재생 횟수)
    groups = [g['test_plays'].values for _, g in zone_df.groupby('zone')]
    if len(groups) >= 2:
        kw_stat, kw_p = stats.kruskal(*groups)
        print(f"\n  Kruskal-Wallis 검정 (향후 재생 횟수):")
        print(f"    H = {kw_stat:.2f}, p = {kw_p:.2e}")
    
    # Zone별 향후 재생 밀도 비교
    groups_intensity = [g['test_intensity'].values for _, g in zone_df.groupby('zone')]
    if len(groups_intensity) >= 2:
        kw_stat2, kw_p2 = stats.kruskal(*groups_intensity)
        print(f"\n  Kruskal-Wallis 검정 (향후 재생 밀도):")
        print(f"    H = {kw_stat2:.2f}, p = {kw_p2:.2e}")
    
    return survival


def visualize(zone_df, survival):
    """시각화"""
    print("\n[3/3] 시각화...")
    
    zone_order = ['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4']
    colors = {'Zone 1': '#4ecdc4', 'Zone 2': '#45b7d1', 'Zone 3': '#ff6b6b', 'Zone 4': '#95a5a6'}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('가설 3 v3: Zone 분류의 미래 예측력\n(관찰 70% → Zone 판별, 검증 30% → 생존 측정)', 
                 fontsize=14, fontweight='bold')
    
    # 1) Zone 분포
    ax1 = axes[0, 0]
    zone_counts = zone_df['zone'].value_counts().reindex([z for z in zone_order if z in zone_df['zone'].values])
    zone_colors = [colors.get(z, '#999') for z in zone_counts.index]
    bars = ax1.bar(zone_counts.index, zone_counts.values, color=zone_colors, edgecolor='white')
    for bar, val in zip(bars, zone_counts.values):
        pct = val / len(zone_df) * 100
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
                 f'{val:,}\n({pct:.1f}%)', ha='center', fontsize=9)
    ax1.set_title('Zone 분포', fontsize=12)
    ax1.set_ylabel('곡 수')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2) Zone별 생존율 (핵심!)
    ax2 = axes[0, 1]
    survival_sorted = survival.reindex([z for z in zone_order if z in survival.index])
    bar_colors = [colors.get(z, '#999') for z in survival_sorted.index]
    bars = ax2.bar(survival_sorted.index, survival_sorted['survival_rate'], 
                   color=bar_colors, edgecolor='white')
    for bar, (idx, row) in zip(bars, survival_sorted.iterrows()):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{row["survival_rate"]:.1%}\n(n={int(row["n_songs"]):,})', 
                 ha='center', fontsize=9)
    ax2.set_title('Zone별 생존율 (검증 30% 기간)\n(높을수록 곡이 미래에도 재생됨)', fontsize=12)
    ax2.set_ylabel('생존율')
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3) Zone별 향후 평균 재생 횟수
    ax3 = axes[1, 0]
    test_plays = zone_df.groupby('zone')['test_plays'].mean().reindex(
        [z for z in zone_order if z in zone_df['zone'].values])
    bar_colors3 = [colors.get(z, '#999') for z in test_plays.index]
    bars = ax3.bar(test_plays.index, test_plays.values, color=bar_colors3, edgecolor='white')
    for bar, val in zip(bars, test_plays.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                 f'{val:.1f}회', ha='center', fontsize=9)
    ax3.set_title('Zone별 향후 평균 재생 횟수\n(검증 구간 30%)', fontsize=12)
    ax3.set_ylabel('평균 재생 횟수')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4) Zone별 전반/후반 밀도 비교 (관찰 구간)
    ax4 = axes[1, 1]
    zone_intensity = zone_df.groupby('zone')[['fh_intensity', 'sh_intensity', 'test_intensity']].mean()
    zone_intensity_sorted = zone_intensity.reindex([z for z in zone_order if z in zone_intensity.index])
    x = np.arange(len(zone_intensity_sorted))
    width = 0.25
    ax4.bar(x - width, zone_intensity_sorted['fh_intensity'], width, 
            label='관찰 전반', color='#ffd93d', edgecolor='white')
    ax4.bar(x, zone_intensity_sorted['sh_intensity'], width, 
            label='관찰 후반', color='#ff6b6b', edgecolor='white')
    ax4.bar(x + width, zone_intensity_sorted['test_intensity'], width, 
            label='검증 구간 (미래)', color='#4ecdc4', edgecolor='white')
    ax4.set_xticks(x)
    ax4.set_xticklabels(zone_intensity_sorted.index)
    ax4.set_title('Zone별 재생 밀도 추이\n(관찰→미래 전이 패턴)', fontsize=12)
    ax4.set_ylabel('일당 재생 횟수')
    ax4.legend(fontsize=9)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'h3_zones_v3_temporal.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved: h3_zones_v3_temporal.png")


if __name__ == "__main__":
    df = load_features()
    zone_df = classify_zones_temporal_split(df)
    survival = analyze_predictive_power(zone_df)
    visualize(zone_df, survival)
    
    print("\n" + "="*60)
    print("가설 3 v3 (시간 분리 검증) 완료!")
    print("="*60)
