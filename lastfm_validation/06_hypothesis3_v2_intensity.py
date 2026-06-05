"""
가설 3 재검증: 재생 빈도(Intensity) 기반 Zone 분류
스킵 데이터 없이, 전반전 vs 후반전의 '재생 밀도(plays/day)' 변화로 Zone 분류

핵심 아이디어:
- 곡의 재생 이력을 시간순으로 전반/후반으로 나눔
- 전반전 재생 밀도 vs 후반전 재생 밀도 비교
- 밀도가 높아졌으면 → "갈수록 좋아짐" (Zone 2)
- 밀도가 낮아졌으면 → "갈수록 질림" (Zone 3)
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


def classify_zones_by_intensity(df, min_plays=15):
    """
    재생 밀도(Intensity) 기반 Zone 분류
    
    - 재생 이력을 시간순으로 정렬
    - 전반전 기간(일)과 재생 수, 후반전 기간(일)과 재생 수로 밀도 계산
    - intensity = plays / days (일당 재생 횟수)
    """
    print("\n[1/3] 재생 빈도 기반 Zone 분류...")
    
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
        half = n // 2
        
        first_half = grp.iloc[:half]
        second_half = grp.iloc[half:]
        
        # 전반전 기간 (일) - 최소 1일
        fh_days = max((first_half['timestamp'].max() - first_half['timestamp'].min()).days, 1)
        sh_days = max((second_half['timestamp'].max() - second_half['timestamp'].min()).days, 1)
        
        # 재생 밀도 (plays per day)
        fh_intensity = len(first_half) / fh_days
        sh_intensity = len(second_half) / sh_days
        
        # 변화율: (후반 - 전반) / 전반
        if fh_intensity > 0:
            change_ratio = (sh_intensity - fh_intensity) / fh_intensity
        else:
            change_ratio = 0
        
        # Zone 분류 (우선순위: Z2 → Z3 → Z1 → Z4)
        # 임계값: 30% 이상 변화를 유의미한 변화로 간주
        THRESHOLD = 0.3
        
        if change_ratio > THRESHOLD:
            zone = 'Zone 2'  # 갈수록 좋아짐 (밀도 증가)
        elif change_ratio < -THRESHOLD:
            zone = 'Zone 3'  # 갈수록 질림 (밀도 감소)
        elif fh_intensity > 0.1 and sh_intensity > 0.1:
            zone = 'Zone 1'  # 꾸준히 좋아함
        else:
            zone = 'Zone 4'  # 그저그럼
        
        zone_results.append({
            'user_id': uid,
            'song_key': skey,
            'total_plays': n,
            'fh_intensity': fh_intensity,
            'sh_intensity': sh_intensity,
            'change_ratio': change_ratio,
            'zone': zone,
            'last_play': grp['timestamp'].max(),
            'first_play': grp['timestamp'].min(),
            'lifespan_days': (grp['timestamp'].max() - grp['timestamp'].min()).days
        })
    
    zone_df = pd.DataFrame(zone_results)
    
    print(f"\n  Zone 분포:")
    zone_counts = zone_df['zone'].value_counts()
    for z, cnt in zone_counts.items():
        print(f"    {z}: {cnt:,} ({cnt/len(zone_df):.1%})")
    
    return zone_df


def compute_survival(zone_df, df):
    """Zone별 생존율/은퇴율 분석"""
    print("\n[2/3] 생존율 계산...")
    
    # 각 사용자의 마지막 활동 시점
    user_last_activity = df.groupby('user_id')['timestamp'].max().reset_index()
    user_last_activity.columns = ['user_id', 'user_last_ts']
    
    zone_df = zone_df.merge(user_last_activity, on='user_id')
    
    # 은퇴 판정: 마지막 재생 후 60일+ 재생 없음 + 사용자는 활동 중
    zone_df['days_since_last'] = (zone_df['user_last_ts'] - zone_df['last_play']).dt.days
    zone_df['is_retired'] = (zone_df['days_since_last'] >= 60).astype(int)
    
    # 판단 가능한 곡만 (사용자 활동이 30일 이상 남아있어야)
    evaluable = zone_df[zone_df['days_since_last'] >= 30].copy()
    print(f"  평가 가능 곡: {len(evaluable):,} / {len(zone_df):,}")
    
    # Zone별 은퇴율
    survival = evaluable.groupby('zone').agg(
        retirement_rate=('is_retired', 'mean'),
        n_songs=('is_retired', 'count'),
        avg_lifespan=('lifespan_days', 'mean'),
        median_lifespan=('lifespan_days', 'median')
    )
    survival['survival_rate'] = 1 - survival['retirement_rate']
    
    print(f"\n  Zone별 결과:")
    print(survival.to_string())
    
    # 카이제곱 검정
    contingency = pd.crosstab(evaluable['zone'], evaluable['is_retired'])
    if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
        chi2, p_chi2, dof, _ = stats.chi2_contingency(contingency)
        print(f"\n  카이제곱 검정:")
        print(f"    chi2 = {chi2:.2f}, df = {dof}, p = {p_chi2:.2e}")
        print(f"    결론: {'PASS' if p_chi2 < 0.05 else 'FAIL'}")
    
    # Kruskal-Wallis (생존 기간)
    groups = [g['days_since_last'].values for _, g in evaluable.groupby('zone')]
    if len(groups) >= 2:
        kw_stat, kw_p = stats.kruskal(*groups)
        print(f"\n  Kruskal-Wallis 검정 (생존 기간):")
        print(f"    H = {kw_stat:.2f}, p = {kw_p:.2e}")
    
    # 추가: Zone별 향후 30일 재생 밀도 비교
    # 이건 예측력의 더 직접적인 지표
    evaluable['future_activity'] = evaluable['days_since_last'].apply(
        lambda x: 'active' if x < 60 else 'retired'
    )
    
    return evaluable, survival


def visualize_zones(evaluable, survival):
    """시각화"""
    print("\n[3/3] 시각화...")
    
    zone_order = ['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4']
    colors = {'Zone 1': '#4ecdc4', 'Zone 2': '#45b7d1', 'Zone 3': '#ff6b6b', 'Zone 4': '#95a5a6'}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # 1) Zone 분포
    ax1 = axes[0, 0]
    zone_counts = evaluable['zone'].value_counts().reindex([z for z in zone_order if z in evaluable['zone'].values])
    zone_colors = [colors.get(z, '#999') for z in zone_counts.index]
    bars = ax1.bar(zone_counts.index, zone_counts.values, color=zone_colors, edgecolor='white')
    for bar, val in zip(bars, zone_counts.values):
        pct = val / len(evaluable) * 100
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
                 f'{val:,}\n({pct:.1f}%)', ha='center', fontsize=9)
    ax1.set_title('Zone 분포 (재생 밀도 기반)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('곡 수')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2) Zone별 은퇴율
    ax2 = axes[0, 1]
    survival_sorted = survival.reindex([z for z in zone_order if z in survival.index])
    bar_colors = [colors.get(z, '#999') for z in survival_sorted.index]
    bars = ax2.bar(survival_sorted.index, survival_sorted['retirement_rate'], 
                   color=bar_colors, edgecolor='white')
    for bar, (idx, row) in zip(bars, survival_sorted.iterrows()):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{row["retirement_rate"]:.1%}\n(n={int(row["n_songs"]):,})', 
                 ha='center', fontsize=9)
    ax2.set_title('Zone별 은퇴율\n(높을수록 곡이 사라짐)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('은퇴율')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3) Zone별 재생 밀도 변화 (전반 vs 후반)
    ax3 = axes[1, 0]
    zone_intensity = evaluable.groupby('zone')[['fh_intensity', 'sh_intensity']].mean()
    zone_intensity_sorted = zone_intensity.reindex([z for z in zone_order if z in zone_intensity.index])
    x = np.arange(len(zone_intensity_sorted))
    width = 0.35
    ax3.bar(x - width/2, zone_intensity_sorted['fh_intensity'], width, 
            label='전반전 밀도', color='#ffd93d', edgecolor='white')
    ax3.bar(x + width/2, zone_intensity_sorted['sh_intensity'], width, 
            label='후반전 밀도', color='#ff6b6b', edgecolor='white')
    ax3.set_xticks(x)
    ax3.set_xticklabels(zone_intensity_sorted.index)
    ax3.set_title('Zone별 전/후반 재생 밀도 비교\n(plays/day)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('일당 재생 횟수')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4) Zone별 곡 수명 분포 (박스플롯)
    ax4 = axes[1, 1]
    zone_data = []
    zone_labels = []
    for z in zone_order:
        data = evaluable[evaluable['zone'] == z]['lifespan_days']
        if len(data) > 0:
            zone_data.append(data.values)
            zone_labels.append(z)
    
    bp = ax4.boxplot(zone_data, labels=zone_labels, patch_artist=True, showfliers=False)
    for patch, z in zip(bp['boxes'], zone_labels):
        patch.set_facecolor(colors.get(z, '#999'))
        patch.set_alpha(0.7)
    ax4.set_title('Zone별 곡 수명 분포 (일)', fontsize=13, fontweight='bold')
    ax4.set_ylabel('수명 (일)')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'h3_zones_v2_intensity.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved: h3_zones_v2_intensity.png")


if __name__ == "__main__":
    df = load_features()
    zone_df = classify_zones_by_intensity(df)
    evaluable, survival = compute_survival(zone_df, df)
    visualize_zones(evaluable, survival)
    
    print("\n" + "="*60)
    print("가설 3 재검증 (재생 밀도 기반) 완료!")
    print("="*60)
