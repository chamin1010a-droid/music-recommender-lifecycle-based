"""
가설 3 검증: 스킵 기반 Zone 분류의 예측력
"전/후반 스킵률 차이로 곡의 은퇴를 예측할 수 있다"

Last.fm 스킵 프록시는 1.7%로 매우 낮음 (scrobble 특성).
→ 대안 전략: 스킵 대신 '완청률(full listen proxy)'의 전/후반 비교로 Zone 분류

완청 프록시: 다음 곡까지 간격이 3분(180초) 이상이면 끝까지 들었을 가능성 높음
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
    df = pd.read_parquet(FEATURES_FILE)
    print(f"  {len(df):,} rows loaded")
    return df


def classify_zones(df, min_plays=15):
    """
    Zone 분류 (Last.fm 적응 버전)
    
    Last.fm은 scrobble 방식이라 스킵된 곡이 거의 기록되지 않음 (스킵률 1.7%).
    → 대안: '짧은 간격(<60초) 재생'을 스킵 프록시로 사용 (기존 방식 유지)
    
    추가 대안: '세션 이탈(abandon)' 개념 도입
    곡 A 재생 후 30분 이상 재생 없음 → 세션 종료 (이탈 or 마지막 곡)
    이것을 '부정적 신호'로 쓰면 프록시를 보완
    
    그러나 우선은 표준 스킵 프록시로 진행 (is_skip_proxy 컬럼)
    """
    print("\n[1/3] Zone 분류...")
    
    # 재생 15회 이상인 사용자-곡 쌍만
    pair_counts = df.groupby(['user_id', 'song_key']).size().reset_index(name='total_plays')
    qualified = pair_counts[pair_counts['total_plays'] >= min_plays]
    print(f"  재생 {min_plays}회 이상 사용자-곡 쌍: {len(qualified):,}")
    
    df_q = df.merge(qualified[['user_id', 'song_key']], on=['user_id', 'song_key'])
    
    # 각 사용자-곡 쌍에 대해 전반/후반 스킵률 계산
    zone_results = []
    
    grouped = df_q.groupby(['user_id', 'song_key'])
    total = len(grouped)
    
    for i, ((uid, skey), grp) in enumerate(grouped):
        if i % 10000 == 0:
            print(f"  Processing {i:,}/{total:,}...", end='\r')
        
        grp = grp.sort_values('timestamp')
        n = len(grp)
        half = n // 2
        
        # 전반전 / 후반전 스킵률
        first_half = grp.iloc[:half]
        second_half = grp.iloc[half:]
        
        fh_skip = first_half['is_skip_proxy'].mean()
        sh_skip = second_half['is_skip_proxy'].mean()
        
        # Zone 분류 (우선순위: Z2 → Z3 → Z1 → Z4)
        delta = fh_skip - sh_skip
        
        if delta > 0.15:
            zone = 'Zone 2'
        elif delta < -0.15:
            zone = 'Zone 3'
        elif fh_skip <= 0.3 and sh_skip <= 0.3:
            zone = 'Zone 1'
        else:
            zone = 'Zone 4'
        
        # 은퇴 여부: 마지막 재생 이후 데이터 끝까지의 기간
        last_play = grp['timestamp'].max()
        
        # 향후 재생 (해당 사용자의 전체 데이터에서 이 곡의 마지막 재생 이후)
        zone_results.append({
            'user_id': uid,
            'song_key': skey,
            'total_plays': n,
            'first_half_skip': fh_skip,
            'second_half_skip': sh_skip,
            'delta': delta,
            'zone': zone,
            'last_play': last_play
        })
    
    zone_df = pd.DataFrame(zone_results)
    
    print(f"\n  Zone 분포:")
    print(zone_df['zone'].value_counts().to_string())
    
    return zone_df, df_q


def compute_survival(zone_df, df):
    """Zone별 생존율 계산: Zone 분류 시점 이후 재생이 계속되는가?"""
    print("\n[2/3] 생존율 계산...")
    
    # 각 사용자의 마지막 활동 시점
    user_last_activity = df.groupby('user_id')['timestamp'].max().reset_index()
    user_last_activity.columns = ['user_id', 'user_last_ts']
    
    zone_df = zone_df.merge(user_last_activity, on='user_id')
    
    # 은퇴 판정: 마지막 재생 후 60일 이상 재생 없으면서, 
    # 사용자는 여전히 활동 중 (사용자 마지막 활동이 곡의 마지막 재생보다 60일+ 뒤)
    zone_df['days_since_last'] = (zone_df['user_last_ts'] - zone_df['last_play']).dt.days
    zone_df['is_retired'] = (zone_df['days_since_last'] >= 60).astype(int)
    
    # 사용자 활동이 곡 마지막 재생과 너무 가까우면 판단 불가 → 제외
    evaluable = zone_df[zone_df['days_since_last'] >= 30].copy()
    print(f"  평가 가능 곡: {len(evaluable):,} / {len(zone_df):,}")
    
    # Zone별 은퇴율
    survival = evaluable.groupby('zone')['is_retired'].agg(['mean', 'count'])
    survival.columns = ['retirement_rate', 'n_songs']
    survival['survival_rate'] = 1 - survival['retirement_rate']
    
    print(f"\n  Zone별 은퇴율/생존율:")
    print(survival.to_string())
    
    # 카이제곱 검정
    contingency = pd.crosstab(evaluable['zone'], evaluable['is_retired'])
    if len(contingency) >= 2 and contingency.shape[1] >= 2:
        chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)
        print(f"\n  카이제곱 검정:")
        print(f"    chi2 = {chi2:.2f}, df = {dof}, p = {p_chi2:.2e}")
        print(f"    결론: {'PASS - Zone별 은퇴율에 유의미한 차이' if p_chi2 < 0.05 else 'FAIL'}")
    
    # Kruskal-Wallis 검정 (향후 재생 기간)
    groups = [group['days_since_last'].values for name, group in evaluable.groupby('zone')]
    if len(groups) >= 2:
        kw_stat, kw_p = stats.kruskal(*groups)
        print(f"\n  Kruskal-Wallis 검정:")
        print(f"    H = {kw_stat:.2f}, p = {kw_p:.2e}")
    
    return evaluable, survival


def visualize_zones(evaluable, survival):
    """Zone 분류 결과 시각화"""
    print("\n[3/3] 시각화...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1) Zone 분포
    ax1 = axes[0]
    zone_counts = evaluable['zone'].value_counts().sort_index()
    colors = {'Zone 1': '#4ecdc4', 'Zone 2': '#45b7d1', 'Zone 3': '#ff6b6b', 'Zone 4': '#95a5a6'}
    zone_colors = [colors.get(z, '#999') for z in zone_counts.index]
    bars = ax1.bar(zone_counts.index, zone_counts.values, color=zone_colors, edgecolor='white')
    for bar, val in zip(bars, zone_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                 f'{val:,}', ha='center', fontsize=10)
    ax1.set_title('Zone 분포')
    ax1.set_ylabel('곡 수')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2) Zone별 은퇴율
    ax2 = axes[1]
    zone_order = ['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4']
    survival_sorted = survival.reindex([z for z in zone_order if z in survival.index])
    bar_colors = [colors.get(z, '#999') for z in survival_sorted.index]
    bars = ax2.bar(survival_sorted.index, survival_sorted['retirement_rate'], 
                   color=bar_colors, edgecolor='white')
    for bar, (idx, row) in zip(bars, survival_sorted.iterrows()):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{row["retirement_rate"]:.1%}\n(n={int(row["n_songs"]):,})', 
                 ha='center', fontsize=9)
    ax2.set_title('Zone별 은퇴율\n(높을수록 곡이 사라짐)')
    ax2.set_ylabel('은퇴율')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3) Zone별 전반/후반 스킵률 비교
    ax3 = axes[2]
    zone_skip_stats = evaluable.groupby('zone')[['first_half_skip', 'second_half_skip']].mean()
    zone_skip_sorted = zone_skip_stats.reindex([z for z in zone_order if z in zone_skip_stats.index])
    x = np.arange(len(zone_skip_sorted))
    width = 0.35
    ax3.bar(x - width/2, zone_skip_sorted['first_half_skip'], width, label='전반전', color='#ffd93d', edgecolor='white')
    ax3.bar(x + width/2, zone_skip_sorted['second_half_skip'], width, label='후반전', color='#ff6b6b', edgecolor='white')
    ax3.set_xticks(x)
    ax3.set_xticklabels(zone_skip_sorted.index)
    ax3.set_title('Zone별 전/후반 스킵 프록시 비교')
    ax3.set_ylabel('추정 스킵률')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'h3_zones.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved: h3_zones.png")


if __name__ == "__main__":
    df = load_features()
    zone_df, df_q = classify_zones(df)
    evaluable, survival = compute_survival(zone_df, df)
    visualize_zones(evaluable, survival)
    
    print("\n" + "="*60)
    print("가설 3 검증 완료!")
    print("="*60)
