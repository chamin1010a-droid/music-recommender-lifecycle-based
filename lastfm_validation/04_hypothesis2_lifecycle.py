"""
가설 2 검증: 비대칭 생애주기 (Asymmetric Lifecycle)
2-A: 곡의 월간 재생 빈도가 단봉(Unimodal) 패턴을 보이는가?
2-B: T_rise < T_fall (성장은 빠르고 쇠퇴는 느린가)?
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
    df = pd.read_parquet(FEATURES_FILE, columns=['user_id', 'timestamp', 'artist_name', 'song_key'])
    print(f"  {len(df):,} rows loaded")
    return df


def build_monthly_timelines(df, min_total_plays=10):
    """사용자-곡별 월간 재생 타임라인 구성"""
    print("\n[1/3] 월간 재생 타임라인 구성...")
    
    # 총 재생 10회 이상인 사용자-곡 쌍만
    pair_counts = df.groupby(['user_id', 'song_key']).size().reset_index(name='total_plays')
    qualified = pair_counts[pair_counts['total_plays'] >= min_total_plays]
    print(f"  총 재생 {min_total_plays}회 이상 사용자-곡 쌍: {len(qualified):,}")
    
    # 자격 있는 쌍만 필터
    df_q = df.merge(qualified[['user_id', 'song_key']], on=['user_id', 'song_key'])
    
    # 연-월 컬럼
    df_q['year_month'] = df_q['timestamp'].dt.to_period('M')
    
    # 월간 재생 횟수
    monthly = df_q.groupby(['user_id', 'song_key', 'year_month']).size().reset_index(name='monthly_plays')
    
    print(f"  월간 타임라인 레코드: {len(monthly):,}")
    return monthly, df_q


def check_unimodal(group_df):
    """단봉 판정: 피크 전 증가 + 피크 후 감소"""
    if len(group_df) < 3:
        return None
    
    plays = group_df['monthly_plays'].values
    months = np.arange(len(plays))
    
    peak_idx = np.argmax(plays)
    
    # 피크가 처음이나 마지막이면 단봉 판단 어려움
    if peak_idx == 0 or peak_idx == len(plays) - 1:
        return {'is_unimodal': False, 'peak_idx': peak_idx, 'n_months': len(plays)}
    
    # 피크 전: 증가 추세?
    pre_peak = plays[:peak_idx + 1]
    if len(pre_peak) >= 2:
        slope_rise, _, _, _, _ = stats.linregress(np.arange(len(pre_peak)), pre_peak)
    else:
        slope_rise = 0
    
    # 피크 후: 감소 추세?
    post_peak = plays[peak_idx:]
    if len(post_peak) >= 2:
        slope_fall, _, _, _, _ = stats.linregress(np.arange(len(post_peak)), post_peak)
    else:
        slope_fall = 0
    
    is_unimodal = slope_rise > 0 and slope_fall < 0
    
    return {
        'is_unimodal': is_unimodal,
        'peak_idx': peak_idx, 
        'n_months': len(plays),
        'slope_rise': slope_rise,
        'slope_fall': slope_fall
    }


def compute_t_rise_t_fall(df_q):
    """T_rise (첫 재생→피크)와 T_fall (피크→마지막 재생) 계산"""
    print("\n[2/3] T_rise / T_fall 계산...")
    
    results = []
    
    grouped = df_q.groupby(['user_id', 'song_key'])
    total = len(grouped)
    
    for i, ((uid, skey), group) in enumerate(grouped):
        if i % 10000 == 0:
            print(f"  Processing {i:,}/{total:,}...", end='\r')
        
        # 날짜 정렬
        ts = group['timestamp'].sort_values()
        
        first_play = ts.iloc[0]
        last_play = ts.iloc[-1]
        
        # 월간 빈도 구하기
        months = ts.dt.to_period('M')
        monthly_counts = months.value_counts().sort_index()
        
        if len(monthly_counts) < 3:
            continue
        
        # 피크 월
        peak_month = monthly_counts.idxmax()
        peak_month_ts = peak_month.to_timestamp()
        
        # T_rise: 첫 재생 → 피크 월 (일)
        t_rise = (peak_month_ts - first_play).days
        
        # T_fall: 피크 월 → 마지막 재생 (일)
        t_fall = (last_play - peak_month_ts).days
        
        # 단봉 판정
        plays_arr = monthly_counts.values
        peak_idx = np.argmax(plays_arr)
        
        if peak_idx == 0 or peak_idx == len(plays_arr) - 1:
            is_unimodal = False
        else:
            pre = plays_arr[:peak_idx + 1]
            post = plays_arr[peak_idx:]
            s_rise = stats.linregress(np.arange(len(pre)), pre).slope if len(pre) >= 2 else 0
            s_fall = stats.linregress(np.arange(len(post)), post).slope if len(post) >= 2 else 0
            is_unimodal = s_rise > 0 and s_fall < 0
        
        results.append({
            'user_id': uid,
            'song_key': skey,
            'artist': group['artist_name'].iloc[0],
            't_rise': max(t_rise, 1),  # 0일 방지
            't_fall': max(t_fall, 1),
            'total_plays': len(ts),
            'n_months': len(monthly_counts),
            'is_unimodal': is_unimodal
        })
    
    print(f"\n  완료: {len(results):,} 곡 분석됨")
    return pd.DataFrame(results)


def analyze_and_visualize(lifecycle_df):
    """가설 2 분석 + 시각화"""
    print("\n[3/3] 분석 및 시각화...")
    
    # 2-A: 단봉 비율
    n_total = len(lifecycle_df)
    n_unimodal = lifecycle_df['is_unimodal'].sum()
    pct_unimodal = n_unimodal / n_total
    
    print(f"\n--- 2-A: 단봉 생애주기 ---")
    print(f"  총 곡 수: {n_total:,}")
    print(f"  단봉 비율: {n_unimodal:,} / {n_total:,} = {pct_unimodal:.1%}")
    print(f"  과반 초과? {'YES' if pct_unimodal > 0.5 else 'NO'}")
    
    # 이항 검정
    binom = stats.binomtest(n_unimodal, n_total, p=0.5, alternative='greater')
    print(f"  이항 검정 p-value: {binom.pvalue:.2e}")
    
    # 2-B: T_rise vs T_fall (단봉 곡만)
    unimodal = lifecycle_df[lifecycle_df['is_unimodal']].copy()
    
    print(f"\n--- 2-B: 비대칭 검정 (단봉 곡 {len(unimodal):,}개) ---")
    
    if len(unimodal) > 0:
        median_rise = unimodal['t_rise'].median()
        median_fall = unimodal['t_fall'].median()
        mean_rise = unimodal['t_rise'].mean()
        mean_fall = unimodal['t_fall'].mean()
        
        print(f"  T_rise 중앙값: {median_rise:.0f}일")
        print(f"  T_fall 중앙값: {median_fall:.0f}일")
        print(f"  T_rise 평균: {mean_rise:.0f}일")
        print(f"  T_fall 평균: {mean_fall:.0f}일")
        
        ratio = unimodal['t_fall'] / unimodal['t_rise']
        ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
        median_ratio = ratio.median()
        print(f"  T_fall/T_rise 중앙값: {median_ratio:.2f}")
        
        # Wilcoxon 부호순위 검정 (T_fall > T_rise?)
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
            unimodal['t_fall'].values, 
            unimodal['t_rise'].values, 
            alternative='greater'  # T_fall > T_rise
        )
        print(f"\n  Wilcoxon 검정 (T_fall > T_rise):")
        print(f"    statistic = {wilcoxon_stat:,.0f}")
        print(f"    p-value = {wilcoxon_p:.2e}")
        print(f"    결론: {'PASS - 쇠퇴가 성장보다 느림 (T_fall > T_rise)' if wilcoxon_p < 0.05 else 'FAIL'}")
    
    # --- 시각화 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1) 단봉 vs 비단봉 파이
    ax1 = axes[0]
    sizes = [n_unimodal, n_total - n_unimodal]
    labels_pie = [f'단봉 (Unimodal)\n{n_unimodal:,} ({pct_unimodal:.1%})', 
                  f'비단봉\n{n_total - n_unimodal:,} ({1-pct_unimodal:.1%})']
    colors_pie = ['#4ecdc4', '#ff6b6b']
    ax1.pie(sizes, labels=labels_pie, colors=colors_pie, autopct='', startangle=90)
    ax1.set_title('2-A: 단봉 생애주기 비율')
    
    # 2) T_rise vs T_fall 산점도
    ax2 = axes[1]
    if len(unimodal) > 0:
        sample = unimodal.sample(min(2000, len(unimodal)), random_state=42)
        ax2.scatter(sample['t_rise'], sample['t_fall'], alpha=0.3, s=10, color='#4ecdc4')
        max_val = max(sample['t_rise'].max(), sample['t_fall'].max())
        ax2.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='T_rise = T_fall')
        ax2.set_xlabel('T_rise (성장 기간, 일)')
        ax2.set_ylabel('T_fall (쇠퇴 기간, 일)')
        ax2.set_title(f'2-B: 성장 vs 쇠퇴 기간\n(대각선 위 = 쇠퇴가 더 느림)')
        ax2.legend()
        ax2.grid(alpha=0.3)
    
    # 3) T_fall/T_rise 비율 히스토그램
    ax3 = axes[2]
    if len(unimodal) > 0:
        ratio_capped = ratio.clip(upper=10)
        ax3.hist(ratio_capped, bins=50, color='#ffd93d', alpha=0.7, edgecolor='white')
        ax3.axvline(1, color='red', linestyle='--', linewidth=2, label='대칭 (비율=1)')
        ax3.axvline(median_ratio, color='blue', linestyle='--', linewidth=2, 
                    label=f'중앙값={median_ratio:.2f}')
        ax3.set_xlabel('T_fall / T_rise 비율')
        ax3.set_ylabel('곡 수')
        ax3.set_title(f'2-B: 비대칭 비율 분포\n(>1이면 쇠퇴가 더 느림)')
        ax3.legend()
        ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'h2_lifecycle.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Chart saved: h2_lifecycle.png")
    
    return lifecycle_df


if __name__ == "__main__":
    df = load_features()
    monthly, df_q = build_monthly_timelines(df)
    lifecycle_df = compute_t_rise_t_fall(df_q)
    analyze_and_visualize(lifecycle_df)
    
    print("\n" + "="*60)
    print("가설 2 검증 완료!")
    print("="*60)
