"""
가설 3 v5: 고정 기간 월간 재생 추이 → 미래 재생 예측 회귀 모델

접근:
1. 사용자-곡별로 재생 이력을 월 단위 빈으로 쪼갬 (모든 빈이 동일 기간)
2. 첫 3개월의 재생 횟수 추이를 Feature로 사용
3. 4번째 달의 재생 횟수를 Target으로 예측
4. 회귀 모델 학습 → "과거 추이가 미래 재생을 예측하는가?"
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score

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


def build_monthly_bins(df, obs_months=3, pred_months=1):
    """
    사용자-곡별 고정 월간 재생 횟수 빈 생성
    
    - 곡의 첫 재생일부터 30일 단위로 동일 기간 빈을 만듦
    - obs_months개 빈 = 관찰 구간 (Feature)
    - 그 다음 pred_months개 빈 = 예측 구간 (Target)
    """
    total_months = obs_months + pred_months
    print(f"\n[1/4] 월간 빈 생성 (관찰 {obs_months}개월 + 예측 {pred_months}개월)...")
    
    # 사용자-곡별 첫 재생일
    first_plays = df.groupby(['user_id', 'song_key'])['timestamp'].min().reset_index()
    first_plays.columns = ['user_id', 'song_key', 'first_play']
    
    df = df.merge(first_plays, on=['user_id', 'song_key'])
    
    # 첫 재생 이후 경과일
    df['days_since_first'] = (df['timestamp'] - df['first_play']).dt.days
    
    # 몇 번째 월 빈인지 (0, 1, 2, 3, ...)
    df['month_bin'] = df['days_since_first'] // 30
    
    # total_months 이내만 사용
    df_window = df[df['month_bin'] < total_months]
    
    # 사용자-곡-월별 재생 횟수
    monthly = df_window.groupby(['user_id', 'song_key', 'month_bin']).size().reset_index(name='plays')
    
    # 피벗: 각 월 빈을 컬럼으로
    pivot = monthly.pivot_table(index=['user_id', 'song_key'], 
                                 columns='month_bin', values='plays', fill_value=0)
    
    # 모든 빈이 존재하는 곡만 (최소한 관찰 구간에 재생이 있어야)
    required_cols = list(range(total_months))
    for c in required_cols:
        if c not in pivot.columns:
            pivot[c] = 0
    pivot = pivot[required_cols]
    pivot.columns = [f'm{i}' for i in range(total_months)]
    
    # 관찰 구간(첫 3개월)에 최소 5회 이상 재생
    obs_cols = [f'm{i}' for i in range(obs_months)]
    pivot['obs_total'] = pivot[obs_cols].sum(axis=1)
    qualified = pivot[pivot['obs_total'] >= 5].copy()
    
    print(f"  관찰 기간 5회+ 사용자-곡: {len(qualified):,}")
    
    # Feature와 Target 분리
    pred_cols = [f'm{i}' for i in range(obs_months, total_months)]
    qualified['target'] = qualified[pred_cols].sum(axis=1)
    
    print(f"  Target(미래 재생) 평균: {qualified['target'].mean():.2f}회")
    print(f"  Target 0인 비율: {(qualified['target'] == 0).mean():.1%}")
    
    return qualified, obs_cols, pred_cols


def build_features(qualified, obs_cols):
    """관찰 구간에서 Feature 추출"""
    print(f"\n[2/4] Feature 생성...")
    
    obs_values = qualified[obs_cols].values  # shape: (n_songs, obs_months)
    
    # Feature 1: 각 월의 재생 횟수 (그대로)
    # Feature 2: 추세 기울기 (slope)
    months = np.arange(len(obs_cols))
    slopes = []
    for row in obs_values:
        if np.std(row) == 0:
            slopes.append(0)
        else:
            slope, _, _, _, _ = stats.linregress(months, row)
            slopes.append(slope)
    
    qualified = qualified.copy()
    qualified['slope'] = slopes
    qualified['obs_mean'] = obs_values.mean(axis=1)
    qualified['obs_std'] = obs_values.std(axis=1)
    qualified['obs_max'] = obs_values.max(axis=1)
    # 마지막 달 / 첫 달 비율
    first_month = obs_values[:, 0].astype(float)
    last_month = obs_values[:, -1].astype(float)
    qualified['last_first_ratio'] = np.where(first_month > 0, last_month / first_month, last_month)
    
    feature_cols = obs_cols + ['slope', 'obs_mean', 'obs_std', 'obs_max', 'last_first_ratio']
    
    print(f"  Features: {feature_cols}")
    print(f"  데이터 shape: {qualified[feature_cols].shape}")
    
    return qualified, feature_cols


def train_and_evaluate(qualified, feature_cols):
    """회귀 모델 학습 및 평가"""
    print(f"\n[3/4] 회귀 모델 학습...")
    
    X = qualified[feature_cols].values
    y = qualified['target'].values
    
    # 1) 단순 선형 회귀
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    r2_train = r2_score(y, y_pred)
    mae_train = mean_absolute_error(y, y_pred)
    
    # 5-fold CV
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    cv_mae = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    
    print(f"\n  Linear Regression 결과:")
    print(f"    Train R² = {r2_train:.4f}")
    print(f"    Train MAE = {mae_train:.2f}")
    print(f"    CV R² = {cv_scores.mean():.4f} (+-{cv_scores.std():.4f})")
    print(f"    CV MAE = {cv_mae.mean():.2f} (+-{cv_mae.std():.2f})")
    
    # Feature 중요도 (계수)
    print(f"\n  Feature 계수:")
    for name, coef in sorted(zip(feature_cols, model.coef_), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {name:20s}: {coef:+.4f}")
    print(f"    {'intercept':20s}: {model.intercept_:+.4f}")
    
    # 2) Baseline: "마지막 달 재생 = 미래 재생" (단순 예측)
    last_month_col = [c for c in feature_cols if c.startswith('m')][-1]
    y_baseline = qualified[last_month_col].values
    r2_baseline = r2_score(y, y_baseline)
    mae_baseline = mean_absolute_error(y, y_baseline)
    
    print(f"\n  Baseline (마지막 달 그대로):")
    print(f"    R² = {r2_baseline:.4f}")
    print(f"    MAE = {mae_baseline:.2f}")
    
    print(f"\n  모델 vs Baseline:")
    print(f"    R² 개선: {r2_baseline:.4f} → {cv_scores.mean():.4f}")
    print(f"    MAE 개선: {mae_baseline:.2f} → {cv_mae.mean():.2f}")
    
    return model, cv_scores, cv_mae, r2_baseline, mae_baseline


def visualize(qualified, model, feature_cols, cv_scores, r2_baseline):
    """시각화"""
    print(f"\n[4/4] 시각화...")
    
    obs_cols = [c for c in feature_cols if c.startswith('m')]
    
    X = qualified[feature_cols].values
    y = qualified['target'].values
    y_pred = model.predict(X)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('가설 3 v5: 월간 재생 추이 → 미래 재생 예측\n'
                 '(동일 기간 30일 빈, 관찰 3개월 → 예측 1개월)',
                 fontsize=13, fontweight='bold')
    
    # 1) 실제 vs 예측 산점도
    ax1 = axes[0, 0]
    sample = np.random.choice(len(y), min(5000, len(y)), replace=False)
    ax1.scatter(y[sample], y_pred[sample], alpha=0.2, s=8, color='#4ecdc4')
    max_val = max(y.max(), y_pred.max())
    ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='완벽한 예측')
    ax1.set_xlabel('실제 미래 재생 횟수')
    ax1.set_ylabel('예측 미래 재생 횟수')
    ax1.set_title(f'실제 vs 예측\nCV R² = {cv_scores.mean():.4f}', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2) 추이 패턴별 평균 (slope 기반 그룹)
    ax2 = axes[0, 1]
    qualified_copy = qualified.copy()
    qualified_copy['trend_group'] = pd.cut(
        qualified_copy['slope'], 
        bins=[-np.inf, -1, -0.2, 0.2, 1, np.inf],
        labels=['급감', '감소', '안정', '증가', '급증']
    )
    
    trend_means = qualified_copy.groupby('trend_group', observed=True).agg(
        obs_mean=('obs_mean', 'mean'),
        target_mean=('target', 'mean'),
        count=('target', 'count')
    )
    
    colors_bar = ['#ff6b6b', '#ffa07a', '#95a5a6', '#87ceeb', '#45b7d1']
    x = np.arange(len(trend_means))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, trend_means['obs_mean'], width, 
                     label='관찰 기간 월평균', color='#ffd93d', edgecolor='white')
    bars2 = ax2.bar(x + width/2, trend_means['target_mean'], width, 
                     label='미래 1개월 재생', color='#4ecdc4', edgecolor='white')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(trend_means.index)
    for i, cnt in enumerate(trend_means['count']):
        ax2.text(i, max(trend_means['obs_mean'].iloc[i], trend_means['target_mean'].iloc[i]) + 0.3,
                 f'n={cnt:,}', ha='center', fontsize=8, color='gray')
    ax2.set_title('추이 그룹별 관찰 vs 미래 재생', fontsize=12)
    ax2.set_ylabel('평균 재생 횟수')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3) Feature 중요도
    ax3 = axes[1, 0]
    importance = pd.Series(dict(zip(feature_cols, model.coef_)))
    importance_sorted = importance.reindex(importance.abs().sort_values(ascending=True).index)
    colors_imp = ['#4ecdc4' if v >= 0 else '#ff6b6b' for v in importance_sorted]
    ax3.barh(importance_sorted.index, importance_sorted.values, color=colors_imp, edgecolor='white')
    ax3.axvline(0, color='black', linewidth=0.5)
    ax3.set_title('Feature 중요도 (회귀 계수)', fontsize=12)
    ax3.set_xlabel('계수')
    ax3.grid(axis='x', alpha=0.3)
    
    # 4) 월별 재생 추이 예시 (추이 그룹별)
    ax4 = axes[1, 1]
    for label, color in [('급감', '#ff6b6b'), ('안정', '#95a5a6'), ('급증', '#45b7d1')]:
        group = qualified_copy[qualified_copy['trend_group'] == label]
        if len(group) == 0:
            continue
        means = list(group[obs_cols].mean().values) + [group['target'].mean()]
        months_label = [f'M{i+1}' for i in range(len(obs_cols))] + ['미래']
        ax4.plot(months_label, means, 'o-', label=f'{label} (n={len(group):,})', 
                 color=color, linewidth=2, markersize=6)
    
    ax4.set_title('추이 그룹별 월간 재생 패턴', fontsize=12)
    ax4.set_xlabel('기간 (30일 단위)')
    ax4.set_ylabel('평균 재생 횟수')
    ax4.axvline(len(obs_cols) - 0.5, color='gray', linestyle='--', alpha=0.5, label='관찰/예측 경계')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'h3_v5_regression.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved: h3_v5_regression.png")


def try_multiple_windows(df):
    """2~5개월 관찰 기간으로 각각 모델을 학습하여 최적 기간 탐색"""
    print("\n" + "="*60)
    print("다양한 관찰 기간 비교")
    print("="*60)
    
    results = []
    
    for obs_m in [2, 3, 4, 5]:
        print(f"\n--- 관찰 {obs_m}개월 + 예측 1개월 ---")
        qualified, obs_cols, pred_cols = build_monthly_bins(df, obs_months=obs_m, pred_months=1)
        qualified, feature_cols = build_features(qualified, obs_cols)
        
        X = qualified[feature_cols].values
        y = qualified['target'].values
        
        model = LinearRegression()
        cv_r2 = cross_val_score(model, X, y, cv=5, scoring='r2')
        cv_mae = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
        
        # Baseline
        last_col = obs_cols[-1]
        y_baseline = qualified[last_col].values
        r2_base = r2_score(y, y_baseline)
        
        results.append({
            'obs_months': obs_m,
            'n_songs': len(qualified),
            'cv_r2': cv_r2.mean(),
            'cv_mae': cv_mae.mean(),
            'baseline_r2': r2_base,
        })
        
        print(f"  n={len(qualified):,}, CV R²={cv_r2.mean():.4f}, Baseline R²={r2_base:.4f}")
    
    results_df = pd.DataFrame(results)
    print("\n\n최종 비교:")
    print(results_df.to_string(index=False))
    
    return results_df


if __name__ == "__main__":
    df = load_features()
    
    # 기본: 관찰 3개월 + 예측 1개월
    qualified, obs_cols, pred_cols = build_monthly_bins(df, obs_months=3, pred_months=1)
    qualified, feature_cols = build_features(qualified, obs_cols)
    model, cv_scores, cv_mae, r2_baseline, mae_baseline = train_and_evaluate(qualified, feature_cols)
    visualize(qualified, model, feature_cols, cv_scores, r2_baseline)
    
    # 다양한 관찰 기간 비교
    comparison = try_multiple_windows(df)
    
    print("\n" + "="*60)
    print("가설 3 v5 (월간 추이 회귀) 완료!")
    print("="*60)
