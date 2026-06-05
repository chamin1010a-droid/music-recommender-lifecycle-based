import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

folder_path = r"c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록"
input_csv = os.path.join(folder_path, "ytm_history_features.csv")

# Set Korean Font
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("Loading data for analysis...")
df = pd.read_csv(input_csv)

# 1. Familiarity vs Satisfaction Score
# To visualize the trend, we group familiarity into bins or plot a smoothed curve.
# Since familiarity is highly right-skewed, we create bins
bins = [-1, 0, 1, 2, 3, 5, 10, 20, 50, 100, float('inf')]
labels = ['0 (처음)', '1', '2', '3', '4-5', '6-10', '11-20', '21-50', '51-100', '101+']

df['familiarity_bin'] = pd.cut(df['familiarity'], bins=bins, labels=labels)

# Metrics to analyze:
# - satisfaction_score (0~2)
# - skip_defense_rate = 1 - is_skipped (0 or 1)
# - relisten_rate = relisten_within_7d (0 or 1)

grouped = df.groupby('familiarity_bin', observed=True).agg(
    satisfaction_mean=('satisfaction_score', 'mean'),
    skip_defense_rate=('is_skipped', lambda x: 1 - x.mean()),
    relisten_rate=('relisten_within_7d', 'mean'),
    count=('satisfaction_score', 'count')
).reset_index()

print("Grouped Results:")
print(grouped)

# Plot 1: Satisfaction Score across Familiarity
plt.figure(figsize=(12, 6))
sns.barplot(data=grouped, x='familiarity_bin', y='satisfaction_mean', color='skyblue')
plt.title('익숙함(누적 청취 횟수)에 따른 종합 만족도(추정치)', fontsize=16)
plt.xlabel('익숙함 수준 (누적 청취 횟수 구간)', fontsize=12)
plt.ylabel('평균 만족도 (0~2)', fontsize=12)

# Add trend line (polynomial fit for the mean points to show inverted U)
x = np.arange(len(grouped))
y = grouped['satisfaction_mean'].values

# Fit a degree 2 polynomial
if len(y[~np.isnan(y)]) > 2:
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--", label=f"추세선 (Poly-2)", linewidth=2.5)
    plt.legend()

save_path1 = os.path.join(folder_path, "familiarity_vs_satisfaction.png")
plt.tight_layout()
plt.savefig(save_path1)
plt.close()

# Plot 2: Detailed Components (Skip Defense & Relisten)
fig, ax1 = plt.subplots(figsize=(12, 6))

ax2 = ax1.twinx()
sns.lineplot(ax=ax1, data=grouped, x='familiarity_bin', y='skip_defense_rate', marker='o', color='b', label='스킵 안 할 확률 (만족)')
sns.lineplot(ax=ax2, data=grouped, x='familiarity_bin', y='relisten_rate', marker='s', color='g', label='7일 내 재청취율')

ax1.set_xlabel('익숙함 수준 (누적 청취 횟수 구간)', fontsize=12)
ax1.set_ylabel('스킵 안 할 확률', color='b', fontsize=12)
ax2.set_ylabel('재청취율', color='g', fontsize=12)
ax1.set_title('익숙함에 따른 개별 만족 지표 변화', fontsize=16)

# Combine legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

save_path2 = os.path.join(folder_path, "familiarity_components.png")
plt.tight_layout()
plt.savefig(save_path2)
plt.close()

# Let's also create the Walkthrough artifact resources if needed.
# Since we can embed these in walkthrough.md, we will copy them to the appDataDir later.

print(f"Artifacts saved to:\n  - {save_path1}\n  - {save_path2}")
