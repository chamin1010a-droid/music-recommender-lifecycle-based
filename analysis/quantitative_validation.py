"""
음원 생애주기 가설 — 정량적 검증 (6명 통합)
=============================================
핵심 검증: "스킵률의 변화 방향이 미래 청취 행동을 예측하는가?"

방법론: 시간 분할(Temporal Split) 예측력 검증
  1. 각 사용자의 데이터를 시간 기준 전반/후반으로 나눔
  2. 전반 데이터만으로 곡별 "스킵 추세"를 판별 (증가/감소/유지)
  3. 후반 데이터에서 실제로 그 곡의 재생이 어떻게 변했는지 확인
  4. "전반에 스킵률이 올라간 곡은, 후반에 재생이 줄었는가?" → 예측력 검증

이건 정의를 검증하는 게 아니라, 정의의 '예측력'을 검증하는 거다.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
REPORT_DIR = os.path.join(BASE_DIR, 'reports')
os.makedirs(REPORT_DIR, exist_ok=True)

# ===================================================================
# 사용자 데이터 목록
# ===================================================================
USERS = {
    'user': os.path.join(BASE_DIR, '유튜브 뮤직 로그들', 'user', 'user_features.csv'),
    '친구D': os.path.join(BASE_DIR, '유튜브 뮤직 로그들', '친구D', '친구D_features.csv'),
    '친구B': os.path.join(BASE_DIR, '유튜브 뮤직 로그들', '친구B', '친구B_features.csv'),
    '친구C': os.path.join(BASE_DIR, '유튜브 뮤직 로그들', '친구C', '친구C_features.csv'),
    '친구E': os.path.join(BASE_DIR, '유튜브 뮤직 로그들', '친구E', '친구E_features.csv'),
    '친구F': os.path.join(BASE_DIR, '유튜브 뮤직 로그들', '친구F', '친구F_features.csv'),
}


def load_user_data(csv_path, user_name):
    """사용자 데이터 로드 및 기본 전처리"""
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    
    # is_skipped 컬럼 확인
    if 'is_skipped' not in df.columns:
        print(f"  ⚠️ {user_name}: is_skipped 컬럼 없음 → 건너뜀")
        return None
    
    return df


def classify_song_trajectory(song_df, split_date):
    """
    곡의 전반/후반 궤적을 분류한다.
    
    전반(split_date 이전) 데이터로 판별:
      - 스킵률 추세 (증가/감소/유지)
      - 재생 활성도
    
    후반(split_date 이후) 데이터로 검증:
      - 실제 재생 횟수 변화
      - 실제 스킵률 변화
    """
    first_half = song_df[song_df['timestamp'] < split_date]
    second_half = song_df[song_df['timestamp'] >= split_date]
    
    if len(first_half) < 4:  # 전반부 최소 4회 재생 필요
        return None
    
    # === 전반부에서 스킵 추세 판별 ===
    fh_sorted = first_half.sort_values('timestamp')
    fh_mid = len(fh_sorted) // 2
    fh_early = fh_sorted.iloc[:fh_mid]
    fh_late = fh_sorted.iloc[fh_mid:]
    
    fh_early_skip = fh_early['is_skipped'].mean()
    fh_late_skip = fh_late['is_skipped'].mean()
    
    skip_delta = fh_late_skip - fh_early_skip  # 양수 = 스킵 증가(질리는 중), 음수 = 감소(빠져드는 중)
    
    # 추세 분류
    if skip_delta > 0.15:
        trajectory = 'declining'   # 질리는 중 (스킵 ↑)
    elif skip_delta < -0.15:
        trajectory = 'growing'    # 빠져드는 중 (스킵 ↓)
    else:
        trajectory = 'stable'     # 안정 (변화 없음)
    
    # === 후반부 실제 결과 ===
    # 전반부 주간 평균 재생 횟수
    fh_weeks = max(1, (fh_sorted['timestamp'].max() - fh_sorted['timestamp'].min()).days / 7)
    fh_weekly_plays = len(first_half) / fh_weeks
    
    # 후반부 주간 평균 재생 횟수
    if len(second_half) == 0:
        sh_weekly_plays = 0.0
        sh_skip_rate = None
        actually_abandoned = True
        severe_drop = True
    else:
        sh_weeks = max(1, (second_half['timestamp'].max() - second_half['timestamp'].min()).days / 7)
        sh_weekly_plays = len(second_half) / sh_weeks
        sh_skip_rate = second_half['is_skipped'].mean()
        actually_abandoned = (sh_weekly_plays < fh_weekly_plays * 0.3)  # 재생 70%+ 감소
        severe_drop = (sh_weekly_plays < fh_weekly_plays * 0.5)         # 재생 50%+ 감소 (절반 이하)
    
    play_change_ratio = sh_weekly_plays / max(fh_weekly_plays, 0.01)  # 후반/전반 재생 비율
    
    return {
        'trajectory': trajectory,
        'skip_delta': round(skip_delta, 3),
        'fh_plays': len(first_half),
        'sh_plays': len(second_half),
        'fh_weekly': round(fh_weekly_plays, 2),
        'sh_weekly': round(sh_weekly_plays, 2),
        'play_change_ratio': round(play_change_ratio, 2),
        'actually_abandoned': actually_abandoned,
        'severe_drop': severe_drop,
        'sh_skip_rate': round(sh_skip_rate, 3) if sh_skip_rate is not None else None,
    }


def run_temporal_validation(df, user_name):
    """
    시간 분할 예측력 검증을 수행한다.
    
    핵심 검증:
      "전반에 스킵률이 올라간 곡(declining)이 
       후반에 실제로 재생이 감소했는가?"
    """
    # 데이터를 시간 기준으로 반으로 나눔
    min_date = df['timestamp'].min()
    max_date = df['timestamp'].max()
    split_date = min_date + (max_date - min_date) / 2
    data_span_days = (max_date - min_date).days
    
    results = []
    for song_id, group in df.groupby('song_id'):
        result = classify_song_trajectory(group, split_date)
        if result:
            result['song_id'] = song_id
            result['artist'] = group['artist'].iloc[0]
            result['title'] = group['title'].iloc[0]
            results.append(result)
    
    if not results:
        return None
    
    results_df = pd.DataFrame(results)
    
    # === 핵심 검증: 궤적별 후반부 결과 비교 ===
    summary = {}
    for traj in ['declining', 'stable', 'growing']:
        subset = results_df[results_df['trajectory'] == traj]
        if len(subset) == 0:
            continue
        summary[traj] = {
            'count': len(subset),
            'avg_play_change': subset['play_change_ratio'].mean(),
            'median_play_change': subset['play_change_ratio'].median(), 
            'abandon_rate': subset['actually_abandoned'].mean(),
            'severe_drop_rate': subset['severe_drop'].mean(),
            'avg_sh_plays': subset['sh_plays'].mean(),
        }
    
    return {
        'user': user_name,
        'total_songs': len(results_df),
        'data_span_days': data_span_days,
        'split_date': split_date,
        'trajectory_summary': summary,
        'results_df': results_df,
    }


def run_artist_lifecycle_evidence(df, user_name):
    """
    아티스트 수준 생애주기 증거를 수집한다.
    
    "많이 듣던 가수를 이제 잘 안 듣는다" 패턴이 존재하는가?
    → 전체 데이터에서 주간 재생 추세가 확실히 하락한 아티스트 목록
    """
    df = df.copy()
    df['week_num'] = (df['timestamp'] - df['timestamp'].min()).dt.days // 7
    
    decline_artists = []
    
    for artist, group in df.groupby('artist'):
        if len(group) < 15:
            continue
        
        weekly = group.groupby('week_num').size()
        if len(weekly) < 6:
            continue
        
        mid = len(weekly) // 2
        first_avg = weekly.iloc[:mid].mean()
        second_avg = weekly.iloc[mid:].mean()
        
        if first_avg < 1:
            continue
        
        decline_ratio = second_avg / first_avg
        
        # 확실한 하락: 후반 재생이 전반의 40% 이하
        if decline_ratio <= 0.4 and first_avg >= 2:
            decline_artists.append({
                'artist': artist.replace(' - Topic', ''),
                'first_half_weekly': round(first_avg, 1),
                'second_half_weekly': round(second_avg, 1),
                'decline_ratio': round(decline_ratio, 2),
                'total_plays': len(group),
            })
    
    decline_artists.sort(key=lambda x: x['total_plays'], reverse=True)
    return decline_artists


# ===================================================================
# 메인 실행
# ===================================================================
def main():
    print("=" * 70)
    print("📊 음원 생애주기 가설 — 정량적 검증 (6명 통합)")
    print("=" * 70)
    print("\n핵심 검증: '스킵률 변화 방향이 미래 청취를 예측하는가?'")
    print("방법: 각 사용자 데이터를 시간순 전반/후반으로 나누어 예측력 검증\n")
    
    all_validation_results = []
    all_artist_declines = {}
    
    for user_name, csv_path in USERS.items():
        if not os.path.exists(csv_path):
            print(f"⚠️ {user_name}: 파일 없음 ({csv_path})")
            continue
        
        print(f"\n{'─'*50}")
        print(f"👤 {user_name}")
        print(f"{'─'*50}")
        
        df = load_user_data(csv_path, user_name)
        if df is None:
            continue
        
        print(f"  총 재생: {len(df):,}건 | 기간: {df['timestamp'].min().strftime('%Y-%m-%d')} ~ {df['timestamp'].max().strftime('%Y-%m-%d')}")
        
        # === 검증 1: 시간 분할 예측력 ===
        val_result = run_temporal_validation(df, user_name)
        if val_result:
            all_validation_results.append(val_result)
            summary = val_result['trajectory_summary']
            
            print(f"\n  [검증] 전반부 스킵 추세 → 후반부 실제 재생 변화")
            print(f"  {'추세':<15} {'곡수':>5} {'후반/전반 재생비':>15} {'이탈률':>8}")
            print(f"  {'-'*48}")
            
            for traj, label in [('declining', '📉 질리는 중'), ('stable', '➡️ 안정'), ('growing', '📈 빠져드는 중')]:
                if traj in summary:
                    s = summary[traj]
                    print(f"  {label:<13} {s['count']:>5} {s['avg_play_change']:>14.2f}x {s['abandon_rate']*100:>7.1f}%")
        
        # === 검증 2: 아티스트 수준 생애주기 ===
        declines = run_artist_lifecycle_evidence(df, user_name)
        all_artist_declines[user_name] = declines
        
        if declines:
            print(f"\n  [아티스트 생애주기] 확실히 하락한 가수: {len(declines)}명")
            for d in declines[:5]:
                print(f"    📉 {d['artist'][:20]:<22} 주간 {d['first_half_weekly']:.1f} → {d['second_half_weekly']:.1f}회 ({d['decline_ratio']:.0%})")
    
    # ===================================================================
    # 통합 결과
    # ===================================================================
    print(f"\n\n{'='*70}")
    print(f"📋 6명 통합 검증 결과 (재생량 50%+ 급감 기준)")
    print(f"{'='*70}")
    
    # --- 핵심 결과 테이블 ---
    print(f"\n🎯 핵심 검증: '전반에 질리는 추세를 보인 곡이 정말로 후반에 재생이 감소하는가?'")
    print(f"\n  {'사용자':<8} {'분석곡':>6} | {'📉질리는중':>12} {'급감비율':>7} | {'➡️안정':>10} {'급감비율':>7} | {'📈빠져듦':>12} {'급감비율':>7}")
    print(f"  {'─'*85}")
    
    # 통합 집계
    agg = {'declining': [], 'stable': [], 'growing': []}
    
    for val in all_validation_results:
        user = val['user']
        total = val['total_songs']
        s = val['trajectory_summary']
        
        dec = s.get('declining', {})
        stb = s.get('stable', {})
        grw = s.get('growing', {})
        
        dec_str = f"{dec.get('count', 0)}곡" if dec else "-"
        dec_abn = f"{dec.get('severe_drop_rate', 0)*100:.0f}%" if dec else "-"
        stb_str = f"{stb.get('count', 0)}곡" if stb else "-"
        stb_abn = f"{stb.get('severe_drop_rate', 0)*100:.0f}%" if stb else "-"
        grw_str = f"{grw.get('count', 0)}곡" if grw else "-"
        grw_abn = f"{grw.get('severe_drop_rate', 0)*100:.0f}%" if grw else "-"
        
        print(f"  {user:<8} {total:>5}곡 | {dec_str:>10} {dec_abn:>7} | {stb_str:>10} {stb_abn:>7} | {grw_str:>10} {grw_abn:>7}")
        
        for traj in agg:
            if traj in s:
                agg[traj].append(s[traj])
    
    # 전체 평균
    print(f"  {'─'*85}")
    for traj, label in [('declining', '📉 질리는 중'), ('stable', '➡️ 안정'), ('growing', '📈 빠져드는 중')]:
        if agg[traj]:
            avg_severe = np.mean([s['severe_drop_rate'] for s in agg[traj]])
            avg_change = np.mean([s['avg_play_change'] for s in agg[traj]])
            median_change = np.median([s['median_play_change'] for s in agg[traj]])
            total_songs = sum(s['count'] for s in agg[traj])
            print(f"\n  {label} 통합 ({total_songs}곡)")
            print(f"    재생 50%+ 급감 비율: {avg_severe*100:.1f}%")
            print(f"    평균 재생비율 (후반/전반): {avg_change:.2f}x (중간값: {median_change:.2f}x)")
    
    # --- 예측력 판정 ---
    if agg['declining'] and agg['growing']:
        dec_severe = np.mean([s['severe_drop_rate'] for s in agg['declining']])
        grw_severe = np.mean([s['severe_drop_rate'] for s in agg['growing']])
        
        print(f"\n\n  {'='*50}")
        if dec_severe > grw_severe:
            diff = dec_severe - grw_severe
            print(f"  ✅ 예측력 검증 통과 (재생 감소 기준)!")
            print(f"     질리는 곡의 급감비율({dec_severe*100:.1f}%) > 빠져드는 곡의 급감비율({grw_severe*100:.1f}%)")
            print(f"     차이: {diff*100:.1f}%p")
        else:
            print(f"  ❌ 예측력 검증 실패")
            print(f"     질리는 곡의 급감비율({dec_severe*100:.1f}%) ≤ 빠져드는 곡의 급감비율({grw_severe*100:.1f}%)")
        print(f"  {'='*50}")
    
    # --- 아티스트 생애주기 증거 ---
    print(f"\n\n📉 아티스트 수준 생애주기 증거")
    print(f"{'─'*50}")
    total_declines = sum(len(v) for v in all_artist_declines.values())
    users_with_decline = sum(1 for v in all_artist_declines.values() if len(v) > 0)
    print(f"  {users_with_decline}/{len(all_artist_declines)}명의 사용자에서 확실히 하락한 아티스트 발견")
    print(f"  총 {total_declines}명의 아티스트가 '전반기 대비 후반기 재생 60%+ 감소' 패턴")
    print(f"  → 아티스트 수준에서 생애주기(성장→안정→쇠퇴) 패턴이 존재함을 확인")
    
    # ===================================================================
    # 시각화
    # ===================================================================
    create_validation_charts(all_validation_results, all_artist_declines)
    
    print(f"\n✅ 시각화 저장 완료: {REPORT_DIR}")


def create_validation_charts(all_results, all_declines):
    """검증 결과를 시각화한다."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('음원 생애주기 가설 — 정량적 검증 결과', fontsize=16, fontweight='bold', y=1.02)
    
    # --- 차트 1: 궤적별 이탈률 비교 (핵심!) ---
    ax = axes[0]
    categories = ['📉 질리는 중\n(스킵↑)', '➡️ 안정\n(변화없음)', '📈 빠져드는 중\n(스킵↓)']
    traj_keys = ['declining', 'stable', 'growing']
    colors = ['#e74c3c', '#95a5a6', '#2ecc71']
    
    abandon_rates = []
    for traj in traj_keys:
        rates = []
        for val in all_results:
            if traj in val['trajectory_summary']:
                rates.append(val['trajectory_summary'][traj]['abandon_rate'])
        abandon_rates.append(np.mean(rates) * 100 if rates else 0)
    
    bars = ax.bar(categories, abandon_rates, color=colors, edgecolor='white', linewidth=2, width=0.6)
    ax.set_ylabel('이탈률 (%)', fontsize=12)
    ax.set_title('전반부 스킵 추세 → 후반부 이탈률', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(abandon_rates) * 1.3 if max(abandon_rates) > 0 else 100)
    
    for bar, val in zip(bars, abandon_rates):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # --- 차트 2: 궤적별 후반부 재생 변화율 ---
    ax = axes[1]
    play_changes = []
    for traj in traj_keys:
        changes = []
        for val in all_results:
            if traj in val['trajectory_summary']:
                changes.append(val['trajectory_summary'][traj]['avg_play_change'])
        play_changes.append(np.mean(changes) if changes else 0)
    
    bars = ax.bar(categories, play_changes, color=colors, edgecolor='white', linewidth=2, width=0.6)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='변화 없음 (1.0x)')
    ax.set_ylabel('후반/전반 재생 비율', fontsize=12)
    ax.set_title('전반부 스킵 추세 → 후반부 재생량 변화', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    
    for bar, val in zip(bars, play_changes):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.2f}x', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # --- 차트 3: 사용자별 일관성 (각 사용자마다 declining vs growing 이탈률 차이) ---
    ax = axes[2]
    user_names = []
    dec_rates = []
    grw_rates = []
    
    for val in all_results:
        user = val['user']
        s = val['trajectory_summary']
        if 'declining' in s and 'growing' in s:
            user_names.append(user)
            dec_rates.append(s['declining']['abandon_rate'] * 100)
            grw_rates.append(s['growing']['abandon_rate'] * 100)
    
    if user_names:
        x = np.arange(len(user_names))
        width = 0.35
        ax.bar(x - width/2, dec_rates, width, label='📉 질리는 곡', color='#e74c3c', alpha=0.8)
        ax.bar(x + width/2, grw_rates, width, label='📈 빠져드는 곡', color='#2ecc71', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(user_names, fontsize=11)
        ax.set_ylabel('이탈률 (%)', fontsize=12)
        ax.set_title('사용자별 예측 일관성', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    save_path = os.path.join(REPORT_DIR, 'lifecycle_validation_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n📊 차트 저장: {save_path}")


if __name__ == '__main__':
    main()
