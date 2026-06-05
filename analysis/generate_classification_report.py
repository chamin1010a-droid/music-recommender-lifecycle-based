import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')
from lifecycle_recommender import run_pipeline

history_csv = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
meta_csv = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\ytm_metadata_cache.csv'

result = run_pipeline(
    csv_path=history_csv,
    user_name='user',
    playlist_size=15,
    preset='default',
    metadata_path=meta_csv,
    user_birth_year=1998
)
temps = result['temp_tracker'].song_temps

categories = ['Rising', 'Steady', 'Warm', 'Cool', 'Frozen']
name_map = {
    'Rising': '🔥 Hot (Rising)',
    'Steady': '🔥 Hot (Steady)',
    'Warm': '🟡 Warm',
    'Cool': '🧊 Cool',
    'Frozen': '❄️ Frozen'
}

with open(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\classification_report.txt', 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("곡 온도 분류(Classification) 검증 리포트\n")
    f.write("="*60 + "\n\n")
    
    for cat in categories:
        songs = [s for s in temps.values() if s['temperature'] == cat]
        songs.sort(key=lambda x: (x['tier'], x['total_plays']), reverse=True)
        
        f.write(f"=== {name_map[cat]} ({len(songs)}곡) ===\n")
        f.write(f"[선정 기준]\n")
        if cat == 'Rising': f.write(" - 가파른 상승세를 보이며 최근 집중적으로 듣는 곡 (기울기 > 0.05)\n")
        elif cat == 'Steady': f.write(" - 급상승은 없으나 꾸준하게 항상 듣는 활성 곡\n")
        elif cat == 'Warm': f.write(" - 꽤 들었던 곡이지만 최근 한동안 재생되지 않아 대기 중인 곡\n")
        elif cat == 'Cool': f.write(" - 아티스트 내 다른 곡 대비 상대적으로 '스킵률'이 너무 높거나 하락세가 명확해 잠시 추천을 쉬는 곡\n")
        elif cat == 'Frozen': f.write(" - 너무 오랫동안(예: 3개월 이상) 재생 기록이 없어 완전히 잊힌 곡\n")
        
        f.write("\n[대표 곡 Top 25 예시]\n")
        for i, s in enumerate(songs[:25], 1):
            f.write(f"  {i}. [{s['tier']}] {s['artist'][:15]:<15} - {s['title'][:30]:<30} (총 {s['total_plays']:>3}회 | 스킵 {s['skip_rate']*100:>2.0f}% | 최근 {s['days_since_last']:>3}일 전)\n")
        f.write("\n\n")

print("리포트 생성 완료: classification_report.txt")
