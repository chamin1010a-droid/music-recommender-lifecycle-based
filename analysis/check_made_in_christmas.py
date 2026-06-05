import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

history_csv = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
df = pd.read_csv(history_csv, encoding='utf-8-sig')

# Made In Christmas 조회
christmas = df[df['title'].str.contains('Made In Christmas', na=False, case=False)]
print("="*60)
print(f"🎵 Made In Christmas 데이터 분석")
print("="*60)
print(f"총 재생수: {len(christmas)}")
if len(christmas) > 0:
    skips = christmas["is_skipped"].sum()
    print(f"스킵 수: {skips} (스킵률: {skips/len(christmas)*100:.1f}%)")
    print(christmas[['timestamp', 'is_skipped']].sort_values('timestamp', ascending=False))
