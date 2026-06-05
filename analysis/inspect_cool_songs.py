import sys, os
sys.path.insert(0, r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\core')
sys.stdout.reconfigure(encoding='utf-8')

from lifecycle_recommender import *

csv = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\ytm_history_features.csv'
df = pd.read_csv(csv, encoding='utf-8-sig')

n = ArtistNameNormalizer()
df = n.normalize_dataframe(df)
tc = ArtistTierClassifier(df)
tm = tc.classify_tiers()
tt = SongTemperatureTracker(df, tm)
st = tt.classify_all_songs()

# 쇠퇴 보정 적용 전/후 비교를 위해 보정 전 온도 저장
pre_decline = {}
for sid, info in st.items():
    pre_decline[sid] = info['temperature']

fd = AsymmetricFlowDetector(df, tm)
fd.detect_growth_bottom_up()
fd.detect_decline_top_down()
corr = fd.apply_decline_correction(st)

# Jay Park, KARA 곡 상세 분석
targets = ['Jay Park', 'KARA']

for target in targets:
    print(f"\n{'='*60}")
    print(f"🔍 {target} 곡별 상세 분석")
    print(f"{'='*60}")
    
    songs = [(sid, info) for sid, info in st.items() 
             if isinstance(info['artist'], str) and target.lower() in info['artist'].lower()]
    
    if not songs:
        print(f"  곡 없음")
        continue
    
    # 아티스트 쇠퇴 정보
    decline_info = [s for s in fd.decline_signals if target.lower() in s['artist'].lower()]
    if decline_info:
        d = decline_info[0]
        print(f"  📉 쇠퇴 감지: severity={d['severity']}, 잔존율={d['decline_ratio']:.0%}")
    else:
        print(f"  ✅ 쇠퇴 감지 없음")
    
    print(f"\n  {'곡명':<30} {'보정전':>7} {'보정후':>7} {'스킵률':>5} {'전반skip':>7} {'후반skip':>7} {'마지막':>5} {'재생':>4} {'Zone':>6}")
    print(f"  {'-'*95}")
    
    for sid, info in sorted(songs, key=lambda x: x[1]['total_plays'], reverse=True):
        old_temp = pre_decline.get(sid, '?')
        new_temp = info['temperature']
        changed = " ⬇️" if old_temp != new_temp else ""
        
        fh = info.get('first_half_skip')
        sh = info.get('second_half_skip')
        
        # Zone 판별
        if fh is not None and sh is not None:
            if info['total_plays'] >= 3:
                fh_val = fh
                sh_val = sh
                first_3 = 0  # approximate
                skip_rate = info['skip_rate']
                
                if fh_val <= 0.33 and skip_rate <= 0.3:
                    zone = "Zone1"
                elif fh_val > sh_val + 0.15:
                    zone = "Zone2"
                elif sh_val > fh_val + 0.15:
                    zone = "Zone3"
                else:
                    zone = "Zone4"
            else:
                zone = "N/A"
        else:
            zone = "N/A"
        
        fh_str = f"{fh*100:.0f}%" if fh is not None else "N/A"
        sh_str = f"{sh*100:.0f}%" if sh is not None else "N/A"
        
        print(f"  {info['title'][:29]:<30} {old_temp:>7} {new_temp:>7}{changed} "
              f"{info['skip_rate']*100:>4.0f}% {fh_str:>7} {sh_str:>7} "
              f"{info['days_since_last']:>4}일 {info['total_plays']:>4} {zone:>6}")
