"""
시드별 플레이리스트를 YouTube Music 링크 포함 HTML로 생성합니다.
"""
import sys, os, urllib.parse
sys.path.append(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\core')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import json

# playlist_comparison.md 파싱 대신, multi_playlist.py 결과를 직접 재생성
from song_title_normalizer import SongTitleNormalizer

norm = SongTitleNormalizer()

# 플레이리스트 데이터 파싱
playlists_raw = """
#1|Love Is All|The Black Skirts
Pole Dance (봉춤을 추네)|JANNABI
Love Is All|The Black Skirts
Gang Gang Schiele|HYUKOH
Romantic Sunday|Car, the Garden
May I Laugh (Instrumental)|JANNABI
H O M E|HANRORO
Summer (뜨거운 여름밤은 가고 남은 건 볼품없지만)|JANNABI
SOMEDAY (어떤 날)|The Black Skirts
비틀비틀 짝짜꿍|HANRORO
Diamond|The Black Skirts
MIRROR|HANRORO
Just as We Were|Car, the Garden
LOVE (로켓트)|JANNABI
Radioactive|Imagine Dragons
Hollywood|The Black Skirts
TEAM 401|Car, the Garden
자처|HANRORO
TOMBOY|HYUKOH
나무 (나무)|Car, the Garden
To. __|HANRORO
---
#2|별이될께|December
Because of love (사랑 참…)|December
니가 떠난 그날|Monday Kiz
Stand Still|The Black Skirts
큰일이다 (큰일이다)|V.O.S
Shout! Shout! (세상에 소리쳐)|December
Puppy|The Black Skirts
별이될께 (별이될께)|December
MIRROR|HANRORO
배운게 사랑이라 (배운게 사랑이라)|December
Antifreeze|The Black Skirts
미안해, 고마워, 사랑해|Monday Kiz
사랑합니다|SG Wannabe
Radioactive|Imagine Dragons
Cry Cry Cry|Realslow
사랑해 그리고 기억해|Monday Kiz
페이지원|SG Wannabe
그대여 (그대여)|Monday Kiz
세글자|SG Wannabe
Aftermath (후유증)|ZE:A
After Love (사랑한 후에)|Park Hyo Shin
---
#3|페이지원|SG Wannabe
그대여 (그대여)|Monday Kiz
세글자|SG Wannabe
추억은 사랑을 닮아|Park Hyo Shin
Dimly (아스라이)|M.C the MAX
내사람 : Partner for Life|SG Wannabe
One Day Only (사계) (하루살이)|M.C the MAX
안녕이라고 말하지마 (안녕이라고 말하지마)|DAVICHI
사랑해 그리고 기억해|Monday Kiz
MIRROR|HANRORO
은|SG Wannabe
애상|COOL
니가 떠난 그날|Monday Kiz
Radioactive|Imagine Dragons
페이지원|SG Wannabe
울어 (울어)|V.O.S
미안해, 고마워, 사랑해|Monday Kiz
After Love (사랑한 후에)|Park Hyo Shin
After You've Gone (넘쳐흘러)|M.C the MAX
큰일이다 (큰일이다)|V.O.S
No matter where (어디에도)|M.C the MAX
---
#4|비틀비틀 짝짜꿍|HANRORO
H O M E|HANRORO
LOVE (로켓트)|JANNABI
Jersey Girl|The Black Skirts
Romantic Sunday|Car, the Garden
[LIVE CLIP] 한로로 - 입춘|HANRORO
Forever Has Always Been|Redoor
TEAM 401|Car, the Garden
MIRROR|HANRORO
나무 (나무)|Car, the Garden
비틀비틀 짝짜꿍|HANRORO
Ron|Redoor
Good Luck to You (행운을 빌어요)|JANNABI
Radioactive|Imagine Dragons
자처|HANRORO
Just as We Were|Car, the Garden
Let me go!|The Volunteers
Blue rain|Redoor
Good Boy Twist|JANNABI
Silently Completely Eternally|Nerd Connection
Sweet memories (그 밤 그 밤)|JANNABI
---
#5|Shoot Me|DAY6
Congratulations|DAY6
Dimly (아스라이)|M.C the MAX
좋아합니다 I Like You|DAY6
Puppy|The Black Skirts
완전 멋지잖아 So Cool|DAY6
MONEYBALL|Xdinary Heroes
Tattoo (Tattoo)|CNBLUE
Test Me|Xdinary Heroes
MIRROR|HANRORO
Shoot Me|DAY6
Love Revolution (Love Revolution)|CNBLUE
세글자|SG Wannabe
Radioactive|Imagine Dragons
Supernatural|Xdinary Heroes
Sweet memories (그 밤 그 밤)|JANNABI
Love Is All|The Black Skirts
그대여 (그대여)|Monday Kiz
Antifreeze|The Black Skirts
Man in the Box|Xdinary Heroes
Stand Still|The Black Skirts
"""

# 파싱
playlists = []
current = None
for line in playlists_raw.strip().split('\n'):
    line = line.strip()
    if not line:
        continue
    if line == '---':
        if current:
            playlists.append(current)
        current = None
        continue
    if line.startswith('#'):
        parts = line.split('|')
        current = {
            'id': parts[0],
            'seed_title': parts[1],
            'seed_artist': parts[2],
            'tracks': []
        }
        continue
    if current and '|' in line:
        parts = line.split('|')
        current['tracks'].append({'title': parts[0], 'artist': parts[1]})

if current:
    playlists.append(current)

# HTML 생성
def ytm_link(title, artist):
    q = f"{title} {artist}".replace(' - Topic', '')
    return f"https://music.youtube.com/search?q={urllib.parse.quote(q)}"

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
seed_emojis = ['🎸', '💿', '🎹', '🎺', '🥁']

html = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Multi-Signal 플레이리스트</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'Inter', sans-serif;
    background: #0a0a0f;
    color: #e0e0e0;
    padding: 20px;
}
h1 {
    text-align: center;
    font-size: 28px;
    margin: 30px 0 10px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    text-align: center;
    color: #888;
    font-size: 14px;
    margin-bottom: 30px;
}
.playlists {
    display: flex;
    gap: 16px;
    overflow-x: auto;
    padding-bottom: 20px;
    scroll-snap-type: x mandatory;
}
.playlist-card {
    flex: 0 0 340px;
    scroll-snap-align: start;
    background: #14141f;
    border-radius: 16px;
    border: 1px solid #222;
    overflow: hidden;
}
.playlist-header {
    padding: 20px;
    text-align: center;
    position: relative;
}
.playlist-header h2 {
    font-size: 18px;
    margin-bottom: 4px;
}
.playlist-header .artist {
    font-size: 13px;
    opacity: 0.6;
}
.playlist-header .emoji {
    font-size: 36px;
    display: block;
    margin-bottom: 8px;
}
.track-list {
    padding: 0 12px 16px;
}
.track {
    display: flex;
    align-items: center;
    padding: 8px 10px;
    border-radius: 8px;
    transition: background 0.2s;
    text-decoration: none;
    color: inherit;
    gap: 10px;
}
.track:hover {
    background: rgba(255,255,255,0.06);
}
.track-num {
    font-size: 12px;
    color: #555;
    width: 20px;
    text-align: right;
    flex-shrink: 0;
}
.track-info {
    flex: 1;
    min-width: 0;
}
.track-title {
    font-size: 13px;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.track-artist {
    font-size: 11px;
    color: #888;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.play-btn {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    opacity: 0;
    transition: opacity 0.2s;
    font-size: 12px;
}
.track:hover .play-btn {
    opacity: 1;
}
.badge {
    display: inline-block;
    font-size: 9px;
    padding: 1px 5px;
    border-radius: 4px;
    margin-left: 4px;
    font-weight: 600;
}
</style>
</head>
<body>
<h1>🎯 Multi-Signal 추천 플레이리스트</h1>
<p class="subtitle">가사(30%) + 오디오(30%) + 메타(15%) + 태그(25%) 기반 • 클릭하면 YouTube Music에서 검색</p>
<div class="playlists">
"""

for i, pl in enumerate(playlists):
    color = colors[i % len(colors)]
    emoji = seed_emojis[i % len(seed_emojis)]
    
    html += f"""
<div class="playlist-card">
    <div class="playlist-header" style="background: linear-gradient(135deg, {color}22, {color}08);">
        <span class="emoji">{emoji}</span>
        <h2 style="color: {color};">{pl['seed_title']}</h2>
        <div class="artist">{pl['seed_artist']}</div>
    </div>
    <div class="track-list">
"""
    for j, track in enumerate(pl['tracks'], 1):
        link = ytm_link(track['title'], track['artist'])
        title_display = track['title'][:35]
        
        html += f"""        <a class="track" href="{link}" target="_blank">
            <span class="track-num">{j}</span>
            <div class="track-info">
                <div class="track-title">{title_display}</div>
                <div class="track-artist">{track['artist']}</div>
            </div>
            <div class="play-btn" style="background: {color}33; color: {color};">▶</div>
        </a>
"""
    
    html += """    </div>
</div>
"""

html += """</div>
</body>
</html>"""

output_path = r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\playlist_viewer.html'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"✅ 저장: {output_path}")
print(f"플레이리스트 {len(playlists)}개, 총 {sum(len(p['tracks']) for p in playlists)}곡")
