import sys
import codecs

# Fix print encoding
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

file_path = r"c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\시청 기록.html"

with open(file_path, 'r', encoding='utf-8') as f:
    html = f.read()

chunks = html.split('class="outer-cell mdl-cell mdl-cell--12-col mdl-shadow--2dp">')

target_song = "주저하는 연인들을 위해"
total_chunks_with_song = 0
music_chunks_with_song = 0

for chunk in chunks[1:]:
    if target_song in chunk:
        total_chunks_with_song += 1
        if 'YouTube Music' in chunk:
            music_chunks_with_song += 1
            
print(f"Total entries containing '{target_song}': {total_chunks_with_song}")
print(f"Entries where 'YouTube Music' is explicitly mentioned: {music_chunks_with_song}")

# Let's also check if they are just regular YouTube
print("\nExample of chunk without YouTube Music:")
for chunk in chunks[1:]:
    if target_song in chunk and 'YouTube Music' not in chunk:
        print(chunk[:500].replace('\n', ' '))
        break
