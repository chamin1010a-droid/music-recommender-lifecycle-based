import re
import pandas as pd
from datetime import datetime

file_path = r"c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 시록\시청 기록.html"
# The folder name might be '시청 기록', let me fallback to it.
import os

folder_path = r"c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록"
file_path = os.path.join(folder_path, "시청 기록.html")

print("Reading HTML file...")
with open(file_path, 'r', encoding='utf-8') as f:
    html_content = f.read()

print("File loaded. Starting regex parsing...")

# Typical pattern: 
# <div class="outer-cell mdl-cell mdl-cell--12-col mdl-shadow--2dp">
# Inside this, there are typically two content-cells (one for title/artist/time, one for app/details)

# Let's split by outer-cell
chunks = html_content.split('class="outer-cell mdl-cell mdl-cell--12-col mdl-shadow--2dp">')

records = []
print(f"Total entries found: {len(chunks)-1}")

for chunk in chunks[1:]:
    # Determine if it's from YouTube Music
    # Sometimes it says 'YouTube Music' directly, or 'YouTube Music 앱에서 시청함' etc.
    app = 'YouTube'
    if 'YouTube Music' in chunk:
        app = 'YouTube Music'
        
    if app != 'YouTube Music':
        continue # For now, we only want YouTube Music
        
    # Extract Title:
    # <a href="watch url">Title</a>
    title_match = re.search(r'<a[^>]*href="https://music\.youtube\.com/watch\?v=[^>]*>([^<]+)</a>', chunk)
    if not title_match:
        # Sometimes the href is just youtube.com
        title_match = re.search(r'<a[^>]*href="https://www\.youtube\.com/watch\?v=[^>]*>([^<]+)</a>', chunk)
        
    title = title_match.group(1).strip() if title_match else None
    
    # Extract Artist(s) / Channel
    # This comes after the <br> typically, but artist is also an <a> tag
    # Often it's <a href="...channel...">Artist</a>
    artist_match = re.search(r'<a[^>]*href="https://(?:music|www)\.youtube\.com/channel/[^>]*>([^<]+)</a>', chunk)
    artist = artist_match.group(1).strip() if artist_match else None
    
    # Alternatively find text nodes before the date
    
    # Extract Date Time
    # Look for the innermost text that looks like a date. Instead, we can just grab all texts 
    # and look for something that ends with KST or similar.
    # But an easy way is to use regex to find dates.
    date_match = re.search(r'<br>(\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.[^<]+)', chunk)
    date_str = date_match.group(1).strip() if date_match else None
    
    if title and date_str:
        records.append({
            'title': title,
            'artist': artist,
            'time_str': date_str,
            'app': app
        })

print(f"Total YouTube Music records identified: {len(records)}")

if len(records) > 0:
    for i in range(min(5, len(records))):
        print(records[i])

# Create dataframe
df = pd.DataFrame(records)
# Save it
output_csv = os.path.join(folder_path, "ytm_history_parsed.csv")
df.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"Saved parsed data to {output_csv}")
