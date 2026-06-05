import codecs, sys, re
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

html = open(r'c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\시청 기록.html', 'r', encoding='utf-8').read()
chunks = html.split('class="outer-cell mdl-cell mdl-cell--12-col mdl-shadow--2dp">')

missing_title = 0
missing_date = 0
total_found = 0

for c in chunks:
    if 'YouTube Music' in c and 'for lovers who hesitate' in c:
        total_found += 1
        
        # Check Title
        m = re.search(r'<a[^>]*href="https://(?:music|www)\.youtube\.com/watch\?v=[^"]*">([^<]+)</a>', c)
        title = m.group(1).replace('\xa0', ' ') if m else None
        if not title:
            missing_title += 1
            
        # Check Date
        date_match = re.search(r'<br>([^<]+(?:KST|UTC|AM|PM|오전|오후)[^<]*)\s*</div>', c)
        if not date_match:
            date_match = re.search(r'<br>([^<]+[0-9]{2}:[0-9]{2}:[0-9]{2}[^<]*)', c)
        if not date_match:
            missing_date += 1
            print("--- CHUNK MISSING DATE ---")
            print(c[:500])

print(f"\nTotal blocks: {total_found}")
print(f"Blocks missing title: {missing_title}")
print(f"Blocks missing date: {missing_date}")
