import codecs, sys, re, collections
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

file_path = r"c:\Users\user\Desktop\데이터분석\음악 프로젝트\Takeout\YouTube 및 YouTube Music\시청 기록\시청 기록.html"
with open(file_path, 'r', encoding='utf-8') as f:
    html = f.read()

chunks = html.split('class="outer-cell mdl-cell mdl-cell--12-col mdl-shadow--2dp">')

titles = []
for c in chunks:
    if 'YouTube Music' in c and '주저하는 연인들을 위해' in c:
        m = re.search(r'<a[^>]*href="https://(?:music|www)\.youtube\.com/watch\?v=[^"]*">([^<]+)</a>', c)
        if m:
            title = m.group(1).replace('\xa0', ' ')
            titles.append(title)

print("Title variants found:")
for t, count in collections.Counter(titles).most_common():
    print(f"[{count} times]: {t}")
