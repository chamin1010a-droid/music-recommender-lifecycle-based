import urllib.parse
import urllib.request
import json
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def test_itunes(artist, title):
    query = f"{artist} {title}"
    url = f"https://itunes.apple.com/search?term={urllib.parse.quote(query)}&entity=song&limit=1"
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, context=ctx) as response:
            data = json.loads(response.read().decode('utf-8'))
            if data['resultCount'] > 0:
                item = data['results'][0]
                print(f"[Match] Artist: {item.get('artistName')}, Title: {item.get('trackName')}")
                print(f"        Release Date: {item.get('releaseDate')}")
                print(f"        Genre: {item.get('primaryGenreName')}")
            else:
                print(f"[No Match] {query}")
    except Exception as e:
        print(f"[Error] {e}")

test_itunes("데이식스", "예뻤어")
test_itunes("LANY", "dna")
test_itunes("잔나비", "주저하는 연인들을 위해")

