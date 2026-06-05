// ── 상태 ──
let selectedSeed = null;
let currentPlaylist = [];
let matchedVideoIds = [];

// ── 검색 ──
const seedSearch = document.getElementById('seedSearch');
const searchResults = document.getElementById('searchResults');
let searchTimeout = null;

seedSearch.addEventListener('input', () => {
    clearTimeout(searchTimeout);
    const q = seedSearch.value.trim();
    if (q.length < 1) {
        searchResults.classList.add('hidden');
        return;
    }
    searchTimeout = setTimeout(() => searchSongs(q), 300);
});

seedSearch.addEventListener('focus', () => {
    if (seedSearch.value.trim().length >= 1) {
        searchSongs(seedSearch.value.trim());
    }
});

document.addEventListener('click', (e) => {
    if (!e.target.closest('.search-box')) {
        searchResults.classList.add('hidden');
    }
});

async function searchSongs(q) {
    const res = await fetch(`/api/songs?q=${encodeURIComponent(q)}`);
    const songs = await res.json();
    
    if (songs.length === 0) {
        searchResults.innerHTML = '<div class="result-item"><div class="r-title">검색 결과 없음</div></div>';
    } else {
        searchResults.innerHTML = songs.map(s => `
            <div class="result-item" onclick="selectSeed('${escapeHtml(s.title)}', '${escapeHtml(s.artist)}')">
                <div class="r-title">${escapeHtml(s.title)}</div>
                <div class="r-artist">${escapeHtml(s.artist)}</div>
            </div>
        `).join('');
    }
    searchResults.classList.remove('hidden');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML.replace(/'/g, "\\'");
}

// ── 시드 선택 ──
function selectSeed(title, artist) {
    selectedSeed = { title, artist };
    
    const el = document.getElementById('selectedSeed');
    el.querySelector('.seed-title').textContent = title;
    el.querySelector('.seed-artist').textContent = `— ${artist}`;
    el.classList.remove('hidden');
    
    seedSearch.value = '';
    searchResults.classList.add('hidden');
    document.getElementById('generateBtn').disabled = false;
}

function clearSeed() {
    selectedSeed = null;
    document.getElementById('selectedSeed').classList.add('hidden');
    document.getElementById('generateBtn').disabled = true;
    document.getElementById('playlistSection').classList.add('hidden');
}

// ── 플레이리스트 생성 ──
async function generatePlaylist() {
    if (!selectedSeed) return;
    
    const count = parseInt(document.getElementById('songCount').value) || 25;
    
    showLoading('추천 플레이리스트 생성 중...');
    hideResult();
    document.getElementById('playlistSection').classList.add('hidden');
    
    try {
        const res = await fetch('/api/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                seed_title: selectedSeed.title,
                seed_artist: selectedSeed.artist,
                count: count
            })
        });
        
        const data = await res.json();
        
        if (data.error) {
            showResult(data.error, 'error');
            hideLoading();
            return;
        }
        
        currentPlaylist = data.playlist;
        renderPlaylist(data.seed, data.playlist);
        
        // YouTube Music 매칭
        showLoading('YouTube Music 곡 매칭 중...');
        await matchYTMusic(data.playlist);
        
    } catch (e) {
        showResult(`오류: ${e.message}`, 'error');
    }
    
    hideLoading();
}

function renderPlaylist(seed, playlist) {
    const section = document.getElementById('playlistSection');
    const body = document.getElementById('playlistBody');
    
    // 플레이리스트 이름 기본값
    document.getElementById('playlistName').value = `${seed.title} 기반 추천`;
    
    body.innerHTML = playlist.map((track, i) => `
        <div class="track-row" data-idx="${i}">
            <span class="track-num">${i + 1}</span>
            <div>
                <div class="track-title">${escapeForDisplay(track.title)}</div>
                <div class="track-artist">${escapeForDisplay(track.artist)}</div>
            </div>
            <span class="track-sim">${(track.similarity * 100).toFixed(0)}%</span>
            <span class="track-status pending" id="status-${i}">⏳</span>
        </div>
    `).join('');
    
    section.classList.remove('hidden');
}

function escapeForDisplay(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ── YouTube Music 매칭 ──
async function matchYTMusic(playlist) {
    const res = await fetch('/api/search_ytmusic', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ songs: playlist })
    });
    
    const results = await res.json();
    matchedVideoIds = [];
    let matchCount = 0;
    
    results.forEach((r, i) => {
        const el = document.getElementById(`status-${i}`);
        if (r.matched) {
            el.textContent = '✅';
            el.className = 'track-status matched';
            matchedVideoIds.push(r.videoId);
            matchCount++;
        } else {
            el.textContent = '❌';
            el.className = 'track-status failed';
        }
    });
    
    const btn = document.getElementById('ytmBtn');
    btn.disabled = matchedVideoIds.length === 0;
    btn.textContent = `▶ YouTube Music에 등록 (${matchCount}곡)`;
}

// ── YouTube Music 등록 ──
async function pushToYTMusic() {
    if (matchedVideoIds.length === 0) return;
    
    const name = document.getElementById('playlistName').value || '추천 플레이리스트';
    const btn = document.getElementById('ytmBtn');
    btn.disabled = true;
    btn.textContent = '등록 중...';
    
    try {
        const res = await fetch('/api/create_playlist', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: name,
                description: `${selectedSeed.title} 기반 추천 플레이리스트 (음악 추천 엔진)`,
                video_ids: matchedVideoIds
            })
        });
        
        const data = await res.json();
        
        if (data.success) {
            showResult(
                `✅ 플레이리스트 "${name}" 생성 완료! (${data.count}곡) — ` +
                `<a href="${data.url}" target="_blank">YouTube Music에서 열기 →</a>`,
                'success'
            );
            btn.textContent = '✅ 등록 완료!';
        } else {
            showResult(`❌ 오류: ${data.error}`, 'error');
            btn.disabled = false;
            btn.textContent = `▶ YouTube Music에 등록 (${matchedVideoIds.length}곡)`;
        }
    } catch (e) {
        showResult(`❌ 오류: ${e.message}`, 'error');
        btn.disabled = false;
        btn.textContent = `▶ YouTube Music에 등록 (${matchedVideoIds.length}곡)`;
    }
}

// ── 유틸 ──
function showLoading(text) {
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('loadingText').textContent = text;
}

function hideLoading() {
    document.getElementById('loading').classList.add('hidden');
}

function showResult(msg, type) {
    const el = document.getElementById('resultMessage');
    el.innerHTML = msg;
    el.className = `result-message ${type}`;
    el.classList.remove('hidden');
}

function hideResult() {
    document.getElementById('resultMessage').classList.add('hidden');
}
