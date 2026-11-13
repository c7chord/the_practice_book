#!/usr/bin/env python3
"""Find where a chord sheet sits in a recording → the sheet's "원곡 싱크" anchors.

    python3 tools/sync_parse.py SHEET.chordsheet.json --url https://youtu.be/ID   [-o OUT.json]
    python3 tools/sync_parse.py SHEET.chordsheet.json --audio song.m4a --video ID [-o OUT.json]

How it works
  1. beat-track the recording (librosa) and take a chroma vector per beat
  2. expand the sheet into the same beat grid (sheet bar = its beats × tempo ratio, ratio 0.5 / 1 / 2 picked from the tempos)
  3. subsequence-DTW the sheet's chord templates against the audio chroma — this skips intros/outros the sheet
     doesn't have and absorbs sections whose bar counts are a little off
  4. read off where bar 1, every section start and the end of the last bar land → anchors (seconds in the video)

Needs: pip install librosa soundfile yt-dlp  (and ffmpeg for --url)
"""
import argparse, json, math, os, re, subprocess, sys, tempfile

import numpy as np

NOTE = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


# ---------- sheet → beat-level chord templates (mirrors the app's splitBar / parseTS) ----------
def parse_ts(s):
    n, d = (list(map(int, str(s or '4/4').split('/'))) + [4])[:2]
    if d == 8 and n % 3 == 0 and n > 3:
        return n // 3
    return n or 4


def split_bar(text):
    t = (text or '').replace('♯', '#').replace('♭', 'b')
    t = re.sub(r'[()\[\]{}]', ' ', t)
    t = re.sub(r'-(?=[A-G])', ' ', t)
    t = re.sub(r',(?=\s|$)', ' ', t).strip()
    return t.split() if t else []


def chord_pcs(tok):
    """pitch classes of a chord symbol (enough for chroma matching), or None for N.C./unknown"""
    m = re.match(r'^([A-G])([#b]?)(.*)$', tok)
    if not m:
        return None
    root = (NOTE[m.group(1)] + {'#': 1, 'b': -1, '': 0}[m.group(2)]) % 12
    q, bass = m.group(3), None
    if '/' in q:
        q, b = q.split('/', 1)
        mb = re.match(r'^([A-G])([#b]?)', b)
        if mb:
            bass = (NOTE[mb.group(1)] + {'#': 1, 'b': -1, '': 0}[mb.group(2)]) % 12
    ql = q.lower()
    iv = {0, 4, 7}
    if re.match(r'^(m(?!aj)|min|-)', q):
        iv = {0, 3, 7}
    if 'dim' in ql or q.startswith(('°', 'o')):
        iv = {0, 3, 6} | ({9} if '7' in q else set())
    if 'm7b5' in ql or 'ø' in q:
        iv = {0, 3, 6, 10}
    if 'aug' in ql or q.startswith('+'):
        iv = {0, 4, 8}
    if 'sus4' in ql or ql in ('sus',):
        iv = (iv - {3, 4}) | {5}
    if 'sus2' in ql:
        iv = (iv - {3, 4}) | {2}
    if re.search(r'(maj|M|Δ)(7|9|11|13)', q):
        iv |= {11}
    elif re.search(r'(^|[^#b])(7|9|11|13)', q) and not ({3, 6, 9} <= iv):
        iv |= {10}
    if re.search(r'(^|[^#b\d])6', q):
        iv |= {9}
    for acc, deg in re.findall(r'([#b])(5|9|11|13)', q):
        iv |= {({'5': 7, '9': 2, '11': 5, '13': 9}[deg] + (1 if acc == '#' else -1)) % 12}
    if re.search(r'add9|(^|[^#b\d])9', q):
        iv |= {2}
    pcs = {(root + i) % 12 for i in iv}
    if bass is not None:
        pcs.add(bass)
    return root, pcs


def sheet_beats(song):
    """→ list of (template or None) per sheet beat, and bar → first-beat index"""
    beats, bar_start, prev, last = [], [], [], None
    for sec in song['sections']:
        nb = parse_ts(sec.get('timeSig') or song.get('timeSig') or '4/4')
        for txt in sec['bars']:
            toks = split_bar(txt)
            if toks == ['%']:
                toks = prev
            else:
                prev = toks
            bar_start.append(len(beats))
            slots = []
            for t in toks:
                if t in ('/', '.', '_', '-', '//'):
                    slots.append(last)
                elif re.match(r'^N\.?C\.?$', t, re.I):
                    last = None
                    slots.append(None)
                else:
                    c = chord_pcs(t)
                    last = c
                    slots.append(c)
            for k in range(nb):
                beats.append(slots[min(len(slots) - 1, k * len(slots) // nb)] if slots else last)
    bar_start.append(len(beats))
    return beats, bar_start


def template(c):
    v = np.zeros(12)
    if c is None:
        return v
    root, pcs = c
    for p in pcs:
        v[p] = 1.0
    v[root] += 0.5
    return v / np.linalg.norm(v)


# ---------- audio ----------
def download(url, out_dir):
    tpl = os.path.join(out_dir, 'audio.%(ext)s')
    subprocess.run(['yt-dlp', '-q', '--no-playlist', '-f', 'bestaudio', '-x', '--audio-format', 'wav', '-o', tpl, url], check=True)
    return os.path.join(out_dir, 'audio.wav')


def video_id(url):
    m = re.search(r'(?:v=|youtu\.be/|embed/|shorts/)([\w-]{6,20})', url or '')
    return m.group(1) if m else ''


def analyse(audio_path, song, ratio=None):
    import librosa
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    hop = 512
    y_h, y_p = librosa.effects.hpss(y)
    tempo, beat_frames = librosa.beat.beat_track(y=y_p, sr=sr, hop_length=hop, trim=False)
    tempo = float(np.atleast_1d(tempo)[0])
    beat_t = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
    chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr, hop_length=hop)
    # column 0 = before the first beat, column j = [beat j-1, beat j), last = after the last beat
    bsync = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
    bounds = np.concatenate([[0.0], beat_t, [len(y) / sr]])                      # start of column j = bounds[j]
    bsync = bsync[:, :len(bounds) - 1]
    A = bsync / (np.linalg.norm(bsync, axis=0, keepdims=True) + 1e-9)

    sb, bar_start = sheet_beats(song)
    sheet_bpm = float(song.get('bpm') or 100)
    if ratio is None:   # audio beats per sheet beat: the sheet may be written in half/double time
        ratio = min((0.5, 1, 2), key=lambda r: abs(math.log(tempo / (sheet_bpm * r))))
    # sheet at audio-beat resolution
    if ratio >= 1:
        q = [c for c in sb for _ in range(int(ratio))]
        q_bar = [b * int(ratio) for b in bar_start]
    else:
        q = sb[::2]
        q_bar = [b // 2 for b in bar_start]
    X = np.stack([template(c) for c in q], axis=1)                               # 12 × sheet beats
    has = np.array([c is not None for c in q])

    C = 1 - X.T @ A                                                              # cost: 1 − cosine
    C[~has, :] = 0.5                                                             # N.C. matches anything weakly
    subseq = X.shape[1] < A.shape[1]
    D, wp = librosa.sequence.dtw(C=C, subseq=subseq,
                                 step_sizes_sigma=np.array([[1, 1], [1, 2], [2, 1]]),
                                 weights_add=np.array([0, 0.15, 0.15]))
    wp = wp[::-1]
    first, last = {}, {}
    for i, j in wp:
        first.setdefault(int(i), int(j))
        last[int(i)] = int(j)
    def audio_time_of(qi):
        if qi >= len(q):                     # end of the last bar = end of the audio beat its last beat ends on
            j = last.get(len(q) - 1)
            return float(bounds[min(j + 1, len(bounds) - 1)]) if j is not None else None
        j = first.get(qi)
        return float(bounds[j]) if j is not None else None
    sim = float(np.mean([1 - C[i, j] for i, j in wp if has[i]])) if has.any() else 0.0
    # harmonic change curve: 1 − cos(chroma just before, chroma just after) — peaks where the chord changes (≈ downbeats)
    fr_t = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)
    w = max(4, int(0.35 * sr / hop))
    cs = np.concatenate([np.zeros((12, 1)), np.cumsum(chroma, axis=1)], axis=1)
    n = chroma.shape[1]
    nov = np.zeros(n)
    for f in range(w, n - w):
        a_, b_ = cs[:, f] - cs[:, f - w], cs[:, f + w] - cs[:, f]
        nov[f] = 1 - a_ @ b_ / (np.linalg.norm(a_) * np.linalg.norm(b_) + 1e-9)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, fmax=250, n_mels=32)   # low end: kick + bass mark downbeats
    return dict(tempo=tempo, ratio=ratio, beat_t=beat_t, q_bar=q_bar, time_of=audio_time_of, sim=sim, path=wp, C=C, has=has,
                nov=nov, fr_t=fr_t, beat_len=60 / tempo, onset=onset, onset_floor=float(np.median(onset)))


def refine(times, res):
    """move each anchor onto the real downbeat near it (±0.6 beat): the moment where the chord changes AND the low end
    (kick/bass) hits. Where the chord doesn't change there, borrow the shift of the nearest anchor that did, then land on
    a clear low-end hit if one is right there. (The beat tracker's phase can be off by a fraction of a beat.)"""
    L = min(len(res['nov']), len(res['onset']))
    nov, on, ft = res['nov'][:L], res['onset'][:L], res['fr_t'][:L]
    win = 0.6 * res['beat_len']
    shifts = []
    for t in times:
        m = (ft >= t - win) & (ft <= t + win) if t is not None else None
        if t is None or not m.any() or nov[m].max() <= 0.12:
            shifts.append(None); continue
        score = (nov / nov[m].max()) * (0.3 + on / (on[m].max() + 1e-9))
        k = int(np.argmax(np.where(m, score, -1)))
        shifts.append(float(ft[k] - t))
    out = []
    for i, t in enumerate(times):
        if t is None:
            out.append(None); continue
        if shifts[i] is not None:
            out.append(t + shifts[i]); continue
        near = sorted((abs(j - i), shifts[j]) for j in range(len(times)) if shifts[j] is not None)
        t += near[0][1] if near else 0.0
        m = (ft >= t - 0.08) & (ft <= t + 0.08)
        if m.any():
            k = int(np.argmax(np.where(m, on, -1)))
            if on[k] > 3 * res['onset_floor']:
                t = float(ft[k])
        out.append(t)
    return out


def anchors_for(song, res):
    total = sum(len(s['bars']) for s in song['sections'])
    want, fi = [0], 0
    for s in song['sections']:
        if fi:
            want.append(fi)
        fi += len(s['bars'])
    want.append(total)
    fis = sorted(set(want))
    times = refine([res['time_of'](res['q_bar'][fi]) for fi in fis], res)
    out = []
    for fi, t in zip(fis, times):
        if t is not None and (not out or t > out[-1]['t'] + 0.2):
            out.append({'fi': fi, 't': round(max(0.0, t), 3), 'auto': True})
    return out


def section_scores(song, res):
    C, has, wp = res['C'], res['has'], res['path']
    per = {}
    fi = 0
    for si, s in enumerate(song['sections']):
        a, b = res['q_bar'][fi], res['q_bar'][fi + len(s['bars'])]
        vals = [1 - C[i, j] for i, j in wp if a <= i < b and has[i]]
        per[si] = float(np.mean(vals)) if vals else None
        fi += len(s['bars'])
    return per


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sheet')
    ap.add_argument('--url')
    ap.add_argument('--audio')
    ap.add_argument('--video', help='YouTube id to store (default: from --url)')
    ap.add_argument('--ratio', type=float, help='audio beats per sheet beat (default: guessed from tempos)')
    ap.add_argument('-o', '--out')
    a = ap.parse_args()
    song = json.load(open(a.sheet, encoding='utf-8'))
    with tempfile.TemporaryDirectory() as tmp:
        path = a.audio or download(a.url, tmp)
        res = analyse(path, song, a.ratio)
    anchors = anchors_for(song, res)
    vid = a.video or video_id(a.url or '') or (song.get('sync') or {}).get('video', '')
    per = section_scores(song, res)
    song['sync'] = {'video': vid, 'anchors': anchors,
                    'src': f"자동 분석 · 일치도 {res['sim']:.2f}"}
    out = a.out or a.sheet
    json.dump(song, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f"tempo {res['tempo']:.1f} BPM (audio) · sheet {song.get('bpm')} BPM · ratio {res['ratio']}")
    print(f"chord match {res['sim']:.2f} {'(good)' if res['sim'] >= 0.6 else '(weak — check by ear)'}")
    fi = 0
    for si, s in enumerate(song['sections']):
        t = next((x['t'] for x in anchors if x['fi'] == fi), None)
        sc = per.get(si)
        print(f"  {s.get('form') or 'section'}: bar {fi + 1} at {t if t is not None else '—'}s · match {sc:.2f}" if sc is not None else f"  {s.get('form')}: bar {fi + 1}")
        fi += len(s['bars'])
    print(f"  end: {next((x['t'] for x in anchors if x['fi'] == fi), '—')}s")
    print('wrote', out)


if __name__ == '__main__':
    main()
