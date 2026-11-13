#!/usr/bin/env node
/* Check a .chordsheet.json in the real app (chord-sheet.html) — run before handing a sheet over.

     node tools/check_sheet.js "Song - Artist.chordsheet.json" [--png shot.png]

   Prints: chords the app can't read (must be 0), bar count and length at the sheet's BPM,
   the app's own analysis comments (best reading + ranked alternatives), and the guitar voicing
   chosen for every chord with the notes it actually sounds (tuning + capo applied).
   Needs Playwright with Chromium (npm i playwright && npx playwright install chromium).
   Exit code 1 when a chord can't be read or the page throws. */
const path = require('path'), fs = require('fs');
const { chromium } = require('playwright');
const args = process.argv.slice(2), file = args.find(a => !a.startsWith('--'));
const png = args.includes('--png') ? args[args.indexOf('--png') + 1] : null;
if (!file) { console.error('usage: node tools/check_sheet.js SHEET.chordsheet.json [--png out.png]'); process.exit(2); }
const app = 'file://' + path.resolve(__dirname, '..', 'chord-sheet.html');
(async () => {
  const raw = JSON.parse(fs.readFileSync(file, 'utf8'));
  const b = await chromium.launch(), p = await b.newPage({ viewport: { width: 1200, height: 900 } });
  const errs = []; p.on('pageerror', e => errs.push(e.message));
  await p.goto(app); await p.waitForTimeout(300);
  const r = await p.evaluate(raw => {
    prefs.showRoman = true; prefs.gtr = true; prefs.staff = true;
    const s = normalizeSong(raw); LIB[s.id] = s; openSheet(s.id);
    const N = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'];
    const bad = [...document.querySelectorAll('.ch.bad')].map(e => e.textContent);
    const an = analyzeSong(song), plan = gtrPlan(), seen = new Map();
    chordOccurrences().forEach(o => { const v = plan.get(o.fi + ':' + o.ti); if (v && !seen.has(o.c.sym)) seen.set(o.c.sym, `${fretsStr(v.f)}${v.fixed ? ' (fixed)' : ''}  sounds ${gtrNotes(v.f).map(m => N[m % 12]).join(' ')}`); });
    const beats = flatList().reduce((a, { si }) => a + parseTS(song.sections[si].timeSig || song.timeSig).beats, 0);
    return {
      title: `${song.title} — ${song.artist}`, key: an.key ? an.key.name + (an.est ? ' (estimated)' : '') : '—', bpm: song.bpm,
      bars: flatCount(), secs: Math.round(beats * 60 / song.bpm), sections: song.sections.map(x => `${x.form}(${x.bars.length})`).join(' '),
      guitar: song.guitar || null, bad,
      notes: an.notes.map(n => `${n.fi + 1}${n.fe !== n.fi ? '–' + (n.fe + 1) : ''}마디 [${n.tag}] ${n.title}` + (n.alts || []).map((a, i) => `\n      ${i + 2}순위 [${a.tag}] ${a.title}`).join('')),
      voicings: [...seen].map(([k, v]) => `${k.padEnd(10)} ${v}`)
    };
  }, raw);
  await p.waitForTimeout(200);
  if (png) { await p.evaluate(() => { cellEl(0).scrollIntoView({ block: 'start' }); scrollBy(0, -150); }); await p.screenshot({ path: png }); }
  console.log(`${r.title}\nkey ${r.key} · ${r.bpm} BPM · ${r.bars} bars ≈ ${Math.floor(r.secs / 60)}:${String(r.secs % 60).padStart(2, '0')} at that tempo\nsections ${r.sections}`);
  if (r.guitar) console.log('guitar', JSON.stringify(r.guitar));
  console.log(`\nunreadable chords: ${r.bad.length ? r.bad.join(', ') : '0 ✓'}`);
  console.log('\napp analysis:' + (r.notes.length ? '\n  ' + r.notes.join('\n  ') : ' (all diatonic — no comments)'));
  console.log('\nguitar voicings (low→high, frets above the capo):\n  ' + r.voicings.join('\n  '));
  if (errs.length) console.log('\npage errors:', errs);
  await b.close();
  process.exit(r.bad.length || errs.length ? 1 : 0);
})();
