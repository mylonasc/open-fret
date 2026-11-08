// Read-only import-completeness audit. Each JSON input line: {name, tex}.
const at = require('@coderline/alphatab');
const readline = require('node:readline');
at.Logger.logLevel = at.LogLevel.None;
readline.createInterface({input: process.stdin, crlfDelay: Infinity}).on('line', line => {
  let name;
  try {
    const input = JSON.parse(line); name = input.name;
    const score = at.importer.ScoreLoader.loadScoreFromBytes(
      new TextEncoder().encode(input.tex), new at.Settings());
    const groups = [];
    for (const track of score.tracks) for (const staff of track.staves)
      for (const bar of staff.bars) for (const voice of bar.voices) for (const beat of voice.beats)
        if (beat.notes.length) groups.push({tick: beat.absolutePlaybackStart / 2,
          duration: beat.playbackDuration / 2, pitches: beat.notes.map(n => n.realValue)});
    process.stdout.write(JSON.stringify({name, ok: true, groups}) + '\n');
  } catch (e) { process.stdout.write(JSON.stringify({name, ok: false, error: String(e.message || e)}) + '\n'); }
});
