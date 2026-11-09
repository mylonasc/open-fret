// Independent AlphaTab import; no repository token/event parser is used.
const fs = require('node:fs');
const at = require('./alphatex_parser/node_modules/@coderline/alphatab');
const settings = new at.Settings();
const score = at.importer.ScoreLoader.loadScoreFromBytes(
  new TextEncoder().encode(fs.readFileSync(process.argv[2], 'utf8')), settings);
const generator = new at.midi.MidiFileGenerator(score, settings, {}), notes = [];
for (const track of score.tracks) for (const staff of track.staves)
  for (const bar of staff.bars) for (const voice of bar.voices)
    for (const beat of voice.beats) for (const note of beat.notes) {
      if (note.isTieDestination) continue;
      const duration = generator.getNoteDuration(note, beat.playbackDuration);
      notes.push({start: beat.absolutePlaybackStart / 2, pitch: note.realValue,
        end: (beat.absolutePlaybackStart + Math.max(duration.untilTieOrSlideEnd, duration.letRingEnd)) / 2});
    }
process.stdout.write(JSON.stringify(notes));
