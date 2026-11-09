// JSON-lines worker. AlphaTab owns the grammar and note-duration semantics.
const readline = require('node:readline');
const at = require('@coderline/alphatab');
at.Logger.logLevel = at.LogLevel.None;

function parse(content) {
  // Corpus octave names are one above scientific notation. Normalize BEFORE
  // importing so default tunings, string numbering and realValue stay native.
  content = content.replace(/^\\tuning\s+([^\r\n]+)/gm, (_, values) =>
    '\\tuning ' + values.trim().split(/\s+/).map(value => {
      if (!/^[A-G][#b]?-?\d+$/.test(value)) throw Error('Unsupported tuning: ' + value);
      return value.replace(/(-?\d+)$/, octave => String(Number(octave) - 1));
    }).join(' '));
  content = content.replace(/^\\track[ \t]*(?=\r?$)/gm, '\\track "Guitar"')
    .replace(/^\\frets[^\r\n]*(?:\r?\n|$)/gm, '');
  const settings = new at.Settings();
  settings.notation.notationMode = at.NotationMode.GuitarPro;
  const score = at.importer.ScoreLoader.loadScoreFromBytes(new TextEncoder().encode(content), settings);
  if (score.masterBars.some(bar => bar.tempoAutomation && bar.tempoAutomation.value !== score.tempo))
    throw Error('Variable tempo is not supported by single-tempo viewer exports');
  if (score.masterBars.some(bar => bar.isRepeatStart || bar.repeatCount > 0))
    throw Error('Repeat expansion is not supported by the training representation');
  const generator = new at.midi.MidiFileGenerator(score, settings, {});
  const tracks = [];
  const diagnostics = {articulations_not_targeted: {}, skipped_unpitched_notes: 0, skipped_staves: 0};
  const count = key => diagnostics.articulations_not_targeted[key] =
    (diagnostics.articulations_not_targeted[key] || 0) + 1;
  for (const track of score.tracks) for (const staff of track.staves) {
    if (staff.isPercussion || !staff.tuning.length) { diagnostics.skipped_staves++; continue; }
    const events = [];
    for (const bar of staff.bars) for (const voice of bar.voices) for (const beat of voice.beats) {
      if (beat.automations.some(a => a.type === at.model.AutomationType.Tempo && a.value !== score.tempo))
        throw Error('Variable tempo is not supported by single-tempo viewer exports');
      for (const note of beat.notes) {
        if (!note.isStringed || note.fret < 0 || note.isDead) { diagnostics.skipped_unpitched_notes++; continue; }
        if (note.fret > 24) throw Error('Fret exceeds supported maximum 24');
        for (const flag of ['isPalmMute', 'isLetRing', 'isStaccato', 'isHammerPullOrigin',
                            'hasBend', 'isTrill', 'isGhost', 'isHarmonic']) if (note[flag]) count(flag);
        if (note.slideInType || note.slideOutType) count('slide');
        if (note.vibrato || beat.vibrato) count('vibrato');
        if (note.isTieDestination || (note.slideOrigin && note.slideOrigin.slideOutType === at.model.SlideOutType.Legato))
          continue;
        // Same duration calculation as AlphaTab's MIDI playback, including
        // ties, let-ring, palm-mute and staccato. No rearticulated tie targets.
        const duration = generator.getNoteDuration(note, beat.playbackDuration);
        const ticks = Math.round(Math.max(duration.untilTieOrSlideEnd, duration.letRingEnd) / 2);
        const start = Math.round(beat.absolutePlaybackStart / 2);
        if (start < 0 || ticks <= 0) throw Error('Unsupported negative/zero note timing');
        const string = staff.tuning.length - note.string + 1;
        if (staff.tuning[string - 1] + note.fret !== note.realValue)
          throw Error('Pitch transposition/harmonic not representable by tuning + fret');
        events.push({start, duration: ticks, voice: voice.index,
                     notes: [{string, fret: note.fret, tech: null}]});
      }
    }
    events.sort((a, b) => a.start - b.start);
    if (events.length) tracks.push({tuning: staff.tuning, tempo: score.tempo, events,
      program: track.playbackInfo.program, name: track.name,
      voices: Math.max(...staff.bars.map(b => b.voices.length))});
  }
  return {tracks, diagnostics};
}

readline.createInterface({input: process.stdin, crlfDelay: Infinity}).on('line', line => {
  try { process.stdout.write(JSON.stringify({ok: true, ...parse(JSON.parse(line).content)}) + '\n'); }
  catch (error) { process.stdout.write(JSON.stringify({ok: false, error: String(error.message || error)}) + '\n'); }
});
