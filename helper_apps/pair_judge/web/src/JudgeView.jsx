import { useCallback, useEffect, useRef, useState } from 'react';
import ViewerPanel from './ViewerPanel.jsx';
import { fetchPair, sendSkip, sendVote } from './api.js';

function useJudgeName() {
  const [judge, setJudge] = useState(() => {
    try {
      return window.localStorage.getItem('ofv-judge') || '';
    } catch (_) {
      return '';
    }
  });
  const update = (v) => {
    setJudge(v);
    try {
      window.localStorage.setItem('ofv-judge', v);
    } catch (_) { /* noop */ }
  };
  return [judge, update];
}

export default function JudgeView({ onVoted }) {
  const [pair, setPair] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  // voted: {reveal, winner} after voting; the pair stays on screen with an
  // explicit Next button so judges can inspect the reveal before moving on.
  const [voted, setVoted] = useState(null);
  const [notes, setNotes] = useState('');
  const [judge, setJudge] = useJudgeName();
  const busyRef = useRef(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    setVoted(null);
    try {
      const p = await fetchPair();
      setPair(p.done ? null : p);
    } catch (e) {
      setError(String(e && e.message ? e.message : e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const act = async (kind, winner) => {
    if (!pair || busyRef.current || voted) return;
    busyRef.current = true;
    try {
      if (kind === 'vote') {
        const res = await sendVote({ pair_id: pair.pair_id, winner, notes, judge: judge || 'anon' });
        setVoted({ reveal: res.reveal, winner });
        if (onVoted) onVoted();
      } else {
        await sendSkip({ pair_id: pair.pair_id, judge: judge || 'anon' });
        setNotes('');
        await load();
      }
    } catch (e) {
      setError(String(e && e.message ? e.message : e));
    } finally {
      busyRef.current = false;
    }
  };

  const next = async () => {
    setNotes('');
    await load();
  };

  if (loading) return <p>Loading a pair…</p>;
  if (error) return <p className="error">Error: {error}</p>;
  if (!pair) {
    return (
      <div className="done">
        <h2>Queue empty — all pairs judged.</h2>
        <p>Check the Results tab for the Elo standings, or export the votes for preference training.</p>
        <button type="button" onClick={load}>Reload</button>
      </div>
    );
  }

  return (
    <div>
      <div className="judge-bar">
        <span>
          Pair <b>{pair.pair_id}</b> · <b>{pair.song}</b>
        </span>
        <span>
          {pair.total - pair.remaining}/{pair.total} judged
        </span>
        <label className="judge-name">
          Judge
          <input value={judge} onChange={(e) => setJudge(e.target.value)} placeholder="anon" />
        </label>
      </div>
      {voted && (
        <div className="reveal">
          A was <b>{voted.reveal.left}</b> · B was <b>{voted.reveal.right}</b>
          {voted.winner !== 'tie' && (
            <span> — you picked <b>{voted.winner === 'left' ? 'A' : 'B'}</b></span>
          )}
          {voted.winner === 'tie' && <span> — recorded as a tie</span>}
          {' '}
          <button type="button" onClick={next}>Next pair →</button>
        </div>
      )}
      <div className="panels">
        <ViewerPanel bundle={pair.left.bundle} title="Version A" side="left" />
        <ViewerPanel bundle={pair.right.bundle} title="Version B" side="right" />
      </div>
      <div className="vote-bar">
        <button type="button" className="vote-a" disabled={!!voted} onClick={() => act('vote', 'left')}>
          A better
        </button>
        <button type="button" className="vote-tie" disabled={!!voted} onClick={() => act('vote', 'tie')}>
          Tie
        </button>
        <button type="button" className="vote-b" disabled={!!voted} onClick={() => act('vote', 'right')}>
          B better
        </button>
        <button type="button" className="vote-skip" onClick={() => act('skip')}>
          Skip
        </button>
        <input
          className="notes"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="Optional note (e.g. B stretches, A cleaner shifts)"
        />
      </div>
    </div>
  );
}
