import { useCallback, useEffect, useState } from 'react';
import { downloadVotes, fetchResults } from './api.js';

export default function ResultsView() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      setData(await fetchResults());
    } catch (e) {
      setError(String(e && e.message ? e.message : e));
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  if (error) return <p className="error">Error: {error}</p>;
  if (!data) return <p>Loading results…</p>;

  const rows = [...(data.trials || [])].sort((a, b) => b.elo - a.elo);
  return (
    <div>
      <div className="results-head">
        <span>
          <b>{data.votes}</b> votes · <b>{data.skipped}</b> skipped · {data.pairs} pairs
        </span>
        <span>
          <button type="button" onClick={load}>Refresh</button>{' '}
          <button type="button" onClick={downloadVotes}>Download votes JSONL</button>
        </span>
      </div>
      <table className="elo">
        <thead>
          <tr>
            <th>#</th>
            <th>Trial</th>
            <th>Elo</th>
            <th>Wins</th>
            <th>Decided</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.trial}>
              <td>{i + 1}</td>
              <td className="mono">{r.trial}</td>
              <td>{r.elo}</td>
              <td>{r.wins}</td>
              <td>{r.decided}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="hint">
        Tally: {Object.entries(data.tally || {}).map(([k, v]) => `${k}: ${v}`).join(' · ') || 'no votes yet'}
      </p>
    </div>
  );
}
