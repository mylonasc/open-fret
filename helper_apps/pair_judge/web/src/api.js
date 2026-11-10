export async function fetchPair() {
  const r = await fetch('/api/next');
  if (!r.ok) throw new Error(`next failed: ${r.status}`);
  return r.json();
}

export async function sendVote({ pair_id, winner, notes, judge }) {
  const r = await fetch('/api/vote', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pair_id, winner, notes, judge }),
  });
  if (!r.ok) throw new Error(`vote failed: ${r.status}`);
  return r.json();
}

export async function sendSkip({ pair_id, judge }) {
  const r = await fetch('/api/skip', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pair_id, judge }),
  });
  if (!r.ok) throw new Error(`skip failed: ${r.status}`);
  return r.json();
}

export async function fetchResults() {
  const r = await fetch('/api/results');
  if (!r.ok) throw new Error(`results failed: ${r.status}`);
  return r.json();
}

export function downloadVotes() {
  window.location.href = '/api/export';
}
