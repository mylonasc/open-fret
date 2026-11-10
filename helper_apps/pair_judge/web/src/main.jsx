import { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import JudgeView from './JudgeView.jsx';
import ResultsView from './ResultsView.jsx';
import './styles.css';

function useSoundfont() {
  // Fetch the vendored SONiVOX soundfont once and hand the viewer a blob
  // URL, which it prefers over the CDN default (see window.__OFV_SOUNDFONT_URL).
  const [ready, setReady] = useState(false);
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch('/vendor/lib/sonivox.sf2');
        if (!res.ok) throw new Error(`soundfont ${res.status}`);
        const blob = await res.blob();
        if (!cancelled) window.__OFV_SOUNDFONT_URL = URL.createObjectURL(blob);
      } catch (err) {
        // eslint-disable-next-line no-console
        console.warn('Local soundfont unavailable, falling back to CDN default.', err);
      } finally {
        if (!cancelled) setReady(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);
  return ready;
}

function App() {
  const [tab, setTab] = useState('judge');
  const audioReady = useSoundfont();

  return (
    <div className="app">
      <header className="topbar">
        <h1>Open-Fret pair judge</h1>
        <nav>
          <button type="button" className={tab === 'judge' ? 'active' : ''} onClick={() => setTab('judge')}>
            Judge
          </button>
          <button type="button" className={tab === 'results' ? 'active' : ''} onClick={() => setTab('results')}>
            Results
          </button>
        </nav>
      </header>
      <main>
        {!audioReady && <p>Preparing audio…</p>}
        {audioReady && tab === 'judge' && <JudgeView />}
        {audioReady && tab === 'results' && <ResultsView />}
      </main>
      <footer>
        Blind A/B judging for HPO comparisons: trial identities are hidden until you vote. Ties and skips are recorded.
      </footer>
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);
