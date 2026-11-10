import { useEffect, useRef } from 'react';

// Mounts one repo TabDataViewer bundle into a div. The viewer UMD (with its
// own React) is loaded once in index.html; we only mount/unmount here.
export default function ViewerPanel({ bundle, title, side }) {
  const hostRef = useRef(null);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return undefined;
    if (typeof window === 'undefined' || !window.TabDataViewer || typeof window.TabDataViewer.mount !== 'function') {
      host.innerHTML = '<p><b>Viewer failed to load.</b> Is the viewer bundle served?</p>';
      return undefined;
    }
    try {
      window.TabDataViewer.mount(host, bundle);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error(err);
    }
    return () => {
      // The viewer component tears down its own player + roll (including
      // cancelling WaveRoll's throttled frame). Just drop the DOM; calling
      // pause/stop/destroy again here races with that teardown.
      try {
        host.innerHTML = '';
      } catch (_) {
        /* noop */
      }
    };
  }, [bundle]);

  return (
    <div className="panel">
      <div className="panel-head">
        <span className={`side-badge side-${side}`}>{side === 'left' ? 'A' : 'B'}</span>
        <span className="panel-title">{title}</span>
      </div>
      <div ref={hostRef} className="ofv-host" />
    </div>
  );
}
