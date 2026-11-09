import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

const FEATURES = [
  {
    title: 'Grounded decoding',
    description:
      'Anchored generation emits one fingering slot per input attack, with pitch-equivalent masking and per-onset string uniqueness — the tab cannot invent notes.',
  },
  {
    title: 'Domain-aware losses',
    description:
      'Cross-entropy plus a pitch marginal, a conditional variant, a macro-averaged event-type term, and equivalence smoothing over same-pitch fingerings.',
  },
  {
    title: 'Verifiable by construction',
    description:
      'Strict AlphaTex parsing against AlphaTab 1.3.0, frozen validation panels, playable viewers, and an independent loss/label audit that recomputes the objective.',
  },
];

function Hero() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link className="button button--secondary button--lg" to="/docs/intro">
            Read the docs
          </Link>
          <Link className="button button--secondary button--lg" to="/docs/examples">
            Hear the examples
          </Link>
        </div>
      </div>
    </header>
  );
}

function Features() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FEATURES.map((f) => (
            <div key={f.title} className={clsx('col col--4')}>
              <div className="text--center padding-horiz--md">
                <Heading as="h3">{f.title}</Heading>
                <p>{f.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description="MIDI-to-guitar-tablature with verifiable decoding">
      <Hero />
      <main>
        <Features />
      </main>
    </Layout>
  );
}
