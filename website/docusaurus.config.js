// @ts-check

import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Open-Fret',
  tagline: 'MIDI-to-guitar-tablature with verifiable decoding',
  favicon: 'img/favicon.svg',

  // Served as a GitHub project page. Keep in sync with the repository name;
  // local `npm start` serves the same paths.
  url: 'https://mylonasc.github.io',
  baseUrl: '/open-fret/',

  organizationName: 'mylonasc',
  projectName: 'open-fret',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-nB0miv6/jRmo5UMMR1wu3Gz6NLsoTkbqJghGIsx//Rlm+ZU03BU6SQNC66uf4l5+',
      crossorigin: 'anonymous',
    },
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: 'Open-Fret',
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'docs',
            position: 'left',
            label: 'Docs',
          },
          {
            href: 'https://github.com/mylonasc/open-fret',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {label: 'Introduction', to: '/docs/intro'},
              {label: 'Dataset', to: '/docs/dataset'},
              {label: 'Losses', to: '/docs/losses'},
            ],
          },
          {
            title: 'Project',
            items: [
              {label: 'HPO insights', to: '/docs/hpo'},
              {label: 'Training', to: '/docs/training'},
              {label: 'Examples', to: '/docs/examples'},
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/mylonasc/open-fret',
              },
              {
                label: 'Interactive demo',
                href: 'https://mylonasc.github.io/open-fret/examples/player.html',
              },
            ],
          },
        ],
        copyright: `Open-Fret docs are MIT to the extent the contributors own the rights. Third-party data and components retain their own terms.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

export default config;
