import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // Force the production constant: bare `process` does not exist in
  // browsers and some bundled deps guard on process.env.NODE_ENV.
  define: { "process.env.NODE_ENV": JSON.stringify("production") },
  build: {
    lib: {
      entry: "src/index.jsx",
      name: "TabDataViewer",
      fileName: () => "tab-data-viewer.umd.js",
      formats: ["umd"],
    },
    outDir: "dist",
    emptyOutDir: true,
    // React + ReactDOM are bundled in so notebook/page embedding needs
    // no extra setup. AlphaTab + html-midi-player stay external (CDN):
    // they need soundfonts and audio backends that must not be bundled.
    rollupOptions: {},
    sourcemap: true,
  },
});
