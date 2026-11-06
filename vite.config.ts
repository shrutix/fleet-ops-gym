import { defineConfig } from "vite";

// Cross-origin isolation lets jax-js use a multi-threaded Wasm backend (SharedArrayBuffer).
// Everything the page loads is same-origin, so these headers cost nothing. If you deploy
// elsewhere, serve the same two headers or Wasm falls back to a single thread.
const isolation = {
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
    base: "./",
    build: { target: "esnext" },
    worker: { format: "es" },
    server: { headers: isolation },
    preview: { headers: isolation },
    optimizeDeps: { exclude: ["@jax-js/jax", "@jax-js/optax"] },
});
