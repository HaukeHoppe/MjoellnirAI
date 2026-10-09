/// <reference types="vitest/config" />
import { createReadStream, existsSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig, type Plugin } from "vite";

// `npm run dev`: serve the exported graph (../data/explorer/graph.json) at /data/graph.json, as the
// container's bind mount does, and a local chat URL at /config.json.
function devData(): Plugin {
  const graph = resolve(__dirname, "../data/explorer/graph.json");
  return {
    name: "dev-data",
    configureServer(server) {
      server.middlewares.use("/data/graph.json", (_req, res) => {
        if (!existsSync(graph)) {
          res.statusCode = 404;
          res.end("Run python data/export_explorer_graph.py first");
          return;
        }
        res.setHeader("Content-Type", "application/json");
        createReadStream(graph).pipe(res);
      });
      server.middlewares.use("/config.json", (_req, res) => {
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify({ chatUrl: "http://localhost:3000/" }));
      });
    },
  };
}

export default defineConfig({
  // Relative asset paths: the same build is served at / (locally, port 3001) and at /graph/ (VPS, behind Caddy).
  base: "./",
  plugins: [devData()],
  test: {
    environment: "node",
  },
});
