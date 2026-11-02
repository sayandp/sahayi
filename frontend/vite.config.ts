import { readFileSync } from "node:fs";
import type { IncomingMessage, ServerResponse } from "node:http";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

interface Recorded {
  id: string;
  question: string;
  events: {
    event: string;
    data: { meta?: { ttfc_ms: number; ttft_ms: number; latency_ms: number } };
  }[];
}

const streams: Recorded[] = JSON.parse(
  readFileSync(new URL("./mock/streams.json", import.meta.url), "utf-8"),
);
const ABSTAIN_ID = "n014";
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** POST /ask replays a recorded answer as real SSE, at its recorded pace. No LLM is called. */
async function replay(req: IncomingMessage, res: ServerResponse) {
  let body = "";
  for await (const part of req) body += part;
  const question = String(JSON.parse(body || "{}").question ?? "").trim();
  const match = streams.find((s) => s.question.trim() === question);
  const stream = match ?? streams.find((s) => s.id === ABSTAIN_ID)!;
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "X-Mock-Match": match ? match.id : "none",
  });
  const meta = stream.events.at(-1)!.data.meta!;
  const tokens = stream.events.filter((e) => e.event === "token").length;
  const gap = Math.max(40, (meta.latency_ms - meta.ttft_ms) / Math.max(tokens, 1));
  let first = true;
  for (const e of stream.events) {
    if (e.event === "sources") await sleep(Math.min(meta.ttfc_ms, 2500));
    if (e.event === "token") {
      await sleep(first ? Math.min(meta.ttft_ms - meta.ttfc_ms, 1500) : gap);
      first = false;
    }
    if (e.event === "result") await sleep(Math.min(gap, 400));
    res.write(`event: ${e.event}\ndata: ${JSON.stringify(e.data)}\n\n`);
  }
  res.end();
}

function mockApi(): Plugin {
  const handler = (req: IncomingMessage, res: ServerResponse, next: () => void) => {
    if (req.method === "POST" && req.url === "/ask") void replay(req, res);
    else next();
  };
  return {
    name: "sahayi-mock-api",
    configureServer: (server) => void server.middlewares.use(handler),
    configurePreviewServer: (server) => void server.middlewares.use(handler),
  };
}

export default defineConfig({
  plugins: [react(), tailwindcss(), mockApi()],
  server: { port: 5173 },
  preview: { port: 5173 },
});
