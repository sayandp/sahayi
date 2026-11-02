export interface Source {
  chunk_id: string;
  source_url: string;
  page: number;
  link: string;
  snippet: string;
  scheme?: string | null;
  downloaded_at?: string | null;
  tag?: string;
}
export interface SchemeCard {
  scheme_id: string;
  name: string;
  eligible: boolean | null;
  reasons: string[];
  benefit: string | null;
  how_to_apply: string | null;
  documents: string[];
  citations: string[];
}
export interface Meta {
  ttfc_ms: number;
  ttft_ms: number;
  latency_ms: number;
  tokens_in: number;
  tokens_out: number;
  cost_inr: number;
  model: string;
  fallbacks?: string[];
  cached?: boolean;
}
export interface AskResult {
  answer: string;
  lang: "ml" | "en";
  schemes: SchemeCard[];
  citations: Source[];
  abstained: boolean;
  degraded?: boolean;
  follow_up_question: string | null;
  as_of: string | null;
  meta: Meta;
}
export interface Handlers {
  onSources(lang: "ml" | "en", sources: Source[]): void;
  onToken(text: string): void;
  onResult(result: AskResult): void;
  onMockMiss(): void;
}

const RAW_API_URL: string = import.meta.env.VITE_API_URL ?? "";
/** Empty: mock mode. "/": the real API on this origin (one-container deploy). */
export const MOCK_MODE = RAW_API_URL === "";
export const API_URL = RAW_API_URL.replace(/\/$/, "");

/** POST /ask and dispatch its server-sent events. Rejects on HTTP or stream errors. */
export async function ask(
  question: string,
  handlers: Handlers,
  signal: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(res.status === 429 ? "rate_limited" : "http_error");
  if (res.headers.get("X-Mock-Match") === "none") handlers.onMockMiss();
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += value.replaceAll("\r\n", "\n");
    let end: number;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      dispatch(buffer.slice(0, end), handlers);
      buffer = buffer.slice(end + 2);
    }
  }
}

function dispatch(block: string, h: Handlers) {
  let event = "";
  let data = "";
  for (const line of block.split("\n")) {
    if (line.startsWith("event: ")) event = line.slice(7);
    if (line.startsWith("data: ")) data += line.slice(6);
  }
  if (!data) return;
  const payload = JSON.parse(data);
  if (event === "sources") h.onSources(payload.lang, payload.sources);
  if (event === "token") h.onToken(payload.text);
  if (event === "result") h.onResult(payload);
  if (event === "error") throw new Error("stream_error");
}
