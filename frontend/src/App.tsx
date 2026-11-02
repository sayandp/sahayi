import { useEffect, useRef, useState, type FormEvent, type ReactNode } from "react";
import { ask, MOCK_MODE, type AskResult, type SchemeCard, type Source } from "./api";
import { EXAMPLES, isYesNo, langOf, UI, type UiLang } from "./text";

type Status = "idle" | "searching" | "writing" | "done" | "error" | "rate_limited";
type Theme = "system" | "light" | "dark";
const TAG = /\[([^\[\]\s]+:\d+:\d+)\]/g;

function loadUiLang(): UiLang {
  try {
    const saved = localStorage.getItem("sahayi-lang");
    if (saved === "en" || saved === "ml") return saved;
  } catch {
    /* private mode */
  }
  return "ml";
}
function loadTheme(): Theme {
  try {
    const saved = localStorage.getItem("sahayi-theme");
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    /* private mode */
  }
  return "system";
}
function remember(key: string, value: string | null) {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch {
    /* private mode */
  }
}

export function App() {
  const [uiLang, setUiLang] = useState<UiLang>(loadUiLang);
  const [theme, setTheme] = useState<Theme>(loadTheme);
  const [question, setQuestion] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [sources, setSources] = useState<Source[]>([]);
  const [streamed, setStreamed] = useState("");
  const [result, setResult] = useState<AskResult | null>(null);
  const [mockMiss, setMockMiss] = useState(false);
  const [openCite, setOpenCite] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const t = UI[uiLang];

  useEffect(() => {
    if (theme === "system") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = theme;
    remember("sahayi-theme", theme === "system" ? null : theme);
  }, [theme]);

  function chooseLang(lang: UiLang) {
    setUiLang(lang);
    remember("sahayi-lang", lang);
  }

  async function submit(q: string) {
    const text = q.trim();
    if (text.length < 2) return;
    abort.current?.abort();
    const controller = new AbortController();
    abort.current = controller;
    setQuestion(text);
    setStatus("searching");
    setSources([]);
    setStreamed("");
    setResult(null);
    setMockMiss(false);
    setOpenCite(null);
    window.scrollTo({ top: 0 });
    try {
      await ask(
        text,
        {
          onSources: (_, s) => {
            setSources(s);
            setStatus("writing");
          },
          onToken: (chunk) => setStreamed((prev) => prev + chunk),
          onResult: (r) => {
            setResult(r);
            setStatus("done");
          },
          onMockMiss: () => setMockMiss(true),
        },
        controller.signal,
      );
    } catch (err) {
      if (controller.signal.aborted) return;
      setStatus(err instanceof Error && err.message === "rate_limited" ? "rate_limited" : "error");
    }
  }

  function reset() {
    abort.current?.abort();
    setStatus("idle");
    setQuestion("");
    setResult(null);
    setSources([]);
    setStreamed("");
  }

  const number = (id: string) => {
    const i = sources.findIndex((s) => s.chunk_id === id);
    return i >= 0 ? i + 1 : null;
  };
  const shown = result ? result.answer : streamed;
  const opened = openCite
    ? (result?.citations.find((c) => c.chunk_id === openCite) ??
      sources.find((c) => c.chunk_id === openCite))
    : null;

  return (
    <div lang={uiLang} className="min-h-dvh flex flex-col">
      <header className="sticky top-0 z-20" style={{ background: "var(--page)" }}>
        <div className="mx-auto w-full max-w-2xl flex items-center gap-2 px-4 py-2">
          <button
            onClick={reset}
            lang="ml"
            className="display text-2xl mr-auto"
            style={{ color: "var(--green)" }}
          >
            സഹായി
          </button>
          <LangToggle value={uiLang} onChange={chooseLang} />
          <button
            onClick={() => setSettingsOpen(true)}
            aria-label={t.settings}
            className="h-11 w-11 flex items-center justify-center rounded-full"
            style={{ border: "1px solid var(--line)" }}
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              aria-hidden="true"
            >
              <path d="M4 7h10M18 7h2M4 17h2M10 17h10" />
              <circle cx="16" cy="7" r="2" />
              <circle cx="8" cy="17" r="2" />
            </svg>
          </button>
        </div>
        <div className="kasavu" />
      </header>

      <main className="mx-auto w-full max-w-2xl flex-1 px-4 pb-10">
        {status === "idle" ? (
          <Home uiLang={uiLang} onAsk={submit} />
        ) : (
          <Answer
            uiLang={uiLang}
            question={question}
            status={status}
            sources={sources}
            shown={shown}
            result={result}
            mockMiss={mockMiss}
            number={number}
            onCite={setOpenCite}
            onReply={(reply) => submit(`${question} ${result?.follow_up_question ?? ""} ${reply}`)}
            onRetry={() => submit(question)}
            onNew={reset}
          />
        )}
      </main>

      {MOCK_MODE && (
        <p className="text-center text-xs py-2 px-4" style={{ color: "var(--muted)" }}>
          {t.mock}
        </p>
      )}

      {opened && (
        <CitationPanel
          uiLang={uiLang}
          source={opened}
          n={number(opened.chunk_id)}
          onClose={() => setOpenCite(null)}
        />
      )}
      {settingsOpen && (
        <SettingsDrawer
          uiLang={uiLang}
          theme={theme}
          setTheme={setTheme}
          result={result}
          onClose={() => setSettingsOpen(false)}
        />
      )}
    </div>
  );
}

function LangToggle({ value, onChange }: { value: UiLang; onChange(l: UiLang): void }) {
  return (
    <div
      role="group"
      aria-label="Language / ഭാഷ"
      className="flex rounded-full overflow-hidden"
      style={{ border: "1px solid var(--line)" }}
    >
      {(["ml", "en"] as const).map((l) => (
        <button
          key={l}
          lang={l}
          onClick={() => onChange(l)}
          aria-pressed={value === l}
          className="h-11 px-3 text-sm font-semibold"
          style={value === l ? { background: "var(--green)", color: "var(--on-green)" } : undefined}
        >
          {l === "ml" ? "മല" : "EN"}
        </button>
      ))}
    </div>
  );
}

function Home({ uiLang, onAsk }: { uiLang: UiLang; onAsk(q: string): void }) {
  const t = UI[uiLang];
  const [text, setText] = useState("");
  const submit = (e: FormEvent) => {
    e.preventDefault();
    onAsk(text);
  };
  return (
    <div className="pt-7">
      <h1 className="display text-[2rem] sm:text-5xl" lang={uiLang}>
        {t.heading}
      </h1>
      <p className="mt-3" style={{ color: "var(--muted)" }}>
        {t.sub}
      </p>
      <form onSubmit={submit} className="mt-5">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) submit(e);
          }}
          rows={3}
          maxLength={1000}
          aria-label={t.ask}
          placeholder={t.placeholder}
          lang={langOf(text || t.placeholder)}
          className="w-full rounded-xl p-4 text-lg resize-none"
          style={{ background: "var(--sheet)", border: "2px solid var(--green)" }}
        />
        <button
          type="submit"
          className="mt-2 w-full h-13 rounded-xl text-lg font-semibold"
          style={{ background: "var(--green)", color: "var(--on-green)" }}
        >
          {t.ask}
        </button>
      </form>
      <h2 className="mt-7 mb-2 font-semibold" style={{ color: "var(--muted)" }}>
        {t.try}
      </h2>
      <ul className="flex flex-col gap-2">
        {EXAMPLES.map((ex) => (
          <li key={ex.q}>
            <button
              onClick={() => onAsk(ex.q)}
              lang={ex.lang}
              className="w-full text-left rounded-lg px-3 py-2"
              style={{ border: "1px solid var(--line)", background: "var(--sheet)" }}
            >
              {ex.q}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Answer(props: {
  uiLang: UiLang;
  question: string;
  status: Status;
  sources: Source[];
  shown: string;
  result: AskResult | null;
  mockMiss: boolean;
  number(id: string): number | null;
  onCite(id: string): void;
  onReply(reply: string): void;
  onRetry(): void;
  onNew(): void;
}) {
  const { uiLang, question, status, sources, shown, result, number, onCite } = props;
  const t = UI[uiLang];
  const failed = status === "error" || status === "rate_limited";
  return (
    <div className="pt-4">
      <p lang={langOf(question)} className="font-semibold" data-testid="question">
        {question}
      </p>

      {props.mockMiss && (
        <p className="mt-3 rounded-lg p-3 text-sm" style={{ background: "var(--notice)" }}>
          {t.mockMiss}
        </p>
      )}

      {status === "searching" && <Waiting label={t.finding} />}

      {sources.length > 0 && !result?.abstained && (
        <section aria-label={t.sources} className="pop-in mt-3" data-testid="sources">
          <h2 className="text-sm font-semibold mb-1" style={{ color: "var(--muted)" }}>
            {t.sources}
          </h2>
          <ol className="no-scrollbar flex gap-2 overflow-x-auto -mx-4 px-4 pb-1">
            {sources.map((s, i) => (
              <li key={s.chunk_id} className="shrink-0">
                <button
                  onClick={() => onCite(s.chunk_id)}
                  className="flex items-center gap-2 rounded-lg pl-1.5 pr-3 h-11 max-w-60 text-sm"
                  style={{ border: "1px solid var(--line)", background: "var(--sheet)" }}
                >
                  <span className="cite" style={{ margin: 0 }}>
                    {i + 1}
                  </span>
                  <span className="truncate" lang={langOf(s.scheme ?? "")}>
                    {s.scheme}
                  </span>
                </button>
              </li>
            ))}
          </ol>
        </section>
      )}

      {status === "writing" && !shown && <Waiting label={t.writing} />}

      {failed && (
        <section
          className="mt-5 rounded-xl p-4"
          role="alert"
          style={{ background: "var(--sheet)", border: "2px solid var(--no)" }}
        >
          <h2 className="display text-xl">{t.errorTitle}</h2>
          <p className="mt-1">{status === "rate_limited" ? t.rateLimited : t.errorBody}</p>
          <button
            onClick={props.onRetry}
            className="mt-3 h-11 px-4 rounded-lg font-semibold"
            style={{ background: "var(--green)", color: "var(--on-green)" }}
          >
            {t.retry}
          </button>
        </section>
      )}

      {shown && !result?.abstained && !result?.degraded && (
        <section
          className="mt-4 text-[1.0625rem]"
          data-testid="answer"
          lang={langOf(shown)}
          aria-live="polite"
        >
          <Cited text={shown} number={number} onCite={onCite} />
        </section>
      )}

      {result?.abstained && (
        <section
          className="pop-in mt-4 rounded-xl p-4"
          data-testid="abstain"
          style={{ background: "var(--sheet)", borderLeft: "6px solid var(--unknown)" }}
        >
          <h2 className="display text-xl">{t.noAnswer}</h2>
          <p className="mt-1" lang={result.lang}>
            {result.answer}
          </p>
        </section>
      )}

      {result?.degraded && (
        <section
          className="pop-in mt-4 rounded-xl p-4"
          data-testid="busy"
          style={{ background: "var(--notice)" }}
        >
          <h2 className="display text-xl">{t.busy}</h2>
          <p className="mt-1" lang={result.lang}>
            {result.answer}
          </p>
        </section>
      )}

      {result?.follow_up_question && (
        <FollowUp uiLang={uiLang} question={result.follow_up_question} onReply={props.onReply} />
      )}

      {result && result.schemes.length > 0 && (
        <section className="mt-6" aria-label={t.schemes}>
          <div className="flex flex-col gap-3">
            {result.schemes.map((c) => (
              <Card
                key={c.scheme_id}
                uiLang={uiLang}
                card={c}
                asked={Boolean(result.follow_up_question)}
                number={number}
                onCite={onCite}
              />
            ))}
          </div>
        </section>
      )}

      {result && (
        <footer
          className="mt-6 pt-3 text-sm"
          data-testid="footer"
          style={{ borderTop: "1px solid var(--line)" }}
        >
          <p className="font-semibold">{t.verify}</p>
          {result.as_of && (
            <p style={{ color: "var(--muted)" }}>
              {t.asOf}: {formatDate(result.as_of, uiLang)}
            </p>
          )}
          <button
            onClick={props.onNew}
            className="mt-3 h-11 px-4 rounded-lg font-semibold"
            style={{ border: "2px solid var(--green)", color: "var(--green)" }}
          >
            {t.newQuestion}
          </button>
        </footer>
      )}
    </div>
  );
}

function Waiting({ label }: { label: string }) {
  return (
    <p className="mt-4 flex items-center gap-2" role="status" style={{ color: "var(--muted)" }}>
      <span
        className="inline-block h-3 w-3 rounded-full motion-safe:animate-pulse"
        style={{ background: "var(--gold-bright)" }}
      />
      {label}
    </p>
  );
}

/** Renders text with [chunk_id] tags as numbered chips. */
function Cited({
  text,
  number,
  onCite,
}: {
  text: string;
  number(id: string): number | null;
  onCite(id: string): void;
}) {
  const parts: ReactNode[] = [];
  let last = 0;
  for (const m of text.matchAll(TAG)) {
    parts.push(text.slice(last, m.index).replace(/\s+$/, ""));
    const n = number(m[1]);
    if (n !== null) {
      parts.push(
        <button
          key={m.index}
          className="cite"
          aria-label={`Source ${n}`}
          onClick={() => onCite(m[1])}
        >
          {n}
        </button>,
      );
    }
    last = m.index + m[0].length;
  }
  parts.push(text.slice(last));
  return <>{parts}</>;
}

function Card({
  uiLang,
  card,
  asked,
  number,
  onCite,
}: {
  uiLang: UiLang;
  card: SchemeCard;
  asked: boolean;
  number(id: string): number | null;
  onCite(id: string): void;
}) {
  const t = UI[uiLang];
  const verdict =
    card.eligible === true
      ? ["stamp-yes", t.yes]
      : card.eligible === false
        ? ["stamp-no", t.no]
        : ["stamp-unknown", asked || card.reasons.length > 0 ? t.unknown : t.unchecked];
  const row = (label: string, value: ReactNode) => (
    <div className="mt-3">
      <h4 className="text-sm font-semibold" style={{ color: "var(--muted)" }}>
        {label}
      </h4>
      <div>{value}</div>
    </div>
  );
  const cited = (s: string) => (
    <span lang={langOf(s)}>
      <Cited text={s} number={number} onCite={onCite} />
    </span>
  );
  const list = (items: string[]) => (
    <ul className="list-disc pl-5">
      {items.map((x) => (
        <li key={x}>{cited(x)}</li>
      ))}
    </ul>
  );
  return (
    <article
      className="pop-in rounded-xl p-4"
      data-testid="scheme-card"
      style={{ background: "var(--sheet)", border: "1px solid var(--line)" }}
    >
      <div className="flex flex-wrap items-start justify-between gap-x-3 gap-y-2">
        <h3 className="display text-lg min-w-0" lang={langOf(card.name)}>
          {card.name}
        </h3>
        <span className={`stamp ${verdict[0]}`} data-testid="verdict" lang={uiLang}>
          {verdict[1]}
        </span>
      </div>
      {card.reasons.length > 0 && row(t.why, list(card.reasons))}
      {card.benefit && row(t.benefit, cited(card.benefit))}
      {card.how_to_apply && row(t.how, cited(card.how_to_apply))}
      {card.documents.length > 0 && row(t.docs, list(card.documents))}
    </article>
  );
}

function FollowUp({
  uiLang,
  question,
  onReply,
}: {
  uiLang: UiLang;
  question: string;
  onReply(r: string): void;
}) {
  const t = UI[uiLang];
  const [text, setText] = useState("");
  const qLang = langOf(question);
  const yes = UI[qLang].yesReply;
  const no = UI[qLang].noReply;
  return (
    <section
      className="pop-in mt-5 rounded-xl p-4"
      data-testid="follow-up"
      style={{ background: "var(--green-soft)" }}
    >
      <h2 className="text-sm font-semibold" style={{ color: "var(--green)" }}>
        {t.followUp}
      </h2>
      <p className="mt-1 font-semibold" lang={qLang}>
        {question}
      </p>
      {isYesNo(question) ? (
        <div className="mt-3 flex gap-2">
          {[yes, no].map((r) => (
            <button
              key={r}
              onClick={() => onReply(r)}
              lang={qLang}
              className="h-11 min-w-20 px-4 rounded-lg font-semibold"
              style={{ background: "var(--green)", color: "var(--on-green)" }}
            >
              {r}
            </button>
          ))}
        </div>
      ) : (
        <form
          className="mt-3 flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            if (text.trim()) onReply(text.trim());
          }}
        >
          <input
            value={text}
            onChange={(e) => setText(e.target.value)}
            aria-label={t.replyPlaceholder}
            placeholder={t.replyPlaceholder}
            className="min-w-0 flex-1 h-11 rounded-lg px-3"
            style={{ background: "var(--sheet)", border: "1px solid var(--line)" }}
          />
          <button
            type="submit"
            className="h-11 px-4 rounded-lg font-semibold"
            style={{ background: "var(--green)", color: "var(--on-green)" }}
          >
            {t.send}
          </button>
        </form>
      )}
    </section>
  );
}

function Drawer({
  label,
  closeLabel,
  onClose,
  children,
  testId,
}: {
  label: string;
  closeLabel: string;
  onClose(): void;
  children: ReactNode;
  testId: string;
}) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;
  useEffect(() => {
    const before = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCloseRef.current();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      before?.focus();
    };
  }, []);
  return (
    <div
      className="fixed inset-0 z-30 flex justify-end"
      role="dialog"
      aria-modal="true"
      aria-label={label}
      data-testid={testId}
    >
      <button
        className="flex-1 bg-black/40 cursor-default"
        aria-label={closeLabel}
        tabIndex={-1}
        onClick={onClose}
      />
      <div
        className="drawer w-[min(88vw,26rem)] h-full overflow-y-auto"
        style={{ background: "var(--sheet)" }}
      >
        <div className="kasavu" />
        <div className="p-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h2 className="display text-xl min-w-0">{label}</h2>
            <button
              ref={closeRef}
              onClick={onClose}
              className="h-11 px-3 rounded-lg font-semibold shrink-0"
              style={{ border: "1px solid var(--line)" }}
            >
              {closeLabel}
            </button>
          </div>
          {children}
        </div>
      </div>
    </div>
  );
}

function CitationPanel({
  uiLang,
  source,
  n,
  onClose,
}: {
  uiLang: UiLang;
  source: Source;
  n: number | null;
  onClose(): void;
}) {
  const t = UI[uiLang];
  const isPdf = source.link.includes("#page=");
  return (
    <Drawer
      label={`${t.source} ${n ?? ""}`}
      closeLabel={t.close}
      onClose={onClose}
      testId="citation-panel"
    >
      <p className="mt-3 font-semibold" lang={langOf(source.scheme ?? "")}>
        {source.scheme}
      </p>
      <dl className="mt-2 text-sm" style={{ color: "var(--muted)" }}>
        <div>
          <dt className="inline">{t.page}: </dt>
          <dd className="inline">{source.page}</dd>
        </div>
        {source.downloaded_at && (
          <div>
            <dt className="inline">{t.asOf}: </dt>
            <dd className="inline">{formatDate(source.downloaded_at, uiLang)}</dd>
          </div>
        )}
      </dl>
      <blockquote
        className="mt-3 pl-3"
        lang={langOf(source.snippet)}
        style={{ borderLeft: "3px solid var(--gold-bright)" }}
      >
        {source.snippet}
      </blockquote>
      <a
        href={source.link}
        target="_blank"
        rel="noreferrer"
        data-testid="open-source"
        className="mt-4 flex items-center justify-center min-h-11 rounded-lg px-4 font-semibold text-center"
        style={{ background: "var(--green)", color: "var(--on-green)" }}
      >
        {isPdf ? t.openPdf(source.page) : t.openPage}
      </a>
    </Drawer>
  );
}

function SettingsDrawer({
  uiLang,
  theme,
  setTheme,
  result,
  onClose,
}: {
  uiLang: UiLang;
  theme: Theme;
  setTheme(t: Theme): void;
  result: AskResult | null;
  onClose(): void;
}) {
  const t = UI[uiLang];
  const m = result?.meta;
  const max = m ? Math.max(m.latency_ms, 1) : 1;
  const bar = (label: string, ms: number) => (
    <div className="mt-3" key={label}>
      <div className="flex justify-between text-sm">
        <span>{label}</span>
        <span className="font-semibold">{(ms / 1000).toFixed(2)} s</span>
      </div>
      <div className="h-2 rounded" style={{ background: "var(--green-soft)" }}>
        <div
          className="h-2 rounded"
          style={{ width: `${(ms / max) * 100}%`, background: "var(--green)" }}
        />
      </div>
    </div>
  );
  return (
    <Drawer label={t.settings} closeLabel={t.close} onClose={onClose} testId="settings">
      <h3 className="mt-5 font-semibold">{t.theme}</h3>
      <div className="mt-2 flex flex-wrap gap-2">
        {(["system", "light", "dark"] as const).map((x) => (
          <button
            key={x}
            onClick={() => setTheme(x)}
            aria-pressed={theme === x}
            className="min-h-11 px-3 rounded-lg text-sm font-semibold"
            style={
              theme === x
                ? { background: "var(--green)", color: "var(--on-green)" }
                : { border: "1px solid var(--line)" }
            }
          >
            {t.themes[x]}
          </button>
        ))}
      </div>
      <h3 className="mt-6 font-semibold">{t.speed}</h3>
      {m ? (
        <>
          {bar(t.ttfc, m.ttfc_ms)}
          {bar(t.ttft, m.ttft_ms)}
          {bar(t.total, m.latency_ms)}
          <dl className="mt-4 text-sm" style={{ color: "var(--muted)" }}>
            <div>
              {t.model}: {m.model}
            </div>
            <div>
              {t.cost}: ₹{m.cost_inr.toFixed(2)}
            </div>
            {m.cached && <div>{t.fromCache}</div>}
          </dl>
        </>
      ) : (
        <p className="mt-2" style={{ color: "var(--muted)" }}>
          {t.noSpeed}
        </p>
      )}
    </Drawer>
  );
}

function formatDate(iso: string, lang: UiLang): string {
  const d = new Date(`${iso}T00:00:00`);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(lang === "ml" ? "ml-IN" : "en-IN", {
    day: "numeric",
    month: "long",
    year: "numeric",
  });
}
