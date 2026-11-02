import { expect, test, type Page } from "@playwright/test";
import { mkdirSync } from "node:fs";

const SHOTS = "screenshots";
mkdirSync(SHOTS, { recursive: true });
const shot = (page: Page, name: string, fullPage = true) =>
  page.screenshot({ path: `${SHOTS}/${name}.png`, fullPage });
const ML_Q = "എന്റെ കുടുംബത്തിൽ NREGA ജോബ് കാർഡ് അപേക്ഷിക്കാൻ എന്തെല്ലാം രേഖകൾ വേണം?";
const ELIG_Q = "I'm 45 years old and have a savings account. Can I still join Atal Pension Yojana?";
const FOLLOW_Q =
  "My husband passed away last year. Can I get the Indira Gandhi National Widow Pension?";

test.beforeEach(async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" }); // screenshots never catch a slide-in
});

/** Serves one canned SSE answer, for states the recordings do not include. */
async function serve(page: Page, result: object) {
  const source = { chunk_id: "a--b:1:0", source_url: "https://x.gov.in/a.pdf", page: 1,
    link: "https://x.gov.in/a.pdf#page=1", snippet: "Rule text", scheme: "Scheme A",
    downloaded_at: "2026-10-12" };
  const events = [
    `event: sources\ndata: ${JSON.stringify({ lang: "en", sources: [source] })}`,
    `event: token\ndata: ${JSON.stringify({ text: "Text. " })}`,
    `event: result\ndata: ${JSON.stringify({ lang: "en", schemes: [], citations: [source],
      abstained: false, as_of: "2026-10-12", follow_up_question: null,
      meta: { ttfc_ms: 1, ttft_ms: 2, latency_ms: 3, tokens_in: 0, tokens_out: 0, cost_inr: 0,
        model: "m" }, ...result })}`,
  ];
  await page.route("**/ask", (r) =>
    r.fulfill({ contentType: "text/event-stream", body: events.join("\n\n") + "\n\n" }),
  );
}

async function ask(page: Page) {
  await page.goto("/");
  await page.getByRole("button", { name: "EN", exact: true }).click();
  await page.getByRole("textbox").fill("Any question");
  await page.getByRole("button", { name: "Ask", exact: true }).click();
}

async function noSideScroll(page: Page) {
  const over = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(over).toBeLessThanOrEqual(0);
}

test("ask, sources first, answer, open citation", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("സർക്കാർ");
  await noSideScroll(page);
  await shot(page, "01-home-ml-mobile");
  await page.getByRole("button", { name: ML_Q }).click();

  // sources appear before any answer text
  await expect(page.getByTestId("sources")).toBeVisible();
  await expect(page.getByTestId("answer")).toHaveCount(0);
  await expect(page.getByTestId("answer")).toContainText("NREGA", { timeout: 15_000 });
  await expect(page.getByTestId("footer")).toContainText("അക്ഷയ");
  await noSideScroll(page);
  await shot(page, "02-answer-ml-mobile");

  await page.getByTestId("answer").locator(".cite").first().click();
  const panel = page.getByTestId("citation-panel");
  await expect(panel).toBeVisible();
  await expect(panel).toContainText("MGNREGA");
  await expect(panel.getByTestId("open-source")).toHaveAttribute("href", /^https:/);
  await noSideScroll(page);
  await shot(page, "03-citation-panel-mobile", false);
  await page.keyboard.press("Escape");
  await expect(panel).toHaveCount(0);

  await page.getByRole("button", { name: "ക്രമീകരണങ്ങൾ" }).click();
  await expect(page.getByTestId("settings")).toContainText("മോഡൽ");
});

test("English eligibility answer shows a stamped card", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "EN", exact: true }).click();
  await page.getByRole("textbox").fill(ELIG_Q);
  await page.getByRole("button", { name: "Ask", exact: true }).click();
  const card = page.getByTestId("scheme-card").first();
  await expect(card.getByTestId("verdict")).toHaveText("Not eligible", { timeout: 15_000 });
  await noSideScroll(page);
  await shot(page, "04-eligibility-en-mobile");
  await page.setViewportSize({ width: 1100, height: 800 });
  await shot(page, "04b-eligibility-en-desktop");
});

test("follow-up question is offered as a reply", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "EN", exact: true }).click();
  await page.getByRole("button", { name: FOLLOW_Q }).click();
  await expect(page.getByTestId("follow-up")).toContainText("poverty line", { timeout: 15_000 });
  await expect(page.getByTestId("scheme-card").first().getByTestId("verdict")).toHaveText(
    "Need more details",
  );
  await shot(page, "05-follow-up-en-mobile");
});

test("abstain shows the department contact and the Akshaya footer", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "EN", exact: true }).click();
  await page.getByRole("textbox").fill("How do I link my PAN card with Aadhaar?");
  await page.getByRole("button", { name: "Ask", exact: true }).click();
  await expect(page.getByTestId("abstain")).toContainText("Akshaya", { timeout: 15_000 });
  await expect(page.getByTestId("footer")).toContainText("Akshaya centre");
  await noSideScroll(page);
  await shot(page, "06-abstain-en-mobile");
});

test("dark mode and settings latency", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "dark", reducedMotion: "reduce" });
  await page.goto("/");
  await page.getByRole("button", { name: ML_Q }).click();
  await expect(page.getByTestId("footer")).toBeVisible({ timeout: 15_000 });
  await page.getByRole("button", { name: "ക്രമീകരണങ്ങൾ" }).click();
  await expect(page.getByTestId("settings")).toContainText("രേഖകൾ കാണിച്ചത്");
  await shot(page, "07-settings-dark-mobile", false);
});

test("a yes/no follow-up becomes quick-reply buttons", async ({ page }) => {
  await serve(page, { answer: "Text. [a--b:1:0]", follow_up_question: "Are you a farmer?" });
  await ask(page);
  const box = page.getByTestId("follow-up");
  await expect(box.getByRole("button", { name: "Yes" })).toBeVisible();
  await expect(box.getByRole("button", { name: "No" })).toBeVisible();
  await expect(box.getByRole("textbox")).toHaveCount(0);
});

test("the busy state keeps the sources and cards", async ({ page }) => {
  await serve(page, {
    answer: "Sahayi is busy right now.",
    degraded: true,
    schemes: [{ scheme_id: "a", name: "Scheme A", eligible: null, reasons: [],
      benefit: "Rule text [a--b:1:0]", how_to_apply: null, documents: [], citations: ["a--b:1:0"] }],
  });
  await ask(page);
  await expect(page.getByTestId("busy")).toContainText("busy");
  await expect(page.getByTestId("sources")).toBeVisible();
  await expect(page.getByTestId("verdict")).toHaveText("Eligibility not checked");
  await shot(page, "08-busy-en-mobile");
});
