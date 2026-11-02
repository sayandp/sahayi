import { expect, test } from "@playwright/test";

// Against a running deploy build in retrieval-only mode: E2E_BASE_URL=http://localhost:7860
test.skip(!process.env.E2E_BASE_URL, "needs a running deploy build");

test("retrieval-only: real sources, busy notice, cards, citation panel", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await page.getByRole("button", { name: "EN", exact: true }).click();
  await expect(page.getByText("Demo: recorded answers only")).toHaveCount(0); // not mock mode
  await page.getByRole("textbox").fill("How much does PM-KISAN pay a farmer family every year?");
  await page.getByRole("button", { name: "Ask", exact: true }).click();

  await expect(page.getByTestId("sources")).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId("sources")).toContainText("PM-KISAN");
  await expect(page.getByTestId("busy")).toContainText("busy", { timeout: 15_000 });
  const cards = page.getByTestId("scheme-card");
  await expect(cards.first()).toBeVisible();
  await expect(cards.first().getByTestId("verdict")).toHaveText("Eligibility not checked");
  await expect(page.getByTestId("footer")).toContainText("Akshaya centre");
  await page.screenshot({ path: "screenshots/09-deployed-busy-mobile.png", fullPage: true });

  await page.getByTestId("sources").locator("button").first().click();
  const panel = page.getByTestId("citation-panel");
  await expect(panel).toBeVisible();
  await expect(panel.getByTestId("open-source")).toHaveAttribute("href", /^https?:/);
  await page.screenshot({ path: "screenshots/10-deployed-citation-mobile.png" });
  await page.keyboard.press("Escape");

  const health = await page.request.get("/health");
  const body = await health.json();
  expect(body.status).toBe("ok");
  expect(body.answer_mode).toBe("retrieval_only");
  expect(body.groq_key).toBe(false);
  expect(body.gemini_key).toBe(false);
});

test("Malayalam question gets the Malayalam busy notice", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("textbox").fill("പിഎം കിസാൻ പദ്ധതിയിൽ ഒരു കുടുംബത്തിന് വർഷം എത്ര രൂപ ലഭിക്കും?");
  await page.getByRole("button", { name: "ചോദിക്കുക", exact: true }).click();
  await expect(page.getByTestId("busy")).toContainText("തിരക്ക", { timeout: 60_000 });
  await expect(page.getByTestId("sources")).toBeVisible();
});
