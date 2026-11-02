import { defineConfig } from "@playwright/test";

const deployed = process.env.E2E_BASE_URL;

export default defineConfig({
  testDir: "./e2e",
  testMatch: deployed ? "deployed.spec.ts" : "smoke.spec.ts",
  timeout: 90_000,
  reporter: "list",
  use: { baseURL: deployed ?? "http://localhost:5173", viewport: { width: 360, height: 780 } },
  webServer: deployed
    ? undefined
    : { command: "npm run dev", url: "http://localhost:5173", reuseExistingServer: true },
});
