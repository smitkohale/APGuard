// Opens each URL in a headless browser and, if Streamlit Community Cloud shows
// its "gone to sleep" page, clicks the wake-up button and waits for the app.
// A plain HTTP ping isn't enough: the sleep page is served over HTTP 200 and
// the app itself only starts when the button is pressed.
//
//   node .github/scripts/keep-awake.mjs https://apguard.streamlit.app
import { chromium } from "playwright";

const urls = process.argv.slice(2);
if (!urls.length) {
  console.error("usage: node keep-awake.mjs <url> [<url> ...]");
  process.exit(2);
}

const WAKE_BUTTON = /get this app back up|wake (it|this app) (back )?up/i;
const APP_READY_MS = 180_000; // a cold start can take a couple of minutes

async function visit(browser, url) {
  const page = await browser.newPage();
  try {
    await page.goto(url, { waitUntil: "domcontentloaded", timeout: 90_000 });
    await page.waitForTimeout(5_000);

    const button = page.getByRole("button", { name: WAKE_BUTTON });
    if (await button.count()) {
      console.log(`${url}: asleep, clicking wake-up`);
      await button.first().click();
      // the sleep page disappears once the container is booting
      await button.first().waitFor({ state: "detached", timeout: APP_READY_MS }).catch(() => {});
    } else {
      console.log(`${url}: awake`);
    }

    // Streamlit renders the app inside an iframe on Community Cloud; look in every frame.
    const deadline = Date.now() + APP_READY_MS;
    while (Date.now() < deadline) {
      for (const frame of page.frames()) {
        if (await frame.locator('[data-testid="stApp"], [data-testid="stAppViewContainer"]').count().catch(() => 0)) {
          console.log(`${url}: app is up`);
          return true;
        }
      }
      await page.waitForTimeout(5_000);
    }
    console.error(`${url}: app did not finish loading within ${APP_READY_MS / 1000}s`);
    return false;
  } catch (err) {
    console.error(`${url}: ${err.message}`);
    return false;
  } finally {
    await page.close();
  }
}

const browser = await chromium.launch();
let ok = true;
for (const url of urls) ok = (await visit(browser, url)) && ok;
await browser.close();
process.exit(ok ? 0 : 1);
