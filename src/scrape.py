#!/usr/bin/env python3
"""
Scraper for the Artificial Analysis Coding Agents benchmark page.

Extracts the full per-combination performance dataset from:
  https://artificialanalysis.ai/agents/coding-agents

The page is a Next.js app whose data lives inside the RSC payload emitted via
`self.__next_f.push(...)` calls. Each model × frontend combination appears as
a JSON object that carries the 4 performance benchmarks shown in the page's
"Performance" section:

    Index             — overall Coding Agent Index (mean of the 3 sub-benchmarks)
    DeepSWE           — reward on the DeepSWE benchmark (datacurve-ai/deep-swe)
    Terminal-Bench v2 — reward on Terminal-Bench v2 (terminal-bench@2.0)
    SWE-Atlas-QnA     — reward on SWE-Atlas-QnA (datasets/swe-atlas-qna)

All 4 are pass@1 reward scores (higher = better).

The Index comes from the top-level `indexScore` field; the 3 sub-benchmark
rewards come from the `evals` array, where each entry has a
`datasetIndexName` ("deep-swe" / "swe-atlas-qna" / "terminal-bench-v2") and
a `mean.reward` value.

This scraper uses Playwright (same approach as the reference repository
https://github.com/Elaman0117/Generic-LLM-Leaderboard/) so the RSC stream is
fully received before extraction. A urllib fallback parses the static HTML
when Playwright is unavailable.

Output: output/raw_data.json — a list of 42 combination objects.
"""

import json
import os
import re
import sys

URL = "https://artificialanalysis.ai/agents/coding-agents"
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "raw_data.json")
MIN_COMBOS_EXPECTED = 40  # we expect ~42


# ──────────────────────────────────────────────────────────────────────
# Shared extraction logic — given the full RSC payload text, locate every
# {"id":"<hex>","agentName":"..."} combination object and parse it as JSON.
# ──────────────────────────────────────────────────────────────────────

def _find_matching_brace(text, start):
    """Walk forward from `start` (an opening `{`) to its matching `}`."""
    depth = 0
    in_str = False
    escape = False
    j = start
    while j < len(text):
        c = text[j]
        if escape:
            escape = False
        elif c == "\\":
            escape = True
        elif c == '"':
            in_str = not in_str
        elif not in_str:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return j + 1
        j += 1
    return None


def extract_combinations_from_payload(payload_text):
    """Parse the RSC payload text and return all combination objects."""
    combos = []
    for m in re.finditer(r'\{"id":"[0-9a-f]+","agentName":"', payload_text):
        end = _find_matching_brace(payload_text, m.start())
        if end is None:
            continue
        obj_text = payload_text[m.start():end]
        try:
            obj = json.loads(obj_text)
        except json.JSONDecodeError:
            continue
        # Keep only objects that look like a real combination row:
        # must have agentName + an aggregate mean with cacheHitRate.
        if (
            "agentName" in obj
            and isinstance(obj.get("mean"), dict)
            and "cacheHitRate" in obj["mean"]
        ):
            combos.append(obj)
    return combos


# ──────────────────────────────────────────────────────────────────────
# Playwright-based scraper (primary) — navigates to the page, lets the RSC
# stream finish, then evaluates the same extraction in-browser.
# ──────────────────────────────────────────────────────────────────────

EXTRACT_JS = r"""
(() => {
  const scripts = document.querySelectorAll('script');
  const fragments = [];
  for (let i = 0; i < scripts.length; i++) {
    const text = scripts[i].textContent || '';
    if (!text.includes('__next_f')) continue;
    const pushPattern = /self\.__next_f\.push\(\s*(\[.*?\])\s*\)\s*;?/gs;
    let m;
    while ((m = pushPattern.exec(text)) !== null) {
      try {
        const arr = eval(m[1]);
        if (arr && arr.length >= 2 && typeof arr[1] === 'string') {
          fragments.push(arr[1]);
        }
      } catch(e) { /* skip */ }
    }
  }
  return fragments.join('');
})()
"""


def scrape_with_playwright():
    from playwright.sync_api import sync_playwright

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1920, "height": 1080})

        print(f"[1/3] Navigating to {URL} ...")
        page.goto(URL, wait_until="networkidle", timeout=90000)
        page.wait_for_timeout(5000)  # let RSC stream complete

        print("[2/3] Extracting RSC payload ...")
        payload_text = page.evaluate(EXTRACT_JS)
        print(f"  Payload length: {len(payload_text)} chars")

        combos = extract_combinations_from_payload(payload_text)
        print(f"  Parsed {len(combos)} combinations")

        browser.close()
    return combos


# ──────────────────────────────────────────────────────────────────────
# Static-HTML fallback — fetch the page with curl/requests and parse the
# RSC payload directly from the HTML. Useful when Playwright isn't available.
# ──────────────────────────────────────────────────────────────────────

def scrape_with_urllib():
    import urllib.request

    print(f"[1/3] Fetching {URL} via urllib ...")
    req = urllib.request.Request(
        URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    print(f"  HTML length: {len(html)} chars")

    print("[2/3] Extracting RSC payload from static HTML ...")
    fragments = []
    for m in re.finditer(
        r"self\.__next_f\.push\(\s*(\[.*?\])\s*\)\s*;?", html, re.DOTALL
    ):
        raw = m.group(1)
        try:
            arr = eval(raw)  # safe-ish: only Next.js literal arrays
            if isinstance(arr, list) and len(arr) >= 2 and isinstance(arr[1], str):
                fragments.append(arr[1])
        except Exception:
            continue
    payload_text = "".join(fragments)
    print(f"  Payload length: {len(payload_text)} chars")

    combos = extract_combinations_from_payload(payload_text)
    print(f"  Parsed {len(combos)} combinations")
    return combos


# ──────────────────────────────────────────────────────────────────────
# Normalisation — slim each combination down to the 4 performance
# benchmarks (Index + 3 sub-benchmarks) plus identifying fields.
# ──────────────────────────────────────────────────────────────────────

# The 4 performance benchmarks shown in the page's "Performance" section.
# All are pass@1 reward scores (higher = better).
# `key`        — the field name we'll use downstream
# `source`     — where in the combination object to find it
# `dataset`    — for sub-benchmarks, the `datasetIndexName` to look up in `evals`
PERF_TESTS = [
    {
        "key": "Index",
        "label": "Index",
        "source": "indexScore",
    },
    {
        "key": "DeepSWE",
        "label": "DeepSWE",
        "source": "eval",
        "dataset": "deep-swe",
    },
    {
        "key": "Terminal-Bench v2",
        "label": "Terminal-Bench v2",
        "source": "eval",
        "dataset": "terminal-bench-v2",
    },
    {
        "key": "SWE-Atlas-QnA",
        "label": "SWE-Atlas-QnA",
        "source": "eval",
        "dataset": "swe-atlas-qna",
    },
]


def _extract_eval_reward(evals, dataset_name):
    """Return the mean.reward for the given datasetIndexName, or None."""
    if not isinstance(evals, list):
        return None
    for e in evals:
        if not isinstance(e, dict):
            continue
        if e.get("datasetIndexName") == dataset_name:
            mean = e.get("mean") or {}
            return mean.get("reward")
    return None


def normalize_combos(combos):
    """Slim each combination to the fields used downstream."""
    out = []
    for c in combos:
        display = c.get("display", {}) or {}
        evals = c.get("evals") or []

        # Build the 4 performance values
        perf = {}
        for test in PERF_TESTS:
            if test["source"] == "indexScore":
                v = c.get("indexScore")
            elif test["source"] == "eval":
                v = _extract_eval_reward(evals, test["dataset"])
            else:
                v = None
            if v is not None:
                try:
                    perf[test["key"]] = float(v)
                except (TypeError, ValueError):
                    pass

        entry = {
            "agent": c.get("agentName"),
            "model": display.get("model"),
            "displayLabel": c.get("displayLabel"),
            "hostModelSlug": c.get("hostModelSlug"),
            "provider": c.get("provider"),
            "perf": perf,
        }
        # Drop if any critical field is missing
        if not entry["agent"] or not entry["model"]:
            continue
        if not entry["perf"]:
            continue
        out.append(entry)
    return out


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    combos = None
    errors = []

    # Try Playwright first
    try:
        combos = scrape_with_playwright()
    except Exception as e:
        errors.append(f"playwright: {e}")
        print(f"  Playwright failed: {e}")

    # Fall back to urllib
    if not combos:
        try:
            combos = scrape_with_urllib()
        except Exception as e:
            errors.append(f"urllib: {e}")
            print(f"  urllib failed: {e}")

    if not combos:
        print("\nERROR: All scraping methods failed.")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print(f"\n[3/3] Normalising {len(combos)} combinations ...")
    normalized = normalize_combos(combos)
    print(f"  Kept {len(normalized)} valid combinations")

    if len(normalized) < MIN_COMBOS_EXPECTED:
        print(
            f"  WARNING: expected at least {MIN_COMBOS_EXPECTED} combinations, "
            f"got {len(normalized)}"
        )

    # Quick stats
    agents = sorted({c["agent"] for c in normalized})
    models = sorted({c["model"] for c in normalized})
    print(f"  Unique agents (frontends): {agents}")
    print(f"  Unique models (with thinking-level): {len(models)}")
    print(f"  Performance benchmarks extracted: {[t['key'] for t in PERF_TESTS]}")
    if normalized:
        print(f"  Sample perf (first combo): {normalized[0]['perf']}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(normalized)} combinations to {OUTPUT_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Scraping failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
