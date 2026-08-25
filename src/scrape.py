#!/usr/bin/env python3
"""
Scraper for the Artificial Analysis Coding Agents benchmark page.
(Fixed Version - Robust against RSC payload structure & key name changes)
"""

import json
import os
import re
import sys

URL = "https://artificialanalysis.ai/agents/coding-agents"
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "raw_data.json")
# 网站目前显示有 56 个模型，保留一定容错空间
MIN_COMBOS_EXPECTED = 30  

# ──────────────────────────────────────────────────────────────────────
# 核心修复：动态查找 JSON 对象，不再依赖硬编码的 id 或 agentName
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
    if not payload_text:
        return combos

    # 匹配所有合法的 JSON 对象起始位置： {"key":
    for m in re.finditer(r'\{"[a-zA-Z0-9_]+"[ \t]*:', payload_text):
        end = _find_matching_brace(payload_text, m.start())
        if end is None:
            continue
        obj_text = payload_text[m.start():end]
        
        # 启发式过滤：跳过明显不包含评测数据的无关 UI 对象 (如 CSS 类名配置)
        keywords = ["agent", "Agent", "evals", "score", "Score", "index", "Index", "mean", "reward", "deep-swe", "terminal-bench"]
        if not any(kw in obj_text for kw in keywords):
            continue
            
        try:
            obj = json.loads(obj_text)
        except json.JSONDecodeError:
            continue
            
        # 智能识别：提取可能变更过的 Agent 名称字段
        agent_name = (
            obj.get("agentName") or obj.get("agent") or obj.get("name") or 
            obj.get("displayName") or obj.get("displayLabel")
        )
        
        # 智能识别：检查是否存在指标字段
        has_metrics = any(k in obj for k in [
            "evals", "indexScore", "score", "index", "mean", "reward", "benchmarks", "metrics"
        ])
        
        if agent_name and has_metrics:
            combos.append(obj)
            
    # 根据 ID 或 Agent+Model 组合进行去重
    seen = set()
    unique_combos = []
    for c in combos:
        display = c.get("display", {}) or {}
        uid = c.get("id") or (c.get("agentName") or c.get("agent") or "") + (display.get("model", ""))
        if uid and uid not in seen:
            seen.add(uid)
            unique_combos.append(c)
            
    return unique_combos

# ──────────────────────────────────────────────────────────────────────
# Playwright 抓取逻辑 (保持不变)
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

def scrape_with_urllib():
    import urllib.request

    print(f"[1/3] Fetching {URL} via urllib ...")
    req = urllib.request.Request(
        URL,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    print(f"  HTML length: {len(html)} chars")

    print("[2/3] Extracting RSC payload from static HTML ...")
    fragments = []
    for m in re.finditer(r"self\.__next_f\.push\(\s*(\[.*?\])\s*\)\s*;?", html, re.DOTALL):
        raw = m.group(1)
        try:
            arr = eval(raw)
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
# 增强版数据标准化逻辑 (兼容网站键名变更)
# ──────────────────────────────────────────────────────────────────────

PERF_TESTS = [
    {"key": "Index", "label": "Index", "source": "indexScore"},
    {"key": "DeepSWE", "label": "DeepSWE", "source": "eval", "dataset": "deep-swe"},
    {"key": "Terminal-Bench v2", "label": "Terminal-Bench v2", "source": "eval", "dataset": "terminal-bench-v2"},
    {"key": "SWE-Atlas-QnA", "label": "SWE-Atlas-QnA", "source": "eval", "dataset": "swe-atlas-qna"},
]

def _extract_eval_reward(evals, dataset_name):
    if not isinstance(evals, list):
        return None
    for e in evals:
        if not isinstance(e, dict):
            continue
        
        # 兼容可能的字段名变更
        e_name = (
            e.get("datasetIndexName") or e.get("name") or e.get("dataset") or 
            e.get("benchmark") or e.get("slug") or e.get("label")
        )
        if e_name and dataset_name.lower() in str(e_name).lower():
            mean = e.get("mean") or e.get("score") or e.get("result") or e.get("value")
            if isinstance(mean, dict):
                return mean.get("reward") or mean.get("score") or mean.get("value") or mean.get("pass@1") or mean.get("mean")
            return mean
    return None

def normalize_combos(combos):
    out = []
    for c in combos:
        display = c.get("display", {}) or c
        evals = c.get("evals") or c.get("benchmarks") or c.get("metrics") or []
        
        perf = {}
        for test in PERF_TESTS:
            v = None
            if test["source"] == "indexScore":
                v = c.get("indexScore") or c.get("score") or c.get("index") or c.get("codingAgentIndex") or c.get("mean")
            elif test["source"] == "eval":
                v = _extract_eval_reward(evals, test["dataset"])
                if v is None:
                    # 兜底方案：直接在顶层寻找包含 benchmark 名字的键
                    for k, val in c.items():
                        if test["dataset"].replace("-", "").lower() in k.lower() and isinstance(val, (int, float)):
                            v = val
                            break
            
            if v is not None:
                try:
                    perf[test["key"]] = float(v)
                except (TypeError, ValueError):
                    pass
                    
        agent = c.get("agentName") or c.get("agent") or c.get("name") or c.get("displayName")
        model = display.get("model") or display.get("modelName") or c.get("model") or c.get("hostModelSlug")
        
        entry = {
            "agent": agent,
            "model": model,
            "displayLabel": c.get("displayLabel") or c.get("label"),
            "hostModelSlug": c.get("hostModelSlug") or c.get("slug"),
            "provider": c.get("provider") or c.get("providerName"),
            "perf": perf,
        }
        
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

    try:
        combos = scrape_with_playwright()
    except Exception as e:
        errors.append(f"playwright: {e}")
        print(f"  Playwright failed: {e}")

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
        print(f"  WARNING: expected at least {MIN_COMBOS_EXPECTED} combinations, got {len(normalized)}")

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
