"""Swappable LLM client (any OpenAI-compatible chat endpoint) with a disk cache.

  LLM_API_KEY   required to enable the LLM path
  LLM_BASE_URL  default Gemini's OpenAI-compatible endpoint (free tier)
  LLM_MODEL     default gemini-2.0-flash          (swap to gpt-4o-mini, a stronger Gemini, OpenRouter models, ...)

If no key is set, `available()` is False and the pipeline uses its offline fallbacks (nothing crashes).
"""
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".llm_cache")


def cfg():
    return {
        "key": os.environ.get("LLM_API_KEY", ""),
        "base": os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/"),
        "model": os.environ.get("LLM_MODEL", "gemini-3.1-flash-lite"),
    }


def available():
    return bool(cfg()["key"])


def chat(prompt, retries=3):
    c = cfg()
    os.makedirs(CACHE, exist_ok=True)
    h = hashlib.sha256((c["model"] + prompt).encode()).hexdigest()
    f = os.path.join(CACHE, h + ".txt")
    if os.path.exists(f):
        return open(f, encoding="utf-8").read()
    native = "generativelanguage.googleapis.com" in c["base"]
    if native:   # Gemini native REST (works with AI Studio keys; the OpenAI-compat route 404'd for our key)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{c['model']}:generateContent"
        body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                           "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}).encode()
        hdr = {"Content-Type": "application/json", "x-goog-api-key": c["key"]}
    else:
        url = c["base"] + "/chat/completions"
        body = json.dumps({"model": c["model"], "temperature": 0,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        hdr = {"Content-Type": "application/json", "Authorization": "Bearer " + c["key"]}
    req = urllib.request.Request(url, body, hdr)
    last = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                j = json.load(r)
            out = j["candidates"][0]["content"]["parts"][0]["text"] if native else j["choices"][0]["message"]["content"]
            open(f, "w", encoding="utf-8").write(out)
            return out
        except urllib.error.HTTPError as e:
            last = f"{e.code} {e.read()[:150]}"
            if e.code in (400, 401, 403, 404):
                break          # not retryable
            time.sleep(3 * (i + 1))
        except Exception as e:
            last = e
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"LLM call failed: {last}")


def chat_json(prompt):
    txt = chat(prompt)
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        raise ValueError("no JSON in LLM reply")
    return json.loads(m.group(0))
