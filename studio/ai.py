"""Talking to the AI service. You bring your own key, and choose the service and model in the dashboard
(Settings, AI) or in .env. What is saved from the dashboard is kept in ai.json in the data folder and wins.

    PEAKCUT_AI_PROVIDER   gemini | openai | xai | groq | openrouter | anthropic | ollama | custom
    PEAKCUT_AI_MODELS     one or more model ids, comma separated, tried in this order
    PEAKCUT_AI_KEY        your key (or PEAKCUT_AI_KEY_FILE, a file holding it)
    PEAKCUT_AI_BASE_URL   only for "custom" (any service that speaks the OpenAI chat format), or to override

Every service is asked for the same thing: JSON that follows a small schema. Gemini takes the schema directly.
The others are told the schema in the prompt and asked for a JSON object, which works with any chat model.

One picture can also be shown (`see`), which is how the webcam is looked for on a gaming stream. A model that
cannot look at pictures answers with something else, and that is simply taken as "it did not look"."""
import base64
import json
import os
import re
import threading

import httpx

from . import config

# name -> (label shown on the pages, address, how it is spoken to, where to get a key, an example model id)
PROVIDERS = {
    "gemini": ("Gemini", "https://generativelanguage.googleapis.com/v1beta/models", "gemini", "https://aistudio.google.com/apikey", "gemini-3.1-flash-lite"),
    "openai": ("OpenAI", "https://api.openai.com/v1", "openai", "https://platform.openai.com/api-keys", "gpt-4o-mini"),
    "xai": ("Grok", "https://api.x.ai/v1", "openai", "https://console.x.ai", "grok-4"),
    "groq": ("Groq", "https://api.groq.com/openai/v1", "openai", "https://console.groq.com/keys", "llama-3.3-70b-versatile"),
    "openrouter": ("OpenRouter", "https://openrouter.ai/api/v1", "openai", "https://openrouter.ai/keys", "google/gemini-2.5-flash"),
    "anthropic": ("Claude", "https://api.anthropic.com/v1", "anthropic", "https://console.anthropic.com", ""),
    "ollama": ("Ollama (on this machine)", "http://127.0.0.1:11434/v1", "openai", "https://ollama.com", "llama3.1"),
    "custom": ("Custom (OpenAI-compatible)", "", "openai", "", ""),
}
ALIASES = {"google": "gemini", "grok": "xai", "claude": "anthropic", "local": "ollama"}
NO_KEY = {"ollama", "custom"}       # a model running on your own machine needs no key

TOP_15_MODELS = [
    {
        "id": "gemini-3.1-flash-lite",
        "name": "Gemini 3.1 Flash Lite",
        "rpm": 15,
        "tpm": "250K",
        "rpd": 500,
        "tokens": "1,048,576",
        "desc": "Top choice: 500 RPD daily limit, 15 RPM, and 1M token context window."
    },
    {
        "id": "gemini-3.5-flash-lite",
        "name": "Gemini 3.5 Flash Lite",
        "rpm": 15,
        "tpm": "250K",
        "rpd": 500,
        "tokens": "1,048,576",
        "desc": "500 RPD daily allowance, 15 RPM, high speed with 1M tokens."
    },
    {
        "id": "gemini-flash-lite-latest",
        "name": "Gemini Flash Lite (Latest)",
        "rpm": 15,
        "tpm": "250K",
        "rpd": 500,
        "tokens": "1,048,576",
        "desc": "Auto-updated Flash Lite tier alias for maximal daily clipping."
    },
    {
        "id": "gemini-3.8-flash",
        "name": "Gemini 3.8 Flash",
        "rpm": 5,
        "tpm": "250K",
        "rpd": 20,
        "tokens": "1,048,576",
        "desc": "Cutting-edge text & video reasoning with 1M context."
    },
    {
        "id": "gemini-3.6-flash",
        "name": "Gemini 3.6 Flash",
        "rpm": 5,
        "tpm": "250K",
        "rpd": 20,
        "tokens": "1,048,576",
        "desc": "High quality viral moment detector with 1M tokens."
    },
    {
        "id": "gemini-3.5-flash",
        "name": "Gemini 3.5 Flash",
        "rpm": 5,
        "tpm": "250K",
        "rpd": 20,
        "tokens": "1,048,576",
        "desc": "Balanced reasoning and structured JSON clip generation."
    },
    {
        "id": "gemini-3-flash-preview",
        "name": "Gemini 3 Flash Preview",
        "rpm": 5,
        "tpm": "250K",
        "rpd": 20,
        "tokens": "1,048,576",
        "desc": "Fast generation with validated schema support."
    },
    {
        "id": "gemini-2.5-flash",
        "name": "Gemini 2.5 Flash",
        "rpm": 5,
        "tpm": "250K",
        "rpd": 20,
        "tokens": "1,048,576",
        "desc": "Proven high-reliability clipping model."
    },
    {
        "id": "gemini-flash-latest",
        "name": "Gemini Flash (Latest)",
        "rpm": 5,
        "tpm": "250K",
        "rpd": 20,
        "tokens": "1,048,576",
        "desc": "Canonical alias for the latest production Flash model."
    },
    {
        "id": "gemini-4-hydrogen",
        "name": "Gemini 4 Hydrogen",
        "rpm": 5,
        "tpm": "250K",
        "rpd": 20,
        "tokens": "1,048,576",
        "desc": "Advanced next-gen model with 1M context."
    },
    {
        "id": "gemma-4-26b-a4b-it",
        "name": "Gemma 4 26B (a4b)",
        "rpm": 30,
        "tpm": "16K",
        "rpd": 14400,
        "tokens": "262,144",
        "desc": "Massive 14,400 RPD daily quota! 30 RPM, 262K tokens."
    },
    {
        "id": "gemma-4-31b-it",
        "name": "Gemma 4 31B",
        "rpm": 30,
        "tpm": "16K",
        "rpd": 14400,
        "tokens": "262,144",
        "desc": "Massive 14,400 RPD daily quota! 30 RPM, 262K tokens."
    },
    {
        "id": "gemini-robotics-er-2-preview",
        "name": "Gemini Robotics ER 2 Preview",
        "rpm": 5,
        "tpm": "250K",
        "rpd": 20,
        "tokens": "131,072",
        "desc": "Specialized event recognition model with 131K tokens."
    },
    {
        "id": "gemini-3.5-transcribe",
        "name": "Gemini 3.5 Transcribe",
        "rpm": 3,
        "tpm": "10K",
        "rpd": 25,
        "tokens": "98,304",
        "desc": "Audio and transcript alignment specialist."
    },
    {
        "id": "antigravity-preview-latest",
        "name": "Antigravity Preview Latest",
        "rpm": 60,
        "tpm": "100K",
        "rpd": 100,
        "tokens": "1,048,576",
        "desc": "60 RPM, 100 RPD, 100K TPM with 1M token context."
    }
]

# ---------------------------------------------------------------- the settings in use
_lock = threading.Lock()
_saved = {"at": None, "data": {}}
_runtime = {
    "active_key_index": 0,
    "active_model": None,
}


def _file():
    return config.DATA_DIR / "ai.json"


def _from_dashboard() -> dict:
    """What was saved from the dashboard, read again only when the file changes."""
    f = _file()
    try:
        at = f.stat().st_mtime_ns
    except OSError:
        at = 0
    if _saved["at"] != at:
        try:
            data = json.loads(f.read_text(encoding="utf-8")) if at else {}
        except (OSError, ValueError):
            data = {}
        _saved.update(at=at, data=data if isinstance(data, dict) else {})
    return _saved["data"]


def keys() -> list:
    """All configured keys, starting with dashboard keys, or .env keys, or defaults."""
    d = _from_dashboard()
    if d.get("keys") and isinstance(d["keys"], list):
        ks = [str(x).strip() for x in d["keys"] if str(x).strip()]
        if ks:
            return ks
    if d.get("key"):
        raw = str(d["key"]).strip()
        ks = [k.strip() for k in re.split(r"[\r\n,]+", raw) if k.strip()]
        if ks:
            return ks
    return config.ai_keys()


def active_key_index() -> int:
    d = _from_dashboard()
    saved_idx = d.get("active_key_index")
    if saved_idx is not None and isinstance(saved_idx, int):
        idx = saved_idx
    else:
        idx = _runtime["active_key_index"]
    ks = keys()
    if not ks:
        return 0
    return max(0, min(idx, len(ks) - 1))


def set_active_key_index(idx: int) -> int:
    ks = keys()
    if not ks:
        return 0
    idx = max(0, min(idx, len(ks) - 1))
    _runtime["active_key_index"] = idx
    f = _file()
    if f.exists():
        try:
            with _lock:
                data = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    data["active_key_index"] = idx
                    data["key"] = ks[idx]
                    f.write_text(json.dumps(data, indent=2), encoding="utf-8")
                    _saved["at"] = None
        except Exception:
            pass
    return idx


def switch_key(target_idx: int | None = None) -> tuple:
    """Quick key switching: toggles to the next key or the target index."""
    ks = keys()
    if not ks:
        return 0, ""
    if target_idx is not None:
        idx = max(0, min(target_idx, len(ks) - 1))
    else:
        idx = (active_key_index() + 1) % len(ks)
    set_active_key_index(idx)
    return idx, ks[idx]


def key() -> str:
    ks = keys()
    if not ks:
        return ""
    return ks[active_key_index()]


def models() -> list:
    d = _from_dashboard()
    ms = list(d.get("models") or [])
    if not ms:
        ms = list(config.AI_MODELS)
    return ms if ms else list(config.DEFAULT_TOP_15_MODELS)


def active_model() -> str:
    ms = models()
    if not ms:
        return ""
    if _runtime["active_model"] and _runtime["active_model"] in ms:
        return _runtime["active_model"]
    return ms[0]


def set_active_model(model_name: str) -> str:
    ms = models()
    m_clean = str(model_name or "").strip()
    if m_clean in ms:
        _runtime["active_model"] = m_clean
        f = _file()
        if f.exists():
            try:
                with _lock:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and "models" in data:
                        existing = [str(x) for x in data["models"]]
                        if m_clean in existing:
                            existing.remove(m_clean)
                            existing.insert(0, m_clean)
                            data["models"] = existing
                            f.write_text(json.dumps(data, indent=2), encoding="utf-8")
                            _saved["at"] = None
            except Exception:
                pass
        return m_clean
    return active_model()


def is_limit_error(e: Exception) -> bool:
    """Detects 429 rate limit, 503 high demand, quota exhaustion, or token limits."""
    if isinstance(e, AIError):
        if getattr(e, "wait", None) is not None or getattr(e, "daily", False):
            return True
    text = str(e).lower()
    m = re.search(r"http\s*(\d{3})", text)
    if m and m.group(1) in ("429", "503"):
        return True
    return any(p in text for p in (
        "429", "503", "quota", "rate limit", "rate_limit", "resource_exhausted",
        "exhausted", "per day", "perday", "daily", "spikes in demand", "high demand",
        "overloaded", "too many requests", "token limit", "tokens", "exceeded your current quota"
    ))


def current() -> dict:
    """Provider, models, address and key: what the dashboard saved, or else what .env says."""
    d = _from_dashboard()
    p = str(d.get("provider") or config.AI_PROVIDER or "").strip().lower()
    return {"provider": ALIASES.get(p, p),
            "models": models(),
            "base_url": (str(d.get("base_url") or "") if d.get("provider") else config.AI_BASE_URL).rstrip("/"),
            "key": key(),
            "keys": keys(),
            "active_key_index": active_key_index(),
            "active_model": active_model()}


def provider() -> str:
    return current()["provider"]


def label() -> str:
    return PROVIDERS.get(provider(), ("",))[0]


def base_url() -> str:
    c = current()
    return c["base_url"] or PROVIDERS.get(c["provider"], ("", ""))[1]


def ready() -> bool:
    c = current()
    return bool(c["provider"] in PROVIDERS and c["models"] and base_url() and (c["key"] or c["provider"] in NO_KEY))


def public() -> dict:
    """What the settings page and UI may see."""
    c = current()
    ks = keys()
    idx = active_key_index()
    curr_k = ks[idx] if ks else ""
    return {
        "provider": c["provider"],
        "models": c["models"],
        "active_model": active_model(),
        "base_url": c["base_url"],
        "has_key": bool(curr_k),
        "key_hint": curr_k[-4:] if len(curr_k) >= 12 else "",
        "ready": ready(),
        "label": label(),
        "saved_here": bool(_from_dashboard().get("provider")),
        "active_key_index": idx,
        "keys_info": [
            {
                "index": i,
                "hint": k[-4:] if len(k) >= 12 else (k[:4] if k else ""),
                "full_preview": (k[:7] + "..." + k[-4:]) if len(k) >= 16 else ("Key " + str(i + 1)),
                "active": i == idx
            }
            for i, k in enumerate(ks)
        ],
        "top_models": TOP_15_MODELS,
        "providers": [
            {"id": k, "label": v[0], "address": v[1], "needs_key": k not in NO_KEY, "key_url": v[3], "example": v[4]}
            for k, v in PROVIDERS.items()
        ]
    }


def save(provider_id: str, model_ids, address: str = "", new_key=None, new_keys=None) -> dict:
    """Save choice in dashboard with multiple keys and models support."""
    p = ALIASES.get(str(provider_id or "").strip().lower(), str(provider_id or "").strip().lower())
    if p not in PROVIDERS:
        raise ValueError("Choose one of the AI services in the list.")
    if isinstance(model_ids, str):
        model_ids = model_ids.split(",")
    ms = [str(m).strip()[:120] for m in (model_ids or []) if str(m).strip()][:30]
    if not ms:
        raise ValueError("Enter the model to use.")
    if any(not re.fullmatch(r"[\w.\-:/@]+", m) for m in ms):
        raise ValueError("A model id has letters, digits and . - : / only.")
    address = str(address or "").strip().rstrip("/")[:300]
    if address and not re.fullmatch(r"https?://[^\s]+", address):
        raise ValueError("The address has to start with http:// or https://")
    if p == "custom" and not address:
        raise ValueError("A custom service needs its address, for example http://127.0.0.1:1234/v1")
    
    old_ks = keys()
    if new_keys is not None:
        if isinstance(new_keys, list):
            ks = [str(x).strip() for x in new_keys if str(x).strip()]
        else:
            ks = [k.strip() for k in re.split(r"[\r\n,]+", str(new_keys)) if k.strip()]
    elif new_key is not None and str(new_key).strip():
        ks = [k.strip() for k in re.split(r"[\r\n,]+", str(new_key)) if k.strip()]
    else:
        ks = old_ks
    
    if not ks and p not in NO_KEY:
        raise ValueError("Paste the key for this service.")
    
    active_idx = min(active_key_index(), max(0, len(ks) - 1))
    primary_k = ks[active_idx] if ks else ""
    
    with _lock:
        f = _file()
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "provider": p,
            "models": ms,
            "base_url": address,
            "key": primary_k,
            "keys": ks,
            "active_key_index": active_idx
        }, indent=2), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        tmp.replace(f)
        _saved["at"] = None
    return public()


def list_models(provider_id=None, address=None, with_key=None) -> list:
    """The model ids a service offers, so the settings page can show a list to pick from."""
    c = current()
    p = ALIASES.get(str(provider_id or c["provider"]).lower(), str(provider_id or c["provider"]).lower())
    if p not in PROVIDERS:
        raise ValueError("Choose one of the AI services in the list.")
    k = str(with_key or "").strip() or (c["key"] if c["provider"] == p else "")
    base = str(address or "").strip().rstrip("/") or (c["base_url"] if c["provider"] == p else "") or PROVIDERS[p][1]
    if not base:
        raise ValueError("Enter the service's address first.")
    how = PROVIDERS[p][2]
    if how == "gemini":
        r = httpx.get(base, params={"pageSize": 200}, headers={"x-goog-api-key": k}, timeout=20)
    elif how == "anthropic":
        r = httpx.get(f"{base}/models", params={"limit": 200}, headers={"x-api-key": k, "anthropic-version": "2023-06-01"}, timeout=20)
    else:
        r = httpx.get(f"{base}/models", headers={"Authorization": f"Bearer {k}"} if k else {}, timeout=20)
    if r.status_code != 200:
        raise AIError(f"HTTP {r.status_code} {r.text[:200].replace(k or chr(0), '<key>')}")
    data = r.json()
    if how == "gemini":
        found = []
        for m in data.get("models", []):
            if "generateContent" not in (m.get("supportedGenerationMethods") or []):
                continue
            mid = m.get("name", "").split("/")[-1]
            low_id = mid.lower()
            if mid and low_id != "aqa" and not any(t in low_id for t in ("-tts", "-image", "imagen-", "veo-", "lyria-", "embedding")):
                found.append(mid)
    else:
        found = [m.get("id", "") for m in data.get("data", []) if isinstance(m, dict)]
    return sorted({m for m in found if m})[:400]


def test() -> dict:
    """One tiny request with the settings in use, to prove the key, the address and the model all work."""
    ms = models()
    if not ready():
        raise ValueError("Save a service, a model and a key first.")
    chosen_model = active_model() or ms[0]
    try:
        out, sent, wrote = call(chosen_model, 'Answer with a JSON array that holds the single string "ok".', {"type": "ARRAY", "items": {"type": "STRING"}})
        return {"ok": True, "model": chosen_model, "tokens": sent + wrote, "answer": str(out)[:80], "key_index": active_key_index()}
    except Exception as e:
        # If active model failed, try first model that works in list
        for alt_m in ms:
            if alt_m == chosen_model:
                continue
            try:
                out, sent, wrote = call(alt_m, 'Answer with a JSON array that holds the single string "ok".', {"type": "ARRAY", "items": {"type": "STRING"}})
                set_active_model(alt_m)
                return {"ok": True, "model": alt_m, "tokens": sent + wrote, "answer": str(out)[:80], "key_index": active_key_index()}
            except Exception:
                continue
        raise e


class AIError(RuntimeError):
    """A failed AI call. `wait` is how many seconds the service asked us to wait before trying again, when it
    said so (a Retry-After header, or Gemini's retryDelay)."""
    def __init__(self, message, wait=None, daily=False):
        super().__init__(message)
        self.wait = wait
        self.daily = daily        # the limit is for the whole day, so waiting a minute won't help


problem = {"text": "", "at": 0.0}       # why the AI last failed, for the dashboard; cleared when it answers again


def explain(e) -> str:
    """What went wrong with an AI call, in words the owner can act on."""
    text, low = str(e), str(e).lower()
    name = (models() or ["?"])[0]
    m = re.match(r"HTTP (\d{3})\b", text)
    status = m.group(1) if m else ""
    if status in ("401", "403") or "api key" in low or "api_key" in low or "unauthorized" in low or "permission" in low:
        return f"{label()} refused the key. Paste a working key under Settings, AI."
    if "no longer available" in low or "deprecated" in low or "retired" in low or "discontinued" in low:
        return f"{label()} has retired \"{name}\" (or it isn't available to new keys). Open Settings, AI, press Find models and pick a newer model."
    # Only a 404 that is about a model says the model is unknown. A 400 like "API key not found" or a stray
    # "404" inside some other message must not be reported as an unknown model.
    if (status == "404" and "model" in low) or (not status and ("404" in text or "not found" in low or "does not exist" in low or "unknown model" in low or "not_found" in low)) \
            or (status and "is not found for api version" in low):
        return f"{label()} doesn't know the model \"{name}\". Open Settings, AI, press Find models and pick one from the list."
    if "429" in text or "quota" in low or "rate limit" in low or "exhausted" in low:
        return f"{label()} says the key has hit its limit or has no credit left. Check the account with {label()}, or try again later."
    if "connect" in low or "timed out" in low or "timeout" in low or "getaddrinfo" in low or "refused" in low or "nodename" in low or "servname" in low or "nameresolutionerror" in low or "name resolution" in low:
        return f"PeakCut couldn't reach {label()}. Check the internet connection" + (" and that Ollama is running." if provider() == "ollama" else ".")
    return ""


def note_problem(e):
    import time
    why = explain(e)
    if why:
        problem.update(text=why, at=time.time())


def clear_problem():
    problem.update(text="", at=0.0)


def asked_wait(r) -> float | None:
    """How long the service asked us to wait, in seconds, or None when it didn't say."""
    try:
        v = (r.headers.get("retry-after") or "").strip()
        if v:
            return max(0.0, float(v))
    except (ValueError, AttributeError):
        pass
    m = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', getattr(r, "text", "") or "")
    return float(m.group(1)) if m else None


def _fail(r, model):
    text = r.text[:300]
    for k in keys():
        if k:
            text = text.replace(k, "<key>")
    limited = r.status_code in (429, 503)
    raise AIError(f"HTTP {r.status_code} {text}", wait=asked_wait(r) if limited else None,
                  daily=limited and bool(re.search(r"PerDay|per day|daily", r.text or "", re.I)))


# ---------------------------------------------------------------- the schema, said in words for chat models
def _json_schema(s):
    """Gemini's schema spelling (ARRAY, OBJECT...) as ordinary JSON Schema."""
    if isinstance(s, dict):
        return {k: (v.lower() if k == "type" and isinstance(v, str) else _json_schema(v)) for k, v in s.items()}
    if isinstance(s, list):
        return [_json_schema(x) for x in s]
    return s


def _wrapped(prompt, schema):
    return (prompt + "\n\nAnswer with JSON only, no other text and no code fences: one object with a single key "
            "\"items\", whose value follows this JSON Schema:\n" + json.dumps(_json_schema(schema)))


def _json_chunks(text: str):
    """Every balanced {...} or [...] in the text, in the order they appear."""
    out, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            if depth == 0:
                start = i
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0 and start is not None:
                out.append(text[start:i + 1])
                start = None
            elif depth < 0:
                depth = 0
    return out


def _parse(text):
    """The JSON in an answer, even if the model wrapped it in a code fence or a sentence.

    A model that is not careful sometimes puts a sentence in front of the JSON,
    or a second object after it, or forgets a comma. The JSON is looked for
    inside the answer rather than demanded of it."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    out = None
    try:
        out = json.loads(text)
    except ValueError:
        for chunk in _json_chunks(text):
            try:
                out = json.loads(chunk)
                break
            except ValueError:
                continue
        if out is None:
            m = re.search(r"[\[{].*[\]}]", text, re.S)
            if not m:
                raise AIError("The AI did not answer with JSON.")
            try:
                out = json.loads(m.group(0))
            except ValueError:
                raise AIError("The AI did not answer with JSON.")
    if isinstance(out, dict) and "items" in out:
        out = out["items"]
    return out


# ---------------------------------------------------------------- one call
def call(model: str, prompt: str, schema: dict, timeout: float | None = None, use_key: str | None = None):
    """Ask one model. Returns (the parsed JSON answer, tokens sent, tokens written back).

    `timeout` is how long one answer may take. A live stream passes a short one, because a slow
    answer must not stop the recording."""
    how = PROVIDERS[provider()][2]
    return {"gemini": _gemini, "openai": _openai, "anthropic": _anthropic}[how](model, prompt, schema, timeout, use_key)


def see(prompt: str, image: bytes, timeout: float | None = None, use_key: str | None = None) -> tuple:
    """Ask one model about one picture: (the answer as text, tokens in, tokens out)."""
    if not ready() or not image or not models():
        return "", 0, 0
    k = use_key or key()
    model_list = models()
    curr_m = active_model()
    ordered = ([curr_m] + [m for m in model_list if m != curr_m]) if curr_m in model_list else model_list
    b64 = base64.b64encode(image).decode()
    how = PROVIDERS[provider()][2]
    for model in ordered:
        try:
            if how == "gemini":
                body = {"contents": [{"role": "user", "parts": [
                    {"text": prompt},
                    {"inline_data": {"mimeType": "image/jpeg", "data": b64}}]}]}
                r = httpx.post(f"{base_url()}/{_gem(model)}:generateContent", json=body,
                               headers={"x-goog-api-key": k}, timeout=timeout or 120)
                if r.status_code != 200:
                    _fail(r, model)
                data = r.json()
                text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
                u = data.get("usageMetadata", {})
                sent = int(u.get("promptTokenCount") or 0)
                return text, sent, max(0, int(u.get("totalTokenCount") or 0) - sent)
            if how == "anthropic":
                body = {"model": model, "max_tokens": 300, "messages": [{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": prompt}]}]}
                r = httpx.post(f"{base_url()}/messages", json=body, timeout=timeout or 120,
                               headers={"x-api-key": k, "anthropic-version": "2023-06-01"})
                if r.status_code != 200:
                    _fail(r, model)
                data = r.json()
                text = "".join(p.get("text", "") for p in data.get("content") or [] if p.get("type") == "text")
                u = data.get("usage") or {}
                return text, int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
            headers = {"Authorization": f"Bearer {k}"} if k else {}
            body = {"model": model, "stream": False, "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}]}]}
            r = httpx.post(f"{base_url()}/chat/completions", json=body, headers=headers, timeout=timeout or 120)
            if r.status_code != 200:
                _fail(r, model)
            data = _openai_body(r)
            text = (data["choices"][0]["message"].get("content") or "")
            u = data.get("usage") or {}
            return text, int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
        except Exception as e:
            if is_limit_error(e):
                continue
            break
    return "", 0, 0


def _gem(model: str) -> str:
    """Gemini names models "models/gemini-2.5-flash"; the address already ends in /models, so only the bare id is used."""
    return model.removeprefix("models/")


def _gemini(model, prompt, schema, timeout=None, use_key=None):
    k = use_key or key()
    url = f"{base_url()}/{_gem(model)}:generateContent"
    headers = {"x-goog-api-key": k}
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.4, "responseMimeType": "application/json", "responseSchema": schema}}
    r = httpx.post(url, json=body, headers=headers, timeout=timeout or 120)
    used_plain = False
    if r.status_code in (400, 422) and "api key" not in r.text.lower() and "api_key" not in r.text.lower():
        # Models like Gemma in the Gemini list support generateContent without responseSchema.
        plain = {"contents": [{"role": "user", "parts": [{"text": _wrapped(prompt, schema)}]}]}
        r = httpx.post(url, json=plain, headers=headers, timeout=timeout or 120)
        used_plain = True
    if r.status_code != 200:
        _fail(r, model)
    data = r.json()
    text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
    u = data.get("usageMetadata", {})
    sent = int(u.get("promptTokenCount") or 0)
    parsed = _parse(text) if used_plain else json.loads(text)
    return parsed, sent, max(0, int(u.get("totalTokenCount") or 0) - sent)   # the answer, and any thinking billed with it


def _openai_body(r):
    """The answer, whether the service sent plain JSON or an event stream.

    Some OpenAI-compatible servers answer with a stream of events unless the
    request says otherwise, so the answer is read from whichever shape arrived.
    A model that answers with a tool call instead of a message leaves the JSON in
    the call's arguments, so that is used too."""
    try:
        data = r.json()
    except ValueError:
        data = _from_events(r.text)
    for ch in data.get("choices") or []:
        msg = ch.get("message") or {}
        if (msg.get("content") or "").strip():
            continue
        for call in msg.get("tool_calls") or []:
            args = (call.get("function") or {}).get("arguments")
            if args:
                msg["content"] = args
                break
    return data


def _from_events(text: str) -> dict:
    """Read an answer out of a server-sent event stream."""
    parts, usage, calls = [], {}, {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        chunk = line[5:].strip()
        if not chunk or chunk == "[DONE]":
            continue
        try:
            d = json.loads(chunk)
        except ValueError:
            continue
        usage = d.get("usage") or usage
        for ch in d.get("choices") or []:
            for piece in ((ch.get("message") or {}), (ch.get("delta") or {})):
                if piece.get("content"):
                    parts.append(piece["content"])
                for call in piece.get("tool_calls") or []:
                    i = call.get("index", 0)
                    calls[i] = calls.get(i, "") + ((call.get("function") or {}).get("arguments") or "")
    message = {"content": "".join(parts)}
    if not message["content"] and calls:
        message["content"] = "".join(calls.values())
    return {"choices": [{"message": message}], "usage": usage}


def _openai(model, prompt, schema, timeout=None, use_key=None):
    k = use_key or key()
    headers = {"Authorization": f"Bearer {k}"} if k else {}
    body = {"model": model, "messages": [{"role": "user", "content": _wrapped(prompt, schema)}], "stream": False}
    # Most services accept these two extras; a few models refuse one of them, so they are dropped on a refusal.
    extras = [{"temperature": 0.4, "response_format": {"type": "json_object"}}, {"response_format": {"type": "json_object"}}, {}]
    r = None
    for extra in extras:
        r = httpx.post(f"{base_url()}/chat/completions", json={**body, **extra}, headers=headers, timeout=timeout or 180)
        if r.status_code not in (400, 422):
            break
    if r.status_code != 200:
        _fail(r, model)
    data = _openai_body(r)
    u = data.get("usage") or {}
    return _parse(data["choices"][0]["message"].get("content")), int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)


def _anthropic(model, prompt, schema, timeout=None, use_key=None):
    k = use_key or key()
    body = {"model": model, "max_tokens": 8192, "messages": [{"role": "user", "content": _wrapped(prompt, schema)}]}
    r = httpx.post(f"{base_url()}/messages", json=body, timeout=timeout or 180,
                   headers={"x-api-key": k, "anthropic-version": "2023-06-01"})
    if r.status_code != 200:
        _fail(r, model)
    data = r.json()
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    u = data.get("usage") or {}
    return _parse(text), int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
