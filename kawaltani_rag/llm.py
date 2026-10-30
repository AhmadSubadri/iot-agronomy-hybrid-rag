"""
Lapisan LLM Generation — Multi-Provider untuk A/B Testing (Bab 3.9.3 & Tabel 6 Proposal).

Mendukung:
  - openai     : GPT-4o-mini (OpenAI Cloud API)
  - gemini     : Gemini Flash (Google AI Studio API)
  - ollama     : Llama 3 8B (Lokal On-Premise)
  - openrouter / groq : Opsional tambahan kompatibel OpenAI-API
"""
import os
import time
import json
import urllib.request
from . import config

try:
    from dotenv import load_dotenv
    load_dotenv(config.BASE_DIR / ".env")
except Exception:
    pass


class LLMError(RuntimeError):
    pass


def _post_json(url: str, payload: dict, headers: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _openai_compatible(base_url: str, key_env: str, system: str, user: str,
                       model: str, temperature: float, max_tokens: int, api_key: str = None):
    key = (api_key or "").strip() or os.getenv(key_env)
    if not key:
        raise LLMError(f"{key_env} belum diset di file .env maupun pada panel Pengaturan UI.")
    data = _post_json(
        base_url,
        dict(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        ),
        {"Authorization": f"Bearer {key}"},
    )
    usage = data.get("usage", {})
    return data["choices"][0]["message"]["content"], dict(
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
    )


def _openai(system, user, model, temperature, max_tokens, api_key=None):
    return _openai_compatible(
        "https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY",
        system, user, model, temperature, max_tokens, api_key=api_key,
    )


def _groq(system, user, model, temperature, max_tokens, api_key=None):
    return _openai_compatible(
        "https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY",
        system, user, model, temperature, max_tokens, api_key=api_key,
    )


def _openrouter(system, user, model, temperature, max_tokens, api_key=None):
    return _openai_compatible(
        "https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY",
        system, user, model, temperature, max_tokens, api_key=api_key,
    )


def _gemini(system, user, model, temperature, max_tokens, api_key=None):
    key = (api_key or "").strip() or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY belum diset di file .env maupun pada panel Pengaturan UI.")
    data = _post_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
        dict(
            systemInstruction={"parts": [{"text": system}]},
            contents=[{"role": "user", "parts": [{"text": user}]}],
            generationConfig=dict(temperature=temperature, maxOutputTokens=max_tokens),
        ),
        {},
    )
    usage = data.get("usageMetadata", {})
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    return text, dict(
        prompt_tokens=usage.get("promptTokenCount"),
        completion_tokens=usage.get("candidatesTokenCount"),
    )


def _ollama(system, user, model, temperature, max_tokens, api_key=None):
    host = (api_key or "").strip() if (api_key and api_key.startswith("http")) else os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        data = _post_json(
            f"{host}/api/chat",
            dict(
                model=model,
                stream=False,
                options=dict(temperature=temperature, num_predict=max_tokens),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            ),
            {},
        )
    except Exception as e:
        raise LLMError(f"Tidak dapat terhubung ke Ollama lokal ({host}): {e}")
    return data["message"]["content"], dict(
        prompt_tokens=data.get("prompt_eval_count"),
        completion_tokens=data.get("eval_count"),
    )


_PROVIDERS = {
    "openai": _openai,
    "gemini": _gemini,
    "ollama": _ollama,
    "groq": _groq,
    "openrouter": _openrouter,
}


def generate(system: str, user: str, provider: str = None, model: str = None,
             temperature: float = None, max_tokens: int = None, api_key: str = None) -> dict:
    provider = provider or config.LLM_PROVIDER
    model = model or config.LLM_MODEL
    temperature = config.LLM_TEMPERATURE if temperature is None else temperature
    max_tokens = max_tokens or config.LLM_MAX_TOKENS
    fn = _PROVIDERS.get(provider)
    if not fn:
        raise LLMError(f"Provider tidak dikenal: '{provider}'. Pilihan: {', '.join(_PROVIDERS)}")
    t0 = time.time()
    text, usage = fn(system, user, model, temperature, max_tokens, api_key=api_key)
    return dict(
        text=text.strip(),
        latency_s=round(time.time() - t0, 3),
        provider=provider,
        model=model,
        **usage,
    )

