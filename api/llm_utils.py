"""
Shared Claude-calling helper for the GenAI feature modules (api/genui.py,
api/subtype_differential.py, and friends). Centralizes the
real-model-with-honest-fallback convention used throughout: returns None
whenever ANTHROPIC_API_KEY isn't set or the call fails, so every caller
handles "unavailable" the same way instead of reimplementing the same
try/except block. Callers own their own prompt construction and response
parsing/validation — this only owns the network call.
"""
import os


def call_claude(prompt: str, max_tokens: int = 500, model: str = "claude-sonnet-5") -> str | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as e:  # anthropic missing, no network, bad key, etc.
        print(f"[llm_utils] call failed ({type(e).__name__}); caller falls back to unavailable.")
        return None
