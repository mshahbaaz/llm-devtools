import os


def get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Get a free key at https://openrouter.ai/keys "
            "and export it: export OPENROUTER_API_KEY=sk-or-..."
        )
    return key
