from __future__ import annotations
import json, os, tempfile

_CACHE_PATH = os.path.join(tempfile.gettempdir(), "noteforge_cache.json")
_cache: dict[str, str] = {}
_loaded = False


def _ensure_loaded() -> None:
    global _cache, _loaded
    if _loaded:
        return
    if os.path.exists(_CACHE_PATH):
        try:
            with open(_CACHE_PATH, encoding="utf-8") as fh:
                _cache = json.load(fh)
        except Exception:
            _cache = {}
    _loaded = True


def get_from_cache(key: str) -> str | None:
    _ensure_loaded()
    return _cache.get(key)


def set_cache(key: str, value: str) -> None:
    _ensure_loaded()
    _cache[key] = value
    try:
        with open(_CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(_cache, fh, indent=2, ensure_ascii=False)
    except Exception:
        pass


def clear_cache() -> None:
    global _cache, _loaded
    _cache, _loaded = {}, True
    try:
        os.remove(_CACHE_PATH)
    except FileNotFoundError:
        pass
