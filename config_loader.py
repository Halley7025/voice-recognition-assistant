import os, yaml

_config = None

def load_config():
    global _config
    if _config is None:
        path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(path, "r", encoding="utf-8") as f:
            _config = yaml.safe_load(f)
    return _config

def get(key_path, default=None):
    """Get config value by dot-separated path, e.g. 'audio.sample_rate'"""
    cfg = load_config()
    keys = key_path.split(".")
    val = cfg
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return default
    return val if val is not None else default
