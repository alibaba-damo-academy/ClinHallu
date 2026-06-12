"""Quick API connectivity check."""
import sys
import time
from pathlib import Path

from openai import OpenAI

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
for candidate in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from common import load_config, resolve_model_api_config

config = load_config(str(CONFIG_PATH))

targets = [
    ("judge", config["models"]["judge"], "judge"),
    ("model_cot", config["models"]["model_cot"], "model_cot"),
    ("gold_cot", config["models"]["gold_cot"][0], "default"),
]

seen = set()
resolved_targets = []
for label, model_ref, purpose in targets:
    resolved = resolve_model_api_config(config, model_ref, purpose=purpose)
    key = (resolved["base_url"], resolved["model_name"])
    if key in seen:
        continue
    seen.add(key)
    resolved_targets.append((label, resolved))

print("Resolved targets:")
for label, resolved in resolved_targets:
    print(f"  {label}: {resolved['model_name']} @ {resolved['base_url']}")
print("=" * 50)

for label, resolved in resolved_targets:
    client = OpenAI(
        base_url=resolved["base_url"],
        api_key=resolved["api_key"],
        timeout=30.0,
        max_retries=0,
    )
    print(f"\nTesting {label}: {resolved['model_name']}")
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=resolved["model_name"],
            messages=[{"role": "user", "content": "Say 'hello' in one word."}],
            max_tokens=10,
        )
        elapsed = time.time() - start
        content = response.choices[0].message.content
        print(f"  ✅ OK | {elapsed:.1f}s | Response: {content}")
    except Exception as error:
        elapsed = time.time() - start
        print(f"  ❌ FAILED | {elapsed:.1f}s | {error}")

print("\n" + "=" * 50)
print("Done.")
