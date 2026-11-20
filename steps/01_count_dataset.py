"""Step 1: Count MMDocRAG questions in the selected domain for each split."""

import json

import requests

from common import DATASET_BASE, DOMAIN, SPLITS

counts = {}
for split, filename in SPLITS.items():
    with requests.get(
        f"{DATASET_BASE}/{filename}", stream=True, timeout=(10, 120)
    ) as response:
        response.raise_for_status()
        counts[split] = sum(
            DOMAIN is None or json.loads(line)["domain"] == DOMAIN
            for line in response.iter_lines()
            if line.strip()
        )

print(f"Domain: {DOMAIN or 'all'}")
for split, count in counts.items():
    print(f"{split:<12} {count}")
print(f"{'total':<12} {sum(counts.values())}")
