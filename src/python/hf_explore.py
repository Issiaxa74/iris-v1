# -*- coding: utf-8 -*-
"""Exploracion HF parquet via HTTP Range (sin descargas completas). F4b."""
import json, urllib.request, sys

API = "https://huggingface.co/api/datasets/{ds}/parquet/{cfg}/{split}"

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "iris-f4b/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

def head_size(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "iris-f4b/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.headers.get("Content-Length"), r.url

CANDIDATES = [
    ("wikimedia/wikipedia", "20231101.es", "train"),
    ("wikimedia/wikipedia", "20231101.en", "train"),
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train"),
    ("allenai/c4", "es", "train"),
    ("OpenAssistant/oasst2", "default", "train"),
]

for ds, cfg, split in CANDIDATES:
    try:
        files = get(API.format(ds=ds, cfg=cfg, split=split))
        print(f"\n=== {ds} [{cfg}/{split}] : {len(files)} parquet(s)")
        # muestra los 3 primeros y cuenta total
        total = 0
        for i, f in enumerate(files):
            size, url = head_size(f)
            n = int(size or 0)
            total += n
            if i < 3 or i == len(files) - 1:
                print(f"  [{i}] {n/1e6:.1f} MB  {f.rsplit('/',1)[-1]}")
                print(f"      {f}")
        print(f"  TOTAL {len(files)} archivos, {total/1e9:.2f} GB")
    except Exception as e:
        print(f"\n=== {ds} [{cfg}/{split}] ERROR: {e}")
