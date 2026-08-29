"""Generate test images with OpenAI image models (DALL-E 3 = the family the contest
benchmark uses; gpt-image-1 = GPT-4o's own image model, a different family).

    export OPENAI_API_KEY=...
    python -m scripts.gen_openai --model dall-e-3 --count 30 --out data/hack/dalle_api
    python -m scripts.gen_openai --model gpt-image-1 --count 30 --out data/hack/gptimage_api

Prompts are drawn at random from a bank of everyday, photo-like subjects (no "digital art"
cues) so the test matches what people actually post. Files are saved as delivered (PNG),
never re-encoded. Images land in a folder that scripts/score_dir.py can score in one go.
"""
from __future__ import annotations

import argparse
import base64
import os
import random
import time

import requests

SUBJECTS = ["a woman waiting at a bus stop in light rain", "a plate of noodles on a cafe table",
            "a golden retriever on a beach at sunset", "a crowded night market street",
            "a man repairing a bicycle in a garage", "a living room with morning light",
            "a mountain lake with a wooden pier", "two friends taking a selfie at a concert",
            "a child blowing out birthday candles", "a parked red scooter on a narrow street",
            "a farmer's market stall with vegetables", "an office desk with a laptop and coffee",
            "a cat sleeping on a windowsill", "a busy train platform in the evening",
            "a bowl of fruit on a kitchen counter", "a family barbecue in a backyard",
            "a street musician playing guitar", "a snowy village road at dusk",
            "a classroom with students at desks", "a fishing boat in a small harbor",
            "a bride and groom in a narrow alley", "a person hiking on a forest trail",
            "an elderly man reading a newspaper on a bench", "a rooftop view of a city skyline",
            "a mechanic under a car lift", "a florist arranging a bouquet",
            "a soccer match on a neighborhood field", "a bakery display of bread",
            "a teenager skateboarding in a parking lot", "a rainy window with city lights"]
STYLES = ["candid smartphone photo", "photo, natural light", "photograph, slightly out of focus",
          "35mm film photo", "documentary photograph", "photo taken by a tourist"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dall-e-3", choices=["dall-e-3", "gpt-image-1"])
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("set OPENAI_API_KEY first")
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    made = 0
    for i in range(args.count):
        prompt = f"{rng.choice(SUBJECTS)}, {rng.choice(STYLES)}"
        body = {"model": args.model, "prompt": prompt, "n": 1, "size": args.size}
        if args.model == "dall-e-3":
            body.update({"quality": "standard", "style": "natural", "response_format": "b64_json"})
        try:
            r = requests.post("https://api.openai.com/v1/images/generations", json=body,
                              headers={"Authorization": f"Bearer {key}"}, timeout=180)
            r.raise_for_status()
            d = r.json()["data"][0]
            if "b64_json" in d:
                img = base64.b64decode(d["b64_json"])
            else:
                img = requests.get(d["url"], timeout=120).content
        except Exception as e:
            print(f"skip {i}: {str(e)[:160]}", flush=True)
            time.sleep(3)
            continue
        p = os.path.join(args.out, f"{args.model}_{args.seed}_{i:03d}.png")
        with open(p, "wb") as fh:
            fh.write(img)
        with open(os.path.join(args.out, "prompts.txt"), "a") as fh:
            fh.write(f"{os.path.basename(p)}\t{prompt}\n")
        made += 1
        print(f"{made}/{args.count} {os.path.basename(p)}  <- {prompt}", flush=True)
    print(f"done: {made} images in {args.out}")


if __name__ == "__main__":
    main()
