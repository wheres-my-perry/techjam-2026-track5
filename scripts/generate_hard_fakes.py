"""Generate hard synthetic training fakes, topic-matched to the canon2 corpus.

    python -m scripts.generate_hard_fakes --manifest data/manifests/canon2_train.csv \
        --count 500

Design (Thinh, 2026-08-29): the model must learn from "smartly generated"
images — the kind that keep photographic structure and imperfections.
- 60% img2img: a REAL image sampled from our own corpus is regenerated
  through SDXL at strength 0.35-0.65 -> real layout, generator pixels.
  On-topic by construction. The hardest known class of fake.
- 40% txt2img: prompt bank mirroring corpus topic proportions (faces 35%,
  churches 20%, objects 20%, animals 15%, landscapes 10%), engineered for
  realism-with-flaws (candid/amateur/motion-blur cues; negative prompt bans
  render/illustration looks).
- Outputs downscaled to a random 200-256 short side (corpus range) — no new
  size leak. NO fakes-only noise/JPEG post-processing (would teach noise=fake).
- Train-split only. generator=sdxl_hard, source=selfgen. Resume-safe.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time

TOPICS = {
    "face": 0.35, "church": 0.20, "object": 0.20, "animal": 0.15,
    "landscape": 0.10,
}

SUBJECTS = {
    "face": ["a middle-aged man", "a young woman", "an elderly person",
             "a teenager", "a bearded man", "a woman wearing glasses",
             "a child laughing", "a person in a winter coat"],
    "church": ["a small stone church", "a whitewashed chapel",
               "a gothic cathedral facade", "a rural wooden church",
               "a brick church with a steeple", "an old mission church"],
    "object": ["a coffee mug on a desk", "a parked bicycle", "a delivery van",
               "a bowl of fruit", "a laptop on a table", "a pair of sneakers",
               "a motorcycle at a curb", "a plate of street food"],
    "animal": ["a tabby cat", "a golden retriever", "a red fox",
               "a horse in a field", "a parrot on a branch", "a brown bear"],
    "landscape": ["a foggy valley", "a rocky coastline", "rolling farmland",
                  "a mountain lake", "a forest trail in autumn",
                  "sand dunes at dusk"],
}

SETTINGS = ["outdoors on an overcast day", "in warm evening light",
            "under harsh midday sun", "indoors with window light",
            "at golden hour", "on a rainy afternoon", "in soft shade",
            "under streetlights at night"]

STYLE = ["candid amateur photo", "casual smartphone photo",
         "photojournalism shot", "family album photo", "street photography",
         "documentary photo"]

FLAWS = ["slight motion blur", "slightly overexposed", "harsh on-camera flash",
         "a bit underexposed", "shallow depth of field, focus slightly missed",
         "visible sensor grain", "mild lens distortion at the edges"]

NEGATIVE = ("illustration, painting, 3d render, cgi, cartoon, anime, "
            "watermark, text, logo, oversaturated, airbrushed, studio "
            "beauty retouch, perfect symmetry")

SRC_TOPIC = [("afhq", "animal"), ("church", "church"), ("lsun", "church"),
             ("celeba", "face"), ("ffhq", "face"), ("sfhq", "face"),
             ("metfaces", "face"), ("landscape", "landscape"),
             ("coco", "object"), ("imagenet", "object")]


def topic_of_source(source: str) -> str:
    s = source.lower()
    for key, t in SRC_TOPIC:
        if key in s:
            return t
    return "object"


def make_prompt(rng, topic):
    return (f"{rng.choice(STYLE)} of {rng.choice(SUBJECTS[topic])}, "
            f"{rng.choice(SETTINGS)}, {rng.choice(FLAWS)}, realistic "
            f"colors, natural skin texture" if topic == "face" else
            f"{rng.choice(STYLE)} of {rng.choice(SUBJECTS[topic])}, "
            f"{rng.choice(SETTINGS)}, {rng.choice(FLAWS)}, realistic colors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/canon2_train.csv",
                    help="corpus manifest: real rows are img2img seeds + "
                         "topic source")
    ap.add_argument("--count", type=int, default=500)
    ap.add_argument("--img2img-frac", type=float, default=0.6)
    ap.add_argument("--out", default="data/hack/claude")
    ap.add_argument("--out-manifest", default="data/manifests/selfgen_raw.csv")
    ap.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-native", action="store_true", default=True,
                    help="save at generator-native 1024 (default). Downscaling fakes only would make the shrink factor a label.")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from src.data import load_image, load_manifest

    rng = random.Random(args.seed)
    reals = [s for s in load_manifest(args.manifest) if s.label == 0]
    rng.shuffle(reals)
    os.makedirs(args.out, exist_ok=True)

    # plan the 500 before loading any model (deterministic, resume-stable)
    plan = []
    topics = list(TOPICS)
    weights = [TOPICS[t] for t in topics]
    for i in range(args.count):
        mode = "i2i" if rng.random() < args.img2img_frac else "t2i"
        if mode == "i2i":
            seed_img = reals[i % len(reals)]
            topic = topic_of_source(seed_img.source)
            plan.append((i, mode, topic, seed_img.path,
                         rng.uniform(0.35, 0.65), rng.randint(0, 2**31)))
        else:
            topic = rng.choices(topics, weights=weights)[0]
            plan.append((i, mode, topic, "", 0.0, rng.randint(0, 2**31)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    from diffusers import (AutoPipelineForImage2Image, AutoPipelineForText2Image)
    t2i = AutoPipelineForText2Image.from_pretrained(
        args.model, torch_dtype=dtype, variant="fp16" if device == "cuda"
        else None).to(device)
    i2i = AutoPipelineForImage2Image.from_pipe(t2i)
    print(f"pipelines ready on {device}", flush=True)

    rows, t0, made = [], time.time(), 0
    for i, mode, topic, seed_path, strength, gseed in plan:
        out_path = os.path.join(args.out, f"sg_{i:04d}.png")
        rows.append({"path": out_path, "label": 1, "generator": "sdxl_hard",
                     "source": "selfgen"})
        if os.path.exists(out_path):
            continue
        prng = random.Random(gseed)
        prompt = make_prompt(prng, topic)
        gen = torch.Generator(device=device).manual_seed(gseed)
        try:
            if mode == "i2i":
                init = load_image(seed_path).resize((1024, 1024),
                                                    Image.LANCZOS)
                img = i2i(prompt=prompt, negative_prompt=NEGATIVE,
                          image=init, strength=strength,
                          num_inference_steps=args.steps, guidance_scale=5.5,
                          generator=gen).images[0]
            else:
                img = t2i(prompt=prompt, negative_prompt=NEGATIVE,
                          width=1024, height=1024,
                          num_inference_steps=args.steps, guidance_scale=5.5,
                          generator=gen).images[0]
        except Exception as e:
            print(f"skip {i}: {e}", flush=True)
            continue
        if not args.keep_native:   # fakes-only downscale = shrink-factor leak (Thinh's bucket rule); default off since 2026-08-29
            short = prng.randint(200, 256)
            img = img.resize((short, short), Image.LANCZOS)
        tmp = out_path + ".tmp.png"
        img.save(tmp, format="PNG")
        os.replace(tmp, out_path)
        made += 1
        if made % 20 == 0:
            rate = made / (time.time() - t0)
            print(f"{made} generated ({rate*3600:.0f}/h, plan {i+1}/"
                  f"{args.count})", flush=True)

    os.makedirs(os.path.dirname(args.out_manifest) or ".", exist_ok=True)
    with open(args.out_manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "label", "generator",
                                           "source"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {args.out_manifest}")


if __name__ == "__main__":
    main()
