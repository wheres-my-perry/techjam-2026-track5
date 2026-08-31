#!/bin/bash
set -u
cd /workspace/techjam-2026-track5/data/wildfake/raw/Images
for z in Diffusion_based/DDIM.zip Diffusion_based/DDPM.zip Real/afhq.zip Real/celebahq.zip Real/church.zip Real/coco.zip Real/ffhq.zip Real/imagenet.zip; do
  out="${z%.zip}"
  ( mkdir -p "$out" && unzip -q -o "$z" -d "$out" && rm -f "$z" && echo "DONE $z" ) &
done
wait
echo WF_EXTRACT_ALL_DONE
