# Data state — `data/manifests/canon6`

| split | total | real | fake | fake generators |
|---|---|---|---|---|
| **train** | 100,204 | 50,102 | 50,102 | 25 |
| **val** | 12,502 | 6,251 | 6,251 | 25 |
| **test** | 157,673 | 76,535 | 81,138 | 33 |

**Images actually pushed into training (train + val): 112,706** (100,204 train / 12,502 val). Test is never trained on.

## train — by native-size bucket

### <=341 px — 65,432 images · 32,716 real / 32,716 fake (1.00:1)

| | sources / generators | subjects |
|---|---|---|
| **REAL** | wildfake 12,121, artifact_ffhq 2,260, artifact_afhq 2,239, artifact_imagenet 2,228, artifact_celebahq 2,226, artifact_pro_gan 2,212 | general scenes 10,930, faces 7,598, animals 5,175, church 2,720 |
| **AI** | glide 1,636, cips 1,636, cycle_gan 1,636, projected_gan 1,636, face_synthetics 1,636, stylegan2 1,636 | faces 13,078, general scenes 10,510, animals 1,567, bedroom 1,418 |

### 342-512 px — 8,420 images · 4,210 real / 4,210 fake (1.00:1)

| | sources / generators | subjects |
|---|---|---|
| **REAL** | flickr30k_web 2,573, afhq_512 779, coco_640 553, lsun_bedroom 305 | other 2,573, animals 779, general scenes 553, bedroom 305 |
| **AI** | sd14 1,501, sdxl 1,429, sd21 1,280 | other 4,210 |

### 513-768 px — 8,466 images · 4,233 real / 4,233 fake (1.00:1)

| | sources / generators | subjects |
|---|---|---|
| **REAL** | coco_640 4,221, openimages_1024 11, sid_real 1 | general scenes 4,221, other 12 |
| **AI** | sd21 1,536, sdxl 1,384, sd14 1,313 | other 4,233 |

### 769-1024 px — 17,886 images · 8,943 real / 8,943 fake (1.00:1)

| | sources / generators | subjects |
|---|---|---|
| **REAL** | openimages_1024 6,255, sid_real 1,609, celebahq_1024 1,079 | other 7,858, faces 1,079, general scenes 6 |
| **AI** | midjourney_v6 7,197, flux_sid 1,746 | other 8,943 |

## val — by native-size bucket

### <=341 px — 8,160 images · 4,080 real / 4,080 fake (1.00:1)

| | sources / generators | subjects |
|---|---|---|
| **REAL** | wildfake 1,501, artifact_pro_gan 293, artifact_afhq 288, artifact_cycle_gan 285, artifact_ffhq 284, artifact_lsun 283 | general scenes 1,353, faces 938, animals 678, church 333 |
| **AI** | latent_diffusion 204, star_gan 204, stylegan1 204, taming_transformer 204, gansformer 204, projected_gan 204 | faces 1,669, general scenes 1,310, animals 182, bedroom 180 |

### 342-512 px — 1,116 images · 558 real / 558 fake (1.00:1)

| | sources / generators | subjects |
|---|---|---|
| **REAL** | flickr30k_web 353, coco_640 84, afhq_512 81, lsun_bedroom 40 | other 353, general scenes 84, animals 81, bedroom 40 |
| **AI** | sd14 212, sdxl 187, sd21 159 | other 558 |

### 513-768 px — 992 images · 496 real / 496 fake (1.00:1)

| | sources / generators | subjects |
|---|---|---|
| **REAL** | coco_640 495, openimages_1024 1 | general scenes 495, other 1 |
| **AI** | sd21 193, sdxl 164, sd14 139 | other 496 |

### 769-1024 px — 2,234 images · 1,117 real / 1,117 fake (1.00:1)

| | sources / generators | subjects |
|---|---|---|
| **REAL** | openimages_1024 785, sid_real 199, celebahq_1024 133 | other 984, faces 133 |
| **AI** | midjourney_v6 899, flux_sid 218 | other 1,117 |

## test — by native-size bucket

### <=341 px — 121,075 images · 44,404 real / 76,671 fake (0.58:1)

| | sources / generators | subjects |
|---|---|---|
| **REAL** | wildfake 16,356, artifact_cycle_gan 3,074, artifact_lsun 3,064, artifact_imagenet 3,051, artifact_celebahq 3,048, artifact_pro_gan 3,036 | general scenes 15,023, faces 10,123, animals 7,060, church 3,645 |
| **AI** | ddpm 30,877, ddim 29,995, deepfloyd_if 3,508, palette 2,046, lama 2,046, mat 2,043 | bedroom 34,625, general scenes 17,877, church 13,264, other 5,631 |

### 342-512 px — 18,069 images · 17,551 real / 518 fake (33.88:1)

| | sources / generators | subjects |
|---|---|---|
| **REAL** | flickr30k_web 11,008, afhq_512 3,140, coco_640 2,134, lsun_bedroom 1,269 | other 11,008, animals 3,140, general scenes 2,134, bedroom 1,269 |
| **AI** | sd14 210, sdxl 161, sd21 147 | other 518 |

### 513-768 px — 12,668 images · 12,128 real / 540 fake (22.46:1)

| | sources / generators | subjects |
|---|---|---|
| **REAL** | coco_640 12,096, openimages_1024 28, sid_real 4 | general scenes 12,096, other 32 |
| **AI** | sd21 205, sdxl 192, sd14 143 | other 540 |

### 769-1024 px — 5,856 images · 2,447 real / 3,409 fake (0.72:1)

| | sources / generators | subjects |
|---|---|---|
| **REAL** | openimages_1024 1,741, sid_real 418, celebahq_1024 288 | other 2,159, faces 288 |
| **AI** | sid_tampered 2,289, midjourney_v6 901, flux_sid 219 | other 3,409 |

### >1024 px — 5 images · 5 real / 0 fake (no fakes)

| | sources / generators | subjects |
|---|---|---|
| **REAL** | openimages_1024 4, sid_real 1 | other 5 |
| **AI** | — |  |

## train — every source and generator

**18 real sources** (50,102 images)

| real source | n |
|---|---|
| wildfake | 12,121 |
| openimages_1024 | 6,266 |
| coco_640 | 4,793 |
| flickr30k_web | 2,580 |
| artifact_ffhq | 2,260 |
| artifact_afhq | 2,239 |
| artifact_imagenet | 2,228 |
| artifact_celebahq | 2,226 |
| artifact_pro_gan | 2,212 |
| artifact_lsun | 2,197 |
| artifact_cycle_gan | 2,178 |
| artifact_coco | 2,166 |
| artifact_landscape | 1,772 |
| sid_real | 1,610 |
| celebahq_1024 | 1,079 |
| lsun_bedroom | 855 |
| afhq_512 | 779 |
| artifact_metfaces | 541 |

**25 fake generators** (50,102 images)

| generator | n |
|---|---|
| midjourney_v6 | 7,197 |
| sd21 | 2,816 |
| sd14 | 2,814 |
| sdxl | 2,813 |
| flux_sid | 1,746 |
| glide | 1,636 |
| cips | 1,636 |
| cycle_gan | 1,636 |
| projected_gan | 1,636 |
| face_synthetics | 1,636 |
| stylegan2 | 1,636 |
| star_gan | 1,636 |
| pro_gan | 1,636 |
| vq_diffusion | 1,636 |
| latent_diffusion | 1,636 |
| taming_transformer | 1,636 |
| gau_gan | 1,636 |
| sfhq | 1,636 |
| stylegan3 | 1,636 |
| diffusion_gan | 1,636 |
| stylegan1 | 1,636 |
| gansformer | 1,636 |
| big_gan | 1,636 |
| denoising_diffusion_gan | 1,636 |
| stable_diffusion | 1,632 |

## Held out of training entirely (test only)

| generator | n in test |
|---|---|
| ddim | 29,995 |
| ddpm | 30,877 |
| deepfloyd_if | 3,508 |
| generative_inpainting | 2,043 |
| lama | 2,046 |
| mat | 2,043 |
| palette | 2,046 |
| sid_tampered | 2,289 |
