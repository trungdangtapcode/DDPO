
# Using Stable Diffusion with Both Text Input and Image Input

### (No VRAM Duplication – Unified Pipeline Guide)

This document explains how to use **Stable Diffusion** for both **text-to-image** and **image-to-image** generation **without doubling VRAM usage**.
It includes details about pipelines, model components, and recommended usage patterns.

---

# 1. Overview

Stable Diffusion can generate images from:

1. **Text Input** → *txt2img*
2. **Image Input** → *img2img*

Both modes **reuse the same model weights**:

* **UNet**
* **VAE**
* **CLIP text encoder**
* **Schedulers**

Because of this, you can use **one pipeline** or **two pipelines sharing the same components**, and VRAM will **not** double.

---

# 2. Does img2img cause VRAM duplication?

### **❌ No — VRAM does NOT duplicate.**

Diffusers provides two pipeline classes:

| Mode          | Pipeline Class                   |
| ------------- | -------------------------------- |
| Text → Image  | `StableDiffusionPipeline`        |
| Image → Image | `StableDiffusionImg2ImgPipeline` |

Even though these are separate classes, they reuse the **same model components** internally.

You only get duplicated VRAM usage if you **load two completely separate pipelines from disk** instead of sharing the modules.

---

# 3. Recommended Setup (One Unified Pipeline)

You can use **one StableDiffusionPipeline** for both text and image inputs:

```python
from diffusers import StableDiffusionPipeline
from PIL import Image

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4"
).to("cuda")
```

### ✓ Text → Image

```python
result = pipe(prompt="a fantasy castle")
image = result.images[0]
image.save("txt2img.png")
```

### ✓ Image → Image (img2img)

```python
init_image = Image.open("input.png").convert("RGB")

result = pipe(
    prompt="cinematic style",
    image=init_image,
    strength=0.6
)
result.images[0].save("img2img.png")
```

**This single pipeline handles both modes without extra VRAM usage.**

---

# 4. Strength Parameter (img2img)

`strength` controls how much the output differs from the input:

| Strength | Effect                   |
| -------- | ------------------------ |
| 0.2–0.4  | Very similar to original |
| 0.5–0.7  | Moderate change          |
| 0.8–1.0  | Large transformation     |

Example:

```python
strength=0.6
```

---

# 5. What if you want two pipeline objects?

You can safely create **two pipelines that share the same model objects**, ensuring **zero duplication**:

```python
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline
)

base = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

img2img = StableDiffusionImg2ImgPipeline(
    vae=base.vae,
    text_encoder=base.text_encoder,
    tokenizer=base.tokenizer,
    unet=base.unet,
    scheduler=base.scheduler,
    safety_checker=base.safety_checker,
    feature_extractor=base.feature_extractor,
)

base.to("cuda")
img2img.to("cuda")
```

Both pipelines now point to **the same weights in VRAM**.

---

# 6. Why you don’t need the CLIP image encoder

Stable Diffusion uses:

* **CLIP text encoder (ViT-L/14)** → for text
* **VAE encoder** → for images
  (NOT CLIP’s image encoder)

Stable Diffusion **does not include** the CLIP image encoder because its image workflow is based on **VAE latent space**, not CLIP space.

This is why:

* You do *not* need a CLIP image encoder for img2img.
* You can directly upload an image and use the **VAE** to encode it.

---

# 7. Summary

### ✔ You can use both text input and image input in one pipeline

### ✔ No VRAM duplication

### ✔ Image-to-image uses VAE, not CLIP

### ✔ `strength` determines similarity to original

### ✔ You can share weights if you want separate pipeline objects
