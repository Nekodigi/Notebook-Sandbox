import os
from io import BytesIO

import PIL
import PIL.Image
import requests
import torch
import tqdm
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image
from torchvision.transforms import functional as T
from transformers import BlipForConditionalGeneration, BlipProcessor

from diffusion.ddim_invert import invert, sample


class DiffusionTools:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4"
        ).to(self.device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large", torch_dtype=torch.float16
        ).to("cuda")

    def load_image(self, url, size=None):
        # also support local image
        if url.startswith("http"):
            response = requests.get(url, timeout=0.2)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            if size is not None:
                img = img.resize(size)
            return img
        else:
            img = Image.open(url).convert("RGB")
            if size is not None:
                img = img.resize(size)
            return img

    def preprocess_img(self, url):
        input_image = self.load_image(
            url,
            size=(512, 512),
        )
        init = "tongue which is "
        inputs = self.processor(input_image, init, return_tensors="pt").to(
            "cuda", torch.float16
        )  # type: ignore
        out = self.model.generate(**inputs)
        prompt = self.processor.decode(out[0], skip_special_tokens=True)
        print(prompt)

        with torch.no_grad():
            latent = self.pipe.vae.encode(
                T.to_tensor(input_image).unsqueeze(0).to(self.device) * 2 - 1
            )
        l = 0.18215 * latent.latent_dist.sample()
        return input_image, l, prompt

    def invert(
        self,
        start_latents,
        prompt,
        guidance_scale=3.5,
        num_inference_steps=80,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt="",
    ):
        return invert(
            self.pipe,
            start_latents,
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            device=self.device,
        )

    def sample(
        self,
        prompt,
        start_step=0,
        start_latents=None,
        guidance_scale=3.5,
        num_inference_steps=30,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt="",
    ):
        return sample(
            self.pipe,
            prompt,
            start_step=start_step,
            start_latents=start_latents,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            device=self.device,
        )

    def decode(self, latent) -> PIL.Image.Image:
        with torch.no_grad():
            im = self.pipe.decode_latents(latent.unsqueeze(0))
        img = self.pipe.numpy_to_pil(im)[0]
        return img

    def cfg_step_multisample(
        self,
        name,
        prompt,
        inverted_latents,
        step_start=20,
        step_step=10,
        cfg_end=20,
        cfg_step=5,
    ):
        for step in range(step_start, inverted_latents.shape[0], step_step):
            for cfg in range(1, cfg_end, cfg_step):
                img = self.sample(
                    prompt,
                    start_latents=inverted_latents[-(step + 1)][None],
                    start_step=step,
                    num_inference_steps=50,
                    guidance_scale=cfg,
                )[0]
                os.makedirs(f"outputs/image/{name}", exist_ok=True)
                img.save(f"outputs/image/{name}/{step}_{cfg}.png")


dt = DiffusionTools()
for i in range(10):
    url = f"inputs/image/{i}.png"  # https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ7L7GME1MADKK-d5K1zKdnTZxNHc0w12yYRw&usqp=CAU

    img, l, prompt = dt.preprocess_img(url)

    inverted_latents = dt.invert(l, prompt, num_inference_steps=50)
    inverted_latents.shape
    print("AAA")

    # # Decode the final inverted latents
    # img = dt.decode(inverted_latents[0])
    # img.save("outputs/image/0.png")
    dt.cfg_step_multisample(i, prompt, inverted_latents)
