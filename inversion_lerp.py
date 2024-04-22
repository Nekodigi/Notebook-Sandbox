from io import BytesIO

import requests
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image
from torchvision.transforms import functional as T
import tqdm


def load_image(url, size=None):
    response = requests.get(url, timeout=0.2)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    if size is not None:
        img = img.resize(size)
    return img


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(
    device
)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

input_image = load_image(
    "https://cdn.pixabay.com/photo/2015/10/09/00/55/lotus-978659_1280.jpg",
    size=(512, 512),
)
input_image_prompt = "Photograph of lotus flower in a pond"

with torch.no_grad():
    latent = pipe.vae.encode(T.to_tensor(input_image).unsqueeze(0).to(device) * 2 - 1)
l = 0.18215 * latent.latent_dist.sample()


## Inversion
@torch.no_grad()
def invert(
    start_latents,
    prompt,
    guidance_scale=3.5,
    num_inference_steps=80,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device=device,
):

    # Encode prompt
    text_embeddings = pipe._encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
    )

    # Latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(pipe.scheduler.timesteps)

    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(
            latent_model_input, t, encoder_hidden_states=text_embeddings
        ).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (
            alpha_t_next.sqrt() / alpha_t.sqrt()
        ) + (1 - alpha_t_next).sqrt() * noise_pred

        # Store
        intermediate_latents.append(latents)

    return torch.cat(intermediate_latents)


inverted_latents = invert(l, input_image_prompt, num_inference_steps=50)
inverted_latents.shape
print("AAA")

# Decode the final inverted latents
with torch.no_grad():
    im = pipe.decode_latents(inverted_latents[-1].unsqueeze(0))
img = pipe.numpy_to_pil(im)[0]

# show this img

img.show()
