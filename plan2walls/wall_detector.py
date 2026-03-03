from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
from PIL import Image
import numpy as np

__all__ = ["WallDetector"]


class WallDetector:
    def __init__(
        self,
        ckpt_path="checkpoint-200000/controlnet",
        stable_diffusion_ckpt = "/models/CompVis/stable-diffusion-v1-4"
    ):
        controlnet = ControlNetModel.from_pretrained(ckpt_path, torch_dtype=torch.float16)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            stable_diffusion_ckpt,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        self.pipe = pipe.to(self.device)

    def detect(self, image_path, hyperparameters):
        image = Image.open(image_path).convert("RGB")
        if hyperparameters["RESOLUTION"]["KEEP_ORIGINAL"]:
            width, height = image.size
        else:
            width, height = hyperparameters["RESOLUTION"]["WIDTH"], hyperparameters["RESOLUTION"]["HEIGHT"]
            image = image.resize((width, height))
        out = self.pipe(
            "A floor plan",
            num_inference_steps=hyperparameters["N_INFERENCE_STEPS"],
            image=image,
            height=height,
            width=width,
            controlnet_conditioning_scale=hyperparameters["CONTROLNET_CONDITIONING_SCALE"],
            guidance_scale=hyperparameters["GUIDANCE_SCALE"],
            generator=[torch.manual_seed(s) for s in range(hyperparameters["N_IMAGES"])],
            num_images_per_prompt=hyperparameters["N_IMAGES"]
        )
        I = np.stack([np.asarray(img) for img in out.images]).mean(axis=0).mean(axis=-1)
        I = np.uint8(I)
        return Image.fromarray(np.uint8((I > 127) * 255))