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

    def detect(self, image_path, hyperparameters, mask_offset=None):
        image = Image.open(image_path).convert("RGB")
        width_original, height_original = image.size
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
        image_detected = Image.fromarray(np.uint8((I > 127) * 255))

        if mask_offset:
            image_detected = image_detected.resize((width_original, height_original))
            image = np.array(image_detected)
            mask_height_factor = mask_offset["vertical"]
            mask_width_factor = mask_offset["horizontal"]
            if mask_height_factor > 0.01:
                mask_height_factor -= 0.01
            if mask_width_factor > 0.01:
                mask_width_factor -= 0.01
            if mask_width_factor > 0:
                image[:, -round(width_original * mask_width_factor):] = 255
            if mask_height_factor > 0:
                image[-round(height_original * mask_height_factor):, :] = 255
            image_detected = Image.fromarray(image)
            image_detected = image_detected.resize((width, height))
        return image_detected