from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image
import numpy as np
import torch

def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0
    
    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image

def inpaint_image(image_url, mask_image, num_inference_steps=50, prompt="a handsome man with ray-ban sunglasses"):
    # Load and preprocess images
    init_image = load_image(image_url).resize((512, 512))
    mask_image = mask_image.resize((512, 512))
    control_image = make_inpaint_condition(init_image, mask_image)
    
    generator = torch.Generator(device="cuda").manual_seed(1)
    
    # Load models
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )
    
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    
    # Generate image
    result_image = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
        eta=1.0,
        image=init_image,
        mask_image=mask_image,
        control_image=control_image,
    ).images[0]
    
    return result_image

if __name__ == "__main__":
    pass