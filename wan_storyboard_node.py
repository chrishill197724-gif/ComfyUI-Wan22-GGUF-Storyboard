import torch
import nodes
import comfy.utils
import comfy.model_management
import comfy.latent_formats

class Wan22StoryboardGGUF:
    """
    Optimized Storyboard Node for Wan 2.2 GGUF (5B).
    Provides First, Double-Middle, and Last image placement for 8GB VRAM users.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE", ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "width": ("INT", {"default": 768, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 1280, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "length_seconds": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 60.0, "step": 0.1}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "start_image": ("IMAGE", ),
                "middle_image_1": ("IMAGE", ),
                "middle_image_2": ("IMAGE", ),
                "end_image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("latent", )
    FUNCTION = "encode"
    CATEGORY = "WanVideo/Storyboard"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def encode(self, vae, positive, negative, width, height, length_seconds, fps, batch_size, 
               start_image=None, middle_image_1=None, middle_image_2=None, end_image=None):
        
        # V14.2 Core Math (Verified Stable)
        raw_frames = int(length_seconds * fps) + 1
        total_frames = ((raw_frames - 1) // 4) * 4 + 1
        t_dim_total = ((total_frames - 1) // 4) + 1
        
        latent_h = height // 16
        latent_w = width // 16
        device = comfy.model_management.intermediate_device()
        latent_format = comfy.latent_formats.Wan22()
        
        latent = torch.zeros([1, 48, t_dim_total, latent_h, latent_w], device=device)
        mask = torch.ones([1, 1, t_dim_total, latent_h, latent_w], device=device)

        active_slots = [img for img in [start_image, middle_image_1, middle_image_2, end_image] if img is not None]
        
        if active_slots:
            num_images = len(active_slots)
            indices = [int(i * (total_frames - 1) / (num_images - 1)) for i in range(num_images)] if num_images > 1 else [0]

            for i, img in enumerate(active_slots):
                current_frame = indices[i]
                latent_frame = current_frame // 4 
                
                if latent_frame >= t_dim_total:
                    latent_frame = t_dim_total - 1
                
                ready = comfy.utils.common_upscale(img[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
                encoded = vae.encode(ready).to(device)
                
                latent[:, :, latent_frame:latent_frame+1, :, :] = encoded[:, :, :1, :, :]
                mask[:, :, latent_frame:latent_frame+1, :, :] = 0.0

                if i == num_images - 1 and num_images > 1:
                    for f_idx in range(max(0, t_dim_total - 2), t_dim_total):
                        latent[:, :, f_idx:f_idx+1, :, :] = encoded[:, :, :1, :, :]
                        mask[:, :, f_idx:f_idx+1, :, :] = 0.0
                
                del ready, encoded
                comfy.model_management.soft_empty_cache()

        processed_samples = latent_format.process_out(latent) * mask + latent * (1.0 - mask)

        out_latent = {}
        out_latent["samples"] = processed_samples.repeat((batch_size,) + (1,) * (processed_samples.ndim - 1))
        out_latent["noise_mask"] = mask.repeat((batch_size,) + (1,) * (mask.ndim - 1))
        
        return (out_latent, )

NODE_CLASS_MAPPINGS = {"Wan22StoryboardGGUF": Wan22StoryboardGGUF}
NODE_DISPLAY_NAME_MAPPINGS = {"Wan22StoryboardGGUF": "Wan2.2 Storyboard: First-Mid-Mid-Last (GGUF 5B)"}