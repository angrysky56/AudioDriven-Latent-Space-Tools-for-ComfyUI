import torch
import math
import numpy as np
from typing import List, Dict, Tuple

class TimestampNoiseGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_params": ("NOISE_PARAMS",),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "noise_type": (["gaussian", "salt_pepper", "perlin"],),
                "analysis_type": ("ANALYSIS_TYPE",),
            },
            "optional": {
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
            }
        }

    RETURN_TYPES = ("LATENT", "TIMESTAMPS")
    FUNCTION = "generate_timestamp_noise"
    CATEGORY = "audio/noise"

    def __init__(self):
        pass

    def rand_perlin_2d(self, shape, res, fade=lambda t: 6*t**5 - 15*t**4 + 10*t**3):
        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1]), indexing="ij"), dim=-1) % 1
        angles = 2*math.pi*torch.rand(res[0]+1, res[1]+1)
        gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

        tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)
        dot = lambda grad, shift: (torch.stack((grid[:shape[0],:shape[1],0] + shift[0], grid[:shape[0],:shape[1], 1] + shift[1]), dim=-1) * grad[:shape[0], :shape[1]]).sum(dim=-1)

        n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
        n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
        n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
        n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])

        t = fade(grid[:shape[0], :shape[1]])
        return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])

    def generate_timestamp_noise(self, noise_params, width, height, noise_type, analysis_type, max_frames=0):
        if not isinstance(noise_params, dict):
            noise_params = {}
        timestamps = noise_params.get("timestamps", [])
        if len(timestamps) == 0:
            return ({"samples": torch.zeros((1, 4, height // 8, width // 8), dtype=torch.float32)}, [0.0])

        # Downsample if max_frames is set and exceeded
        if max_frames > 0 and len(timestamps) > max_frames:
            indices = np.linspace(0, len(timestamps) - 1, max_frames, dtype=int)
            timestamps = [timestamps[i] for i in indices]

        batch_size = len(timestamps)
        latent_height, latent_width = height // 8, width // 8
        noise_batch = torch.zeros((batch_size, 4, latent_height, latent_width), dtype=torch.float32)

        type_params = noise_params.get(noise_type, {})
        base_intensity = float(type_params.get("intensity", 1.0))
        base_persistence = float(type_params.get("persistence", 0.8))
        base_grain = float(type_params.get("grain", 0.5))

        last_ts = float(timestamps[-1]) if len(timestamps) > 0 else 0.0

        for i, timestamp in enumerate(timestamps):
            time_scale = (float(timestamp) / last_ts) if last_ts > 0 else 0.0
            rand_factor = torch.rand(1).item() * 0.3

            modified_params = {
                noise_type: {
                    "intensity": base_intensity * (1.0 + time_scale + rand_factor),
                    "persistence": base_persistence * (1.2 - time_scale * 0.3),
                    "grain": base_grain * (1.0 + time_scale * 0.8 + rand_factor)
                }
            }

            frame = self.generate_basic_noise(modified_params, latent_height, latent_width, noise_type, analysis_type)
            noise_batch[i] = frame[0]

        return ({"samples": noise_batch}, [float(t) for t in timestamps])

    def generate_basic_noise(self, params, height, width, noise_type, analysis_type):
        noise = torch.zeros((1, 4, height, width), dtype=torch.float32)
        p = params.get(noise_type, {})
        intensity = float(p.get("intensity", 1.0))
        persistence = float(p.get("persistence", 0.8))
        grain = float(p.get("grain", 0.5))

        if noise_type == "gaussian":
            if analysis_type in ["mel", "spectral"]:
                base = torch.randn((1, 4, height, width))
                freq = torch.randn((1, 4, height, width))
                noise = (base + freq * grain) * intensity * persistence
            else:
                noise = torch.randn((1, 4, height, width)) * intensity * persistence
        elif noise_type == "salt_pepper":
            threshold = grain
            if analysis_type in ["onset", "segment"]:
                threshold = min(grain * 1.5, 0.9)
            mask = torch.rand((1, 4, height, width)) < threshold
            noise[mask] = intensity
            noise[~mask] = -intensity
            noise *= persistence
        elif noise_type == "perlin":
            freq_multiplier = 1.0
            if analysis_type in ["mel", "spectral"]:
                freq_multiplier = 2.0
            elif analysis_type in ["onset", "segment"]:
                freq_multiplier = 0.5
            for j in range(4):
                n_val = self.rand_perlin_2d((height, width), (1, 1))
                n_val = intensity * n_val * persistence
                if analysis_type in ["mel", "spectral"]:
                    detail = self.rand_perlin_2d((height, width), (2, 2)) * 0.3 * intensity * persistence
                    n_val = n_val + detail
                noise[0, j] = n_val * freq_multiplier
        else:
            noise = torch.randn((1, 4, height, width)) * intensity * persistence

        return noise

NODE_CLASS_MAPPINGS = {
   "TimestampNoiseGenerator": TimestampNoiseGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
   "TimestampNoiseGenerator": "Timestamp Noise Generator"
}