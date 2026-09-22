import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class NoiseParams:
    intensity: float
    grain: float
    persistence: float
    distribution: str

class AudioNoiseMapper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "energy_levels": ("AUDIO_ENERGY",),
                "timestamps": ("TIMESTAMPS",),
                "analysis_type": ("ANALYSIS_TYPE",),
            }
        }

    RETURN_TYPES = ("NOISE_PARAMS", "STRING")
    RETURN_NAMES = ("noise_params", "debug_info")
    FUNCTION = "process_energy_to_noise"
    CATEGORY = "audio/noise"

    def __init__(self):
        self._setup_analysis_params()

    def _setup_analysis_params(self):
        self.params = {
            "default": {"scale": 1.0, "smoothing": 0.2},
            "onset": {"scale": 2.0, "smoothing": 0.1},
            "segment": {"scale": 1.5, "smoothing": 0.15},
            "tempo": {"scale": 1.8, "smoothing": 0.12},
            "mel": {"scale": 2.5, "smoothing": 0.08},
            "spectral": {"scale": 2.2, "smoothing": 0.05},
            "second": {"scale": 2.0, "smoothing": 0.1},
            "half_second": {"scale": 1.5, "smoothing": 0.3},
            "beat": {"scale": 1.8, "smoothing": 0.2}
        }

    def process_energy_to_noise(self,
                              energy_levels: List[float],
                              timestamps: List[float],
                              analysis_type: str) -> Tuple[Dict[str, NoiseParams], str]:
        try:
            if len(energy_levels) == 0 or len(timestamps) == 0:
                raise ValueError("Empty energy levels or timestamps received")

            params = self.params.get(analysis_type, self.params["default"])
            energy_array = np.nan_to_num(np.array(energy_levels, dtype=np.float64), nan=0.0)

            # Calculate noise parameters based on analysis type
            intensity_scale = params["scale"]
            smoothing = params["smoothing"]

            if len(energy_array) > 1:
                diff_std = float(np.std(np.diff(energy_array)))
                std_val = float(np.std(energy_array))
            else:
                diff_std = 0.0
                std_val = 0.0

            persistence = float(np.exp(-smoothing * diff_std))
            mean_energy = float(np.mean(energy_array))
            max_energy = float(np.max(energy_array))
            p90_energy = float(np.percentile(energy_array, 90))
            median_energy = float(np.median(energy_array))
            var_energy = float(np.var(energy_array)) if len(energy_array) > 1 else 0.0

            # Generate different noise distributions
            noise_params = {
                "gaussian": {
                    "intensity": float(mean_energy * intensity_scale * 3.0),
                    "grain": float(std_val * 5.0),
                    "persistence": persistence,
                    "distribution": "gaussian"
                },
                "salt_pepper": {
                    "intensity": float(p90_energy * intensity_scale),
                    "grain": float(np.mean(energy_array > median_energy)),
                    "persistence": persistence,
                    "distribution": "salt_pepper"
                },
                "perlin": {
                    "intensity": float(max_energy * intensity_scale * 4.0),
                    "grain": float(std_val * 10.0),
                    "persistence": persistence,
                    "distribution": "perlin"
                },
                "timestamps": [float(t) for t in timestamps]
            }

            debug_info = (
                f"Analysis: {analysis_type}\n"
                f"Scale: {intensity_scale:.2f}\n"
                f"Mean Energy: {mean_energy:.3f}\n"
                f"Energy Variance: {var_energy:.3f}"
            )

            return (noise_params, debug_info)

        except Exception as e:
            return ({}, f"Error: {str(e)}")

NODE_CLASS_MAPPINGS = {
    "AudioNoiseMapper": AudioNoiseMapper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioNoiseMapper": "Audio To Noise Parameters",
}
