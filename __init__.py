import importlib
import logging

logger = logging.getLogger(__name__)

# List of node files to include
node_list = [
    "librosa_analysis_node",
    "audio_noise_nodes",
    "timestamp-noise-node",
    "AdvancedNoisePatterns",
    "NoiseToLatentConverter"
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Dynamically load each node file
for module_name in node_list:
    try:
        imported_module = importlib.import_module(f".{module_name}", __name__)
        if hasattr(imported_module, "NODE_CLASS_MAPPINGS"):
            NODE_CLASS_MAPPINGS.update(imported_module.NODE_CLASS_MAPPINGS)
        if hasattr(imported_module, "NODE_DISPLAY_NAME_MAPPINGS"):
            NODE_DISPLAY_NAME_MAPPINGS.update(imported_module.NODE_DISPLAY_NAME_MAPPINGS)
    except Exception as e:
        logger.error(f"[AudioDriven Latent Tools] Failed to load module {module_name}: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
