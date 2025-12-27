# ComfyUI-Wan22-GGUF-Storyboard
# Optimized for LowVRAM (8GB) and GGUF workflows.

from .wan_storyboard_node import Wan22StoryboardGGUF

# This dictionary maps the technical class name to a unique string ID for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Wan22StoryboardGGUF": Wan22StoryboardGGUF
}

# This dictionary defines what you actually see in the 'Add Node' menu
NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan22StoryboardGGUF": "Wan2.2 Storyboard: First-Mid-Mid-Last (GGUF 5B)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]