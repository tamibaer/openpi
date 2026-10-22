"""UR5e + Robotiq-Greifer: Abbildung zwischen LeRobot-Datensatz / Roboter und dem Modell.

Datensatz (erzeugt mit examples/ur5e/rosbag_to_lerobot.py):
    image        Basiskamera (D415)
    wrist_image  Handgelenkkamera (D405)
    state        7 Werte: 6 Gelenkwinkel [rad] (shoulder_pan .. wrist_3) + Greifer 0 (offen) .. 1 (zu)
    actions      7 absolute Zielwerte (6 Gelenke + Greifer)
    task         Sprachanweisung (wird über prompt_from_task zu "prompt")

Bei der Inference schickt der Roboter-Client dieselben Schlüssel wie nach dem Repack:
    base_rgb, wrist_rgb (uint8, H×W×3), joints (6,), gripper (1,), prompt (str)
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_ur5e_example() -> dict:
    """Zufälliges Beispiel im Format, das der Roboter-Client bei der Inference schickt."""
    return {
        "base_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "wrist_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "joints": np.random.rand(6).astype(np.float32),
        "gripper": np.random.rand(1).astype(np.float32),
        "prompt": "unstack the red block from the blue block",
    }


def _parse_image(image) -> np.ndarray:
    # LeRobot liefert float32 (C, H, W) in [0, 1]; der Roboter-Client uint8 (H, W, C).
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class UR5eInputs(transforms.DataTransformFn):
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # Training: Datensatz liefert "state" (7,) = 6 Gelenke + Greifer.
        # Inference: Roboter-Client schickt "joints" (6,) und "gripper" (1,) getrennt.
        if "state" in data:
            state = np.asarray(data["state"])
        else:
            state = np.concatenate([np.asarray(data["joints"]), np.atleast_1d(np.asarray(data["gripper"]))])

        base_image = _parse_image(data["base_rgb"])
        wrist_image = _parse_image(data["wrist_rgb"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Kein rechtes Handgelenk -> leeres Bild, Maske aus (außer bei pi0-FAST).
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class UR5eOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # 6 Gelenke + Greifer; der Rest des Modell-Aktionsvektors ist Padding.
        return {"actions": np.asarray(data["actions"][..., :7])}
