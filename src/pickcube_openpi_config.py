"""
PickCube-v1 用 openpi 訓練コンフィグ登録スクリプト

このファイルを import することで "pi0_pickcube_lora" コンフィグが
openpi の config registry に登録される。

train_pickcube.py から import して使用する。
"""
from __future__ import annotations

import dataclasses
try:
    from typing import override          # Python 3.12+
except ImportError:
    from typing_extensions import override  # Python 3.11 fallback

import numpy as np
import torch
import flax.traverse_util
import jax

from openpi.models.pi0 import pi0_config  # Pi0Config lives in pi0_config submodule
from openpi.models import model as _model
from openpi.shared import download
import openpi.transforms as transforms
from openpi.training import config as _cfg
from openpi.training import weight_loaders
from openpi.training.config import (
    AssetsConfig,
    DataConfigFactory,
    ModelTransformFactory,
)


@dataclasses.dataclass(frozen=True)
class PartialCheckpointWeightLoader(weight_loaders.WeightLoader):
    """Like CheckpointWeightLoader but skips params with shape mismatches.

    Useful when fine-tuning from a checkpoint trained with a different action_dim.
    Shape-mismatched params (e.g. action_in_proj) are kept at random initialization.
    """

    params_path: str

    def load(self, params: weight_loaders.at.Params) -> weight_loaders.at.Params:
        loaded = _model.restore_params(
            download.maybe_download(self.params_path), restore_type=np.ndarray
        )
        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        flat_loaded = flax.traverse_util.flatten_dict(loaded, sep="/")

        # Start from reference params (keeps full structure including LoRA and init values)
        result = dict(flat_ref)
        skipped = []
        for k, v in flat_loaded.items():
            if k not in flat_ref:
                continue
            ref = flat_ref[k]
            ref_shape = ref.shape if hasattr(ref, "shape") else tuple()
            if hasattr(v, "shape") and v.shape != ref_shape:
                # Shape mismatch: keep reference (model-init) value
                skipped.append(f"  {k}: ckpt {v.shape} != model {ref_shape} → keeping init")
                continue
            result[k] = v.astype(ref.dtype) if hasattr(v, "dtype") and v.dtype != ref.dtype else v

        if skipped:
            print(f"[PartialCheckpointWeightLoader] {len(skipped)} shape-mismatched params kept at init:")
            for s in skipped:
                print(s)

        return flax.traverse_util.unflatten_dict(result, sep="/")

@dataclasses.dataclass(frozen=True)
class TensorImagesToNumpy(transforms.DataTransformFn):
    """Convert lerobot image tensors (C,H,W) to numpy uint8 arrays (H,W,C) for openpi,
    and add image_mask (all True) as required by Observation.from_dict."""

    def __call__(self, data: dict) -> dict:
        if "image" in data:
            data = dict(data)
            converted = {}
            for k, v in data["image"].items():
                if isinstance(v, torch.Tensor):
                    arr = v.permute(1, 2, 0).numpy().astype(np.uint8)
                else:
                    arr = np.asarray(v, dtype=np.uint8)
                converted[k] = arr
            data["image"] = converted
            data["image_mask"] = {k: np.True_ for k in converted}
        return data


DATASET_REPO_ID    = "dataset"                     # repo_id; with HF_LEROBOT_HOME=/opt/pickcube_lerobot_v2 → /opt/pickcube_lerobot_v2/dataset
ASSETS_LOCAL_PATH  = "/opt/pickcube_lerobot_v2/dataset"   # norm_stats.json is here
ASSET_ID           = "pickcube"                # sub-dir under assets_dir

CHECKPOINT_BASE    = "/opt/checkpoints"
# Use local pi0_libero_low_mem_finetune checkpoint as starting point
# (avoids GCS auth issues; same LoRA architecture gemma_2b_lora + gemma_300m_lora)
PRETRAINED_PARAMS  = "/opt/checkpoints/pi0_libero_low_mem_finetune/libero_ft/29999/params"

PROMPT = "pick up the red cube"


@dataclasses.dataclass(frozen=True)
class PickCubeDataConfig(DataConfigFactory):
    """
    LeRobot dataset config for PickCube-v1.

    Dataset columns expected:
      observation.images.front  (PIL Image, H×W×3)
      observation.images.wrist  (PIL Image, H×W×3)
      observation.state         (float32, 8)  [eef_pos(3) + eef_quat(4) + gripper(1)]
      action                    (float32, 7)  [delta_pos(3) + delta_rot_euler(3) + gripper(1)]
      task_description          (str)
    """

    repo_id: str = DATASET_REPO_ID
    default_prompt: str = PROMPT

    @override
    def create(self, assets_dirs, model_config):
        repack = transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "image": {
                            "front": "observation.images.front",
                            "wrist": "observation.images.wrist",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        # prompt comes from InjectDefaultPrompt in model_transforms
                    }
                )
            ]
        )
        # Convert image tensors (C,H,W) → numpy uint8 (H,W,C) for openpi
        data_transforms = transforms.Group(inputs=[TensorImagesToNumpy()])

        model_transforms = ModelTransformFactory(
            default_prompt=self.default_prompt
        )(model_config)

        base = self.create_base_config(assets_dirs, model_config)
        return dataclasses.replace(
            base,
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )


# ── Register config ───────────────────────────────────────────────────────────

_PICKCUBE_CONFIG = _cfg.TrainConfig(
    name="pi0_pickcube_lora",
    exp_name="lora_ft_v1",
    # LoRA fine-tuning of pi0 base
    model=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_dim=7,
        action_horizon=10,
    ),
    data=PickCubeDataConfig(
        repo_id=DATASET_REPO_ID,
        assets=AssetsConfig(
            assets_dir=ASSETS_LOCAL_PATH,
            asset_id=ASSET_ID,
        ),
    ),
    # Load pretrained weights; skip action layers with different action_dim
    weight_loader=PartialCheckpointWeightLoader(PRETRAINED_PARAMS),
    # LoRA freeze: only train LoRA adapters
    freeze_filter=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),
    ema_decay=None,            # off for LoRA
    num_train_steps=10_000,
    batch_size=16,             # L4 24GB is comfortable with 16
    log_interval=50,
    save_interval=500,
    keep_period=1000,
    checkpoint_base_dir=CHECKPOINT_BASE,
    assets_base_dir=ASSETS_LOCAL_PATH,
    project_name="ur3e-pickcube",
    wandb_enabled=True,
    seed=42,
    resume=False,
    overwrite=True,
)


def get_pickcube_config() -> _cfg.TrainConfig:
    return _PICKCUBE_CONFIG
