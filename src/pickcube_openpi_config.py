"""
PickCube-v1 用 openpi 訓練コンフィグ登録スクリプト

このファイルを import することで "pi0_pickcube_lora" コンフィグが
openpi の config registry に登録される。

train_pickcube.py から import して使用する。
"""
from __future__ import annotations

import dataclasses
from typing import override

import openpi.models.pi0 as pi0_config
import openpi.transforms as transforms
from openpi.training import config as _cfg
from openpi.training import weight_loaders
from openpi.training.config import (
    AssetsConfig,
    DataConfig,
    DataConfigFactory,
    ModelTransformFactory,
)

DATASET_LOCAL_PATH = "/opt/pickcube_lerobot"   # LeRobot dataset inside container
ASSETS_LOCAL_PATH  = "/opt/pickcube_lerobot"   # norm_stats.json is here
ASSET_ID           = "pickcube"                # sub-dir under assets_dir

CHECKPOINT_BASE    = "/opt/openpi/checkpoints"
PRETRAINED_PARAMS  = "gs://openpi-assets/checkpoints/pi0_base/params"

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

    repo_id: str = DATASET_LOCAL_PATH
    default_prompt: str = PROMPT

    @override
    def create(self, assets_dirs, model_config):
        repack = transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "images": {
                            "front": "observation.images.front",
                            "wrist": "observation.images.wrist",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "task_description",
                    }
                )
            ]
        )
        # No delta transform needed (actions already in delta space)
        data_transforms = transforms.Group()

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
    # LoRA fine-tuning of pi0 base
    model=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_dim=7,
        action_horizon=10,
    ),
    data=PickCubeDataConfig(
        repo_id=DATASET_LOCAL_PATH,
        assets=AssetsConfig(
            assets_dir=ASSETS_LOCAL_PATH,
            asset_id=ASSET_ID,
        ),
    ),
    # Load pretrained pi0 base weights
    weight_loader=weight_loaders.CheckpointWeightLoader(PRETRAINED_PARAMS),
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
