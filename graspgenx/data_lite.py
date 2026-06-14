# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight inference-only data helpers.

This module holds the handful of symbols the inference path needs from the
(training-oriented) ``graspgenx.dataset`` package, so that importing the grasp
server / samplers does not drag in the data-generation + training dependency
stack (webdataset, renderer/pyrender, visualization, etc.).

Kept intentionally dependency-light: only ``torch`` + the project logger.
"""

import torch

from graspgenx.utils.logging_config import get_logger

logger = get_logger(__name__)


# Grasp-label id <-> name mapping used by the discriminator. (Mirrors the table
# previously defined in graspgenx.dataset.visualize_utils.)
MAPPING_ID2NAME = {
    0: "pos_true",
    1: "neg_true",
    2: "neg_hncolliding",
    3: "neg_freespace",
    4: "neg_hnretract",
    5: "pos_true_onpolicy",
    6: "neg_true_onpolicy",
}
MAPPING_NAME2ID = {val: key for key, val in MAPPING_ID2NAME.items()}


# Keys whose per-sample tensors are stacked into a batch tensor at collate time.
_STACK_KEYS = [
    "inputs",
    "points",
    "seg",
    "z_offset",
    "onehot",
    "sweep_volume",
    "sweep_volume_open_and_mid",
    "object_inputs",
    "gripper_open_ptc",
    "gripper_close_ptc",
    "gripper_selected_ptc",
    "gripper_selected_open_ptc",
    "gripper_selected_mid_ptc",
    "gripper_selected_close_ptc",
    "gripper_vol_tsdf",
    "gripper_pointnet_repr",
    "bottom_center",
    "cam_pose",
    "ee_pose",
    "placement_masks",
    "placement_region",
]
_CAT_KEYS = ["contact_dirs", "approach_dirs"]


def collate_batch_keys(batch):
    if len(batch) < 1:
        return
    batch = {key: [data[key] for data in batch] for key in batch[0]}
    if "task" in batch:
        task = batch.pop("task")
        batch["task_is_pick"] = torch.stack([torch.tensor(t == "pick") for t in task])
        batch["task_is_place"] = torch.stack([torch.tensor(t == "place") for t in task])
    for key in batch:
        if key in _STACK_KEYS:
            batch[key] = torch.stack(batch[key])
        if key in _CAT_KEYS:
            batch[key] = torch.cat(batch[key])
    return batch


def collate(batch):
    initial_batch_size = len(batch)
    batch = [data for data in batch if not data.get("invalid", False)]
    final_batch_size = len(batch)
    num_dropped = initial_batch_size - final_batch_size
    if num_dropped > 0:
        logger.warning(
            f"[COLLATE] Dropped {num_dropped}/{initial_batch_size} invalid samples"
        )
    return collate_batch_keys(batch)


def sample_points(xyz, num_points):
    """Index sampler that up/down-samples a point cloud to exactly ``num_points``.

    (Relocated from the removed ``graspgenx.dataset.dataset_utils``; used by the
    mesh-input demo.)
    """
    num_replica = num_points // xyz.shape[0]
    num_remain = num_points % xyz.shape[0]
    pt_idx = torch.randperm(xyz.shape[0])
    pt_idx = torch.cat([pt_idx for _ in range(num_replica)] + [pt_idx[:num_remain]])
    return pt_idx
