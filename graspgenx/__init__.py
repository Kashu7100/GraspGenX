# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Library logging: importing logging_config attaches a NullHandler to the
# "graspgenx" logger so importing this package never configures the root logger
# or prints (which previously duplicated/clobbered host Genesis/Eden output).
# Call graspgenx.utils.logging_config.setup_logging() for opt-in console output.
try:
    import graspgenx.utils.logging_config  # noqa: F401
except ImportError:
    pass

# Re-export the asset resolvers. Assets (gripper descriptions + checkpoints) are
# downloaded lazily on first use from the Hugging Face cache — NOT at import time —
# so importing graspgenx neither does network I/O nor writes into site-packages.
try:
    from graspgenx._setup_dependencies import (
        ensure_checkpoints,
        ensure_gripper_descriptions,
        get_checkpoints_root,
        get_checkpoints_version_dir,
        get_gripper_descriptions_assets,
        get_gripper_descriptions_root,
    )
except Exception:  # noqa: BLE001 - never let the import break the package
    import logging

    logging.getLogger(__name__).debug(
        "graspgenx asset resolvers unavailable; continuing.", exc_info=True
    )

__version__ = "0.1.0"
