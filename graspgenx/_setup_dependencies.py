# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locate external assets (gripper descriptions + checkpoints) on demand.

Both are resolved lazily via the Hugging Face Hub into the standard HF cache
(``$HF_HOME``/``~/.cache/huggingface``) — a writable, per-user location — rather
than being git-cloned into the package directory. This makes a plain
``pip install`` (incl. ``git+`` / wheel installs) self-sufficient: nothing is
written into ``site-packages``, and no ``git`` executable is required.

Two assets are managed:

1. ``gripper_descriptions`` — per-gripper URDFs, meshes, ``config.json``.
   Resolution order:
     a. ``$GRASPGENX_GRIPPER_CFG_DIR`` if set (must exist on disk; used as-is).
     b. HF dataset :data:`GRIPPER_DESCRIPTIONS_HF_REPO` (downloaded + cached).

2. ``graspgenx_checkpoints`` — versioned generator + discriminator checkpoints.
   Resolution order:
     a. ``$GRASPGENX_CHECKPOINT_DIR`` if set **and** exists on disk.
     b. HF model :data:`CHECKPOINTS_HF_REPO` (downloaded + cached). Callers use
        the per-version subdir via :data:`DEFAULT_CHECKPOINT_VERSION`
        (e.g. ``release/{gen,dis}/``).

Downloads happen on first access (``get_*`` / ``ensure_*``), NOT at import time.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Optional

from graspgenx.utils.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Hugging Face source repos
# ---------------------------------------------------------------------------

# Generator + discriminator checkpoints (upstream, public, ungated).
CHECKPOINTS_HF_REPO = "adithyamurali/GraspGenXModel"
# Per-version subdir inside the checkpoint repo (``<root>/<version>/{gen,dis}``).
DEFAULT_CHECKPOINT_VERSION = "release"

# Per-gripper URDFs/meshes/configs. Mirror of github.com/beininghan/gripper_descriptions
# (permissively licensed; see LICENSE_ASSETS), hosted as an HF dataset so it can be
# fetched + cached with the same mechanism as the checkpoints.
GRIPPER_DESCRIPTIONS_HF_REPO = "Kashu7100/graspgenx_gripper_descriptions"

# Env-var overrides: point these at an existing local checkout to skip the
# download entirely (e.g. air-gapped clusters, or a manual clone).
_GRIPPER_DESCRIPTIONS_ENV_VAR = "GRASPGENX_GRIPPER_CFG_DIR"
_CHECKPOINTS_ENV_VAR = "GRASPGENX_CHECKPOINT_DIR"
# Backwards-compatibility alias for older code paths.
_PATH_ENV_VAR = _GRIPPER_DESCRIPTIONS_ENV_VAR

# ---------------------------------------------------------------------------
# Resolution machinery
# ---------------------------------------------------------------------------

_setup_lock = threading.Lock()
_gripper_root_cache: Optional[Path] = None
_checkpoints_root_cache: Optional[Path] = None


def _snapshot(repo_id: str, repo_type: str, dep_label: str) -> Optional[Path]:
    """Download ``repo_id`` into the HF cache and return its local path.

    Returns ``None`` (with a warning) on failure, so a transient network issue
    degrades gracefully rather than crashing the caller. Set the corresponding
    ``GRASPGENX_*`` env var to a local checkout to work fully offline.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.warning("huggingface_hub is required to fetch %s; install it or set the override env var.", dep_label)
        return None

    try:
        logger.info("Resolving %s from Hugging Face (%s); downloading on first use, cached thereafter.", dep_label, repo_id)
        return Path(snapshot_download(repo_id=repo_id, repo_type=repo_type))
    except Exception as exc:  # noqa: BLE001 - network/auth/etc; never hard-crash here
        logger.warning(
            "Could not fetch %s from %s (%s). Set the override env var to a local checkout to proceed offline.",
            dep_label,
            repo_id,
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# gripper_descriptions
# ---------------------------------------------------------------------------


def _register_gripper_descriptions_on_sys_path(root: Path) -> None:
    """Make ``import gripper_descriptions`` work without ``pip install``.

    The asset bundle is laid out as ``<root>/gripper_descriptions/__init__.py``
    (a standard package next to its assets). Prepending ``<root>`` to
    ``sys.path`` lets the inner package be imported. Idempotent.
    """
    if not root.exists():
        return
    if not (root / "gripper_descriptions").is_dir():
        return
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def get_gripper_descriptions_root() -> Path | None:
    """Return the gripper_descriptions root (contains ``gripper_descriptions/``).

    Uses ``$GRASPGENX_GRIPPER_CFG_DIR`` if set (must exist), else downloads the
    HF dataset into the cache. Returns ``None`` if neither is available.
    """
    global _gripper_root_cache

    override = os.environ.get(_GRIPPER_DESCRIPTIONS_ENV_VAR)
    if override:
        path = Path(override).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"{_GRIPPER_DESCRIPTIONS_ENV_VAR} points to {path}, which does not exist. "
                f"Create that directory or unset {_GRIPPER_DESCRIPTIONS_ENV_VAR} to download "
                f"{GRIPPER_DESCRIPTIONS_HF_REPO} from Hugging Face."
            )
        _register_gripper_descriptions_on_sys_path(path)
        return path

    with _setup_lock:
        if _gripper_root_cache is None:
            _gripper_root_cache = _snapshot(GRIPPER_DESCRIPTIONS_HF_REPO, "dataset", "gripper_descriptions")
    if _gripper_root_cache is not None:
        _register_gripper_descriptions_on_sys_path(_gripper_root_cache)
    return _gripper_root_cache


def get_gripper_descriptions_assets() -> Path | None:
    """Path to ``gripper_descriptions/assets/x_grippers/`` inside the bundle."""
    root = get_gripper_descriptions_root()
    if root is None:
        return None
    return root / "gripper_descriptions" / "assets" / "x_grippers"


def ensure_gripper_descriptions(force: bool = False) -> Path | None:
    """Ensure the gripper_descriptions bundle is available locally (downloads if needed)."""
    global _gripper_root_cache
    if force:
        _gripper_root_cache = None
    return get_gripper_descriptions_root()


# ---------------------------------------------------------------------------
# graspgenx_checkpoints
# ---------------------------------------------------------------------------


def get_checkpoints_root() -> Path | None:
    """Return the checkpoint repo root (contains ``<version>/{gen,dis}``).

    Uses ``$GRASPGENX_CHECKPOINT_DIR`` if set **and** existing, else downloads
    the HF model into the cache. Returns ``None`` if neither is available.
    """
    global _checkpoints_root_cache

    override = os.environ.get(_CHECKPOINTS_ENV_VAR)
    if override:
        path = Path(override).expanduser().resolve()
        if path.exists():
            return path
        logger.warning(
            "%s points to %s, which does not exist. Falling back to the Hugging Face download (%s).",
            _CHECKPOINTS_ENV_VAR,
            path,
            CHECKPOINTS_HF_REPO,
        )

    with _setup_lock:
        if _checkpoints_root_cache is None:
            _checkpoints_root_cache = _snapshot(CHECKPOINTS_HF_REPO, "model", "graspgenx_checkpoints")
    return _checkpoints_root_cache


def get_checkpoints_version_dir(version: Optional[str] = None) -> Path | None:
    """Return ``<root>/<version>/`` containing ``gen/`` and ``dis/`` subdirs."""
    root = get_checkpoints_root()
    if root is None:
        return None
    return root / (version or DEFAULT_CHECKPOINT_VERSION)


def ensure_checkpoints(force: bool = False) -> Path | None:
    """Ensure the checkpoint repo is available locally (downloads if needed)."""
    global _checkpoints_root_cache
    if force:
        _checkpoints_root_cache = None
    return get_checkpoints_root()
