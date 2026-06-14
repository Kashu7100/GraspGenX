# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import sys

# Top-level logger for the package. Library code logs under this; the only thing
# attached at import time is a NullHandler (see bottom), per stdlib guidance for
# libraries — so importing graspgenx never touches the root logger or emits
# output. Host applications (e.g. Eden/Genesis) keep full control of logging.
PACKAGE_LOGGER_NAME = "graspgenx"

# Global flag to track if console logging has been explicitly enabled.
_logging_initialized = False


def setup_logging(level=logging.INFO):
    """Opt-in console logging for standalone use (CLIs / demo scripts).

    Adds a stdout handler to the ``graspgenx`` package logger only — it does NOT
    configure the root logger, so it won't duplicate or clobber a host
    application's logging (e.g. Genesis/Eden). Safe to call multiple times.

    Importing ``graspgenx`` does NOT call this; scripts that want console output
    should call it explicitly in their ``main``.
    """
    global _logging_initialized

    if _logging_initialized:
        return

    pkg_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    pkg_logger.setLevel(level)

    if not any(
        isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout
        for h in pkg_logger.handlers
    ):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        pkg_logger.addHandler(handler)

    # Handle records here; don't also bubble them to a host app's root logger,
    # which would double-print.
    pkg_logger.propagate = False

    _logging_initialized = True


def get_logger(name):
    """Return a module logger (no side effects).

    Args:
        name: The name of the module (usually ``__name__``).

    Returns:
        A standard ``logging.Logger``. It emits nothing unless the host
        application has configured handlers, or :func:`setup_logging` was called.
    """
    return logging.getLogger(name)


# Attach a NullHandler to the package logger so the library never prints unless
# the application opts in (stdlib best practice for libraries). Replaces the
# previous import-time root-logger configuration that duplicated/clobbered host
# (Genesis/Eden) log output.
logging.getLogger(PACKAGE_LOGGER_NAME).addHandler(logging.NullHandler())
