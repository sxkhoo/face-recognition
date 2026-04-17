"""Hailo-8 hardware-specific inference wrappers."""

from __future__ import annotations

import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

_hailo = None


def import_hailo():
    """Lazy-import hailo_platform symbols. Cached after first call."""
    global _hailo
    if _hailo is None:
        try:
            from hailo_platform import (
                HEF,
                ConfigureParams,
                FormatType,
                HailoStreamInterface,
                InferVStreams,
                InputVStreamParams,
                OutputVStreamParams,
                VDevice,
            )
            _hailo = {
                "HEF": HEF,
                "VDevice": VDevice,
                "ConfigureParams": ConfigureParams,
                "InputVStreamParams": InputVStreamParams,
                "OutputVStreamParams": OutputVStreamParams,
                "InferVStreams": InferVStreams,
                "FormatType": FormatType,
                "HailoStreamInterface": HailoStreamInterface,
            }
        except ImportError as e:
            raise ImportError(
                "hailo_platform not installed. "
                "Use CPU fallbacks for laptop testing."
            ) from e
    return _hailo


def resolve_vstream_name(vstream_params) -> str:
    """Support both list-like and mapping-like Hailo vstream params."""
    if hasattr(vstream_params, "keys"):
        return str(next(iter(vstream_params.keys())))

    first = next(iter(vstream_params))
    return str(getattr(first, "name", first))


@contextmanager
def activate_network_group(network_group):
    """Handle HailoRT API variants that may or may not require explicit params."""
    try:
        with network_group.activate():
            yield
            return
    except TypeError:
        params_factory = getattr(network_group, "create_params", None)
        if params_factory is None:
            raise
        params = params_factory()
        with network_group.activate(params):
            yield


def create_shared_vdevice():
    """Create one Hailo VDevice to share across multiple network groups."""
    from hailo_platform import VDevice

    return VDevice()
