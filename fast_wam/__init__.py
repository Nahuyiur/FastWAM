"""Megatron Core inference patch for Fast-WAM."""

from .config import ActionExpertConfig, FastWAMConfig, VideoExpertConfig
from .model import FastWAMModel

__all__ = ["ActionExpertConfig", "FastWAMConfig", "FastWAMModel", "VideoExpertConfig"]
