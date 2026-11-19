"""Continuous batching for SAM 2-family video trackers: many cameras, one GPU.

Names are loaded on first use, so that `tamserve.boxes`, which runs YOLOX on the CPU
through OpenVINO, can be imported on a machine without PyTorch."""
from importlib import import_module

_WHERE = {
    "Box": "engine", "Clicks": "engine", "Engine": "engine", "prepare": "engine",
    "Finder": "finder", "PeopleFinder": "finder",
    "MODELS": "models", "build_video_predictor": "models",
    "PeopleTracker": "people", "Router": "router", "Tracker": "tracker",
    "Mask": "wire", "Update": "wire",
    "BoxTracker": "boxes", "Detector": "boxes", "Found": "boxes", "gray_model": "boxes",
}
_MODULES = {"engine", "finder", "models", "people", "router", "tracker", "wire", "worker", "compare", "reference",
            "boxes"}

__all__ = ["Engine", "Router", "Tracker", "Finder", "PeopleTracker", "PeopleFinder", "Update", "Mask", "Box",
           "Clicks", "prepare", "build_video_predictor", "MODELS", "BoxTracker", "Detector", "Found", "gray_model"]


def __getattr__(name):
    if name in _WHERE:
        value = getattr(import_module(f"tamserve.{_WHERE[name]}"), name)
    elif name in _MODULES:  # `from tamserve import wire`, as the tests do
        value = import_module(f"tamserve.{name}")
    else:
        raise AttributeError(f"module 'tamserve' has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
