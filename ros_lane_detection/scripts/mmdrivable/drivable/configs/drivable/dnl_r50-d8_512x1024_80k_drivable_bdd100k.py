"""DNL with ResNet-50-d8."""

_base_ = [
    "../_base_/models/dnl_r50-d8.py",
    "../_base_/datasets/bdd100k_512x1024.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_80k.py",
]
load_from = "https://dl.cv.ethz.ch/bdd100k/drivable/models/dnl_r50-d8_512x1024_80k_drivable_bdd100k.pth"
