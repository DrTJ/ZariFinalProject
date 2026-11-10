"""Cascade SSD detection + Deep SORT tracking for real-time multi-object tracking.

Detection strategies:
    full     - heavy SSD300-VGG16 on every frame (baseline, "non-cascade")
    lite     - light SSDLite320-MobileNetV3 on every frame
    cascade  - SSDLite on every frame, SSD300 only when SSDLite is unsure,
               the object count changes, or a refresh keyframe is due
"""

__version__ = "1.0.0"
