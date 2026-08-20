"""Subtitle studio engine: fonts, presets, styles, timing, positioning, animation, rendering.

Keeps the subtitle subsystem out of ``app/services/video.py``: the video service
orchestrates rendering, while every subtitle concern (font discovery/resolution,
style presets, word timing, positioning, animation, text measurement) lives here.
"""