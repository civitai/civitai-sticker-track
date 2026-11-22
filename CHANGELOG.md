# Changelog

## [1.2.0]

- StickerTrack: `hide_when_eye_lost` (default off). A sticker placed on an eye hides while that eye is not found (2 or more eye samples in a row, about 1 s, for example when the head turns to profile and the eye is behind the snout) and shows again when the eye is found. No effect on a sticker that is not on an eye.

## [1.1.0]

First Civitai release, forked from rmatif/ComfyUI-StickerTrack 1.0.3.

- StickerAnchorPick: optional `fallback_masks`, used when no head is under the sticker (for example SAM3 Detect with a point prompt at the sticker centre), and a new `on_head` output.
- StickerTrack: `reid` input. Wire `on_head` to it so a sticker on something that is not a head hides when its subject is lost, instead of re-identifying onto a person after a cut.
- StickerTrack: `anchor_eyes` (default on). A sticker placed on an eye takes its position from that eye's SAM 3.1 detection; the head box and points carry it only between detections.
- StickerTrack: `hide_occluded`, `occluded_below`, `occluded_frames` (off by default). Hide the sticker while the points under it are covered.
