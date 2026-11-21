# Civitai Sticker Track

Civitai's fork of [ComfyUI-StickerTrack](https://github.com/rmatif/ComfyUI-StickerTrack) by rmatif (Apache-2.0, see NOTICE), published to the Comfy registry as `civitai/civitai-sticker-track`. What 1.1.0 adds is in CHANGELOG.md: a point-prompt fallback anchor for stickers that are not on a head, no re-identification takeover for those, eye anchoring, and an optional hide-while-covered mode. Tests: `python tests/test_tracking.py`.

The rest of this file is the upstream description.


Keeps a sticker that was placed on a frame of a video on the same spot of a character (eyes, mouth, head, anywhere on the face) for the rest of the clip, and exports the pose per frame as JSON. It works the same way for people, anime, cartoons, animals and costumes: nothing in it picks a model per character type.

How it works:

1. Core `SAM3_Detect` finds the heads on frame 0 (the prompt is `head:16`: a bare `head` returns only the best-scoring one); `StickerAnchorPick` keeps the one under the sticker and fails rather than take a head far from it.
2. Core `SAM3_VideoTrack` (SAM 3.1) follows that head through the clip.
3. `StickerTrack` seeds TAPNext++ points on the head around the sticker and fits an affine transform from the seed frame on every frame. Where SAM 3.1 sees both eyes under the sticker (prompt `eye:4`, sampled twice a second), the sticker turns with the eye line, because the points read a nod seen side-on as a tilt; elsewhere the rotation comes from the points. Cuts are confirmed against the points. SAM 3.1 keeps the character's identity across cuts; a head that ever shares a frame with it is never taken for it, and the sticker hides while the character is off screen or its centre is outside the frame. Each shot is smoothed, then resampled to fixed ticks.
4. Core `SaveText` (format `json`) writes the track.

## Nodes

| Node | Inputs | Output |
|---|---|---|
| `StickerLoadVideo` | video path, `fps`, `max_side`, optional `start_ms` and `max_ms` | frames on an fps grid from `start_ms`, `fps` |
| `StickerAnchorPick` | per-instance `MASK` of frame 0, `placement` | `initial_mask` for `SAM3_VideoTrack` |
| `TAPNextLoader` | TAPNext++ checkpoint, `resolution` (256 or 512) | `TAPNEXT` |
| `StickerTrack` | frames, `SAM3_TRACK_DATA`, SAM 3.1 `MODEL`, `TAPNEXT`, `placement`, `fps`, `tick_ms`, optional eye `CONDITIONING` | `track_json`, `stats_json` |
| `StickerTrackPreview` | frames, `track_json`, sticker image and mask | frames with the sticker drawn on |

`placement` is the sticker on the first loaded frame (`start_ms`): `{"x": 0.42, "y": 0.41, "scale": 0.3, "rotation": -11}`. `x`/`y` are the sticker centre as fractions of the frame width/height, `scale` is the sticker width as a fraction of the frame width, `rotation` is in degrees, clockwise.

## Output

```json
{"v":1,"ms":250,"show":[[0,1],[19,0]],"x":[[0,0.42],[1,0.4253]],"y":[[0,0.412],[1,0.4111]],"s":[[0,0.3],[1,0.3059]],"r":[[0,-11.0],[1,-10.4]]}
```

Each channel is a list of `[tick, value]` pairs, time = `start_ms + tick * ms`. Values are linear between pairs and held after the last one; `show` is a step channel (1 shown, 0 hidden) and is left out when the sticker is shown throughout. Scale is clamped to 0.05–0.4 and rotation wrapped to [-180, 180).

## Models

| Model | File | Licence |
|---|---|---|
| SAM 3.1 | [Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1) `checkpoints/sam3.1_multiplex_fp16.safetensors` | Meta SAM License |
| TAPNext++ | [google/tapnet](https://huggingface.co/google/tapnet) `tapnextpp_ckpt.pt` (256) or `tapnextpp_512.ckpt` (512) | Apache-2.0 |

Put the TAPNext++ file in `models/tapnext/`, or pass an absolute path. The TAPNext++ model code in `tapnext/` is vendored from [google-deepmind/tapnet](https://github.com/google-deepmind/tapnet) (Apache-2.0).

## Workflow

[`workflows/sticker_track_api.json`](workflows/sticker_track_api.json) is the API-format graph with the fast settings (8 fps, TAPNext++ 256, detection every second frame). It needs ComfyUI with the SAM 3 nodes. `StickerLoadVideo` decodes with PyAV, so AV1 uploads work (OpenCV-based loaders often cannot decode AV1), and keeps only the frames on the fps grid. `start_ms` seeks to the keyframe before that point instead of decoding from the start, and `max_ms` stops after that much video (0 runs to the end). On an RTX 4090 it takes about 0.9 s per second of video once the models are loaded; 12 fps with the 512 checkpoint and detection on every frame takes about 1.8 s.

## Licence

Apache-2.0.
