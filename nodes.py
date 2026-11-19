import itertools
import json
import math
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management as mm
import folder_paths

from . import tracking
from .tapnext.tapnext_torch import TAPNext

if "tapnext" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["tapnext"] = (
        [os.path.join(folder_paths.models_dir, "tapnext")], {".pt", ".ckpt", ".pth"})

PLACEMENT_DEFAULT = '{"x": 0.5, "y": 0.4, "scale": 0.2, "rotation": 0}'


class StickerLoadVideo:
    """Decodes only the frames on an fps grid, with PyAV (AV1 included), downscaled to max_side."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video": ("STRING", {"tooltip": "Path to the video file (a customComfy resource AIR resolves to one)."}),
            "fps": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 60.0, "step": 0.5}),
            "max_side": ("INT", {"default": 1024, "min": 0, "max": 8192,
                                 "tooltip": "Longest side of the returned frames; 0 keeps the source size."}),
        }, "optional": {
            "start_ms": ("INT", {"default": 0, "min": 0, "max": 86_400_000,
                                 "tooltip": "First frame returned, in ms from the start of the video. Track tick 0 lands here."}),
            "max_ms": ("INT", {"default": 0, "min": 0, "max": 86_400_000,
                               "tooltip": "Length to return from start_ms; 0 runs to the end of the video."}),
        }}

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("frames", "fps")
    FUNCTION = "load"
    CATEGORY = "civitai/sticker"

    def load(self, video, fps, max_side, start_ms=0, max_ms=0):
        import av

        path = video if os.path.isabs(video) else folder_paths.get_annotated_filepath(video)
        step = 1.0 / fps
        span = max_ms / 1000.0 if max_ms else math.inf
        frames, k = [], 0
        with av.open(path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            decoded = (f for f in container.decode(stream) if f.time is not None)
            first = next(decoded, None)
            if first is None:
                raise ValueError(f"No frames decoded from {video}")
            start = first.time + start_ms / 1000.0
            if start_ms:
                container.seek(int(start / stream.time_base), stream=stream)
                decoded = (f for f in container.decode(stream) if f.time is not None)
            else:
                decoded = itertools.chain([first], decoded)
            for frame in decoded:
                at = frame.time - start
                if at + 1e-4 < k * step:
                    continue
                if at + 1e-4 >= span:
                    break
                img = frame.to_ndarray(format="rgb24")
                h, w = img.shape[:2]
                if max_side and max(h, w) > max_side:
                    s = max_side / max(h, w)
                    img = cv2.resize(img, (max(2, round(w * s)), max(2, round(h * s))), interpolation=cv2.INTER_AREA)
                frames.append(torch.from_numpy(img))
                while k * step <= at + 1e-4:
                    k += 1
        if not frames:
            raise ValueError(f"No frames in {video} from {start_ms} ms: the video is shorter than that")
        return (torch.stack(frames).float().div_(255.0), float(fps))


class StickerAnchorPick:
    """Keeps the first-frame instance the user placed the sticker on.

    SAM3 Detect returns one instance per prompt unless the prompt asks for more ("head:16").
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "masks": ("MASK", {"tooltip": "Per-instance masks of one frame, best score first (SAM3 Detect, individual_masks)."}),
            "placement": ("STRING", {"multiline": True, "default": PLACEMENT_DEFAULT}),
        }}

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("initial_mask",)
    FUNCTION = "pick"
    CATEGORY = "civitai/sticker"

    def pick(self, masks, placement):
        p = tracking.parse_placement(placement)
        if masks.ndim == 2:
            masks = masks[None]
        H, W = masks.shape[1:]
        ax, ay = p["x"] * W, p["y"] * H
        best = None
        for i in range(masks.shape[0]):
            m = masks[i] > 0.5
            ys, xs = torch.nonzero(m, as_tuple=True)
            if len(xs) == 0:
                continue
            inside = bool(m[min(int(ay), H - 1), min(int(ax), W - 1)])
            dx = max(float(xs.min()) - ax, 0.0, ax - float(xs.max()))
            dy = max(float(ys.min()) - ay, 0.0, ay - float(ys.max()))
            dist = math.hypot(dx, dy)
            size = math.sqrt((float(xs.max() - xs.min()) + 1) * (float(ys.max() - ys.min()) + 1))
            if not inside and dist > 0.5 * max(size, p["scale"] * W):
                continue
            key = (0 if inside else 1, dist, i)
            if best is None or key < best[0]:
                best = (key, i)
        if best is None:
            raise ValueError("No head was found under the sticker on the first frame.")
        i = best[1]
        return ((masks[i:i + 1] > 0.5).float(),)


class TAPNextLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "ckpt_name": (folder_paths.get_filename_list("tapnext"),),
            "resolution": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64,
                                   "tooltip": "512 for tapnextpp_512.ckpt, 256 for tapnextpp_ckpt.pt."}),
        }}

    RETURN_TYPES = ("TAPNEXT",)
    FUNCTION = "load"
    CATEGORY = "civitai/sticker"

    def load(self, ckpt_name, resolution):
        path = folder_paths.get_full_path_or_raise("tapnext", ckpt_name)
        model = TAPNext(image_size=(256, 256))
        sd = torch.load(path, map_location="cpu", weights_only=True)
        sd = sd.get("state_dict", sd)
        model.load_state_dict({k.removeprefix("tapnext."): v for k, v in sd.items()})
        model.eval()
        return ({"model": model, "resolution": resolution},)


class _PointTracker:
    MODEL_SIZE = 256

    def __init__(self, tapnext, images, device, chunk=16):
        self.model = tapnext["model"]
        self.res = tapnext["resolution"]
        self.images = images
        self.device = device
        self.chunk = chunk
        self.H, self.W = images.shape[1:3]
        self.seconds = 0.0

    def _frames(self, a, b):
        v = self.images[a:b, ..., :3].to(self.device, non_blocking=True).movedim(-1, 1)
        v = F.interpolate(v, size=(self.res, self.res), mode="bilinear", align_corners=False) * 2.0 - 1.0
        return v.movedim(1, -1)[None]

    def __call__(self, start, end, pts):
        q = np.zeros((len(pts), 3), np.float32)
        q[:, 1] = pts[:, 1] * self.MODEL_SIZE / self.H
        q[:, 2] = pts[:, 0] * self.MODEL_SIZE / self.W
        queries = torch.from_numpy(q)[None].to(self.device)
        state = None
        for a in range(start, end, self.chunk):
            b = min(end, a + self.chunk)
            t0 = time.perf_counter()
            with torch.no_grad(), torch.autocast(self.device.type, dtype=torch.float16):
                if state is None:
                    tracks, _, vis_logits, state = self.model(video=self._frames(a, b), query_points=queries)
                else:
                    tracks, _, vis_logits, state = self.model(video=self._frames(a, b), state=state)
            tr = tracks[0].float().cpu().numpy()
            vis = (vis_logits[0, :, :, 0] > 0).cpu().numpy()
            self.seconds += time.perf_counter() - t0
            for j in range(b - a):
                pos = np.stack([tr[j, :, 1] * self.W / self.MODEL_SIZE, tr[j, :, 0] * self.H / self.MODEL_SIZE], -1)
                yield a + j, pos, vis[j]


class _HeadEmbedder:
    """Mean SAM3 image-encoder feature of a padded crop around a head.

    Subtracting the clip's own frame-0 mean feature removes the style and lighting every head in
    the clip shares, which otherwise puts any two heads above 0.9 cosine.
    """

    def __init__(self, sam3, images, view, device):
        mm.load_model_gpu(sam3)
        self.trunk = sam3.model.diffusion_model.detector.backbone["vision_backbone"].trunk
        self.dtype = sam3.model.get_dtype()
        self.images, self.view, self.device = images, view, device
        self.cache = {}
        with torch.no_grad():
            frame = images[0:1, ..., :3].to(device).movedim(-1, 1)
            frame = F.interpolate(frame, size=(1008, 1008), mode="bilinear", align_corners=False).to(self.dtype)
            self.context = self.trunk(frame).float()[0].mean((1, 2))

    @torch.no_grad()
    def __call__(self, t, k):
        if (t, k) in self.cache:
            return self.cache[(t, k)]
        v = self.view
        x0, y0, x1, y1 = v.box(t, k)
        px, py = 0.15 * (x1 - x0), 0.15 * (y1 - y0)
        x0, y0 = max(0, int(x0 - px)), max(0, int(y0 - py))
        x1, y1 = min(v.W, int(x1 + px) + 1), min(v.H, int(y1 + py) + 1)
        crop = self.images[t, y0:y1, x0:x1, :3].to(self.device).movedim(-1, 0)[None]
        crop = F.interpolate(crop, size=(1008, 1008), mode="bilinear", align_corners=False).to(self.dtype)
        e = self.trunk(crop).float()[0].mean((1, 2)) - self.context
        e = (e / e.norm()).cpu().numpy()
        self.cache[(t, k)] = e
        return e


def _eye_lines(sam3, eyes_cond, images, poses, tgt, view, fps, size=504):
    """Eye-line angle on frames sampled twice a second, from SAM 3.1 eye masks in a crop around the sticker."""
    from comfy_extras.nodes_sam3 import SAM3_Detect

    n, H, W = images.shape[:3]
    step = max(1, int(round(fps / 2)))
    starts = {t for t in range(1, n) if poses[t] is not None and poses[t - 1] is None}
    frames, crops = [], []
    for t in sorted(set(range(step, n, step)) | starts):
        p = poses[t]
        if p is None or tgt[t] is None:
            continue
        side = int(min(max(2.4 * p.w, 96), H, W))
        x0 = int(min(max(p.cx - side / 2, 0), W - side))
        y0 = int(min(max(p.cy - side / 2, 0), H - side))
        crop = images[t:t + 1, y0:y0 + side, x0:x0 + side, :3].movedim(-1, 1)
        crops.append(F.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False).movedim(1, -1))
        frames.append((t, x0, y0, side / size))
    lines = {}
    if not crops:
        return lines
    masks, boxes = [], []
    for c in range(0, len(crops), 32):
        m, bb = SAM3_Detect.execute(model=sam3, image=torch.cat(crops[c:c + 32]), conditioning=eyes_cond,
                                    threshold=0.25, refine_iterations=0, individual_masks=True).result
        masks.append(m)
        boxes += bb
    masks = torch.cat(masks)
    at = 0
    for (t, x0, y0, s), found in zip(frames, boxes):
        eyes = []
        for m in masks[at:at + len(found)]:
            ys, xs = torch.nonzero(m > 0.5, as_tuple=True)
            if len(xs) >= 6:
                eyes.append((x0 + s * float(xs.float().mean()), y0 + s * float(ys.float().mean()), s * s * float(len(xs))))
        at += len(found)
        if eyes:
            inside = view.contains(t, tgt[t], np.array([e[:2] for e in eyes], np.float32))
            a = tracking.eye_line([e for e, ok in zip(eyes, inside) if ok], poses[t])
            if a is not None:
                lines[t] = a
    return lines


class StickerTrack:
    """SAM3 head track + TAPNext++ points on the head -> per-frame sticker pose, as the app's v1 track JSON."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "track_data": ("SAM3_TRACK_DATA", {"tooltip": "SAM3 Video Track seeded with the StickerAnchorPick mask."}),
            "sam3": ("MODEL", {"tooltip": "The SAM3 model; its image encoder re-identifies the character after cuts."}),
            "tapnext": ("TAPNEXT",),
            "placement": ("STRING", {"multiline": True, "default": PLACEMENT_DEFAULT}),
            "fps": ("FLOAT", {"default": 12.0, "min": 1.0, "max": 120.0, "step": 0.5,
                              "tooltip": "Frame rate of `images`."}),
            "tick_ms": ("INT", {"default": 250, "min": 10, "max": 1000}),
            "points_per_side": ("INT", {"default": 16, "min": 4, "max": 48}),
            "cut_threshold": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 255.0,
                                        "tooltip": "Mean grey difference (0-255, at 64x48) that starts a new shot."}),
            "reid_threshold": ("FLOAT", {"default": 0.4, "min": -1.0, "max": 1.0, "step": 0.01,
                                         "tooltip": "Cosine similarity to the frame-0 head a head needs to take over after a cut."}),
        }, "optional": {
            "eyes": ("CONDITIONING", {"tooltip": "SAM3 text conditioning for eyes (\"eye:4\"). Glasses then turn with the eye line, not the head's nod."}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("track_json", "stats_json")
    FUNCTION = "track"
    CATEGORY = "civitai/sticker"

    def track(self, images, track_data, sam3, tapnext, placement, fps, tick_ms, points_per_side, cut_threshold, reid_threshold, eyes=None):
        from comfy.ldm.sam3.tracker import unpack_masks

        t_start = time.perf_counter()
        p = tracking.parse_placement(placement)
        n, H, W = images.shape[:3]
        device = mm.get_torch_device()
        scale = 384 / max(H, W)
        hm, wm = max(8, round(H * scale)), max(8, round(W * scale))

        packed = track_data["packed_masks"]
        if packed is None:
            raise ValueError("SAM3 tracked nothing; was the initial mask empty?")
        k = packed.shape[1]
        masks = np.zeros((n, k, hm, wm), bool)
        for a in range(0, n, 32):
            m = unpack_masks(packed[a:a + 32].to(device)).float()
            m = F.interpolate(m, size=(hm, wm), mode="bilinear", align_corners=False) > 0.5
            masks[a:a + 32] = m.cpu().numpy()

        small = torch.cat([
            (F.interpolate(images[a:a + 32, ..., :3].to(device).movedim(-1, 1), size=(hm, wm), mode="area") * 255)
            .round().byte().movedim(1, -1).cpu() for a in range(0, n, 32)]).numpy()
        grey = np.stack([cv2.resize(f.mean(-1).astype(np.float32), (64, 48), interpolation=cv2.INTER_AREA) for f in small])
        cuts = tracking.detect_cuts(grey, cut_threshold)

        view = tracking.MaskView(masks, H, W)
        embed = _HeadEmbedder(sam3, images, view, device)
        tgt, reid_log = tracking.assign_target(masks, embed, cuts, min_area=2e-4, reid_threshold=reid_threshold, fps=fps)
        t_prep = time.perf_counter()

        model = tapnext["model"]
        model.to(device)
        tracker = _PointTracker(tapnext, images, device)
        try:
            poses, boundaries, stats = tracking.solve(n, H, W, fps, view, tgt, cuts, p, tracker, points_per_side)
        finally:
            model.to(mm.unet_offload_device())
        t_eyes = time.perf_counter()
        lines = {}
        if eyes is not None:
            lines = _eye_lines(sam3, eyes, images, poses, tgt, view, fps)
            poses = tracking.snap_rotation(poses, boundaries, lines)
        t_eyes = time.perf_counter() - t_eyes
        smoothed = tracking.smooth(poses, boundaries, fps)
        track = tracking.encode_v1(smoothed, H, W, fps, tick_ms)

        stats.update({
            "objects": k,
            "reid": reid_log,
            "eye_lines": len(lines),
            "seconds": {"prep": round(t_prep - t_start, 3), "points": round(tracker.seconds, 3), "eyes": round(t_eyes, 3),
                        "total": round(time.perf_counter() - t_start, 3)},
        })
        return (json.dumps(track, separators=(",", ":")), json.dumps(stats))


class StickerTrackPreview:
    """Draws a sticker along a v1 track, sampled the way the web player samples it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "track_json": ("STRING", {"forceInput": True}),
            "fps": ("FLOAT", {"default": 12.0, "min": 1.0, "max": 120.0, "step": 0.5}),
            "sticker": ("IMAGE",),
        }, "optional": {
            "sticker_mask": ("MASK", {"tooltip": "LoadImage's MASK (1 = transparent)."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render"
    CATEGORY = "civitai/sticker"

    def render(self, images, track_json, fps, sticker, sticker_mask=None):
        track = json.loads(track_json)
        rgb = sticker[0, ..., :3].cpu().numpy().astype(np.float32)
        sh, sw = rgb.shape[:2]
        if sticker_mask is not None:
            alpha = 1.0 - sticker_mask[0].cpu().numpy().astype(np.float32)
            if alpha.shape != (sh, sw):
                alpha = cv2.resize(alpha, (sw, sh))
        else:
            alpha = np.ones((sh, sw), np.float32)
        rgba = np.dstack([rgb * alpha[..., None], alpha])
        out = images.clone()
        n, H, W = images.shape[:3]
        for i in range(n):
            pose = tracking.sample_v1(track, i / fps * 1000.0)
            if not pose["show"]:
                continue
            k = pose["scale"] * W / sw
            M = cv2.getRotationMatrix2D((sw / 2, sh / 2), -pose["rotation"], k)
            M[:, 2] += (pose["x"] * W - sw / 2, pose["y"] * H - sh / 2)
            warped = cv2.warpAffine(rgba, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
            frame = out[i, ..., :3].numpy()
            a = warped[..., 3:4]
            out[i, ..., :3] = torch.from_numpy(frame * (1 - a) + warped[..., :3])
        return (out,)


NODE_CLASS_MAPPINGS = {
    "StickerLoadVideo": StickerLoadVideo,
    "StickerAnchorPick": StickerAnchorPick,
    "TAPNextLoader": TAPNextLoader,
    "StickerTrack": StickerTrack,
    "StickerTrackPreview": StickerTrackPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StickerLoadVideo": "Sticker Load Video",
    "StickerAnchorPick": "Sticker Anchor Pick",
    "TAPNextLoader": "TAPNext++ Loader",
    "StickerTrack": "Sticker Track (SAM3 + TAPNext++)",
    "StickerTrackPreview": "Sticker Track Preview",
}
