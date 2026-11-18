import json
import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

MIN_SCALE, MAX_SCALE = 0.05, 0.4


@dataclass
class Pose:
    cx: float
    cy: float
    w: float
    r: float


def parse_placement(text):
    p = json.loads(text) if isinstance(text, str) else dict(text)
    return {
        "x": float(p["x"]),
        "y": float(p["y"]),
        "scale": float(p["scale"]),
        "rotation": float(p.get("rotation", 0.0)),
    }


def wrap_deg(d):
    return ((d + 180.0) % 360.0 + 360.0) % 360.0 - 180.0


def detect_cuts(grey_small, threshold):
    d = np.abs(np.diff(grey_small.astype(np.float32), axis=0)).mean(axis=(1, 2))
    return [int(i) + 1 for i in np.nonzero(d > threshold)[0]]


def assign_target(masks, embed, cuts, min_area, reid_threshold, fps):
    """Object index of the tracked character per frame (None when it is not on screen).

    Object 0 is the frame-0 head the user placed on. Without a cut only that same object
    may come back; after a cut, or a long absence, any head that looks like it may take over.
    """
    n, k = masks.shape[:2]
    area = masks.reshape(n, k, -1).mean(-1)
    present = area > min_area
    ref = embed(0, 0)
    cutset = set(cuts)
    patience = max(1, int(round(fps)))
    log = []

    def best_match(t, keep=None):
        scored = {j: float(ref @ embed(t, j)) for j in range(k) if present[t, j]}
        log.append({"t": t, "keep": keep, "sims": {str(j): round(s, 3) for j, s in scored.items()}})
        if keep in scored and scored[keep] >= reid_threshold:
            return keep
        if not scored:
            return None
        j = max(scored, key=scored.get)
        return j if scored[j] >= reid_threshold else None

    tgt = [None] * n
    tgt[0] = 0
    cur, missing = 0, 0
    for t in range(1, n):
        if t in cutset:
            cur = best_match(t, keep=cur)
            missing = 0
        elif cur is not None and present[t, cur]:
            missing = 0
        else:
            missing += 1
            if cur is not None and missing <= patience:
                tgt[t] = None
                continue
            if cur is not None or missing % max(1, patience // 4) == 0:
                cur = best_match(t)
        tgt[t] = cur if cur is not None and present[t, cur] else None
    return tgt, log


class MaskView:
    """Masks live on a small grid; geometry happens in image pixels."""

    def __init__(self, masks, H, W):
        self.masks = masks
        self.hm, self.wm = masks.shape[2:]
        self.sx, self.sy = W / self.wm, H / self.hm
        self.H, self.W = H, W
        self._box = {}
        self._dil = {}

    def mask(self, t, k):
        return self.masks[t, k]

    def box(self, t, k):
        key = (t, k)
        if key not in self._box:
            ys, xs = np.nonzero(self.masks[t, k])
            self._box[key] = None if len(xs) == 0 else (
                xs.min() * self.sx, ys.min() * self.sy, (xs.max() + 1) * self.sx, (ys.max() + 1) * self.sy)
        return self._box[key]

    def contains(self, t, k, pts, dilate=2):
        key = (t, k)
        if key not in self._dil:
            m = self.masks[t, k].astype(np.uint8)
            self._dil[key] = cv2.dilate(m, np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)) > 0
        m = self._dil[key]
        xi = np.clip((pts[:, 0] / self.sx).astype(int), 0, self.wm - 1)
        yi = np.clip((pts[:, 1] / self.sy).astype(int), 0, self.hm - 1)
        return m[yi, xi]


def head_rel(pose, box):
    bw, bh = box[2] - box[0], box[3] - box[1]
    unit = math.sqrt(bw * bh)
    return ((pose.cx - (box[0] + box[2]) / 2) / bw, (pose.cy - (box[1] + box[3]) / 2) / bh, pose.w / unit, pose.r)


def from_head_rel(rel, box):
    bw, bh = box[2] - box[0], box[3] - box[1]
    unit = math.sqrt(bw * bh)
    return Pose((box[0] + box[2]) / 2 + rel[0] * bw, (box[1] + box[3]) / 2 + rel[1] * bh, rel[2] * unit, rel[3])


def seed_points(view, t, k, pose, per_side):
    """Lattice points on the head, dense in an ellipse around the sticker and sparse elsewhere."""
    th = math.radians(pose.r)
    ax, ay = 0.8 * pose.w, 0.5 * pose.w
    step = 2 * ax / per_side
    span = np.arange(-ax, ax + 1e-6, step)
    gx, gy = np.meshgrid(span, np.arange(-ay, ay + 1e-6, step))
    inside = (gx / ax) ** 2 + (gy / ay) ** 2 <= 1.0
    lx, ly = gx[inside], gy[inside]
    roi = np.stack([pose.cx + lx * math.cos(th) - ly * math.sin(th),
                    pose.cy + lx * math.sin(th) + ly * math.cos(th)], -1)

    box = view.box(t, k)
    bw, bh = box[2] - box[0], box[3] - box[1]
    cstep = max(bw, bh) / 10
    cx, cy = np.meshgrid(np.arange(box[0] + cstep / 2, box[2], cstep), np.arange(box[1] + cstep / 2, box[3], cstep))
    ctx = np.stack([cx.ravel(), cy.ravel()], -1)

    def keep(p):
        if len(p) == 0:
            return p
        ok = (p[:, 0] >= 0) & (p[:, 0] < view.W) & (p[:, 1] >= 0) & (p[:, 1] < view.H)
        p = p[ok]
        return p[view.contains(t, k, p, dilate=0)]

    roi, ctx = keep(roi), keep(ctx)
    if len(ctx):
        d = np.hypot(ctx[:, 0] - pose.cx, ctx[:, 1] - pose.cy)
        ctx = ctx[d > ax]
    pts = np.concatenate([roi, ctx]) if len(ctx) else roi
    w = np.concatenate([np.ones(len(roi)), np.full(len(ctx), 0.3)])
    d = np.hypot(pts[:, 0] - pose.cx, pts[:, 1] - pose.cy) if len(pts) else np.zeros(0)
    fit_w = w * np.exp(-0.5 * (d / max(0.6 * pose.w, 1.0)) ** 2)
    return pts.astype(np.float32), w, fit_w


def fit_affine(src, dst, w, thr):
    A, inl = cv2.estimateAffine2D(src, dst, method=cv2.USAC_MAGSAC, ransacReprojThreshold=thr,
                                  maxIters=2000, confidence=0.995)
    if A is None or inl is None:
        return None
    m = inl.ravel().astype(bool)
    if m.sum() < 6:
        return None
    sw = np.sqrt(w[m] + 1e-6)[:, None]
    X = np.c_[src[m], np.ones(m.sum())] * sw
    sol, *_ = np.linalg.lstsq(X, dst[m] * sw, rcond=None)
    return sol.T


def pose_from_affine(A, seed):
    c = A @ np.array([seed.cx, seed.cy, 1.0])
    th = math.radians(seed.r)
    au = A[:, :2] @ np.array([math.cos(th), math.sin(th)])
    av = A[:, :2] @ np.array([-math.sin(th), math.cos(th)])
    # Reading roll off the sticker's own horizontal axis keeps yaw shear out of the angle,
    # and the longer axis keeps a turning head from shrinking the sticker.
    k = max(np.hypot(*au), np.hypot(*av))
    return Pose(float(c[0]), float(c[1]), seed.w * k, math.degrees(math.atan2(au[1], au[0]))), k


def leash(pose, anchor, box, frac):
    """Pull `pose` back to within `frac` of the head size of where the head box puts it.

    Points move the sticker freely inside that band, so their precision is kept while slow
    drift on fur and hair stays bounded.
    """
    unit = math.sqrt((box[2] - box[0]) * (box[3] - box[1]))
    dx, dy = anchor.cx - pose.cx, anchor.cy - pose.cy
    d = math.hypot(dx, dy)
    tol = frac * unit
    cx, cy = pose.cx, pose.cy
    if d > tol:
        cx, cy = cx + dx * (1 - tol / d), cy + dy * (1 - tol / d)
    lr = math.log(anchor.w / pose.w)
    lw = math.log(pose.w) + math.copysign(max(0.0, abs(lr) - 0.2), lr)
    return Pose(cx, cy, math.exp(lw), pose.r)


def solve(n, H, W, fps, view, tgt, cuts, placement, track_points, per_side=16, leash_frac=0.2):
    p0 = Pose(placement["x"] * W, placement["y"] * H, placement["scale"] * W, placement["rotation"])
    poses = [None] * n
    poses[0] = p0
    rel0 = head_rel(p0, view.box(0, 0))
    cutset = set(cuts)
    switches = {t for t in range(1, n) if tgt[t] is not None and tgt[t] != tgt[t - 1]}
    stats = {"frames": n, "cuts": sorted(cuts), "true_cuts": [], "segments": 0, "point_frames": 0,
             "held_frames": 0, "hidden_frames": 0, "reseeds": 0}

    thr = max(2.0, 0.006 * max(H, W))
    reseed_after = max(4, int(round(fps)))
    last_w = p0.w

    def guarded(pose, same_shot):
        # A head mask that briefly swallows a paw or loses an ear must not resize the sticker.
        if same_shot:
            pose.w = min(max(pose.w, last_w / 1.33), last_w * 1.33)
        return pose

    seed_t, seed_pose, rel_last, rel_shot = 0, p0, rel0, rel0
    while seed_t is not None and seed_t < n:
        stats["segments"] += 1
        nb = min([b for b in switches if b > seed_t], default=n)
        k0 = tgt[seed_t]
        pts, w, fit_w = seed_points(view, seed_t, k0, seed_pose, per_side)
        stop, next_pose, cut_at = nb, None, None
        if len(pts) >= 8 and nb - seed_t > 1:
            for t, pos, vis in track_points(seed_t, nb, pts):
                if t == seed_t:
                    continue
                k = tgt[t]
                if k is None:
                    if t in cutset:
                        stop, cut_at = t, t
                        break
                    continue
                ok = vis & (pos[:, 0] >= 0) & (pos[:, 0] < W) & (pos[:, 1] >= 0) & (pos[:, 1] < H)
                ok &= view.contains(t, k, pos)
                frac = float((w * ok).sum() / w.sum())
                pose = None
                if ok.sum() >= 8 and frac >= 0.25:
                    A = fit_affine(pts[ok], pos[ok], fit_w[ok], thr)
                    if A is not None:
                        cand, kscale = pose_from_affine(A, seed_pose)
                        if abs(wrap_deg(cand.r - seed_pose.r)) <= 60 and 0.3 <= kscale <= 3.5:
                            pose = cand
                if t in cutset and (pose is None or frac < 0.5):
                    stop, cut_at = t, t
                    break
                if pose is None:
                    poses[t] = guarded(from_head_rel(rel_last, view.box(t, k)), True)
                    last_w = poses[t].w
                    stats["held_frames"] += 1
                    continue
                pose = leash(pose, from_head_rel(rel_shot, view.box(t, k)), view.box(t, k), leash_frac)
                poses[t] = pose
                last_w = pose.w
                rel_last = head_rel(pose, view.box(t, k))
                stats["point_frames"] += 1
                if t - seed_t >= reseed_after and frac < 0.5:
                    stop, next_pose = t, pose
                    break
        else:
            for t in range(seed_t + 1, nb):
                if tgt[t] is not None:
                    poses[t] = guarded(from_head_rel(rel_last, view.box(t, tgt[t])), True)
                    stats["held_frames"] += 1

        if next_pose is not None:
            stats["reseeds"] += 1
            seed_t, seed_pose = stop, next_pose
            continue
        if cut_at is not None:
            stats["true_cuts"].append(cut_at)
        nxt = next((t for t in range(stop, n) if tgt[t] is not None), None)
        if nxt is None:
            break
        new_shot = cut_at is not None or any(stop <= c <= nxt for c in cutset)
        rel = rel0 if new_shot else rel_last
        seed_t, seed_pose = nxt, guarded(from_head_rel(rel, view.box(nxt, tgt[nxt])), not new_shot)
        if new_shot:
            rel_shot = head_rel(seed_pose, view.box(nxt, tgt[nxt]))
        poses[nxt] = seed_pose
        last_w = seed_pose.w
        switches.add(nxt)

    stats["hidden_frames"] = sum(p is None for p in poses)
    return poses, sorted(switches | set(stats["true_cuts"])), stats


def smooth(poses, boundaries, fps, sigma_seconds=0.08):
    sigma = max(0.5, sigma_seconds * fps)
    n = len(poses)
    out = list(poses)
    cut = set(boundaries)
    i = 0
    while i < n:
        if poses[i] is None:
            i += 1
            continue
        j = i + 1
        while j < n and poses[j] is not None and j not in cut:
            j += 1
        run = poses[i:j]
        if len(run) >= 3:
            cx = gaussian_filter1d(np.array([p.cx for p in run]), sigma, mode="nearest")
            cy = gaussian_filter1d(np.array([p.cy for p in run]), sigma, mode="nearest")
            lw = gaussian_filter1d(np.log([p.w for p in run]), sigma, mode="nearest")
            rr = gaussian_filter1d(np.unwrap(np.radians([p.r for p in run])), sigma, mode="nearest")
            for m in range(len(run)):
                out[i + m] = Pose(float(cx[m]), float(cy[m]), float(np.exp(lw[m])), math.degrees(rr[m]))
        i = j
    out[0] = poses[0]
    min_run = max(2, int(round(0.15 * fps)))
    i = 1
    while i < n:
        if out[i] is None:
            i += 1
            continue
        j = i
        while j < n and out[j] is not None:
            j += 1
        if j - i < min_run:
            for m in range(i, j):
                out[m] = None
        i = j
    return out


def _collapse(values, step):
    pairs, last = [], len(values) - 1
    for i, v in enumerate(values):
        changed = i == 0 or v != values[i - 1]
        ends_hold = (not step) and i < last and v != values[i + 1]
        if changed or ends_hold:
            pairs.append([i, v])
    return pairs


def encode_v1(poses, H, W, fps, tick_ms):
    """Track in the web app's v1 channel format (sticker-track.ts encodeStickerTrack)."""
    n = len(poses)
    ticks = int(math.floor((n / fps * 1000.0) / tick_ms)) + 1
    samples = []
    for t in range(ticks):
        f = min(t * tick_ms / 1000.0 * fps, n - 1)
        i0 = int(math.floor(f))
        i1 = min(i0 + 1, n - 1)
        a = f - i0
        p, q = poses[i0], poses[i1]
        if p is not None and q is not None:
            dr = wrap_deg(q.r - p.r)
            samples.append(Pose(p.cx + (q.cx - p.cx) * a, p.cy + (q.cy - p.cy) * a, p.w + (q.w - p.w) * a, p.r + dr * a))
        else:
            samples.append(poses[int(round(f))] if poses[int(round(f))] is not None else None)

    last = next((s for s in samples if s is not None), None)
    xs, ys, ss, rs, show = [], [], [], [], []
    for s in samples:
        if s is not None:
            last = s
        cur = last
        xs.append(round(min(max(cur.cx / W, 0.0), 1.0), 4))
        ys.append(round(min(max(cur.cy / H, 0.0), 1.0), 4))
        ss.append(round(min(max(cur.w / W, MIN_SCALE), MAX_SCALE), 4))
        rs.append(round(wrap_deg(cur.r), 1))
        show.append(0 if s is None else 1)
    track = {"v": 1, "ms": int(tick_ms)}
    show_pairs = _collapse(show, True)
    if not (len(show_pairs) == 1 and show_pairs[0][1] == 1):
        track["show"] = show_pairs
    track["x"] = _collapse(xs, False)
    track["y"] = _collapse(ys, False)
    track["s"] = _collapse(ss, False)
    track["r"] = _collapse(rs, False)
    return track


def _sample_channel(pairs, tick, angular):
    idx = -1
    for i, (t, _) in enumerate(pairs):
        if t <= tick:
            idx = i
        else:
            break
    if idx < 0:
        return pairs[0][1]
    if idx == len(pairs) - 1:
        return pairs[idx][1]
    (t0, v0), (t1, v1) = pairs[idx], pairs[idx + 1]
    a = (tick - t0) / (t1 - t0)
    if not angular:
        return v0 + (v1 - v0) * a
    return wrap_deg(v0 + wrap_deg(v1 - v0) * a)


def sample_v1(track, time_ms):
    """Port of sticker-track.ts sampleChannels: what the player draws at a moment."""
    tick = time_ms / track["ms"]
    show = True
    if "show" in track:
        idx = max(0, max((i for i, (t, _) in enumerate(track["show"]) if t <= tick), default=0))
        show = track["show"][idx][1] == 1
    return {
        "show": show,
        "x": _sample_channel(track["x"], tick, False),
        "y": _sample_channel(track["y"], tick, False),
        "scale": _sample_channel(track["s"], tick, False),
        "rotation": _sample_channel(track["r"], tick, True),
    }
