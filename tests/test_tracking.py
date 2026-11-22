"""Unit tests for tracking.py. Run: python tests/test_tracking.py (needs numpy, scipy, opencv-python-headless)."""
import importlib.util
import os

import numpy as np

_spec = importlib.util.spec_from_file_location("tracking", os.path.join(os.path.dirname(__file__), "..", "tracking.py"))
tr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tr)
Pose = tr.Pose


def test_reid_off_does_not_take_over_after_a_cut():
    n, k = 8, 2
    masks = np.zeros((n, k, 10, 10), bool)
    masks[:4, 0, 2:5, 2:5] = True
    masks[4:, 1, 5:8, 5:8] = True  # after the cut at 4 only another head is on screen
    embed = lambda t, j: np.array([1.0, 0.0])  # and it looks identical to the first
    on, _ = tr.assign_target(masks, embed, [4], 1e-3, 0.4, 8.0, reid=True)
    off, _ = tr.assign_target(masks, embed, [4], 1e-3, 0.4, 8.0, reid=False)
    assert on == [0] * 4 + [1] * 4
    assert off == [0] * 4 + [None] * 4


def test_eye_anchor_holds_the_sticker_on_its_eye():
    n = 20
    poses = [Pose(100.0 + 3 * t, 100.0, 20.0, 0.0) for t in range(n)]  # drifts right 3 px a frame
    eyes = {t: [(100.0, 100.0, 225.0), (160.0, 100.0, 225.0)] for t in range(0, n, 4)}  # eyes still, seen every 4th frame
    out = tr.eye_anchor(poses, [], eyes)
    assert max(abs(p.cx - 100.0) for p in poses) > 50
    assert max(abs(p.cx - 100.0) for p in out[:17]) < 1e-6
    assert all(abs(p.cx - 160.0) > 30 for p in out[:17])  # never hops to the other eye


def test_eye_anchor_leaves_a_sticker_that_is_not_on_an_eye():
    eyes = {t: [(100.0, 100.0, 225.0)] for t in range(0, 20, 4)}
    off = [Pose(130.0 + 3 * t, 100.0, 20.0, 0.0) for t in range(20)]
    assert tr.eye_anchor(off, [], eyes) == off
    hid = [Pose(100.0, 100.0, 20.0, 0.0)] * 20
    hid[5] = None
    assert tr.eye_anchor(hid, [], eyes)[5] is None



def test_hide_when_eye_lost_hides_while_its_eye_is_missing():
    n = 28
    poses = [Pose(100.0, 100.0, 20.0, 0.0) for _ in range(n)]
    eye = [(100.0, 100.0, 225.0)]
    other = [(160.0, 100.0, 225.0)]  # the other eye stays visible; it must not count as found
    eyes = {0: eye, 4: eye, 8: other, 12: other, 16: [], 20: eye, 24: eye}
    out = tr.eye_anchor(poses, [], eyes, hide_lost=True)
    assert all(out[t] is not None for t in range(8)), "shown while the eye is found"
    assert all(out[t] is None for t in range(8, 20)), "hidden from the first miss until the eye is back"
    assert all(out[t] is not None for t in range(20, n)), "back once the eye is found again"
    assert all(p is not None for p in tr.eye_anchor(poses, [], eyes)), "off by default"
    one_miss = {0: eye, 4: eye, 8: [], 12: eye, 16: eye}
    assert all(p is not None for p in tr.eye_anchor(poses, [], one_miss, hide_lost=True)), "a single miss does not hide"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
