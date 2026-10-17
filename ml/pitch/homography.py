import cv2
import numpy as np
import pandas as pd

from ml.pitch.pitch_config import EXCLUDED_KEYPOINTS, PITCH


def fit_homographies(kps, width, height, kp_conf=0.5, margin=10, thr=8.0, exclude=EXCLUDED_KEYPOINTS):
    Hs, rows = {}, []
    for f, g in kps.groupby("frame"):
        g = g[(g.conf >= kp_conf) & g.x.between(margin, width - margin) & g.y.between(margin, height - margin) & ~g.kp.isin(exclude)]
        if len(g) < 4:
            continue
        world = PITCH[g.kp.to_numpy()]
        image = g[["x", "y"]].to_numpy(np.float32)
        H, mask = cv2.findHomography(world, image, cv2.RANSAC, thr)
        if H is None:
            continue
        Hs[int(f)] = H
        rows.append({"frame": int(f), "n": len(g), "inliers": int(mask.sum())})
    return Hs, pd.DataFrame(rows, columns=["frame", "n", "inliers"])


class HomographyProvider:
    def __init__(self, Hs, max_gap=10):
        self.Hs = Hs
        self.frames = np.array(sorted(Hs), dtype=int)
        self.max_gap = max_gap

    def get(self, frame):
        if frame in self.Hs:
            return self.Hs[frame]
        prev = self.frames[self.frames < frame]
        if len(prev) and frame - prev[-1] <= self.max_gap:
            return self.Hs[int(prev[-1])]
        return None


def to_pitch(H, points):
    pts = np.asarray(points, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.linalg.inv(H)).reshape(-1, 2)
