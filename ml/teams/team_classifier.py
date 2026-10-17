import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ml.utils.video import iter_frames

PLAYER_CLASS = 2


def sample_boxes(tracks, n_per_track=8, min_height=40):
    players = tracks[tracks.cls_smooth == PLAYER_CLASS]
    players = players[(players.y2 - players.y1) >= min_height]
    return players.sort_values("conf", ascending=False).groupby("track_id").head(n_per_track)


def jersey_color(frame, box):
    x1, y1, x2, y2 = (max(int(v), 0) for v in box)
    h, w = y2 - y1, x2 - x1
    crop = frame[y1 + int(0.15 * h):y1 + int(0.55 * h), x1 + int(0.2 * w):x2 - int(0.2 * w)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grass = (hsv[..., 0] >= 35) & (hsv[..., 0] <= 85) & (hsv[..., 1] > 40)
    if (~grass).sum() < 20:
        return None
    return np.median(cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[~grass], axis=0)


def track_features(video, tracks, n_per_track=8, min_height=40):
    sample = sample_boxes(tracks, n_per_track, min_height)
    needed = {f: g for f, g in sample.groupby("frame")}
    feats = []
    if needed:
        last = max(needed)
        for i, frame in iter_frames(video):
            if i > last:
                break
            g = needed.get(i)
            if g is None:
                continue
            for r in g.itertuples():
                lab = jersey_color(frame, (r.x1, r.y1, r.x2, r.y2))
                if lab is not None:
                    feats.append([r.track_id, *lab])
    return pd.DataFrame(feats, columns=["track_id", "L", "a", "b"]).groupby("track_id").median()


def assign_teams(features, n_teams=2):
    if len(features) < n_teams:
        return pd.Series(dtype=int, name="team")
    km = KMeans(n_clusters=n_teams, n_init=10, random_state=0).fit(features.values)
    return pd.Series(km.labels_, index=features.index, name="team")
