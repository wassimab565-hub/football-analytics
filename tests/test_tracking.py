import pandas as pd

from ml.tracking.tracker import clean_tracks


def rows(track_id, frames, cls, conf=0.8):
    return [{"frame": f, "track_id": track_id, "cls": c, "conf": conf, "x1": 0, "y1": 0, "x2": 10, "y2": 20} for f, c in zip(frames, cls)]


def test_short_tracks_are_removed():
    df = pd.DataFrame(rows(1, range(20), [2] * 20) + rows(2, range(5), [2] * 5))
    assert set(clean_tracks(df).track_id) == {1}


def test_vote_is_weighted_by_confidence():
    r = rows(1, range(20), [2] * 10 + [3] * 10, conf=0.4)
    for row in r[10:]:
        row["conf"] = 0.9
    out = clean_tracks(pd.DataFrame(r))
    assert (out.cls_smooth == 3).all()


def test_goalkeeper_share_override():
    df = pd.DataFrame(rows(1, range(20), [1] * 4 + [2] * 16))
    assert (clean_tracks(df).cls_smooth == 2).all()
    assert (clean_tracks(df, gk_share=0.15).cls_smooth == 1).all()


def test_empty_input():
    df = pd.DataFrame(columns=["frame", "track_id", "cls", "conf", "x1", "y1", "x2", "y2"])
    assert clean_tracks(df).empty
