import json

import numpy as np
import pandas as pd

from ml.export import export_json


def test_export_json_structure(tmp_path):
    positions = pd.DataFrame({
        "frame": [0, 0, 2], "track_id": [1, 2, 1], "cls": [2, 3, 2], "team": [0.0, np.nan, 0.0],
        "X": [10.04, 52.5, 11.0], "Y": [20.0, 34.0, 21.0],
    })
    path = export_json(positions, {2: (50.0, 30.0)}, {"fps": 25.0, "n_frames": 4}, tmp_path / "r.json")
    data = json.loads(path.read_text())
    assert data["n_frames"] == 4 and data["pitch"] == {"length": 105.0, "width": 68.0}
    assert len(data["frames"]) == 4
    assert data["frames"][0]["players"] == [[1, 0, 2, 10.0, 20.0], [2, -1, 3, 52.5, 34.0]]
    assert data["frames"][1] == {"players": [], "ball": None}
    assert data["frames"][2]["ball"] == [50.0, 30.0]
