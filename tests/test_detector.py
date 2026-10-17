import pandas as pd

from ml.detection.detector import Detector
from ml.tracking.tracker import track_people


class FakeModel:
    def track(self, **kwargs):
        return iter([])

    def predict(self, **kwargs):
        return iter([])


class CountingDetector(Detector):
    def __init__(self):
        super().__init__("unused.pt")
        self.models = []

    def new_model(self):
        self.models.append(FakeModel())
        return self.models[-1]


def test_each_task_gets_its_own_model():
    det = CountingDetector()
    assert track_people(det, "v.mp4", "cfg.yaml").empty
    assert isinstance(det.ball_candidates("v.mp4"), pd.DataFrame)
    assert len(det.models) == 2
    assert det.models[0] is not det.models[1]
