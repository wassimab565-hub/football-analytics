import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    detector_weights: Path
    pitch_weights: Path
    jobs_dir: Path
    demo_json: Path
    max_upload_mb: int = 100
    max_seconds: float = 20.0
    cors_origins: tuple = ("http://localhost:4200",)


def load_settings():
    origins = os.getenv("CORS_ORIGINS", "http://localhost:4200")
    return Settings(
        detector_weights=Path(os.getenv("DETECTOR_WEIGHTS", ROOT / "artifacts/detector_yolo26n_1280.pt")),
        pitch_weights=Path(os.getenv("PITCH_WEIGHTS", ROOT / "artifacts/pitch_yolo26s_pose_960.pt")),
        jobs_dir=Path(os.getenv("JOBS_DIR", ROOT / "backend/data/jobs")),
        demo_json=Path(os.getenv("DEMO_JSON", ROOT / "backend/demo/clip01_minimap.json")),
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "100")),
        max_seconds=float(os.getenv("MAX_SECONDS", "20")),
        cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
    )
