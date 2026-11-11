from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.app.jobs import JobManager
from backend.app.runner import default_runner
from backend.app.schemas import Health, JobCreated, JobStatus
from backend.app.settings import load_settings
from ml.utils.video import video_info

CHUNK = 1 << 20


def create_app(settings=None, runner=None):
    settings = settings or load_settings()
    manager = JobManager(settings.jobs_dir, runner or default_runner(settings))

    app = FastAPI(title="football-analytics")
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["*"], allow_headers=["*"])

    def models_available():
        return settings.detector_weights.exists() and settings.pitch_weights.exists()

    def get_job(job_id):
        job = manager.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job")
        return job

    @app.get("/health", response_model=Health)
    def health():
        return Health(status="ok", live=models_available())

    @app.post("/api/jobs", status_code=202, response_model=JobCreated)
    async def create_job(file: UploadFile = File(...)):
        if not models_available():
            raise HTTPException(503, "Model weights not found on the server")
        if not (file.filename or "").lower().endswith(".mp4"):
            raise HTTPException(400, "Only .mp4 files are accepted")

        job = manager.new_job()
        limit, size, too_big = settings.max_upload_mb * 1024 * 1024, 0, False
        with open(job.video, "wb") as f:
            while chunk := await file.read(CHUNK):
                size += len(chunk)
                if size > limit:
                    too_big = True
                    break
                f.write(chunk)
        if too_big:
            manager.discard(job)
            raise HTTPException(413, f"File larger than {settings.max_upload_mb} MB")

        info = video_info(job.video)
        if info["fps"] <= 0 or info["n_frames"] <= 0:
            manager.discard(job)
            raise HTTPException(400, "Could not read the video")
        if info["n_frames"] / info["fps"] > settings.max_seconds:
            manager.discard(job)
            raise HTTPException(400, f"Video longer than {settings.max_seconds:.0f} s")

        manager.submit(job)
        return JobCreated(id=job.id)

    @app.get("/api/jobs/{job_id}", response_model=JobStatus)
    def job_status(job_id: str):
        job = get_job(job_id)
        return JobStatus(
            id=job.id,
            status=job.status,
            stage=job.stage,
            progress=round(job.progress, 3),
            error=job.error,
            queue_position=manager.queue_position(job),
        )

    @app.get("/api/jobs/{job_id}/result")
    def job_result(job_id: str):
        job = get_job(job_id)
        if job.status != "done":
            raise HTTPException(409, job.error or f"Job is {job.status}")
        return FileResponse(job.result, media_type="application/json")

    @app.get("/api/demo")
    def demo():
        if not settings.demo_json.exists():
            raise HTTPException(404, "Demo data not found")
        return FileResponse(settings.demo_json, media_type="application/json")

    return app
