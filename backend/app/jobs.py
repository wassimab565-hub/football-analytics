import queue
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Job:
    id: str
    dir: Path
    status: str = "uploading"
    stage: str = ""
    progress: float = 0.0
    error: str | None = None
    created: float = field(default_factory=time.time)

    @property
    def video(self):
        return self.dir / "input.mp4"

    @property
    def result(self):
        return self.dir / "result.json"


class JobManager:
    def __init__(self, jobs_dir, runner):
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.runner = runner
        self.jobs = {}
        self.pending = queue.Queue()
        threading.Thread(target=self._work, daemon=True).start()

    def new_job(self):
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, dir=self.jobs_dir / job_id)
        job.dir.mkdir(parents=True)
        self.jobs[job_id] = job
        return job

    def submit(self, job):
        job.status = "queued"
        self.pending.put(job.id)

    def discard(self, job):
        shutil.rmtree(job.dir, ignore_errors=True)
        self.jobs.pop(job.id, None)

    def queue_position(self, job):
        if job.status != "queued":
            return 0
        ahead = sum(1 for j in self.jobs.values() if j.status == "queued" and j.created < job.created)
        running = sum(1 for j in self.jobs.values() if j.status == "running")
        return ahead + running

    def _progress(self, job, stage, fraction):
        job.stage = stage
        job.progress = fraction

    def _work(self):
        while True:
            job = self.jobs.get(self.pending.get())
            if job is None:
                continue
            job.status = "running"
            try:
                self.runner(job.video, job.dir, lambda stage, p: self._progress(job, stage, p))
                job.status, job.stage, job.progress = "done", "done", 1.0
            except Exception as e:
                job.status, job.error = "failed", f"{type(e).__name__}: {e}"[:300]
