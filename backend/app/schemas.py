from pydantic import BaseModel


class JobCreated(BaseModel):
    id: str


class JobStatus(BaseModel):
    id: str
    status: str
    stage: str
    progress: float
    error: str | None = None
    queue_position: int = 0


class Health(BaseModel):
    status: str
    live: bool
