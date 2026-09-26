"""Eternal jobs: a fly doing an absurd job forever (docs/JOBS.md).

    from perpetualfly.jobs import create_job_session, JobRunner
    session, job = create_job_session("sisyphus")
    JobRunner(session, job).run(max_seconds=60)
"""

from perpetualfly.jobs.base import CameraPreset, EternalJob, JobConfig, Steering, format_uptime
from perpetualfly.jobs.registry import available_jobs, get_job, make_job, register_job
from perpetualfly.jobs.runtime import (
    JobCamera,
    JobRunner,
    create_job_session,
    install_job,
    job_props_present,
)

__all__ = [
    "CameraPreset", "EternalJob", "JobCamera", "JobConfig", "JobRunner", "Steering",
    "available_jobs", "create_job_session", "format_uptime", "get_job", "install_job",
    "job_props_present", "make_job", "register_job",
]
