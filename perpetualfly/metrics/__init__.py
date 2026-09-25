"""Runtime metrics: locomotion stats, fall detection, run metrics and run logging.

Typical wiring (see scripts/demo_falls.py)::

    detector = FallDetector(sim)                      # post-step + reset hook
    metrics = RunMetrics().attach(sim, detector)      # counts falls/recoveries/hits
    logger = RunLogger(LoggingConfig(), config=cfg).attach(sim, detector, metrics)
    ...
    metrics.record_hit(hit_event)                     # from the perturbation module
    ...
    logger.close()                                    # in a finally
"""

from perpetualfly.metrics.contacts import ContactClassifier, ContactSummary
from perpetualfly.metrics.falls import FallDetector, FallDetectorConfig, FallEvent, FallState
from perpetualfly.metrics.locomotion import LocomotionStats
from perpetualfly.metrics.run_logger import LoggingConfig, RunLogger
from perpetualfly.metrics.run_metrics import HitRecord, RunMetrics

__all__ = [
    "ContactClassifier",
    "ContactSummary",
    "FallDetector",
    "FallDetectorConfig",
    "FallEvent",
    "FallState",
    "HitRecord",
    "LocomotionStats",
    "LoggingConfig",
    "RunLogger",
    "RunMetrics",
]
