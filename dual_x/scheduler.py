"""Bounded, stoppable scheduler with retry and backoff."""
import logging
import random
import threading
import time

logger = logging.getLogger(__name__)

def run_job(job, attempts=3, base_delay=1.0, sleeper=time.sleep):
    if attempts < 1 or base_delay < 0:
        raise ValueError("invalid retry configuration")
    for index in range(attempts):
        try:
            return job()
        except Exception:
            logger.exception("Job failed attempt %s/%s", index + 1, attempts)
            if index == attempts - 1:
                raise
            sleeper(min(60.0, base_delay * (2 ** index)) + random.uniform(0, 0.1))

def run_periodically(job, interval_seconds, stop_event=None):
    if interval_seconds <= 0:
        raise ValueError("interval must be positive")
    stop_event = stop_event or threading.Event()
    while not stop_event.is_set():
        try:
            run_job(job)
        except Exception:
            logger.exception("Periodic job exhausted retries")
        stop_event.wait(interval_seconds)
