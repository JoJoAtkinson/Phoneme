"""Bridges non-Qt pipeline events to Qt signals on the GUI thread.

The pipeline runs in background threads; it calls ``emit`` (a plain callable)
for every event. We set ``emit`` to a function that posts the event object
into a queued Qt signal, ensuring delivery on the main thread.
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QObject, Signal

from ..config import PipelineMode, Settings
from ..pipeline.aligned import AlignedPipeline
from ..pipeline.base import Pipeline
from ..pipeline.compress import CompressPipeline
from ..pipeline.streaming import StreamingPipeline
from ..pipeline.utterance import UtterancePipeline

log = logging.getLogger(__name__)


def build_pipeline(settings: Settings, emit) -> Pipeline:
    mode = settings.pipeline_mode
    if mode is PipelineMode.STREAMING:
        return StreamingPipeline(settings, emit)
    if mode is PipelineMode.UTTERANCE:
        return UtterancePipeline(settings, emit)
    if mode is PipelineMode.ALIGNED:
        return AlignedPipeline(settings, emit)
    if mode is PipelineMode.COMPRESS:
        return CompressPipeline(settings, emit)
    raise ValueError(f"Unknown pipeline mode: {mode}")


class PipelineRunner(QObject):
    event = Signal(object)
    pipeline_ready = Signal()

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.pipeline: Pipeline | None = None

    def start(self) -> None:
        self.stop()
        self.pipeline = build_pipeline(self.settings, self._emit)
        # Warm up models on a background thread to keep UI responsive
        import threading

        def _warmup():
            pipeline = self.pipeline
            if pipeline is None:
                return
            warmup = getattr(pipeline, "warmup", None)
            if callable(warmup):
                try:
                    warmup()
                except Exception as e:
                    log.exception("warmup failed: %s", e)
            try:
                pipeline.start()
                self.pipeline_ready.emit()
            except Exception as e:
                log.exception("pipeline start failed: %s", e)

        threading.Thread(target=_warmup, daemon=True).start()

    def stop(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception as e:
                log.warning("pipeline stop failed: %s", e)
            self.pipeline = None

    def force_endpoint(self) -> None:
        if self.pipeline is not None:
            self.pipeline.force_endpoint_now()

    def _emit(self, event: Any) -> None:
        # Qt.QueuedConnection for cross-thread signal delivery is automatic
        # when emitter and receiver live in different threads.
        self.event.emit(event)

    # ---- exposed mic buffer for spectrogram / level meters ------------

    def mic_source(self):
        """Return (read_latest, total_written) callables, or None if mic
        isn't running yet. Safe to call any time; returns fresh callables
        each call, which always point at the current pipeline's buffer."""
        if self.pipeline is None or self.pipeline.mic is None:
            return None
        buf = self.pipeline.mic.buffer
        return buf.read_latest, (lambda: buf.total_written)
