"""Alert manager with per-identity cooldown deduplication."""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import numpy as np

from pipeline.types import Alert, MatchResult

logger = logging.getLogger(__name__)


class AlertManager:
    """Tracks watchlist match alerts and enforces per-identity cooldowns.

    Prevents the same person from generating repeated alerts every frame.
    """

    MAX_LOG_SIZE = 10_000

    def __init__(
        self,
        cooldown_seconds: float = 30.0,
        camera_id: str = "cam-01",
    ) -> None:
        self.cooldown_seconds = cooldown_seconds
        self.camera_id = camera_id

        # identity_id -> last alert timestamp
        self._last_alert: Dict[str, float] = {}

        # Running log for current session (capped to prevent unbounded growth)
        self.alert_log: List[Alert] = []

    def try_alert(
        self,
        match: MatchResult,
        bbox: np.ndarray,
        frame_index: int = 0,
    ) -> Optional[Alert]:
        """Create alert if match is positive and cooldown has elapsed.

        Args:
            match: MatchResult from watchlist search.
            bbox: Detection bounding box (x1,y1,x2,y2).
            frame_index: Current frame number.

        Returns:
            Alert if emitted, None if suppressed by cooldown or no match.
        """
        if not match.matched:
            return None

        now = time.time()
        last = self._last_alert.get(match.identity_id, 0.0)

        if (now - last) < self.cooldown_seconds:
            return None

        alert = Alert(
            identity_id=match.identity_id,
            name=match.name,
            similarity=match.similarity,
            bbox=bbox.copy(),
            timestamp=now,
            camera_id=self.camera_id,
            frame_index=frame_index,
        )

        self._last_alert[match.identity_id] = now
        if len(self.alert_log) >= self.MAX_LOG_SIZE:
            self.alert_log = self.alert_log[self.MAX_LOG_SIZE // 2:]
        self.alert_log.append(alert)

        logger.warning(
            "ALERT: %s (id=%s) matched with similarity %.3f @ frame %d",
            alert.name,
            alert.identity_id,
            alert.similarity,
            frame_index,
        )

        return alert

    def reset(self, identity_id: Optional[str] = None) -> None:
        """Reset cooldown for one or all identities."""
        if identity_id:
            self._last_alert.pop(identity_id, None)
        else:
            self._last_alert.clear()
