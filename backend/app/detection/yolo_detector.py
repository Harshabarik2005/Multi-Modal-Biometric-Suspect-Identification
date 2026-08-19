"""YOLOv8 person detector.

Thin wrapper over Ultralytics that hands the rest of the pipeline plain
`Detection` objects instead of Ultralytics `Results`, so nothing downstream is
coupled to the detector library.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.core.config import DetectionSettings, Settings, get_settings
from app.core.logging import get_logger
from app.core.types import Detection

logger = get_logger(__name__)


class YOLOPersonDetector:
    """Detects people (COCO class 0) in BGR frames.

    Weights are downloaded by Ultralytics on first use and cached under
    `paths.models_dir`.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        detection: DetectionSettings | None = None,
        device: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cfg = detection or self.settings.detection
        self.device = device or self.settings.resolve_device()

        # fp16 only makes sense on CUDA; forcing it on CPU raises in torch.
        self.half = self.cfg.half and self.device.startswith("cuda")

        weights = self._resolve_weights(self.cfg.model)
        logger.info("Loading YOLO weights %s on %s", weights, self.device)

        from ultralytics import YOLO  # imported lazily: heavy, and optional in tests

        self.model = YOLO(str(weights))
        self.model.to(self.device)

    def _resolve_weights(self, model: str) -> Path | str:
        """Prefer a local copy in models_dir; otherwise let Ultralytics fetch it."""
        candidate = Path(model)
        if candidate.is_file():
            return candidate

        local = self.settings.paths.models_dir / candidate.name
        if local.is_file():
            return local

        # Ultralytics resolves a bare name like "yolov8n.pt" by downloading it.
        self.settings.paths.models_dir.mkdir(parents=True, exist_ok=True)
        return model

    def _predict(self, source):
        return self.model.predict(
            source,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            classes=[self.cfg.person_class_id],
            imgsz=self.cfg.imgsz,
            device=self.device,
            half=self.half,
            verbose=False,
        )

    def _parse(self, result) -> list[Detection]:
        """Convert one Ultralytics `Results` into our `Detection` list."""
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        detections: list[Detection] = []
        for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, classes):
            if (y2 - y1) < self.cfg.min_box_height:
                # Too small to yield a usable face/gait/re-ID crop later.
                continue
            detections.append(
                Detection(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=float(conf),
                    class_id=int(cls),
                )
            )
        return detections

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect people in one BGR frame."""
        results = self._predict(frame)
        return self._parse(results[0]) if results else []

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """Detect over several frames in one forward pass.

        Ultralytics batches a list of same-shaped frames, which keeps the GPU
        busier than calling `detect` in a loop. Returns one list per input
        frame, in order.
        """
        if not frames:
            return []
        results = self._predict(frames)
        return [self._parse(result) for result in results]
