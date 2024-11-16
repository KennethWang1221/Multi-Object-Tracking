from postprocess.engine.model import Model
from postprocess.models import yolo

class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""

    def __init__(self, model, yolov8_classes_names = None, task=None, verbose=False):
        """Initialize YOLO model"""
        super().__init__(model=model, yolov8_classes_names=yolov8_classes_names, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model predictor classes."""
        return {
            "detect": {
                "model": None,
                "predictor": yolo.detect.DetectionPredictor,
            },
        }