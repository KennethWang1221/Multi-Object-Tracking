
from postprocess.utils import (callbacks,)

class Model():
    """
    This class provides a  interface for YOLO models prediction.
    """

    def __init__(
        self,
        model,
        yolov8_classes_names,
        task,
        verbose,
    ):
        """
        Initializes a new instance of the YOLO model class.

        """
        super().__init__()
        model = str(model).strip()
        self.callbacks = callbacks.get_default_callbacks()
        self.predictor = None  
        self.model = model  
        self.overrides = {} 
        self.overrides["model"] = model
        self.overrides["yolov8_classes_names"] = yolov8_classes_names
        self.overrides["task"] = "detect" 

    def predict(
        self,
        source = None,
        stream = False,
        predictor=None,
        **kwargs,
    ):
        """
        Performs predictions on the given image source using the YOLO model.

        Args:
            source numpy arrays.
            stream (bool): If True, treats the input source as a continuous stream for predictions.
            predictor (BasePredictor | None): An instance of a custom predictor class for making predictions.
                If None, the method uses a default predictor.
            **kwargs (Any): Additional keyword arguments for configuring the prediction process.
        """
        is_cli = False
        args = {**self.overrides, **kwargs}  # highest priority args on the right

        if not self.predictor:
            self.predictor = self.task_map["detect"]["predictor"](overrides=args, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=is_cli)

        return self.predictor(source=source, stream=stream)

    def track(
        self,
        source = None,
        stream = False,
        persist = False,
        **kwargs,
    ):
        """
        Conducts object tracking on the specified input source using the registered trackers.
        """
        if not hasattr(self.predictor, "trackers"):
            from postprocess.trackers import register_tracker

            register_tracker(self, persist)
        kwargs["batch"] = 1  # batch-size 1 for tracking in videos
        kwargs["mode"] = "track"
        return self.predict(source=source, stream=stream, **kwargs)

    def add_callback(self, event: str, func) -> None:
        """
        Adds a callback function for a specified event.

        """
        self.callbacks[event].append(func)