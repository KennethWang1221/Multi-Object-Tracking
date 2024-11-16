import numpy as np
from postprocess.utils import ops

class BaseTensor():
    """
    Base tensor class with additional methods for easy manipulation and device handling.

    Attributes:
        data (np.ndarray): Prediction data such as bounding boxes, masks, or keypoints.
        orig_shape (Tuple[int, int]): Original shape of the image, typically in the format (height, width).
    """

    def __init__(self, data, orig_shape) -> None:
        """
        Initialize BaseTensor with prediction data and the original shape of the image.

        Args:
            data (np.ndarray): Prediction data such as bounding boxes, masks, or keypoints.
            orig_shape (Tuple[int, int]): Original shape of the image in (height, width) format.

        """
        self.data = data
        self.orig_shape = orig_shape

    @property
    def shape(self):
        """
        Returns the shape of the underlying data tensor.

        Returns:
            (Tuple[int, ...]): The shape of the data tensor.
        """
        return self.data.shape

    def __len__(self):  # override len(results)
        """
        Returns the length of the underlying data tensor.

        Returns:
            (int): The number of elements in the first dimension of the data tensor.

        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a new BaseTensor instance containing the specified indexed elements of the data tensor.

        Args:
            idx (int | List[int] ): Index or indices to select from the data tensor.

        Returns:
            (BaseTensor): A new BaseTensor instance containing the indexed data.

        """
        return self.__class__(self.data[idx], self.orig_shape)


class Results():
    """
    A class for storing and manipulating inference results.

    This class encapsulates the functionality for handling detection, segmentation, pose estimation,
    and classification results from YOLO models.

    """

    def __init__(
        self, orig_img, path, names, boxes=None, masks=None, probs=None, keypoints=None, obb=None, speed=None
    ) -> None:
        """
        Initialize the Results class for storing and manipulating inference results.

        """
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None  # native size boxes
        self.masks = None  # native size or imgsz masks
        self.probs = None
        self.keypoints = None
        self.obb = None
        self.speed = {"preprocess": None, "inference": None, "postprocessing": None}
        self.names = names
        self.path = path
        self.save_dir = None
        self._keys = "boxes", "masks", "probs", "keypoints", "obb"

    def __getitem__(self, idx):
        """
        Return a Results object for a specific index of inference results.
        """
        return self._apply("__getitem__", idx)

    def __len__(self):
        """
        Return the number of detections in the Results object.

        """
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                return len(v)

    def update(self, boxes=None, masks=None, probs=None, obb=None):
        """
        Updates the Results object with new detection data.

        This method allows updating the boxes, masks, probabilities, and oriented bounding boxes (OBB) of the
        Results object. It ensures that boxes are clipped to the original image shape.

        """
        if boxes is not None:
            self.boxes = Boxes(ops.clip_boxes(boxes, self.orig_shape), self.orig_shape)


    def _apply(self, fn, *args, **kwargs):
        """
        Applies a function to all non-empty attributes and returns a new Results object with modified attributes.

        This method is internally called by methods like .to(), .cuda(), .cpu(), etc.
        """
        r = self.new()
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                setattr(r, k, getattr(v, fn)(*args, **kwargs))
        return r

    def new(self):
        """
        Creates a new Results object with the same image, path, names, and speed attributes.

        Returns:
            (Results): A new Results object with copied attributes from the original instance.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> new_result = results[0].new()
        """
        return Results(orig_img=self.orig_img, path=self.path, names=self.names, speed=self.speed)

class Boxes(BaseTensor):
    """
    A class for managing and manipulating detection boxes.

    This class provides functionality for handling detection boxes, including their coordinates, confidence scores,
    class labels, and optional tracking IDs. It supports various box formats and offers methods for easy manipulation
    and conversion between different coordinate systems.

    Attributes:
        data (numpy.ndarray): The raw tensor containing detection boxes and associated data.
        orig_shape (Tuple[int, int]): The original image dimensions (height, width).
        is_track (bool): Indicates whether tracking IDs are included in the box data.
        xyxy (numpy.ndarray): Boxes in [x1, y1, x2, y2] format.
        conf (numpy.ndarray): Confidence scores for each box.
        cls (numpy.ndarray): Class labels for each box.
        id (numpy.ndarray): Tracking IDs for each box (if available).
        xywh (numpy.ndarray): Boxes in [x, y, width, height] format.
        xyxyn (numpy.ndarray): Normalized [x1, y1, x2, y2] boxes relative to orig_shape.
        xywhn (numpy.ndarray): Normalized [x, y, width, height] boxes relative to orig_shape.

    """

    def __init__(self, boxes, orig_shape) -> None:
        """
        Initialize the Boxes class with detection box data and the original image shape.

        This class manages detection boxes, providing easy access and manipulation of box coordinates,
        confidence scores, class identifiers, and optional tracking IDs. It supports multiple formats
        for box coordinates, including both absolute and normalized forms.

        Args:
            boxes (np.ndarray): A tensor or numpy array with detection boxes of shape
                (num_boxes, 6) or (num_boxes, 7). Columns should contain
                [x1, y1, x2, y2, confidence, class, (optional) track_id].
            orig_shape (Tuple[int, int]): The original image shape as (height, width). Used for normalization.

        Attributes:
            data (): The raw tensor containing detection boxes and their associated data.
            orig_shape (Tuple[int, int]): The original image size, used for normalization.
            is_track (bool): Indicates whether tracking IDs are included in the box data.
        """
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {6, 7}, f"expected 6 or 7 values but got {n}"  # xyxy, track_id, conf, cls
        super().__init__(boxes, orig_shape)
        self.is_track = n == 7
        self.orig_shape = orig_shape

    @property
    def xyxy(self):
        """
        Returns bounding boxes in [x1, y1, x2, y2] format.

        """
        return self.data[:, :4]

    @property
    def conf(self):
        """
        Returns the confidence scores for each detection box.

        Returns:
            (Tensor | numpy.ndarray): A 1D tensor or array containing confidence scores for each detection,
                with shape (N,) where N is the number of detections.

        Examples:
            >>> boxes = Boxes(tensor([[10, 20, 30, 40, 0.9, 0]]), orig_shape=(100, 100))
            >>> conf_scores = boxes.conf
            >>> print(conf_scores)
            tensor([0.9000])
        """
        return self.data[:, -2]

    @property
    def cls(self):
        """
        Returns the class ID tensor representing category predictions for each bounding box.

        Returns:
            (Tensor | numpy.ndarray): A tensor or numpy array containing the class IDs for each detection box.
                The shape is (N,), where N is the number of boxes.

        Examples:
            >>> results = model("image.jpg")
            >>> boxes = results[0].boxes
            >>> class_ids = boxes.cls
            >>> print(class_ids)  # tensor([0., 2., 1.])
        """
        return self.data[:, -1]

    @property
    def id(self):
        """
        Returns the tracking IDs for each detection box if available.

        Returns:
            (Tensor | None): A tensor containing tracking IDs for each box if tracking is enabled,
                otherwise None. Shape is (N,) where N is the number of boxes.

        Examples:
            >>> results = model.track("path/to/video.mp4")
            >>> for result in results:
            ...     boxes = result.boxes
            ...     if boxes.is_track:
            ...         track_ids = boxes.id
            ...         print(f"Tracking IDs: {track_ids}")
            ...     else:
            ...         print("Tracking is not enabled for these boxes.")

        Notes:
            - This property is only available when tracking is enabled (i.e., when `is_track` is True).
            - The tracking IDs are typically used to associate detections across multiple frames in video analysis.
        """
        return self.data[:, -3] if self.is_track else None

    @property
    def xywh(self):
        """
        Convert bounding boxes from [x1, y1, x2, y2] format to [x, y, width, height] format.

        Returns:
            (Tensor | numpy.ndarray): Boxes in [x_center, y_center, width, height] format, where x_center, y_center are the coordinates of
                the center point of the bounding box, width, height are the dimensions of the bounding box and the
                shape of the returned tensor is (N, 4), where N is the number of boxes.

        Examples:
            >>> boxes = Boxes(tensor([[100, 50, 150, 100], [200, 150, 300, 250]]), orig_shape=(480, 640))
            >>> xywh = boxes.xywh
            >>> print(xywh)
            tensor([[100.0000,  50.0000,  50.0000,  50.0000],
                    [200.0000, 150.0000, 100.0000, 100.0000]])
        """
        return ops.xyxy2xywh(self.xyxy)

    @property
    def xyxyn(self):
        """
        Returns normalized bounding box coordinates relative to the original image size.

        This property calculates and returns the bounding box coordinates in [x1, y1, x2, y2] format,
        normalized to the range [0, 1] based on the original image dimensions.

        Returns:
            (Tensor | numpy.ndarray): Normalized bounding box coordinates with shape (N, 4), where N is
                the number of boxes. Each row contains [x1, y1, x2, y2] values normalized to [0, 1].

        Examples:
            >>> boxes = Boxes(tensor([[100, 50, 300, 400, 0.9, 0]]), orig_shape=(480, 640))
            >>> normalized = boxes.xyxyn
            >>> print(normalized)
            tensor([[0.1562, 0.1042, 0.4688, 0.8333]])
        """
        xyxy = np.copy(self.xyxy)
        xyxy[..., [0, 2]] /= self.orig_shape[1]
        xyxy[..., [1, 3]] /= self.orig_shape[0]
        return xyxy

    @property
    def xywhn(self):
        """
        Returns normalized bounding boxes in [x, y, width, height] format.

        This property calculates and returns the normalized bounding box coordinates in the format
        [x_center, y_center, width, height], where all values are relative to the original image dimensions.

        Returns:
            (Tensor | numpy.ndarray): Normalized bounding boxes with shape (N, 4), where N is the
                number of boxes. Each row contains [x_center, y_center, width, height] values normalized
                to [0, 1] based on the original image dimensions.

        Examples:
            >>> boxes = Boxes(tensor([[100, 50, 150, 100, 0.9, 0]]), orig_shape=(480, 640))
            >>> normalized = boxes.xywhn
            >>> print(normalized)
            tensor([[0.1953, 0.1562, 0.0781, 0.1042]])
        """
        xywh = ops.xyxy2xywh(self.xyxy)
        xywh[..., [0, 2]] /= self.orig_shape[1]
        xywh[..., [1, 3]] /= self.orig_shape[0]
        return xywh


