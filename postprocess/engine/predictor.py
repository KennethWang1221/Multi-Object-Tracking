import cv2
import numpy as np
import os
from types import SimpleNamespace
from postprocess.data import load_inference_source
from postprocess.utils import callbacks
import onnxruntime

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x, axis=None):
    """
    Compute the softmax of an array `x` along a specified axis.

    Parameters:
        x (numpy.ndarray): Input array.
        axis (int, optional): Axis along which the softmax is computed.
            If not provided, the softmax is computed over the entire array.

    Returns:
        numpy.ndarray: Softmax output array with the same shape as the input array.
    """
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = np.arange(w) + grid_cell_offset  # shift x
        sy = np.arange(h) + grid_cell_offset  # shift y
        sy, sx = np.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(np.stack((sx, sy), axis=-1).reshape(-1, 2))
        stride_tensor.append(np.full((h * w, 1), stride))
    return np.concatenate(anchor_points), np.concatenate(stride_tensor)

def decode_bboxes(distance, anchor_points, xywh=True, dim=1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance[:,0:2,:],distance[:,2:,:]
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return np.concatenate((c_xy, wh), axis=dim)  # xywh bbox
    return np.concatenate((x1y1, x2y2), axis=dim)  # xyxy bbox

def concat_tile(infer_res):
    no = 144
    stride = [8, 16, 32]
    reg_max = 16
    nc = 80
    c1 = 16
    shape = infer_res[0].shape  # BCHW
    x_cat = np.concatenate([xi.reshape(shape[0], no, -1) for xi in infer_res], axis=2)
    anchors, strides = make_anchors(infer_res, stride, 0.5)
    anchors = np.transpose(anchors, (1, 0))
    strides = np.transpose(strides, (1, 0))
    box, cls = np.split(x_cat, [reg_max * 4], axis=1)
    
    b, _, a = box.shape
    box = box.reshape(b, 4, c1, a) 
    box = np.transpose(box, (0,2,1,3)) 
    box = softmax(box,axis=1) 
    model_22_dfl_conv_weight = np.arange(16).reshape(1,16,1,1) 
    box = box * model_22_dfl_conv_weight
    box = np.sum(box, axis=1)
    box = box.reshape(b,4,a)
    anchors = np.expand_dims(anchors, axis=0)
    dbox = decode_bboxes(box, anchors) * strides
    y = np.concatenate((dbox, 1 / (1 + np.exp(-cls))), axis=1)
    return y

class IterableSimpleNamespace(SimpleNamespace):
    """
    An iterable SimpleNamespace class that provides enhanced functionality for attribute access and iteration.

    This class extends the SimpleNamespace class with additional methods for iteration, string representation,
    and attribute access. It is designed to be used as a convenient container for storing and accessing
    configuration parameters.
    """

    def __iter__(self):
        """Return an iterator of key-value pairs from the namespace's attributes."""
        return iter(vars(self).items())

    def __str__(self):
        """Return a human-readable string representation of the object."""
        return "\n".join(f"{k}={v}" for k, v in vars(self).items())

    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__

    def get(self, key, default=None):
        """Return the value of the specified key if it exists; otherwise, return the default value."""
        return getattr(self, key, default)

class AutoBackend():
    def __init__(
        self,
        weights=None,
        device="cpu",
        dnn=False,
        data=None,
        fp16=False,
        batch=1,
        fuse=False,
        verbose=False,
    ):
        """
        Initialize the AutoBackend for inference.
        """
        super().__init__()


    def forward(self, im, augment=False, visualize=False, embed=None):
        """
        Runs inference on the YOLOv8.
        """
        pass


class BasePredictor:
    """
    BasePredictor.

    A base class for creating predictors.
    """

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        """
        Initializes the BasePredictor class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        
        self.args = IterableSimpleNamespace(**overrides) # get_cfg(cfg, overrides)
        self.callbacks = _callbacks or callbacks.get_default_callbacks()

        callbacks.add_integration_callbacks(self)

    def preprocess(self, im):
        """
        Prepares input image before inference.
        """
        im = np.stack(self.pre_transform(im))
        im = np.transpose(im, (0, 3, 1, 2))
        im = im.astype(np.float32)
        im /= 255  # 0 - 255 to 0.0 - 1.0
        return im

    def inference(self, im, *args, **kwargs):
        """Runs inference on a given image using the specified model and arguments."""
        suffix = os.path.splitext(self.args.model)[-1][1:]
        providers = ["CPUExecutionProvider"]
        session = onnxruntime.InferenceSession(self.args.model, providers=providers)
        y = session.run([output.name for output in session.get_outputs()], {session.get_inputs()[0].name: im})
        y = concat_tile(y)

        return y
    
    def pre_transform(self, im):
        """
        Pre-transform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Returns:
            (list): A list of transformed images.
        """
        
        return [cv2.resize(x, ((640, 352))) for x in im]

    def postprocessing(self, preds, img, orig_imgs):
        """Post-processes predictions for an image and returns them."""
        return preds

    def __call__(self, source=None, model=None, stream=False, *args, **kwargs):
        """Performs inference on an image or stream."""
        self.stream = stream
        return list(self.stream_inference(source, model, *args, **kwargs))  # merge list of Result into one

    def setup_source(self, source):
        """Sets up source and inference mode."""
        self.imgsz = [self.args.imgsz[0], self.args.imgsz[1]] 
        self.transforms = None

        self.dataset = load_inference_source(
            source=source,
        )
        self.source_type = self.dataset.source_type
        self.vid_writer = {}

    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """Streams real-time inference on camera feed and saves results to file."""

        self.setup_source(source if source is not None else self.args.source)

        self.seen, self.windows, self.batch = 0, [], None
        self.run_callbacks("on_predict_start")
        for self.batch in self.dataset:
            self.run_callbacks("on_predict_batch_start")
            paths, im0s, s = self.batch

            # Preprocess
            im = self.preprocess(im0s)
            # Inference
            preds = self.inference(im, *args, **kwargs)
            # Postprocess
            self.results = self.postprocessing(preds, im, im0s)
            self.run_callbacks("on_predict_postprocess_end")

            # Visualize, save, write results
            n = len(im0s)
            for i in range(n):
                self.seen += 1

            self.run_callbacks("on_predict_batch_end")
            yield from self.results

        # Release assets
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()

        # if self.args.verbose and self.seen:
        self.run_callbacks("on_predict_end")

    def setup_model(self, model, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        self.model = AutoBackend(
            weights=self.args.model,
            device='cpu',
            dnn=False,
            data=None,
            fp16=False,
            batch=1,
            fuse=False,
            verbose=verbose,
        )

        self.device = 'cpu'  # update device

    def run_callbacks(self, event: str):
        """Runs all registered callbacks for a specific event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def add_callback(self, event: str, func):
        """Add callback."""
        self.callbacks[event].append(func)
