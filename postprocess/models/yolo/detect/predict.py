from postprocess.engine.predictor import BasePredictor
from postprocess.engine.results import Results
from postprocess.utils import ops

class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.
    """

    def postprocessing(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(
            preds,
            self.args.conf_thres,
            self.args.iou_thres,
            agnostic=self.args.agnostic_nms,
            max_det=1000,
            classes=self.args.classes,
        )

        results = []
        
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.args.yolov8_classes_names, boxes=pred))
        return results
