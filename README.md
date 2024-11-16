# Multi-Object Tracking

This repository provides a end-to-end solution for Multi-Object Tracking.

## Requirements

To set up the environment, clone the repository and install the necessary packages:

```bash
conda create --name obj_track_conda

conda activate obj_track_conda

pip3 install ipykernel

python3 -m ipykernel install --user --name obj_track_conda --display-name obj_track_conda

pip3 install -r requirements.txt
```

## Data

The [demo.mp4](./data/demo.mp4) from Internet (Only for demo purpose)

Please prepare your own video clip for demo.

## Usage

To run the model on a video, use the following command:

```bash
python demo.py \
  --model_file      ./model/object_tracking.onnx \
  --video_path      ./data/veh2.mp4 \
  --line_points     6 793 1675 777 \
  --classes_interest person motor-vehicle non-motor-vehicle \
  --opts_dir        ./res \
  --conf_thres      0.1 \
  --iou_thres       0.7 \
  --draw_tracks \
  --draw_boxes
```

The scripts will save the visualized results.

### Visualized Image

![assets](./assets/demo.png)

*Note: This is a demonstration. For higher accuracy, please customize the training strategy.*

## References

The torch weight is from [Yolov8](https://docs.ultralytics.com/modes/track/#tracking)

