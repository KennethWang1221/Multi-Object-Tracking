import cv2
import argparse
from postprocess import YOLO, solutions
import os 

DEBUG = False

def main(**args):
    if not os.path.exists(args['opts_dir']):
        os.makedirs(args['opts_dir'])
    
    if DEBUG:
        video_frames = os.path.join(args['opts_dir'], 'video_frames') 
        image_results_path = os.path.join(args['opts_dir'], 'image_res') 

        if not os.path.exists(image_results_path):
            os.makedirs(image_results_path)
        if not os.path.exists(video_frames):
            os.makedirs(video_frames)

    mp4_res = os.path.join(args['opts_dir'], '{}.mp4'.format(os.path.basename(args['opts_dir'])))

    # Video writer
    cap = cv2.VideoCapture(args['video_path'])
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(mp4_res, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Init Object Counter and Define line points
    names2index = {v: k for k, v in args['class_name_to_id'].items()}
    line_points = [(args['line_points'][0], args['line_points'][1]), (args['line_points'][2], args['line_points'][3])]
    counter = solutions.ObjectCounter(
        reg_pts=line_points,
        names=names2index,
        class_id_to_category = args['class_id_to_category'], 
        filtered_classes = args['filtered_classes'],
        classes_interest = args['classes_interest'],    
        draw_tracks=args['draw_tracks'], 
        draw_boxes = args['draw_boxes'],
        line_thickness=2,
    )
    
    frame_count = 0
    model = YOLO(args['model_file'], args['class_name_to_id'])

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        if DEBUG:
            frame_filename = os.path.join(video_frames, f'frame_{frame_count:04d}.png')
            cv2.imwrite(frame_filename, im0)

        tracks = model.track(im0, persist=args['persist'], imgsz=args['imgsz'], conf_thres=args['conf_thres'], iou_thres=args['iou_thres'], agnostic_nms=args['agnostic_nms'])
        im0 = counter.start_counting(im0, tracks)
        video_writer.write(im0)
        
        if DEBUG:
            frame_filename = os.path.join(image_results_path, f'frame_{frame_count:04d}.png')
            cv2.imwrite(frame_filename, im0)
        
        frame_count += 1
    cap.release()
    video_writer.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo')
    # Load file
    parser.add_argument("--model_file", type=str, default='./model/object_tracking.onnx',
                        help='path to NBG model')
    parser.add_argument("--video_path", type=str, default="./data/veh2.mp4",
                        help='path to video')
    parser.add_argument("--line_points", type=int, nargs=4, default=[6,793,1675,777],
                        help='coordinates for line points')
    parser.add_argument("--classes_interest", nargs='+', choices=['person', 'motor-vehicle', 'non-motor-vehicle'], 
                        default=['motor-vehicle'],
                        help='(List[str]): A list of class names to consider tracking. Multiple choices are allowed.')
    # Dirs
    parser.add_argument("--opts_dir", type=str, default="./res", \
                        help='path of outputs files ')

    parser.add_argument("--persist", action='store_false', default=True,
                        help='Whether to persist the trackers if they already exist. True tells the tracker than the current image or frame is the next in a sequence and to expect tracks from the previous image in the current image.')
    parser.add_argument("--conf_thres", type=float, default=0.1,
                        help='Confidence threshold for predictions')
    parser.add_argument("--iou_thres", type=float, default=0.7,
                        help='IoU threshold for non-max suppression')
    parser.add_argument("--agnostic_nms", action='store_true', default=False,
                        help='Whether to use class-agnostic NMS')
    parser.add_argument("--draw_tracks", action='store_true', default=False,
                        help='Flag to control whether to draw the object tracks.')
    parser.add_argument("--draw_boxes", action='store_true', default=False,
                        help='Flag to control whether to draw the object boxes.')
    argspar = parser.parse_args()

    print("\n### Test model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    argspar.class_name_to_id = {
        'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4, 'bus': 5, 'train': 6, 'truck': 7, 'boat': 8, 
        'traffic light': 9, 'fire hydrant': 10, 'stop sign': 11, 'parking meter': 12, 'bench': 13, 'bird': 14, 'cat': 15, 
        'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19, 'elephant': 20, 'bear': 21, 'zebra': 22, 'giraffe': 23, 
        'backpack': 24, 'umbrella': 25, 'handbag': 26, 'tie': 27, 'suitcase': 28, 'frisbee': 29, 'skis': 30, 
        'snowboard': 31, 'sports ball': 32, 'kite': 33, 'baseball bat': 34, 'baseball glove': 35, 'skateboard': 36, 
        'surfboard': 37, 'tennis racket': 38, 'bottle': 39, 'wine glass': 40, 'cup': 41, 'fork': 42, 'knife': 43, 
        'spoon': 44, 'bowl': 45, 'banana': 46, 'apple': 47, 'sandwich': 48, 'orange': 49, 'broccoli': 50, 'carrot': 51, 
        'hot dog': 52, 'pizza': 53, 'donut': 54, 'cake': 55, 'chair': 56, 'couch': 57, 'potted plant': 58, 'bed': 59, 
        'dining table': 60, 'toilet': 61, 'tv': 62, 'laptop': 63, 'mouse': 64, 'remote': 65, 'keyboard': 66, 
        'cell phone': 67, 'microwave': 68, 'oven': 69, 'toaster': 70, 'sink': 71, 'refrigerator': 72, 'book': 73, 
        'clock': 74, 'vase': 75, 'scissors': 76, 'teddy bear': 77, 'hair drier': 78, 'toothbrush': 79
    }

    argspar.class_id_to_category = {
        0: ['person'],
        1: ['car', 'motorcycle', 'bus', 'train', 'truck'],
        2: ['bicycle'],
        3: [
            'airplane', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    }

    argspar.filtered_classes = ['person', 'motor-vehicle', 'non-motor-vehicle', 'IGNORE']

    argspar.imgsz = [352, 640]

    main(**vars(argspar))