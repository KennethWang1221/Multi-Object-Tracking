import numpy as np
from collections import defaultdict
from postprocess.utils.plotting import Annotator, colors

class Point:
    def __init__(self, x, y):
        self.coords = np.array([x, y])

    def __getitem__(self, index):
        return self.coords[index]

class LineString:
    def __init__(self, points):
        self.coords = np.array(points)

    def intersects(self, other):
        p1, p2 = self.coords
        p3, p4 = other.coords
        
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

class Polygon:
    def __init__(self, points):
        self.coords = np.array(points)

    def contains(self, point):
        n = len(self.coords)
        inside = False
        p1x, p1y = self.coords[0]
        for i in range(n + 1):
            p2x, p2y = self.coords[i % n]
            if point[1] > min(p1y, p2y):
                if point[1] <= max(p1y, p2y):
                    if point[0] <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (point[1] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or point[0] <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    @property
    def centroid(self):
        return Point(*np.mean(self.coords, axis=0))
  
class ObjectCounter:
    """A class to manage the counting of objects in a real-time video stream based on their tracks."""

    def __init__(
        self,
        names,
        class_id_to_category,
        filtered_classes,
        classes_interest,
        reg_pts=None,
        line_thickness=2,
        view_in_counts=True,
        view_out_counts=True,
        draw_tracks=False,
        draw_boxes = False,
    ):
        """
        Initializes the ObjectCounter with various tracking and counting parameters.

        Args:
            names (dict): Dictionary of class names.
            reg_pts (list): List of points defining the counting region.
            line_thickness (int): Line thickness for bounding boxes.
            view_in_counts (bool): Flag to control whether to display the in counts on the video stream.
            view_out_counts (bool): Flag to control whether to display the out counts on the video stream.
            draw_tracks (bool): Flag to control whether to draw the object tracks.
        """
        # Mouse events
        self.is_drawing = False
        self.selected_point = None

        # Region & Line Information
        self.reg_pts = reg_pts
        self.counting_region = None

        # Image and annotation Information
        self.im0 = None
        self.tf = line_thickness
        self.view_in_counts = view_in_counts
        self.view_out_counts = view_out_counts

        self.names = names  # Classes names
        self.class_id_to_category = class_id_to_category
        self.filtered_classes = filtered_classes
        self.classes_interest = classes_interest
        # Object counting Information
        self.in_counts = 0
        self.out_counts = 0
        self.count_ids = []
        self.class_wise_count = {}

        # Tracks info
        self.track_history = defaultdict(list)
        self.draw_tracks = draw_tracks
        self.draw_boxes = draw_boxes
        self.env_check = False
        # Initialize counting region
        if len(self.reg_pts) == 2:
            print("Line Counter Initiated.")
            self.counting_region = LineString(self.reg_pts)
        elif len(self.reg_pts) >= 3:
            print("Polygon Counter Initiated.")
            self.counting_region = Polygon(self.reg_pts)
        else:
            print("Invalid Region points provided, region_points must be 2 for lines or >= 3 for polygons.")
            print("Using Line Counter Now")
            self.counting_region = LineString(self.reg_pts)

        # Define the counting line segment
        self.counting_line_segment = LineString(
            [
                (self.reg_pts[0][0], self.reg_pts[0][1]),
                (self.reg_pts[1][0], self.reg_pts[1][1]),
            ]
        )

    def extract_and_process_tracks(self, tracks):
        """Extracts and processes tracks for object counting in a video stream."""
        # Annotator Init and region drawing
        annotator = Annotator(self.im0, self.tf, self.names)
        # Draw region or line
        annotator.draw_region(reg_pts=self.reg_pts, color=(104, 0, 123), thickness=self.tf * 2)

        if tracks[0].boxes.id is not None:
            boxes = tracks[0].boxes.xyxy
            clss = tracks[0].boxes.cls.tolist()
            track_ids = tracks[0].boxes.id.tolist()

            # Extract tracks
            for box, track_id, cls in zip(boxes, track_ids, clss):
                # Draw bounding box
                pred_label = self.names[cls]
                index = next((key for key, value in self.class_id_to_category.items() if pred_label in value), None)
                filterd_class = self.filtered_classes[index]
                pred_label = filterd_class
                if filterd_class != 'IGNORE' and filterd_class in self.classes_interest:
                    if self.draw_boxes:
                        annotator.box_label(box, label=pred_label, color=colors(int(track_id), True))

                    # Store class info
                    if pred_label not in self.class_wise_count:
                        self.class_wise_count[pred_label] = {"IN": 0, "OUT": 0}

                    # Draw Tracks
                    track_line = self.track_history[track_id]
                    track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                    if len(track_line) > 30:
                        track_line.pop(0)

                    # Draw track trails
                    if self.draw_tracks:
                        annotator.draw_centroid_and_tracks(
                            track_line,
                            color=colors(int(track_id), True),
                            track_thickness=self.tf,
                        )

                    prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None
                    cur_position = self.track_history[track_id][-1] if len(self.track_history[track_id]) > 1 else None
                    # Count objects in any polygon
                    if len(self.reg_pts) >= 3:
                        is_inside = self.counting_region.contains(Point(track_line[-1]))

                        if prev_position is not None and is_inside and track_id not in self.count_ids:
                            self.count_ids.append(track_id)

                            if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
                                self.in_counts += 1
                                self.class_wise_count[pred_label]["IN"] += 1
                            else:
                                self.out_counts += 1
                                self.class_wise_count[pred_label]["OUT"] += 1

                    # Count objects using line
                    elif len(self.reg_pts) == 2:
                        if prev_position is not None and len(self.track_history[track_id]) > 5 and track_id not in self.count_ids:

                            if LineString([(prev_position[0], prev_position[1]), (cur_position[0], cur_position[1])]).intersects(
                                self.counting_line_segment
                            ):
                                self.count_ids.append(track_id)

                                # Determine the direction of movement (IN or OUT)
                                line_vector = [self.reg_pts[1][0] - self.reg_pts[0][0], self.reg_pts[1][1] - self.reg_pts[0][1]]
                                movement_vector = [prev_position[1] - self.reg_pts[0][1], prev_position[0] - self.reg_pts[0][0]]
                                
                                cross_product = line_vector[0] * movement_vector[0] - line_vector[1] * movement_vector[1]


                                if cross_product <= 0:
                                    self.out_counts += 1
                                    self.class_wise_count[pred_label]["OUT"] += 1
                                else:
                                    self.in_counts += 1
                                    self.class_wise_count[pred_label]["IN"] += 1
                                                
        labels_dict = {}

        for key, value in self.class_wise_count.items():
            if value["IN"] != 0 or value["OUT"] != 0:
                if not self.view_in_counts and not self.view_out_counts:
                    continue
                elif not self.view_in_counts:
                    labels_dict[str.capitalize(key)] = f"OUT {value['OUT']}"
                elif not self.view_out_counts:
                    labels_dict[str.capitalize(key)] = f"IN {value['IN']}"
                else:
                    labels_dict[str.capitalize(key)] = f"IN {value['IN']} OUT {value['OUT']}"

        if labels_dict:
            annotator.display_analytics(self.im0, labels_dict, (104, 31, 17), (255, 255, 255), 10)

    def start_counting(self, im0, tracks):
        """
        Main function to start the object counting process.

        Args:
            im0 (ndarray): Current frame from the video stream.
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0  # store image
        self.extract_and_process_tracks(tracks)  # draw region even if no objects

        return self.im0


if __name__ == "__main__":
    classes_names = {0: "person", 1: "car"}  # example class names
    ObjectCounter(classes_names)
