import cv2
import numpy as np
from PIL import Image, ImageDraw

from postprocess.utils import ops

class Colors:
    """

    This class provides methods to work with color palette, including converting hex color codes to
    RGB values.

    Attributes:
        palette (list of tuple): List of RGB color values.
        n (int): The number of colors in the palette.
        pose_palette (np.ndarray): A specific color palette array with dtype np.uint8.
    """

    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = (
            "042AFF",
            "0BDBEB",
            "F3F3F3",
            "00DFB7",
            "111F68",
            "FF6FDD",
            "FF444F",
            "CCED00",
            "00F344",
            "BD00FF",
            "00B4FF",
            "DD00BA",
            "00FFFF",
            "26C000",
            "01FFB3",
            "7D24FF",
            "7B0068",
            "FF1B6C",
            "FC6D2F",
            "A2FF0B",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


def is_ascii(s) -> bool:
    """
    Check if a string is composed of only ASCII characters.

    Args:
        s (str): String to be checked.

    Returns:
        (bool): True if the string is composed only of ASCII characters, False otherwise.
    """
    # Convert list, tuple, None, etc. to string
    s = str(s)

    # Check if the string is composed of only ASCII characters
    return all(ord(c) < 128 for c in s)


class Annotator:
    """
    Annotator for train/val mosaics and JPGs and predictions annotations.

    Attributes:
        im (Image.Image or numpy array): The image to annotate.
        pil (bool): Whether to use PIL or cv2 for drawing annotations.
        font (ImageFont.truetype or ImageFont.load_default): Font used for text annotations.
        lw (float): Line width for drawing.
        skeleton (List[List[int]]): Skeleton structure for keypoints.
        limb_color (List[int]): Color palette for limbs.
        kpt_color (List[int]): Color palette for keypoints.
    """

    def __init__(self, im, line_width=None, font_size=None, font="Arial.ttf", pil=False, example="abc"):
        """Initialize the Annotator class with image and line width along with color palette for keypoints and limbs."""
        non_ascii = not is_ascii(example)  # non-latin labels, i.e. asian, arabic, cyrillic
        input_is_pil = isinstance(im, Image.Image)
        self.pil = pil or non_ascii or input_is_pil
        self.lw = line_width or max(round(sum(im.size if input_is_pil else im.shape) / 2 * 0.003), 2)

        assert im.data.contiguous, "Image not contiguous. Apply np.ascontiguousarray(im) to Annotator input images."
        self.im = im if im.flags.writeable else im.copy()
        self.tf = max(self.lw - 1, 1)  # font thickness
        self.sf = self.lw / 3  # font scale
        # Pose
        self.skeleton = [
            [16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [2, 3],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7],
        ]

        self.limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        self.kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
        self.dark_colors = {
            (235, 219, 11),
            (243, 243, 243),
            (183, 223, 0),
            (221, 111, 255),
            (0, 237, 204),
            (68, 243, 0),
            (255, 255, 0),
            (179, 255, 1),
            (11, 255, 162),
        }
        self.light_colors = {
            (255, 42, 4),
            (79, 68, 255),
            (255, 0, 189),
            (255, 180, 0),
            (186, 0, 221),
            (0, 192, 38),
            (255, 36, 125),
            (104, 0, 123),
            (108, 27, 255),
            (47, 109, 252),
            (104, 31, 17),
        }

    def get_txt_color(self, color=(128, 128, 128), txt_color=(255, 255, 255)):
        """Assign text color based on background color."""
        if color in self.dark_colors:
            return 104, 31, 17
        elif color in self.light_colors:
            return 255, 255, 255
        else:
            return txt_color

    def circle_label(self, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255), margin=2):
        """
        Draws a label with a background circle centered within a given bounding box.

        Args:
            box (tuple): The bounding box coordinates (x1, y1, x2, y2).
            label (str): The text label to be displayed.
            color (tuple, optional): The background color of the rectangle (B, G, R).
            txt_color (tuple, optional): The color of the text (R, G, B).
            margin (int, optional): The margin between the text and the rectangle border.
        """
        # If label have more than 3 characters, skip other characters, due to circle size
        if len(label) > 3:
            print(
                f"Length of label is {len(label)}, initial 3 label characters will be considered for circle annotation!"
            )
            label = label[:3]

        # Calculate the center of the box
        x_center, y_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        # Get the text size
        text_size = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, self.sf - 0.15, self.tf)[0]
        # Calculate the required radius to fit the text with the margin
        required_radius = int(((text_size[0] ** 2 + text_size[1] ** 2) ** 0.5) / 2) + margin
        # Draw the circle with the required radius
        cv2.circle(self.im, (x_center, y_center), required_radius, color, -1)
        # Calculate the position for the text
        text_x = x_center - text_size[0] // 2
        text_y = y_center + text_size[1] // 2
        # Draw the text
        cv2.putText(
            self.im,
            str(label),
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.sf - 0.15,
            self.get_txt_color(color, txt_color),
            self.tf,
            lineType=cv2.LINE_AA,
        )

    def text_label(self, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255), margin=5):
        """
        Draws a label with a background rectangle centered within a given bounding box.

        Args:
            box (tuple): The bounding box coordinates (x1, y1, x2, y2).
            label (str): The text label to be displayed.
            color (tuple, optional): The background color of the rectangle (B, G, R).
            txt_color (tuple, optional): The color of the text (R, G, B).
            margin (int, optional): The margin between the text and the rectangle border.
        """
        # Calculate the center of the bounding box
        x_center, y_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        # Get the size of the text
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.sf - 0.1, self.tf)[0]
        # Calculate the top-left corner of the text (to center it)
        text_x = x_center - text_size[0] // 2
        text_y = y_center + text_size[1] // 2
        # Calculate the coordinates of the background rectangle
        rect_x1 = text_x - margin
        rect_y1 = text_y - text_size[1] - margin
        rect_x2 = text_x + text_size[0] + margin
        rect_y2 = text_y + margin
        # Draw the background rectangle
        cv2.rectangle(self.im, (rect_x1, rect_y1), (rect_x2, rect_y2), color, -1)
        # Draw the text on top of the rectangle
        cv2.putText(
            self.im,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.sf - 0.1,
            self.get_txt_color(color, txt_color),
            self.tf,
            lineType=cv2.LINE_AA,
        )

    def box_label(self, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255), rotated=False):
        """
        Draws a bounding box to image with label.

        Args:
            box (tuple): The bounding box coordinates (x1, y1, x2, y2).
            label (str): The text label to be displayed.
            color (tuple, optional): The background color of the rectangle (B, G, R).
            txt_color (tuple, optional): The color of the text (R, G, B).
            rotated (bool, optional): Variable used to check if task is OBB
        """
        txt_color = self.get_txt_color(color, txt_color)
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
        if label:
            w, h = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]  # text width, height
            h += 3  # add pixels to pad text
            outside = p1[1] >= h  # label fits outside box
            if p1[0] > self.im.shape[1] - w:  # shape is (h, w), check if label extend beyond right side of image
                p1 = self.im.shape[1] - w, p1[1]
            p2 = p1[0] + w, p1[1] - h if outside else p1[1] + h
            cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(
                self.im,
                label,
                (p1[0], p1[1] - 2 if outside else p1[1] + h - 1),
                0,
                self.sf,
                txt_color,
                thickness=self.tf,
                lineType=cv2.LINE_AA,
            )

    def rectangle(self, xy, fill=None, outline=None, width=1):
        """Add rectangle to image (PIL-only)."""
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255), anchor="top", box_style=False):
        """Adds text to an image using PIL or cv2."""
        if anchor == "bottom":  # start y from font bottom
            w, h = self.font.getsize(text)  # text width, height
            xy[1] += 1 - h
        if self.pil:
            if box_style:
                w, h = self.font.getsize(text)
                self.draw.rectangle((xy[0], xy[1], xy[0] + w + 1, xy[1] + h + 1), fill=txt_color)
                # Using `txt_color` for background and draw fg with white color
                txt_color = (255, 255, 255)
            if "\n" in text:
                lines = text.split("\n")
                _, h = self.font.getsize(text)
                for line in lines:
                    self.draw.text(xy, line, fill=txt_color, font=self.font)
                    xy[1] += h
            else:
                self.draw.text(xy, text, fill=txt_color, font=self.font)
        else:
            if box_style:
                w, h = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[0]  # text width, height
                h += 3  # add pixels to pad text
                outside = xy[1] >= h  # label fits outside box
                p2 = xy[0] + w, xy[1] - h if outside else xy[1] + h
                cv2.rectangle(self.im, xy, p2, txt_color, -1, cv2.LINE_AA)  # filled
                # Using `txt_color` for background and draw fg with white color
                txt_color = (255, 255, 255)
            cv2.putText(self.im, text, xy, 0, self.sf, txt_color, thickness=self.tf, lineType=cv2.LINE_AA)

    def fromarray(self, im):
        """Update self.im from a numpy array."""
        self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
        self.draw = ImageDraw.Draw(self.im)

    def result(self):
        """Return annotated image as array."""
        return np.asarray(self.im)

    def show(self, title=None):
        """Show the annotated image."""
        Image.fromarray(np.asarray(self.im)[..., ::-1]).show(title)

    def save(self, filename="image.jpg"):
        """Save the annotated image to 'filename'."""
        cv2.imwrite(filename, np.asarray(self.im))

    def get_bbox_dimension(self, bbox=None):
        """
        Calculate the area of a bounding box.

        Args:
            bbox (tuple): Bounding box coordinates in the format (x_min, y_min, x_max, y_max).

        Returns:
            angle (degree): Degree value of angle between three points
        """
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        return width, height, width * height

    def draw_region(self, reg_pts=None, color=(0, 255, 0), thickness=5):
        """
        Draw region line.

        Args:
            reg_pts (list): Region Points (for line 2 points, for region 4 points)
            color (tuple): Region Color value
            thickness (int): Region area thickness value
        """
        cv2.polylines(self.im, [np.array(reg_pts, dtype=np.int32)], isClosed=True, color=color, thickness=thickness)

        # Draw small circles at the corner points
        for point in reg_pts:
            cv2.circle(self.im, (point[0], point[1]), thickness * 2, color, -1)  # -1 fills the circle

    def draw_centroid_and_tracks(self, track, color=(255, 0, 255), track_thickness=2):
        """
        Draw centroid point and track trails.

        Args:
            track (list): object tracking points for trails display
            color (tuple): tracks line color
            track_thickness (int): track line thickness value
        """
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(self.im, [points], isClosed=False, color=color, thickness=track_thickness)
        cv2.circle(self.im, (int(track[-1][0]), int(track[-1][1])), track_thickness * 2, color, -1)

    def display_objects_labels(self, im0, text, txt_color, bg_color, x_center, y_center, margin):
        """
        Display the bounding boxes labels in parking management app.

        Args:
            im0 (ndarray): inference image
            text (str): object/class name
            txt_color (bgr color): display color for text foreground
            bg_color (bgr color): display color for text background
            x_center (float): x position center point for bounding box
            y_center (float): y position center point for bounding box
            margin (int): gap between text and rectangle for better display
        """
        text_size = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[0]
        text_x = x_center - text_size[0] // 2
        text_y = y_center + text_size[1] // 2

        rect_x1 = text_x - margin
        rect_y1 = text_y - text_size[1] - margin
        rect_x2 = text_x + text_size[0] + margin
        rect_y2 = text_y + margin
        cv2.rectangle(im0, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
        cv2.putText(im0, text, (text_x, text_y), 0, self.sf, txt_color, self.tf, lineType=cv2.LINE_AA)

    def display_analytics(self, im0, text, txt_color, bg_color, margin):
        """
        Display the overall statistics for parking lots.

        Args:
            im0 (ndarray): inference image
            text (dict): labels dictionary
            txt_color (bgr color): display color for text foreground
            bg_color (bgr color): display color for text background
            margin (int): gap between text and rectangle for better display
        """
        horizontal_gap = int(im0.shape[1] * 0.02)
        vertical_gap = int(im0.shape[0] * 0.01)
        text_y_offset = 0
        for label, value in text.items():
            txt = f"{label}: {value}"
            text_size = cv2.getTextSize(txt, 0, self.sf, self.tf)[0]
            if text_size[0] < 5 or text_size[1] < 5:
                text_size = (5, 5)
            text_x = im0.shape[1] - text_size[0] - margin * 2 - horizontal_gap
            text_y = text_y_offset + text_size[1] + margin * 2 + vertical_gap
            rect_x1 = text_x - margin * 2
            rect_y1 = text_y - text_size[1] - margin * 2
            rect_x2 = text_x + text_size[0] + margin * 2
            rect_y2 = text_y + margin * 2
            cv2.rectangle(im0, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
            cv2.putText(im0, txt, (text_x, text_y), 0, self.sf, txt_color, self.tf, lineType=cv2.LINE_AA)
            text_y_offset = rect_y2

def save_one_box(xyxy, im, file=None, gain=1.02, pad=10, square=False, BGR=False, save=True):
    """
    Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop.

    This function takes a bounding box and an image, and then saves a cropped portion of the image according
    to the bounding box. Optionally, the crop can be squared, and the function allows for gain and padding
    adjustments to the bounding box.

    Args:
        xyxy: A tensor or list representing the bounding box in xyxy format.
        im (numpy.ndarray): The input image.
        file (Path, optional): The path where the cropped image will be saved. Defaults to 'im.jpg'.
        gain (float, optional): A multiplicative factor to increase the size of the bounding box. Defaults to 1.02.
        pad (int, optional): The number of pixels to add to the width and height of the bounding box. Defaults to 10.
        square (bool, optional): If True, the bounding box will be transformed into a square. Defaults to False.
        BGR (bool, optional): If True, the image will be saved in BGR format, otherwise in RGB. Defaults to False.
        save (bool, optional): If True, the cropped image will be saved to disk. Defaults to True.

    Returns:
        (numpy.ndarray): The cropped image.


    """
    if not isinstance(xyxy, np.ndarray):  # may be list
        xyxy = np.array(xyxy)
    b = ops.xyxy2xywh(xyxy.view(-1, 4))  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = ops.xywh2xyxy(b).long()
    xyxy = ops.clip_boxes(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]) : int(xyxy[0, 3]), int(xyxy[0, 0]) : int(xyxy[0, 2]), :: (1 if BGR else -1)]
    return crop
