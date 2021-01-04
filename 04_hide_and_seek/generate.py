import functools
import json
import numpy as np
import math
import imageio
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
from svg.path import parse_path, Line, CubicBezier

# Note for users: I have exported the svg files from Illustrator with the
# "SVG Tiny" setting. Using functions not recognized by my very basic parser
# might result in errors or weird-looking drawings

FILES = ["EasyStandard.dat", "NormalStandard.dat", "HardStandard.dat"]
SCALE = 10  # Scaling factor for the x and y axes
TEXT_SCALE = 0.2
SCROLL_DIST = 60
TEXT_HEIGHT = 5
LETTER_SPACING = 2
SPACETIME_RATIO = 5
THICKNESS_SPACE = 0.05  # In final units
THICKNESS_TIME = THICKNESS_SPACE / SPACETIME_RATIO  # In time units
RESOLUTION = 0.02  # In normalized units
# TODO: multiple levels of quality?
NS = {"xmlns": "http://www.w3.org/2000/svg"}

# filename, coordinates: time, x, y, ay, scale, spawn angle
IMAGES = [
    ("reading", [15, 5, 0, -20, 0.8, 0]),
    ("dog_sitting", [16, -4, 0, 20, 1, 0]),
    ("back_to_back", [24, -7, 0, 10, 0.8, 0]),
    ("throwing_stick", [30, 6, 0, -10, 1, 0]),
    ("mother_arguing", [40, 4.5, 0, -15, 1, 0]),
    ("father_arguing", [40.5, -4.5, 0, 15, 1, 0]),
    ("mother_crying", [50, -4, 0, 20, 1, 0]),
    ("kid_sad", [50.5, 4, 0, -5, 1, 0]),
    ("father_smoking", [60, 4, 0, -20, 1, 0]),
    ("dog_sleeping", [60.5, -6, 0, 20, 1, 0]),
    ("mother_pensive", [70, 6, 0, -20, 1, 0]),
    ("sad_with_dog", [80, -5, 0, 20, 1, 0]),
    ("removing_sofa", [90, 9, 0, -20, 1, 0]),
    ("kid_sinking", [100, -5, 2, 20, 1, 0]),
    ("dog_diving", [101, 0.5, 7, -20, 1, 0]),
    ("spinning", [116, 6, 0, -20, 1, 0]),
    ("rubbing_eyes", [126, -6, 0, 20, 1, 0]),
    ("separation_left", [136, -10, 0, 20, 1, 5]),
    ("separation_right", [136.5, 10, 0, -20, 1, -5]),
    ("street_person_left1", [150, -7, 0, 20, 1, 2]),
    ("street_person_right1", [155, 7, 0, -20, 1, -2]),
    ("street_person_left2", [160, -7, 0, 20, 1, 2]),
    ("street_person_right2", [165, 7, 0, -20, 1, -2]),
    ("seek", [181, -6, 0, 20, 1, 0]),
    ("hide", [181.5, 6, 0, -20, 1, 0]),
    ("trains", [196, -7, 0, 20, 1, 0]),
    ("sewing_machine", [205, 7, 0, -20, 1, 0]),
    ("mother_removes_picture", [228, -7, 0, 20, 1, 0]),
    ("kid_looking_at_picture", [243, 7, 0, -20, 1, 0]),
    ("trial", [258, 8, 0, -20, 1, 0]),
    ("cardboard_stack", [268, -7, 0, 20, 1, 0]),
    ("still_life", [278, -8, 0, 20, 1, 0]),
    ("seek", [289, -6, 0, 20, 1, 0]),
    ("hide", [289.5, 6, 0, -20, 1, 0]),
    ("trains", [305, -7, 0, 20, 1, 0]),
    ("sewing_machine", [313, 7, 0, -20, 1, 0]),
    ("blood", [325, 0, 5, 20, 1, 0]),
    ("time", [345, -7, 0, 20, 1, 0]),
    ("kid_tel_angry", [370, -6, 0, 20, 1, 0]),
    ("father_tel", [370.5, 6, 0, -20, 1, 0]),
    ("kid_sitting_angry", [385, -6, 0, 20, 1, 0]),
    ("mother_sitting", [385.5, 6, 0, -20, 1, 0]),
    ("kid_throwing_tel", [401, -6, 0, 20, 1, 0]),
    ("father_tel_2", [401.5, 6, 0, -20, 1, 0]),
    ("kid_kicking_chair", [418, -7, 0, 20, 1, 0]),
    ("mother_sitting_2", [418.5, 6, 0, -20, 1, 0]),
    ("repli1", [452, -7, 0, 20, 0.8, 0]),
    ("repli2", [484, 7, 0, -20, 0.8, 0]),
    ("repli3", [516, -7, 0, 20, 0.8, 0]),
    ("repli4", [549, 7, 0, -20, 0.8, 0]),
]

def normalize(x, y, viewbox):
    """Normalize so that the origin is at the bottom center of the image,
    and the height of the image is 1
    """
    xi, yi, width, height = viewbox
    return (x - xi - width / 2) / height, (yi + height - y) / height


@functools.lru_cache
def load_image(filename):
    root = ET.parse("{}.svg".format(filename)).getroot()
    viewbox = tuple(map(float, root.attrib["viewBox"].split()))
    all_lines = []
    for line in root.findall("xmlns:line", NS):
        x1 = float(line.attrib["x1"])
        y1 = float(line.attrib["y1"])
        x2 = float(line.attrib["x2"])
        y2 = float(line.attrib["y2"])
        x1, y1 = normalize(x1, y1, viewbox)
        x2, y2 = normalize(x2, y2, viewbox)
        all_lines.append((x1, y1, x2, y2))
    for polyline in root.findall("xmlns:polyline", NS):
        points = [normalize(*map(float, pt.split(',')), viewbox)
                  for pt in polyline.attrib["points"].split()]
        x1, y1 = points[0]
        for x2, y2 in points[1:]:
            all_lines.append((x1, y1, x2, y2))
            x1, y1 = x2, y2
    for polygon in root.findall("xmlns:polygon", NS):
        points = [normalize(*map(float, pt.split(',')), viewbox)
                  for pt in polygon.attrib["points"].split()]
        points.append(points[0])
        x1, y1 = points[0]
        for x2, y2 in points[1:]:
            all_lines.append((x1, y1, x2, y2))
            x1, y1 = x2, y2
    for path in root.findall("xmlns:path", NS):
        path_spec = parse_path(path.attrib["d"])
        # TODO: is support for QuadraticBezier and Arc needed?
        for segment in path_spec:
            if type(segment) is Line:
                x1, y1 = normalize(segment.start.real,
                                   segment.start.imag, viewbox)
                x2, y2 = normalize(segment.end.real,
                                   segment.end.imag, viewbox)
                all_lines.append((x1, y1, x2, y2))
            elif type(segment) is CubicBezier:
                length = segment.length()
                n_div = max(2, int(math.ceil(length
                                             / (RESOLUTION
                                                * (viewbox[3] - viewbox[1])))))
                points = []
                for i in range(n_div + 1):
                    t = i / n_div
                    c = segment.point(t)
                    x, y = normalize(c.real, c.imag, viewbox)
                    points.append((x, y))
                x1, y1 = points[0]
                for x2, y2 in points[1:]:
                    all_lines.append((x1, y1, x2, y2))
                    x1, y1 = x2, y2

    return all_lines


def add_image(walls, image, offset):
    time, x0, y0, ay, scale, angle = offset
    scale *= SCALE
    for line in image:
        x1, y1, x2, y2 = line
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1

        pos0 = np.array(
            [x1 * scale - THICKNESS_SPACE / 2,
             y1 * scale,
             0])
        convert_scale = np.array([SPACETIME_RATIO, SPACETIME_RATIO, -1])
        rotation = Rotation.from_euler(
            "xyz", [0, ay, 0], degrees=True).as_matrix()

        pos = convert_scale * np.matmul(rotation, pos0 / convert_scale)

        walls.append({
            "_time": time + pos[2],
            "_lineIndex": 0,
            "_type": 0,
            "_duration": THICKNESS_TIME,
            "_width": 0,
            "_customData": {
                "_position": [x0 + pos[0], y0 + pos[1]],
                "_scale": [THICKNESS_SPACE,
                           scale * math.sqrt((x2 - x1)**2 + (y2 - y1) ** 2)],
                "_rotation": angle,
                "_localRotation": [0, -ay, math.degrees(math.atan2(y2 - y1, x2 - x1)) - 90],
            }
        })


def generate_path(walls):
    time = 3
    time_interval = 0.5
    thickness = 0.1
    width = 1
    conv_dist = 2.5
    while time < 588:
        time += time_interval
        angle = np.random.normal(0, 2)
        adjustment = conv_dist * math.sin(math.radians(angle))
        walls.append({
            "_time": time,
            "_lineIndex": 0,
            "_type": 0,
            "_duration": np.random.uniform(0.6, 0.8) * time_interval,
            "_width": 0,
            "_customData": {
                "_position": [-adjustment - width / 2,
                              -thickness / 2],
                "_scale": [width, thickness],
                "_rotation": angle,
                "_localRotation": [0, -angle, 0],
                "_color": [0.0, 0.0, 0.0, 0.5],
            }
        })


def load_font():
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,'?-&0"
    font_img = imageio.imread("font.png")
    img_dim = font_img.shape[:2]

    # Separate letters
    letter_offsets = []
    letter_widths = []
    start = 0
    for i in range(1, img_dim[1]):
        if font_img[0, i, 2] > 100:
            if i > start:
                letter_offsets.append(start)
                letter_widths.append(i - start)
            start = i + 1
    assert len(letter_offsets) == len(alphabet)

    # Convert the letters to walls (1 pixel = 1 unit)
    font = {}
    for k in range(len(alphabet)):
        letter = alphabet[k]
        letter_img = font_img[
            :, letter_offsets[k]:letter_offsets[k] + letter_widths[k], :3]
        letter_matrix = (
            (letter_img[:, :, 0]
                + letter_img[:, :, 1]
                + letter_img[:, :, 2]) > 100)
        letter_walls = []  # list of x, y, w, h
        while True:
            h_lines = []
            v_lines = []
            # Horizontal lines
            for i in range(img_dim[0]):
                start_j = None
                for j in range(letter_widths[k]):
                    if letter_matrix[i, j] and start_j is None:
                        start_j = j
                    if not letter_matrix[i, j] and start_j is not None:
                        h_lines.append((j - start_j, i, start_j))
                        start_j = None
                if start_j is not None:
                    h_lines.append(
                        (letter_widths[k] - start_j, i, start_j))
            # Vertical lines
            for j in range(letter_widths[k]):
                start_i = None
                for i in range(img_dim[0]):
                    if letter_matrix[i, j] and start_i is None:
                        start_i = i
                    if not letter_matrix[i, j] and start_i is not None:
                        v_lines.append((i - start_i, start_i, j))
                        start_i = None
                if start_i is not None:
                    v_lines.append(
                        (img_dim[0] - start_i, start_i, j))
            # Choose the longest line, add it and clear the pixels
            h_lines.sort(key=lambda x: x[0], reverse=True)
            v_lines.sort(key=lambda x: x[0], reverse=True)
            if not h_lines and not v_lines:
                break
            elif not h_lines or v_lines[0][0] >= h_lines[0][0]:
                h, i, j = v_lines[0]
                letter_walls.append((j, img_dim[0] - i - h, 1, h))
                for r in range(i, i + h):
                    letter_matrix[r, j] = False
            else:
                w, i, j = h_lines[0]
                letter_walls.append((j, img_dim[0] - i - 1, w, 1))
                for c in range(j, j + w):
                    letter_matrix[i, c] = False
        font[letter] = (letter_widths[k], letter_walls)

    font[" "] = (5, [])

    return (img_dim[0], font)


def add_lyrics(walls, font):
    with open("lyrics.json", encoding='raw_unicode_escape') as json_file:
        lyrics_json = json.loads(
            json_file.read().encode('raw_unicode_escape').decode())

    time_offset = 0
    for text in lyrics_json["text"]:
        offset = 0
        scale = TEXT_SCALE * (text["scale"] if "scale" in text else 1)
        text_height = TEXT_HEIGHT + (text["height_offset"]
                                     if "height_offset" in text else 0)
        for letter in text["content"]:
            for letter_wall in font[letter][1]:
                walls.append({
                    "_time": (text["time"] + time_offset
                              + scale * (letter_wall[0] + offset)
                                / SPACETIME_RATIO),
                    "_lineIndex": 0,
                    "_type": 0,
                    "_duration": scale * letter_wall[2] / SPACETIME_RATIO,
                    "_width": 0,
                    "_customData": {
                        "_position": [-SCROLL_DIST,
                                      text_height + scale * letter_wall[1]],
                        "_scale": [scale,
                                   scale * letter_wall[3]],
                        "_rotation": 90,
                        "_localRotation": [0, 0, 0],
                    }
                })
            offset += font[letter][0] + LETTER_SPACING


def main():
    walls = []

    # Random spawns
    for filename, coord in IMAGES:
        all_lines = load_image(filename)
        add_image(walls, all_lines, coord)

    # Path
    generate_path(walls)

    # Lyrics
    _, font = load_font()
    add_lyrics(walls, font)

    walls.sort(key=lambda x: x["_time"])

    for filename in FILES:
        with open(filename, "r") as json_file:
            song_json = json.load(json_file)

        song_json["_obstacles"] = walls

        with open(filename, "w") as json_file:
            json.dump(song_json, json_file)


main()
