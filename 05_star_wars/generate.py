import numpy as np
import math
import json
import imageio
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
from svg.path import parse_path, Line, CubicBezier

BPM = 108
TIME_TO_BEATS = BPM / 60

FILES = ["EasyStandard.dat", "ExpertStandard.dat"]

BASE_HJ = 2
RESOLUTION = 0.02  # In normalized units
NS = {"xmlns": "http://www.w3.org/2000/svg"}

OPENING_CRAWL = [
    "It is a period of civil war.",
    "Rebel spaceships, striking",
    "from a hidden base, have",
    "won their first victory",
    "against the evil Galactic",
    "Empire.",
    "",
    "",
    "During the battle, rebel",
    "spies managed to steal",
    "secret plans to the Empire's",
    "ultimate weapon, the",
    "DEATH STAR, an armored",
    "space station with enough",
    "power to destroy an entire",
    "planet.",
    "",
    "",
    "Pursued by the Empire's",
    "sinister agents, Princess",
    "Leia races home aboard her",
    "starship, custodian of the",
    "stolen plans that can save",
    "her people and restore",
    "freedom to the galaxy...."
]


def load_font():
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,'."
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


def add_intro_text(walls, custom_events, font):
    intro_text = ["A long time ago in a galaxy far,", "far away...."]
    line_heights = [1.5, -0.5]
    x_offset = -9
    color = [0, 0.4, 0.8, 0.6]
    start = 0.5 * TIME_TO_BEATS
    duration = 5.5 * TIME_TO_BEATS
    text_dist = 30
    letter_spacing = 2

    for text, line_height in zip(intro_text, line_heights):
        offset = 0
        scale = 0.1
        nj_offset = duration / 2 - BASE_HJ
        for letter in text:
            for letter_wall in font[letter][1]:
                pos = [x_offset + scale * (offset + letter_wall[0]),
                       line_height + scale * letter_wall[1],
                       text_dist]
                effect = scale * (offset + letter_wall[0]) / 40
                walls.append({
                    "_time": start + BASE_HJ + nj_offset + effect,
                    "_duration": 0,
                    "_lineIndex": 0,
                    "_type": 0,
                    "_width": 0,
                    "_customData": {
                        "_position": [0, 0],
                        "_scale": [scale * letter_wall[2],
                                   scale * letter_wall[3],
                                   scale],
                        "_interactable": False,
                        "_noteJumpStartBeatOffset": nj_offset,
                        "_color": color,
                        "_animation": {
                            "_definitePosition": [
                                [*pos, 0.0],
                            ],
                            "_dissolve": [
                                [0, 0],
                                [1, 0],
                            ],
                        }
                    }
                })
            offset += font[letter][0] + letter_spacing


def normalize(x, y, viewbox):
    """Normalize so that the origin is at the bottom center of the image,
    and the height of the image is 1
    """
    xi, yi, width, height = viewbox
    return (x - xi - width / 2) / height, (yi + height - y) / height


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


def add_logo(walls, custom_events):
    image = load_image("logo")

    scale = 5
    thickness = 0.1
    x0, y0 = 0, 0
    start = 7 * TIME_TO_BEATS
    duration = 18 * TIME_TO_BEATS
    start_z = 1
    end_z = 50
    color = [1, 0.9, 0.1, 0.0]

    for line in image:
        x1, y1, x2, y2 = line
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        x1, x2 = - x1, -x2

        pos = [x0 + x1 * scale - thickness / 2, y0 + y1 * scale]

        base_hj = 6
        nj_offset = 14
        walls.append({
            "_time": start,
            "_duration": 0,
            "_lineIndex": 0,
            "_type": 0,
            "_width": 0,
            "_customData": {
                "_scale": [thickness,
                           scale * math.sqrt((x2 - x1)**2 + (y2 - y1) ** 2),
                           thickness],
                "_interactable": False,
                "_noteJumpMovementSpeed": 4,
                "_noteJumpStartBeatOffset": nj_offset,
                "_color": color,
                "_position": [*pos],
                "_localRotation": [0, 0, math.degrees(math.atan2(y2 - y1, x2 - x1)) - 90],
                "_rotation": 180,
            }
        })


def add_crawl(walls, custom_events, font):
    letter_spacing = 2
    min_word_spacing = 1
    scale = 0.12
    thickness = 0.05
    angle = 10
    start = 16 * TIME_TO_BEATS
    duration = 22 * TIME_TO_BEATS
    color = [1, 0.9, 0.1, 0]
    line_time_sep = 2 * TIME_TO_BEATS
    start_z = 0
    end_z = 50
    y_offset = -5

    angle_rad = math.radians(angle)
    rot_zy = np.array([
        [math.cos(angle_rad), -math.sin(angle_rad)],
        [math.sin(angle_rad), math.cos(angle_rad)],
    ])

    mw = 0
    widths = []
    for line in OPENING_CRAWL:
        widths.append(sum(font[letter][0] if letter != " "
                          else min_word_spacing
                          for letter in line)
                      + letter_spacing * (len(line) - 1))
        mw = max(mw, widths[-1])

    x_offset = - scale * mw / 2
    nj_offset = duration / 2 - BASE_HJ

    for i in range(len(OPENING_CRAWL)):
        line = OPENING_CRAWL[i]
        word_spacing = (min_word_spacing
                        + (mw - widths[i])
                        / max(1, sum(1 if letter == " " else 0
                                     for letter in line)))
        offset = 0
        for letter in line:
            for letter_wall in font[letter][1]:
                time_offset = line_time_sep * (offset + letter_wall[0]) / mw

                x = (x_offset + scale * (offset + letter_wall[0]))
                z_base = scale * letter_wall[1]

                z_adj = (time_offset / duration) * (end_z - start_z)
                zy_start_flat = np.array([start_z + z_base + z_adj, y_offset])
                zy_end_flat = np.array([end_z + z_base + z_adj, y_offset])

                zy_start = np.matmul(rot_zy, zy_start_flat)
                zy_end = np.matmul(rot_zy, zy_end_flat)

                walls.append({
                    "_time": (start + i * line_time_sep + nj_offset + 1
                              + time_offset),
                    "_duration": 0,
                    "_lineIndex": 0,
                    "_type": 0,
                    "_width": 0,
                    "_customData": {
                        "_position": [0, 0],
                        "_scale": [scale * letter_wall[2],
                                   thickness,
                                   scale * letter_wall[3]],
                        "_interactable": False,
                        "_noteJumpStartBeatOffset": nj_offset,
                        "_localRotation": [-angle, 0, 0],
                        "_color": color,
                        "_animation": {
                            "_definitePosition": [
                                [x, zy_start[1], zy_start[0], 0.0],
                                [x, zy_end[1], zy_end[0], 1.0],
                            ],
                            "_dissolve": [
                                [0, 0],
                                [1, 0],
                                [1, 0.9],
                                [0, 1],
                            ],
                        }
                    }
                })
            offset += (font[letter][0] if letter != " " else word_spacing
                       ) + letter_spacing


def add_stars(walls, custom_events):
    n_stars = 80
    angle_ranges = [(-80, 80), (-70, 80)]
    dist = 50
    start = 7 * TIME_TO_BEATS
    duration = 82 * TIME_TO_BEATS

    nj_offset = duration / 2 - BASE_HJ

    for i in range(n_stars):
        a0 = np.random.uniform(angle_ranges[0][0], angle_ranges[0][1])
        a1 = np.random.uniform(angle_ranges[1][0], angle_ranges[1][1])
        x = dist * math.cos(math.radians(90 + a0))
        z = dist * math.sin(math.radians(90 + a0))
        y = dist * math.sin(math.radians(a1))
        ecart = math.sqrt(a0**2 + a1**2)
        effect = ecart / 60

        walls.append({
            "_time": start + nj_offset + 1 + effect,
            "_duration": 0,
            "_lineIndex": 0,
            "_type": 0,
            "_width": 0,
            "_customData": {
                "_position": [0, 0],
                "_scale": [0.05, 0.05, 0.05],
                "_interactable": False,
                "_noteJumpStartBeatOffset": nj_offset,
                # "_localRotation": [],
                "_color": [1, 1, 1, 0],
                "_animation": {
                    "_definitePosition": [
                        [x, y, z, 0.0],
                    ],
                    "_dissolve": [
                        [0, 0],
                        [1, 0],
                    ],
                }
            }
        })


def main():
    walls = []
    custom_events = []

    # Lyrics
    _, font = load_font()
    add_intro_text(walls, custom_events, font)
    add_logo(walls, custom_events)
    add_crawl(walls, custom_events, font)
    add_stars(walls, custom_events)

    walls.sort(key=lambda x: x["_time"])

    for filename in FILES:
        with open(filename, "r") as json_file:
            song_json = json.load(json_file)

        song_json["_obstacles"] = walls
        song_json["_customData"]["_customEvents"] = custom_events

        with open(filename, "w") as json_file:
            json.dump(song_json, json_file)


main()
