from collections import defaultdict, OrderedDict
import copy
from enum import Enum
import functools
import itertools
import json
import math
import numpy as np
from scipy.spatial.transform import Rotation
import shutil
from svg.path import parse_path, Line, CubicBezier
import xml.etree.ElementTree as ET
from tqdm import tqdm
import random
# import imageio

INPUT_FILES = ["template", "WIP"]
OUTPUT_FILES = ["ExpertStandard"]

#
# Set seed for determinism
#
np.random.seed(42)

#
# Some constants
#

BPM = 130
BASE_HJ = 2
NS = {"xmlns": "http://www.collada.org/2005/11/COLLADASchema"}
SNS = {"xmlns": "http://www.w3.org/2000/svg"}
NODE_EXCLUSION = ["empty", "metarig", "armature", "camera"]
DEFAULT_COL_INFO = [0.5, 0.5, 0.5, 1, 0, 0.5]

#
# Util functions
#


def name_to_type(name):
    return name_dic[name.split(".")[0].lower()]


def trunc(obj, precision=5):
    if isinstance(obj, np.int32):
        return int(obj)
    elif isinstance(obj, float):
        return round(obj, precision)
    elif isinstance(obj, dict):
        return dict((k, trunc(v)) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, np.ndarray)):
        return list(map(trunc, obj))
    return obj


def lerp(a, b, t):
    if type(a) == list:
        a = np.array(a)
    if type(b) == list:
        b = np.array(b)

    if t <= 0:
        c = a
    elif t >= 1:
        c = b
    else:
        c = (1 - t) * a + t * b

    if type(c) == np.ndarray:
        c = list(c)
    return c


def clamp(x, a, b):
    if a > b:
        a, b = b, a
    if x < a:
        return a
    elif x > b:
        return b
    else:
        return x


def vec_equal(a, b, eps=1e-5):
    return np.linalg.norm(np.array(a) - np.array(b)) < eps


def random_on_sphere(cx, cy, cz, r):
    x0, y0, z0 = np.random.normal(0, 1, 3)
    n0 = math.sqrt(x0**2 + y0**2 + z0**2)
    x = cx + r * x0 / n0
    y = cy + r * y0 / n0
    z = cz + r * z0 / n0
    return x, y, z


n_tracks = 0


def get_unique_track():
    global n_tracks
    n_tracks += 1
    return f"t{n_tracks}"

# name, bi, bf, px, py, pz, rx, ry, rz, ax, ay, az, sx, sy, sz
MODELS = [
    ["main", 0, 500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
]

a_cycle = [f"a/a{i}" for i in range(7)]
# track, scale, list[time, name]
SVGs = [
    ["a", 4, [*[(lerp(0, 40, i/175), a_cycle[i % 7]) for i in range(175)],
              *[(40 + 0.3 * i, f"a/a{i + 8}") for i in range(17)],
              (45.1, None)]],
]

#
# Bongo cats
#

def frange(a, b, step):
    return np.arange(a, b+step/2, step)

guitar = [
    *frange(387, 389, 1/3), 389+2/3,
    *frange(395, 397, 1/3), 397+2/3, 398+1/3,
    *frange(403, 405, 1/3), 405+2/3,
    *frange(411, 413, 1/3), 413+2/3, 414+1/3,
    *frange(423+2/3, 430+1/3, 2/3)
]
guitar_down = [
    (392, 395-1/3), (399, 403-1/3), (408, 411-1/3), (411, 419-1/3),
]
sdrums = [
    *frange(387, 389, 1/3), 389+2/3,
    *frange(395, 397, 1/3), 397+2/3,
    399, 399.25, 399.75, 400, 400.5, 400.75, *frange(400+1/3, 403-1/6, 1/6),
    *frange(403, 405, 1/3), 405+2/3,
    *frange(411, 413, 1/3), 413+2/3,
    419, 419+2/3, 420+1/3, 420+2/3,
    423, 423+1/3, 423+2/3,
    424+1/3, 424+2/3,
    426+1/3, 426+2/3, 427,
    428+1/3, 428+2/3,
]
ddrums = [
    398+1/3, 414+1/3, 421, 425, 429,
]
drums = sorted([(tt, 1) for tt in sdrums] + [(tt, 2) for tt in ddrums])
BONGO_START = 387
BONGO_END = 430+1/3

guitar_frames = []
drum_frames = []
for tt in guitar:
    guitar_down.append((tt, tt+1/6))
guitar_down.sort()
for tt0, tt1 in guitar_down:
    guitar_frames.append((tt0, "bongo/guitar_d"))
    guitar_frames.append((tt1, "bongo/guitar_u"))
guitar_frames.append((BONGO_END, None))
last_hand = 0
for tt, n_hands in drums:
    if n_hands == 2:
        new_hand = 2
    elif last_hand:
        new_hand = 0
    else:
        new_hand = 1
    if new_hand == 0: 
        drum_frames.append((tt, "bongo/drum_du"))
    elif new_hand == 1:
        drum_frames.append((tt, "bongo/drum_ud"))
    else:
        drum_frames.append((tt, "bongo/drum_dd"))
    drum_frames.append((tt+1/6, "bongo/drum_uu"))
    last_hand = new_hand
drum_frames.append((BONGO_END, None))

SVGs.append(["bgg", 10, guitar_frames])
SVGs.append(["bgd", 10, drum_frames])

#
# Add plant lights to the models list
#

PLANT_TIMES = [7, 7+2/3, 8+1/3, 9, 23, 23+2/3, 24+1/3, 25]
PLANT_MODELS = ["plant0", "plant1"]
PLANT_EXCLUSION_ZONES = [(-65, -27), (-60, -33), (-87, -17), (-73, -3),
                         (-58, 10), (-65, -65), (-80, -56), (-80, 27), (-45, 105)]
PLANT_EXCLUSION_RADIUS = 10
PLANT_COUNT = 25
# PLANT_SPAWN_RANGE = (0, 1)

def ok_plant_pos(x, y):
    for xe, ye in PLANT_EXCLUSION_ZONES:
        if (xe - x)**2 + (ye - y)**2 < PLANT_EXCLUSION_RADIUS**2:
            return False
    return True

for ip in range(PLANT_COUNT):
    x, y = PLANT_EXCLUSION_ZONES[0]
    # theta = 90 + lerp(30, 150, ip / PLANT_COUNT)
    t = 2*(ip / PLANT_COUNT)-1
    theta = 180 + 90 * t**3
    while not ok_plant_pos(x, y):
        # uneven arc distribution
        r = np.random.uniform(65, 100)
        # theta = 90 + np.random.normal(90, 42)
        x = r * math.cos(math.radians(theta))
        y = r * math.sin(math.radians(theta))
    if abs(y) < 3:
        continue
    rz = 90 - math.degrees(math.atan2(-x, y))
    s = np.random.uniform(.8, 1.4)
    # plant_time = np.random.uniform(*PLANT_SPAWN_RANGE)
    plant_model = random.choice(PLANT_MODELS)
    MODELS.append([
        plant_model, 0, 500,
        x, y, 0, 0, 0, -rz, 0, 0, 0, s, s, s
    ])


#
# Enum to represent object types
#

class CubeType(Enum):
    WALL = 0    # wall
    NOTE = 1    # fake note
    ENVT = 2    # BTS pillar
    LIGHT = 3   # BTS glowline
    SVG = 4     # position and rotation of track for SVG or similar (uses parenting)
    PLAYER = 5  # player track
    TRACK = 6   # position of track for env or similar (no parenting)


name_dic = {
    "wall": CubeType.WALL,
    "note": CubeType.NOTE,
    "cube": CubeType.ENVT,
    "block": CubeType.ENVT,
    "light": CubeType.LIGHT,
    "svg": CubeType.SVG,
    "player": CubeType.PLAYER,
    "track": CubeType.TRACK,
}


# For the Wait font system. Yea I'm being lazy.
def load_model_old(filename):
    root = ET.parse(f"{filename}.dae").getroot()
    nodes = (root.find("xmlns:library_visual_scenes", NS)
             .find("xmlns:visual_scene", NS)
             .findall("xmlns:node", NS))
    model = []
    for node in nodes:
        name = node.get("name")
        transform = np.array(
            list(map(float, node.find("xmlns:matrix", NS).text.split()))
        ).reshape((4, 4))
        model.append((transform, name))
    return model


@functools.lru_cache(maxsize=None)
def load_model(model_name):
    root = ET.parse("{}.dae".format(model_name)).getroot()

    # Read all the animations and store them temporarily in a map
    node_anims = defaultdict(lambda: [None] * 7)
    library_animations = root.find("xmlns:library_animations", NS)
    if library_animations:
        animations = library_animations.findall("xmlns:animation", NS)
        for animation in animations:
            name = animation.get("name")
            if any(name.lower().startswith(exclu) for exclu in NODE_EXCLUSION):
                continue
            # order: transform, R, G, B, A, metallic, roughness
            node_anim = [None] * 7
            i = 0
            for sub_anim in animation.findall("xmlns:animation", NS):
                anim_array = list(map(
                    float,
                    sub_anim
                    .findall("xmlns:source", NS)[1]
                    .find("xmlns:float_array", NS)
                    .text.split()))
                if i == 0:
                    transforms = [
                        np.reshape(anim_array[16*i:16*(i+1)], (4, 4))
                        for i in range(len(anim_array) // 16)]
                    node_anim[0] = transforms
                else:
                    node_anim[i] = anim_array
                i += 1
            node_anims[name] = node_anim

    # Add all the nodes (aka walls) to the model
    model = []
    nodes = (root.find("xmlns:library_visual_scenes", NS)
             .find("xmlns:visual_scene", NS)
             .findall("xmlns:node", NS))
    for node in nodes:
        name = node.get("name")
        if any(name.lower().startswith(exclu) for exclu in NODE_EXCLUSION):
            continue
        transform = np.array(
            list(map(float, node.find("xmlns:matrix", NS).text.split()))
        ).reshape((4, 4))
        anim = node_anims[name]
        transforms, red, green, blue, alpha, metal, rough = anim
        if transforms is None:
            transforms = [transform]
        n_keys = len(transforms)
        if red is None:
            red = [DEFAULT_COL_INFO[0]] * n_keys
        if green is None:
            green = [DEFAULT_COL_INFO[1]] * n_keys
        if blue is None:
            blue = [DEFAULT_COL_INFO[2]] * n_keys
        if alpha is None:
            alpha = [DEFAULT_COL_INFO[3]] * n_keys
        if metal is None:
            metal = [DEFAULT_COL_INFO[4]] * n_keys
        if rough is None:
            rough = [DEFAULT_COL_INFO[5]] * n_keys
        col_infos = [
            (red[i], green[i], blue[i], alpha[i], metal[i], rough[i])
            for i in range(n_keys)]
        model.append((transforms, col_infos, name))
    return model


def add_model(walls, notes, custom_data, model, info):
    model_name, ti, tf, px, py, pz, rx, ry, rz, ax, ay, az, sx, sy, sz = info

    # Additional rotation applied to the object
    add_rotation = np.zeros((4, 4))
    add_rotation[:3, :3] = Rotation.from_euler(
        "xyz", [rx, ry, rz], degrees=True).as_matrix()
    add_rotation[3, 3] = 1.0

    pbar = tqdm(total=len(model))
    pbar.set_description(model_name)
    for transforms, col_infos, name in model:
        cube_type = name_to_type(name)

        pbar.update(1)

        dt = 1 / max(1, len(transforms) - 1)

        # Only spawn the object when dissolve is non-zero.
        # Note that this code will fail if dissolve is always zero!
        i = -1
        while col_infos[i + 1][3] == 0:
            i += 1
        pib = max(0, i * dt)
        i = len(transforms)
        while col_infos[i - 1][3] == 0:
            i -= 1
        pfb = min((len(transforms) - 1) * dt, i * dt)
        tib = ti + pib * (tf - ti)
        tfb = ti + pfb * (tf - ti)

        definite_positions = []
        local_rotations = []
        scales = []
        colors = []
        dissolves = [[0, 0]]
        for i in range(len(transforms)):
            add_position = np.array(
                [[1, 0, 0, px],
                 [0, 1, 0, py],
                 [0, 0, 1, pz],
                 [0, 0, 0, 1]]
            )
            rescale = np.array(
                [[sx, 0,  0,  0],
                 [0,  sy, 0,  0],
                 [0,  0,  sz, 0],
                 [0,  0,  0,  1]]
            )
            t_mat = np.matmul(add_position, np.matmul(add_rotation, rescale))
            transform = np.matmul(t_mat, transforms[i])

            position = transform[: 3, 3]
            scale = np.array([np.linalg.norm(transform[:3, i])
                              for i in range(3)])
            rotation = transform[: 3, : 3] / scale
            euler = Rotation.from_matrix(
                rotation).as_euler('xyz', degrees=True)

            pivot_blender = np.array([0, 0, 0])
            if cube_type == CubeType.ENVT:
                regex = "\\]PillarPair \\(1\\)\\.\\[0\\]PillarL\\.\\[0\\]Pillar$"
                pivot_rotation = np.array([0, 0, 1])
                rescale = np.array([0.34, 0.34, 0.34/32])
                pos_offset = np.array([-1, 0, 0.17])
            elif cube_type == CubeType.LIGHT:
                regex = "GlowLineL$"
                pivot_rotation = np.array([0, 0, -1])
                rescale = np.array([23.8, 23.8, 0.00048])
                pos_offset = np.array([-1, 0, 0.17])
            elif cube_type == CubeType.WALL:
                pivot_rotation = np.array([1, 0, -1])
                rescale = np.array([2, 2, 2])
                pos_offset = np.array([0, -0.5, 0])
            elif cube_type == CubeType.NOTE:
                pivot_rotation = np.array([0, 0, 0])
                rescale = np.array([2, 2, 2])
                pos_offset = np.array([0, -0.5, -1.2])
            elif cube_type == CubeType.SVG:
                track = name.split(".")[1]
                pivot_rotation = np.array([0, 0, 0])
                rescale = np.array([1, 1, 1])
                pos_offset = np.array([0, 0, 0])
            elif cube_type == CubeType.TRACK:
                track = name.split(".")[1]
                pivot_rotation = np.array([0, 0, 0])
                rescale = np.array([1, 1, 1])
                pos_offset = np.array([0, 0, 0])
            elif cube_type == CubeType.PLAYER:
                pivot_rotation = np.array([0, 0, 0])
                rescale = np.array([1, 1, 1])
                pos_offset = np.array([0, 0, 0])
            rotation_to_blender = scale * (pivot_blender - pivot_rotation)
            correction = - np.matmul(rotation, rotation_to_blender) + pos_offset
            scale *= rescale

            new_position = position + correction

            definite_position = [new_position[1],
                                 new_position[2], -new_position[0]]
            local_rotation = [-euler[1], -euler[2], euler[0]]
            double_scale = [scale[1], scale[2], scale[0]]

            col_info = col_infos[i]
            if cube_type == CubeType.WALL:
                col_coef = 1 + 100 * col_info[4]
                alpha = 10 * (col_info[5] - 0.5)
                color = [
                    col_coef * col_info[0],
                    col_coef * col_info[1],
                    col_coef * col_info[2],
                    alpha
                ]
            elif cube_type == CubeType.NOTE:
                color = [
                    *col_info[:3],
                    1
                ]
            else:
                color = [0] * 4
            dissolve = [col_info[3]]

            for value, values, vlen in [(definite_position, definite_positions, 3),
                                        (local_rotation, local_rotations, 3),
                                        (double_scale, scales, 3),
                                        (color, colors, 4),
                                        (dissolve, dissolves, 1)]:
                tt = max(0, i * dt)
                ttb = (tt - pib) / (pfb - pib)
                if cube_type == CubeType.NOTE:
                    ttb /= 2
                ttb = max(0, min(ttb, 1))
                if i < 2 or not vec_equal(values[-1][:vlen], value) \
                   or not vec_equal(values[-1][:vlen], values[-2][:vlen]):
                    values.append([*value, ttb])
                else:
                    values[-1] = [*value, ttb]

        if cube_type == CubeType.NOTE:
            nj_offset = tfb - tib - BASE_HJ
            # dissolves.append([0, 0.5])
        else:
            nj_offset = (tfb - tib) / 2 - BASE_HJ

        if cube_type == CubeType.ENVT or cube_type == CubeType.LIGHT:
            tti, ttf = 1, 0
            arrays = (definite_positions, local_rotations, scales)
            for values in arrays:
                if len(values) > 1 and vec_equal(values[0][:-1], values[1][:-1]):
                    values.pop(0)
                if len(values) > 1 and vec_equal(values[-2][:-1], values[-1][:-1]):
                    values.pop()
                tti = min(tti, values[0][-1])
                ttf = max(ttf, values[-1][-1])
            custom_data["_environment"].append({
                "_id": regex,
                "_lookupMethod": "Regex",
                "_duplicate": 1,
                "_position": definite_positions[0][:3],
                "_scale": scales[0][:3],
                "_rotation": local_rotations[0][:3],
            })
            if ttf == tti: # The object is not animated
                pass
            else:
                tic = lerp(tib, tfb, tti)
                tfc = lerp(tib, tfb, ttf)
                for values in arrays:
                    for i in range(len(values)):
                        tt = values[i][-1]
                        values[i][-1] = (tt-tti) / (ttf-tti)
                track = get_unique_track()
                custom_data["_environment"][-1]["_track"] = track
                nj_offset = (tfc - tic) / 2 - BASE_HJ
                custom_data["_customEvents"].append({
                    "_time": tic,
                    "_type": "AnimateTrack",
                    "_data": {
                        "_track": track,
                        "_duration": tfc - tic,
                        "_position": definite_positions,
                        "_rotation": local_rotations,
                        "_scale": scales,
                    }
                })

        elif cube_type == CubeType.WALL:
            walls.append({
                "_time": tib + BASE_HJ + nj_offset,
                "_duration": 0,
                "_lineIndex": 0,
                "_type": 0,
                "_width": 0,
                "_customData": {
                    "_interactable": False,
                    "_noteJumpStartBeatOffset": nj_offset,
                    "_position": [0, 0],
                    "_scale": [1, 1, 1],
                    "_rotation": [ax, ay, az],
                    "_animation": {
                        "_definitePosition": definite_positions,
                        "_localRotation": local_rotations,
                        "_color": colors,
                        "_dissolve": dissolves,
                        "_scale": scales,
                    }
                }
            })
        elif cube_type == CubeType.NOTE:
            # TODO: encode dissolve arrow in metalness?
            notes.append({
                "_time": tib + BASE_HJ + nj_offset,
                "_lineIndex": 0,
                "_lineLayer": 0,
                "_type": 0,
                "_cutDirection": 1,
                "_customData": {
                    "_fake": True,
                    "_interactable": False,
                    "_disableSpawnEffect": True,
                    "_disableNoteGravity": True,
                    "_disableNoteLook": True,
                    "_position": [0, 0],
                    "_noteJumpStartBeatOffset": nj_offset,
                    "_animation": {
                        "_definitePosition": definite_positions,
                        "_localRotation": local_rotations,
                        "_color": colors,
                        "_dissolve": dissolves,
                        "_dissolveArrow": dissolves,
                        "_scale": scales,
                    }
                }
            })
        elif cube_type == CubeType.SVG:
            custom_data["_customEvents"].append({
                "_time": 0,
                "_type": "AssignTrackParent",
                "_data": {
                    "_childrenTracks": [
                        track
                    ],
                    "_parentTrack": f"{track}_p"
                }
            })
            custom_data["_customEvents"].append({
                "_time": tib,
                "_type": "AnimateTrack",
                "_data": {
                    "_track": track,
                    "_duration": tfb - tib,
                    "_rotation": local_rotations,
                    # "_dissolve": dissolves,
                }
            })
            custom_data["_customEvents"].append({
                "_time": tib,
                "_type": "AnimateTrack",
                "_data": {
                    "_track": f"{track}_p",
                    "_duration": tfb - tib,
                    "_position": definite_positions,
                }
            })
        elif cube_type == CubeType.TRACK:
            custom_data["_customEvents"].append({
                "_time": tib,
                "_type": "AnimateTrack",
                "_data": {
                    "_track": track,
                    "_duration": tfb - tib,
                    "_position": definite_positions,
                }
            })
        elif cube_type == CubeType.PLAYER:
            custom_data["_customEvents"].append({
                "_time": tib,
                "_type": "AssignPlayerToTrack",
                "_data": {
                    "_track": "player",
                }
            })
            custom_data["_customEvents"].append({
                "_time": tib,
                "_type": "AnimateTrack",
                "_data": {
                    "_track": "player",
                    "_duration": tfb - tib,
                    "_position": definite_positions,
                    "_rotation": local_rotations,
                }
            })


def normalize(x, y, viewbox):
    """Normalize so that the origin is at the center of the image,
    and the width and height of the image are 1
    """
    xi, yi, width, height = viewbox
    return (x - xi) / width - 0.5, (yi + height - y) / height - 0.5


@functools.lru_cache(maxsize=None)
def load_image(filename, n_div=3):
    root = ET.parse("{}.svg".format(filename)).getroot()
    viewbox = tuple(map(float, root.attrib["viewBox"].split()))
    all_lines = []
    for line in root.iter("{http://www.w3.org/2000/svg}line"):
        x1 = float(line.attrib["x1"])
        y1 = float(line.attrib["y1"])
        x2 = float(line.attrib["x2"])
        y2 = float(line.attrib["y2"])
        sw = float(line.attrib["stroke-width"]
                   if "stroke-width" in line.attrib else 1)
        x1, y1 = normalize(x1, y1, viewbox)
        x2, y2 = normalize(x2, y2, viewbox)
        all_lines.append((x1, y1, x2, y2, sw))

    for polyline in root.iter("{http://www.w3.org/2000/svg}polyline"):
        points = [normalize(*map(float, pt.split(',')), viewbox)
                  for pt in polyline.attrib["points"].split()]
        x1, y1 = points[0]
        sw = float(polyline.attrib["stroke-width"]
                   if "stroke-width" in polyline.attrib else 1)
        for x2, y2 in points[1:]:
            all_lines.append((x1, y1, x2, y2, sw))
            x1, y1 = x2, y2
    for polygon in root.iter("{http://www.w3.org/2000/svg}polygon"):
        points = [normalize(*map(float, pt.split(',')), viewbox)
                  for pt in polygon.attrib["points"].split()]
        points.append(points[0])
        x1, y1 = points[0]
        sw = float(polygon.attrib["stroke-width"]
                   if "stroke-width" in polygon.attrib else 1)
        for x2, y2 in points[1:]:
            all_lines.append((x1, y1, x2, y2, sw))
            x1, y1 = x2, y2
    for path in root.iter("{http://www.w3.org/2000/svg}path"):
        path_spec = parse_path(path.attrib["d"])
        sw = float(path.attrib["stroke-width"]
                   if "stroke-width" in path.attrib else 1)
        for segment in path_spec:
            if type(segment) is Line:
                x1, y1 = normalize(segment.start.real,
                                   segment.start.imag, viewbox)
                x2, y2 = normalize(segment.end.real,
                                   segment.end.imag, viewbox)
                all_lines.append((x1, y1, x2, y2, sw))
            elif type(segment) is CubicBezier:
                points = []
                for i in range(n_div + 1):
                    t = i / n_div
                    c = segment.point(t)
                    x, y = normalize(c.real, c.imag, viewbox)
                    points.append((x, y))
                x1, y1 = points[0]
                for x2, y2 in points[1:]:
                    all_lines.append((x1, y1, x2, y2, sw))
                    x1, y1 = x2, y2

    return all_lines


def add_image(walls, image_data):
    track, scale, times = image_data
    thickness = 0.005

    models = [load_image(times[i][1]) for i in range(len(times) - 1)]
    ti = times[0][0]
    tf = times[-1][0]

    max_wall_count = max([len(model) for model in models])

    for i in range(max_wall_count):
        start = ti - 1 + i / max_wall_count
        end = tf + i / max_wall_count

        positions = []
        rotations = []
        scales = []

        for j in range(len(times) - 1):
            tt, _ = times[j]
            ttn = times[j + 1][0]

            model = models[j]
            line = model[i % len(model)]
            x1, y1, x2, y2, _ = line
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            x1, y1 = scale * x1, scale * y1
            x2, y2 = scale * x2, scale * y2

            pos = [x1, y1, 0]
            rot = [0, 0, math.degrees(math.atan2(y2 - y1, x2 - x1)) - 90]
            scl = [0.5, math.sqrt((x2 - x1)**2 + (y2 - y1) ** 2), 0.5]

            ttr = (tt - start) / (end - start)
            ttnr = (ttn - start) / (end - start)
            positions.append([*pos, ttr])
            rotations.append([*rot, ttr],)
            scales.append([*scl, ttr])
            positions.append([*pos, ttnr])
            rotations.append([*rot, ttnr],)
            scales.append([*scl, ttnr])

        nj_offset = (end - start) / 2 - BASE_HJ
        walls.append({
            "_time": start + BASE_HJ + nj_offset,
            "_duration": 0,
            "_lineIndex": 0,
            "_type": 0,
            "_width": 0,
            "_customData": {
                "_track": track,
                "_interactable": False,
                "_noteJumpStartBeatOffset": nj_offset,
                "_position": [0, 0],
                "_color": [1, 1, 1, 0],
                "_scale": [thickness * 2, 1, thickness * 2],
                "_animation": {
                    "_position": positions,
                    "_localRotation": rotations,
                    "_definitePosition": [[0, 0, 0, 0]],
                    "_scale": scales,
                    "_dissolve": [
                        [0, (ti - start) / (end - start)],
                        [1, (ti - start) / (end - start)],
                        [1, (tf - start) / (end - start)],
                        [0, (tf - start) / (end - start)],
                    ],
                }
            }
        })


def add_chimney_smoke(walls):
    smoke_start, smoke_end = 1, 112
    smoke_pos = 10.8, 8, 18.7
    n_cubes = 14
    cube_duration = 10
    n_repet = int((smoke_end - cube_duration - smoke_start) / cube_duration)
    pbar = tqdm(range(n_cubes))
    pbar.set_description("chimney smoke")
    for cnt in pbar:
        cube_time = smoke_start + cube_duration * cnt / n_cubes
        nj_offset = (smoke_end - cube_duration - smoke_start) / 2 - BASE_HJ
        scale = list(np.random.uniform(0.4, 0.6, 3))

        definite_positions = []
        local_rotations = [[*np.random.uniform(0, 45, 3), 0]]
        dissolves = []
        scales = []
        ti = 0
        dt = 1 / n_repet
        for _ in range(n_repet):
            tf = ti + dt
            pos_off_base = np.random.normal(0, 0.1, 2)
            pos_off_top = (0.5 * np.array([math.sin(200 * ti), math.sin(50 * ti)])
                           + np.random.normal(0, 0.5, 2))
            top_height_off = np.random.normal(3, 0.5)
            definite_positions.extend([
                [smoke_pos[0] + pos_off_base[0], smoke_pos[1],
                    smoke_pos[2] + pos_off_base[1], ti],
                [smoke_pos[0] + pos_off_top[0], smoke_pos[1] + top_height_off,
                    smoke_pos[2] + pos_off_top[1], tf],
            ])
            dissolves.extend([
                [0, ti],
                [1, lerp(ti, tf, 0.1)],
                [1, lerp(ti, tf, 0.6)],
                [0, lerp(ti, tf, 0.95)],
            ])
            scales.extend([
                [0.3, 0.3, 0.3, ti],
                [1, 1, 1, tf],
            ])
            ti = tf

        walls.append({
            "_time": cube_time + BASE_HJ + nj_offset,
            "_duration": 0,
            "_lineIndex": 0,
            "_type": 0,
            "_width": 0,
            "_customData": {
                "_interactable": False,
                "_noteJumpStartBeatOffset": nj_offset,
                "_position": [0, 0],
                "_scale": scale,
                "_color": [0.5, 0.5, 0.6, 0],
                "_animation": {
                    "_definitePosition": definite_positions,
                    "_localRotation": local_rotations,
                    "_dissolve": dissolves,
                    "_scale": scales,
                }
            }
        })


def add_rocket_fire(walls):
    fire_start, fire_end = 127, 144
    n_cubes = 25
    cube_duration = 2
    n_repet = int((fire_end - cube_duration - fire_start) / cube_duration)
    pbar = tqdm(range(n_cubes))
    pbar.set_description("rocket fire")
    for cnt in pbar:
        cube_time = fire_start + cube_duration * cnt / n_cubes
        nj_offset = (fire_end - cube_duration - fire_start) / 2 - BASE_HJ
        scale = list(np.random.uniform(0.4, 0.6, 3))

        definite_positions = []
        local_rotations = [[*np.random.uniform(0, 45, 3), 0]]
        dissolves = []
        scales = []
        colors = []
        ti = 0
        dt = 1 / n_repet
        for _ in range(n_repet):
            tf = ti + dt
            pos_off_base = np.random.normal(0, 0.5, 2)
            pos_off_top = np.random.normal(0, 0.25, 2)
            top_height_off = np.random.uniform(-5, -8)
            definite_positions.extend([
                [pos_off_base[0], -0.5, pos_off_base[1], ti],
                [pos_off_top[0], top_height_off, pos_off_top[1], tf],
            ])
            dissolves.extend([
                [0, ti],
                [1, ti],
                [1, lerp(ti, tf, 0.6)],
                [0, lerp(ti, tf, 0.95)],
            ])
            scales.extend([
                [1, 1, 1, ti],
                [.5, .5, .5, tf],
            ])
            colors.extend([
                [1, 0.8, 0, 5, ti],
                [1, 0.6, 0, 1, lerp(ti, tf, 0.4)],
                [1, 0.5, 0, 0.5, tf],
            ])
            ti = tf

        walls.append({
            "_time": cube_time + BASE_HJ + nj_offset,
            "_duration": 0,
            "_lineIndex": 0,
            "_type": 0,
            "_width": 0,
            "_customData": {
                "_track": "fire",
                "_interactable": False,
                "_noteJumpStartBeatOffset": nj_offset,
                "_position": [0, 0],
                "_scale": scale,
                "_animation": {
                    "_definitePosition": definite_positions,
                    "_localRotation": local_rotations,
                    "_dissolve": dissolves,
                    "_scale": scales,
                    "_color": colors,
                }
            }
        })


def add_bh_dust(walls):
    dust_start, dust_end = 280, 289
    n_cubes = 48
    cube_duration = 1
    # palette = [[100, 100, 100, 10], [200, 200, 0, 10]]
    palette = [[0.5, 0.5, 0.5, 0.2]]
    n_repet = int((dust_end - cube_duration - dust_start) / cube_duration)
    pbar = tqdm(range(n_cubes))
    pbar.set_description("bh dust")
    for cnt in pbar:
        cube_time = dust_start + cube_duration * cnt / n_cubes
        nj_offset = (dust_end - cube_duration - dust_start) / 2 - BASE_HJ
        s = np.random.uniform(0.05, 0.2)
        d = np.random.uniform(0.6, 0.9)

        definite_positions = []
        local_rotations = [[*np.random.uniform(0, 45, 3), 0]]
        dissolves = []
        ti = 0
        dt = 1 / n_repet
        for _ in range(n_repet):
            tf = ti + dt
            r = np.random.uniform(4, 8)
            theta = np.random.uniform(0, 2 * np.pi)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            definite_positions.extend([
                [x, y, 22, ti],
                [x, y, -2, tf],
            ])
            dissolves.extend([
                [0, ti],
                [d, lerp(ti, tf, 0.4)],
                [d, lerp(ti, tf, 0.9)],
                [0, lerp(ti, tf, 0.95)],
            ])
            ti = tf

        walls.append({
            "_time": cube_time + BASE_HJ + nj_offset,
            "_duration": 0,
            "_lineIndex": 0,
            "_type": 0,
            "_width": 0,
            "_customData": {
                "_track": "player",
                "_interactable": False,
                "_noteJumpStartBeatOffset": nj_offset,
                "_position": [0, 0],
                "_scale": [s, s, s],
                "_color": random.choice(palette),
                "_animation": {
                    "_definitePosition": definite_positions,
                    "_localRotation": local_rotations,
                    "_dissolve": dissolves,
                }
            }
        })


def add_sea(walls, custom_data):
    ti, tf = 0, 140
    nj_offset = (tf - ti) / 2 - BASE_HJ
    walls.append({
        "_time": ti + BASE_HJ + nj_offset,
        "_duration": 0,
        "_lineIndex": 0,
        "_type": 0,
        "_width": 0,
        "_customData": {
            "_interactable": False,
            "_noteJumpStartBeatOffset": nj_offset,
            "_position": [0, 0],
            "_scale": [800, 0.2, 800],
            "_color": [.6, .7, 1., -42],
            "_animation": {
                "_definitePosition": [[-400, -0.2, -400, 0]],
                "_dissolve": [[1, 1], [0, 1]],
            }
        }
    })


def add_eruption(notes):
    # tsi, tsf, n
    spawn_ranges = [
        (46, 47, 30),
        (47, 62, 70),
        (126, 127, 15),
        (127, 134, 30),
    ]
    lifetime = 6

    pbar = tqdm(total=sum(n for _, _, n in spawn_ranges))
    pbar.set_description("eruption")
    for tsi, tsf, n in spawn_ranges:
        for i in range(n):
            pbar.update(1)
            spawn_x = np.random.uniform(-4, 4)
            spawn_y = np.random.uniform(15, 20)
            spawn_z = np.random.uniform(134, 142)
            spawn_a = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(15, 30)
            final_x = spawn_x + radius * np.cos(spawn_a)
            final_z = spawn_z + radius * np.sin(spawn_a)
            max_y = np.random.uniform(45, 55)
            off = np.random.uniform(0, 2)
            # ti = np.random.uniform(tsi, tsf)
            ti = lerp(tsi, tsf, i / (n-1))
            tf = ti + lifetime
            nj_offset = (tf - ti + off) / 2 - BASE_HJ
            point0 = off / (lifetime + off)
            point1 = (off + 0.4 * lifetime) / (lifetime + off)
            notes.append({
                "_time": ti + BASE_HJ + nj_offset,
                "_lineIndex": 0,
                "_lineLayer": 0,
                "_type": 0,
                "_cutDirection": 1,
                "_customData": {
                    "_fake": True,
                    "_interactable": False,
                    "_disableSpawnEffect": True,
                    "_disableNoteGravity": True,
                    "_disableNoteLook": True,
                    "_position": [0, 0],
                    "_noteJumpStartBeatOffset": nj_offset,
                    "_localRotation": list(np.random.uniform(0, 60, 3)),
                    "_animation": {
                        "_scale": [[3, 3, 3, 1]],
                        "_definitePosition": [
                            [spawn_x, 0, spawn_z, point0],
                            [final_x, 0, final_z, 1],
                        ],
                        "_position": [
                            [0, spawn_y, 0, point0],
                            [0, max_y, 0, point1, "easeOutQuad"],
                            [0, -2, 0, 1, "easeInQuad"],
                        ],
                        "_color": [
                            [1, 0, 0, 1, point1],
                            [0.5, 0, 0, 1, 1],
                        ],
                        "_dissolve": [
                            [0, point0],
                            [1, point0 + 0.02],
                        ],
                        # "_dissolveArrow": [[0, 0]],
                    }
                }
            })


def add_black_hole(walls, custom_data, notes):
    white = [100, 100, 100, 10]
    yellow = [200, 200, 0, 100]
    # dark_orange = [1., .6, 0, -42]

    # Dark and light disks
    circles = [
        (18, CubeType.ENVT, 0, None, "bh", None),
        (19, CubeType.WALL, 1, white, "bh", 0),
        (24, CubeType.ENVT, 0, None, "bh2", None),
        (25.5, CubeType.WALL, 3, yellow, "bh", 1),
        # (21, CubeType.WALL, 1, [100, 100, 100, 10]),
        # (21.5, CubeType.WALL, 1, [100, 42, 0, 10]),
    ]
    n_div = 32
    ti, tf = 232, 281
    anim_dur = 1

    theta_rad = 2 * np.pi / n_div
    theta_deg = 360 / n_div

    pbar = tqdm(total = len(circles) * n_div)
    pbar.set_description("black hole")
    for r, cube_type, z_off, color, track, sd in circles:
        s = 2 * r * np.sin(theta_rad)
        # ti2 = ti + (anim_dur if sd else 0)
        ti2 = ti
        nj_offset = (tf - ti2) / 2 - BASE_HJ
        for i in range(n_div):
            pbar.update(1)

            if cube_type == CubeType.ENVT:
                regex = "\\]PillarPair \\(1\\)\\.\\[0\\]PillarL\\.\\[0\\]Pillar$"
                rescale = np.array([0.17, 0.17/32, 0.17])
                pos_offset = np.array([0, 0.17, 1])
            elif cube_type == CubeType.WALL:
                rescale = np.array([1, 1, 1])
                pos_offset = np.array([-0.5, 0, -1])
            
            scale = [1.05*s/2, r, 1]

            scale *= rescale
            euler = [0, 0, theta_deg * i]
            new_position = [0, 0, z_off] + pos_offset

            scales = [[*scale, 0]]
            definite_positions = [[*new_position, 0]]
            local_rotations = [[*euler, 0]]

            if cube_type == CubeType.WALL:
                off = i / (n_div - 1)
                if sd:
                    off = 1 - off
                off *= anim_dur
                walls.append({
                    "_time": ti2 + BASE_HJ + nj_offset + off,
                    "_duration": 0,
                    "_lineIndex": 0,
                    "_type": 0,
                    "_width": 0,
                    "_customData": {
                        "_track": track,
                        "_interactable": False,
                        "_noteJumpStartBeatOffset": nj_offset,
                        "_position": [0, 0],
                        "_scale": [1, 1, 1],
                        "_color": color,
                        "_animation": {
                            "_definitePosition": definite_positions,
                            "_localRotation": local_rotations,
                            "_dissolve": [[0, 0], [1, 0]],
                            "_scale": scales,
                        }
                    }
                })
            elif cube_type == CubeType.ENVT:
                # track = get_unique_track()
                custom_data["_environment"].append({
                    "_id": regex,
                    "_lookupMethod": "Regex",
                    "_duplicate": 1,
                    "_position": new_position,
                    "_scale": scale,
                    "_rotation": euler,
                    "_track": track,
                })
    
    # Inner accretion disk
    r1, r2, r3 = 23, 36, 40
    n_div = 64
    n_rot = 1
    n_sec = 72
    n_sec_skip = 9
    ti3 = ti # + 2

    theta_rad = 2 * np.pi / n_div
    theta_deg = 360 / n_div
    nj_offset = (tf - ti3) / 2 - BASE_HJ

    pbar = tqdm(total=n_div)
    pbar.set_description("accretion inner")
    s = 2 * r3 * np.sin(theta_rad)
    tracks = []
    for i in range(n_div):
        off = i / (n_div-1)

        pbar.update(1)

        track = f"A{i}"
        tracks.append(track)

        rescale = np.array([1, 1, 1])
        pos_offset = np.array([-0.5, 0, 0])

        scale = [1.05*s/2, 1, r3 - r1]

        scale *= rescale
        # euler = [0, -theta_deg * i, 0]
        # new_position = [r1 * np.cos(theta_rad * i + np.pi/2), 0, r1 * np.sin(theta_rad * i + np.pi/2)] + pos_offset

        scales = [[*scale, 0]]
        # definite_positions = [[*new_position, 0]]
        # local_rotations = [[*euler, 0]]

        pos = [0, 0, r1] + pos_offset
        definite_positions = [[*pos, 0]]

        alpha0 = theta_deg * i
        a_inc = 360 / n_sec
        rotations = []
        track_rotations = []
        # scale = [[]]
        dt = 1 / (n_sec * n_rot)
        to = off / (tf - ti3)
        for j in range(n_sec * n_rot + 1):
            alpha_deg = (alpha0 + a_inc * j) % 360
            # alpha_rad = np.math.radians(alpha_deg)
            dis2z = min(abs(alpha_deg % 360), abs((-alpha_deg) % 360))
            beta_deg = lerp(20, 0, (dis2z - 40) / 40)
            # beta_deg = 30 if dis2z < 90 else 0
            # rm = lerp(r2, r3, (dis2z - 60) / 20)
            tt = clamp(j * dt - to, 0, 1)
            if j % n_sec_skip == 0:
                rotations.append([0, -alpha_deg, 0, tt])
            track_rotations.append([-beta_deg, 0, 0, j * dt])
            # scale = [1.05*s/2, 1, rm - r1] * rescale
            # if len(scales) > 1 and scale[2] == scales[-1][2] == scales[-2][2]:
            #     scales.pop()
            # scales.append([*scale, j * dt])

        walls.append({
            "_time": ti3 + BASE_HJ + nj_offset + off,
            "_duration": 0,
            "_lineIndex": 0,
            "_type": 0,
            "_width": 0,
            "_customData": {
                "_track": track,
                "_interactable": False,
                "_noteJumpStartBeatOffset": nj_offset,
                "_position": [0, 0],
                "_scale": [1, 1, 1],
                "_color": yellow,
                "_animation": {
                    "_definitePosition": definite_positions,
                    "_rotation": rotations,
                    "_dissolve": [[0, 0], [1, 0]],
                    "_scale": scales,
                }
            }
        })

        custom_data["_customEvents"].append({
            "_time": ti3,
            "_type": "AnimateTrack",
            "_data": {
                "_track": track,
                "_duration": tf - ti3,
                "_rotation": track_rotations,
            }
        })

    custom_data["_customEvents"].append({
        "_time": 0,
        "_type": "AssignTrackParent",
        "_data": {
            "_childrenTracks": tracks,
            "_parentTrack": "accr2"
        }
    })
    custom_data["_customEvents"].append({
        "_time": 0,
        "_type": "AssignTrackParent",
        "_data": {
            "_childrenTracks": [
                "accr2",
            ],
            "_parentTrack": "bh"
        }
    })
    custom_data["_customEvents"].append({
        "_time": ti3,
        "_type": "AnimateTrack",
        "_data": {
            "_track": "accr2",
            "_duration": tf - ti3,
            "_rotation": [[-10, 0, 10, 0]],
        }
    })

    # Objects that get "sucked" into the black hole
    ti, tf = 227, 232
    n_items = 32
    xf, yf, zf = 0, 100, 100

    for i in range(n_items):
        tib = ti - i / (n_items - 1)
        nj_offset = (tf - tib) / 2 - BASE_HJ
        # xi, yi = 0, 0
        # while abs(xio) < 10 and abs(yio) < 10:
        #     xio = np.random.uniform(-100, 100)
        #     yio = np.random.uniform(-100, 100)
        xi, yi, zi = random_on_sphere(xf, yf, zf, 80)
        palette = [[100, 100, 100, 10], [200, 200, 0, 100]]
        s = max(np.random.normal(1, 0.5), 0.1)
        # notes.append({
        #     "_time": tib + BASE_HJ + nj_offset,
        #     "_lineIndex": 0,
        #     "_lineLayer": 0,
        #     "_type": 0,
        #     "_cutDirection": 1,
        #     "_customData": {
        #         "_fake": True,
        #         "_interactable": False,
        #         "_disableSpawnEffect": True,
        #         "_disableNoteGravity": True,
        #         "_disableNoteLook": True,
        #         "_position": [0, 0],
        #         "_noteJumpStartBeatOffset": nj_offset,
        #         "_animation": {
        #             "_definitePosition": [
        #                 [xi, yi, zi, 0],
        #                 [xf, yf, zf, 1, "easeInQuad"],
        #             ],
        #             "_localRotation": [[*np.random.uniform(0, 60, 3), 0]],
        #             "_color": [[*random.choice(palette), 0]],
        #             "_dissolve": [[0, 0], [1, 0.3], [1, 0.8], [0.5, 1]],
        #             "_dissolveArrow": [[0, 0]],
        #             "_scale": [[s, s, s, 0]],
        #         }
        #     }
        # })
        walls.append({
            "_time": tib + BASE_HJ + nj_offset,
            "_duration": 0,
            "_lineIndex": 0,
            "_type": 0,
            "_width": 0,
            "_customData": {
                "_interactable": False,
                "_noteJumpStartBeatOffset": nj_offset,
                "_position": [0, 0],
                "_scale": [1, 1, 1],
                "_animation": {
                    "_definitePosition": [
                        [xi, yi, zi, 0],
                        [xf, yf, zf, 1, "easeInQuad"],
                    ],
                    "_localRotation": [[*np.random.uniform(0, 60, 3), 0]],
                    "_color": [[*random.choice(palette), 0]],
                    "_dissolve": [[0, 0], [1, 0.6], [1, 0.8], [0.5, 1]],
                    "_scale": [[s, s, s, 0]],
                }
            }
        })


def add_stars(walls):
    ti, tf = 138, 281
    s = 0.1
    n_stars = 128
    nj_offset = (tf - ti) / 2 - BASE_HJ
    pbar = tqdm(range(n_stars))
    pbar.set_description("stars")
    for i in pbar:
        r = 110
        z = -1
        while z < 0:
            x, y, z = random_on_sphere(0, 100, 0, r)
        off = 2 * i / n_stars
        walls.append({
            "_time": ti + BASE_HJ + nj_offset + off,
            "_duration": 0,
            "_lineIndex": 0,
            "_type": 0,
            "_width": 0,
            "_customData": {
                # "_track": track,
                "_interactable": False,
                "_noteJumpStartBeatOffset": nj_offset,
                "_position": [0, 0],
                "_scale": [s, s, s],
                "_color": [10, 10, 10, 1],
                "_animation": {
                    "_definitePosition": [[x, y, z, 0]],
                    "_dissolve": [[0, 0], [1, 0]],
                }
            }
        })


def add_notes(notes_in, notes_out, custom_data):
    track_off = {
        "bh_left": (-8, 0),
        "bh_right": (8, 0),
        "bh_up": (0, 8),
        "bh_down": (0, -8),
        "bh_up_left": (-8, 8),
        "bh_up_right": (8, 8),
        "bh_down_left": (-8, -8),
        "bh_down_right": (8, -8),
    }

    pbar = tqdm(notes_in)
    pbar.set_description("notes")
    for note_in in pbar:
        note = note_in.copy()

        note["_customData"] = {
            "_disableSpawnEffect": True,
            "_disableNoteGravity": True,
            # "_disableNoteLook": True,
            "_animation": {
                "_dissolve": [[0, 0], [1, 0.2]],
                "_dissolveArrow": [[0, 0], [1, 0.2]],
            }
        }

        note["_customData"]["_track"] = "player"

        x = note["_lineIndex"] -2
        y = note["_lineLayer"]

        orange = [.8, .5, .1]
        black = [.3, .3, .3]
        blue = [.3, .5, .7]
        # orange = [1., .7, 0]
        grey = [.5, .5, .5]
        # light_grey = [.6, .6, .6]
        # yellow = [.8, .6, .1]
        # light_purple = [.6, .7, 1.]
        # if note["_time"] < 30 or 63 <= note["_time"] <= 117.5:
        if note["_time"] < 127.5:
            njs = lerp(12, 16, (note["_time"] - 119) / 8)
            note["_customData"]["_noteJumpMovementSpeed"] = njs
            note["_customData"]["_noteJumpStartBeatOffset"] = 3
            note["_customData"]["_position"] = [0, 0]
            note["_customData"]["_animation"]["_position"] = [
                [x, -1.2, 0, 0],
                [x, -0.5, 0, 0.25, "easeOutElastic"],
                [x, y, 0, 0.45, "easeInOutQuad"],
            ]
            if note["_type"] == 0:
                note["_customData"]["_color"] = grey
            else:
                note["_customData"]["_color"] = black
            note["_customData"]["_animation"]["_dissolve"] = [[0, 0], [1, 0]]
            note["_customData"]["_animation"]["_dissolveArrow"] = [[0, 0], [1, 0]]
        # elif note["_time"] < 127.5:
        #     # note["_customData"]["_noteJumpMovementSpeed"] = 16  # default
        #     note["_customData"]["_noteJumpStartBeatOffset"] = 1
        #     note["_customData"]["_position"] = [0, 0]
        #     if note["_type"] == 1:
        #         note["_customData"]["_color"] = black
        #     note["_customData"]["_animation"]["_position"] = [
        #         [-0.5, 1, 2, 0],
        #         [x, y, 0, 0.25, "easeOutSine"],
        #     ]
        #     note["_customData"]["_animation"]["_localRotation"] = [
        #         [0, 0, 90 if x < 0 else -90, 0],
        #         [0, 0, 0, 0.25, "easeOutSine"],
        #     ]
        #     note["_customData"]["_animation"]["_dissolveArrow"] \
        #         = [[0, 0], [1, 0.25]]
        #     note["_customData"]["_animation"]["_dissolve"] \
        #         = [[0, 0], [np.random.uniform(0.7, 0.9), 0.1]]
        elif note["_time"] <= 137.1:
            note["_customData"]["_noteJumpMovementSpeed"] = 16
            td = min(max(0, note["_time"] - 127), max(0, 137 - note["_time"]))
            yo = lerp(0, 2, td / 8)
            note["_customData"]["_animation"]["_position"] = [
                [0, yo, 0, 0],
                [0, 0, 0, 0.4, "easeOutQuad"],
            ]
            if note["_type"] == 0:
                note["_customData"]["_color"] = grey
            else:
                note["_customData"]["_color"] = black
        elif note["_time"] <= 234:
            # if note["_type"] == 1:
            #     note["_customData"]["_color"] = blue
            # note["_customData"]["_noteJumpStartBeatOffset"] = 1
            # fx = 1.5 if note["_time"] < 230 else 1
            # fy = 1 if note["_time"] < 230 else 0.5
            fx, fy = 1.5, 1
            xo = fx * (x + 0.5)
            yo = fy * (y - 1)
            note["_customData"]["_position"] = [0, 0]
            note["_customData"]["_animation"]["_position"] = [
                [x + xo, y + yo, 0, 0],
                [x, y, 0, 0.5],
            ]
            note["_customData"]["_animation"]["_scale"]  = [
                [.1, .1, .1, 0],
                [1, 1, 1, .2, "easeOutQuad"]
            ]
            # note["_customData"]["_animation"]["_dissolveArrow"] = dissolves
            # note["_customData"]["_animation"]["_localRotation"] = [
            #     [np.random.normal(0, 10), 0, np.random.normal(0, 20), 0],
            #     [0, 0, 0, 0.3, "easeOutSine"],
            # ]
        elif note["_time"] <= 289.1:
            note["_customData"]["_noteJumpStartBeatOffset"] = 1

            # Create clone without arrow that gets sucked into black hole
            note_clone = copy.deepcopy(note)

            note["_customData"]["_animation"]["_dissolve"] = [
                [0, 0],
                [1, 0.1],
                [1, 0.2],
                [0.6, 0.4],
            ]
            note["_customData"]["_animation"]["_dissolveArrow"] = [[0, 0], [1, 0.1]]
            note["_customData"]["_animation"]["_position"] = [[0, 0, -0.05, 0]]
            note_clone["_customData"]["_fake"] = True
            note_clone["_customData"]["_interactable"] = False
            note_clone["_customData"]["_animation"]["_dissolveArrow"] = [[0, 0]]
            note_clone["_customData"]["_animation"]["_position"] = [
                [0, 0, 0, 0.2],
                [0, 0, 25, 0.5, "easeInSine"],
            ]
            note_clone["_customData"]["_animation"]["_dissolve"] = [
                [0, 0.15],
                [0.5, 0.2],
                [0, 0.5],
            ]
            notes_out.append(note_clone)

            # note["_customData"]["_track"] = "bh_ar" if x > -0.1 else "bh_al"
            # if note["_type"] == 0:
            #     note["_customData"]["_color"] = light_grey
            # else:
            #     note["_customData"]["_color"] = yellow
        elif note["_time"] <= 327:
            # if note["_type"] == 1:
            #     note["_customData"]["_color"] = blue
            nj_offset = 2
            note["_customData"]["_noteJumpStartBeatOffset"] = nj_offset

            # Dissolve magic for the clones
            ti = note["_time"] - (BASE_HJ + nj_offset)
            tf = note["_time"] + (BASE_HJ + nj_offset)
            if ti <= 289:
                dissolves = [
                    [0, max(0, (288 - ti) / (tf - ti))],
                    [1, min(1, (289 - ti) / (tf - ti))],
                ]
                # print(ti, tf, dissolves)
            else:
                dissolves = note["_customData"]["_animation"]["_dissolve"]
            
            # Create fake clones of the note and put them on tracks
            for track in track_off.keys():
                note_clone = copy.deepcopy(note)
                note_clone["_customData"]["_fake"] = True
                note_clone["_customData"]["_interactable"] = False
                note_clone["_customData"]["_color"] = grey
                note_clone["_customData"]["_track"] = track
                note_clone["_customData"]["_animation"]["_dissolveArrow"] \
                    = [[0, 0]]
                note_clone["_customData"]["_animation"]["_dissolve"] = dissolves
                notes_out.append(note_clone)

            note["_customData"]["_animation"]["_dissolve"] = [
                [0, 0],
                [1, 0.2],
                [0.8, 0.4],
            ]
            
            # Fake notes diverging
            fx, fy = 1.5, 1
            xo = fx * (1 if x > -0.1 else -1)
            yo = fy * (1 if y > 1.5 else (-1 if y < 0.5 else 0))

            for track in track_off.keys():
                note_clone = copy.deepcopy(note)
                note_clone["_customData"]["_fake"] = True
                note_clone["_customData"]["_interactable"] = False
                note_clone["_customData"]["_animation"]["_dissolveArrow"] = [[0, 0]]
                note_clone["_customData"]["_animation"]["_dissolve"] = [
                    [0, 0.2],
                    [0.2, 0.3],
                    [0.2, 0.45],
                    [0, 0.52],
                ]
                note_clone["_customData"]["_position"] = [0, 0]
                note_clone["_customData"]["_animation"]["_position"] = [
                    [x, y, 0.05, 0.2],
                    [x + xo, y + yo, 5, 0.52, "easeInSine"],
                ]
                notes_out.append(note_clone)
            
            # note["_customData"]["_animation"]["_dissolveArrow"] = [[0, 0]]

            # Arrows converging towards real note
            # color = orange if note["_type"] == 0 else blue
            # if note["_time"] >= 292:
            #     for track in track_off.keys():
            #         x, y = track_off[track]
            #         note_clone = copy.deepcopy(note)
            #         note_clone["_customData"]["_fake"] = True
            #         # note_clone["_customData"]["_color"] = grey
            #         note_clone["_customData"]["_track"] = "player"
            #         note_clone["_customData"]["_animation"]["_dissolve"] \
            #             = [[0, 0]]
            #         note_clone["_customData"]["_animation"]["_color"]  = [
            #             [*grey, 1, 0.3],
            #             [*color, 1, 0.4],
            #         ]
            #         note_clone["_customData"]["_animation"]["_position"] = [
            #             [x, y, 0, 0],
            #             [0, 0, 0, 0.4, "easeOutSine"],
            #         ]
            #         notes_out.append(note_clone)
            #     note["_customData"]["_animation"]["_dissolveArrow"] \
            #         = [[0, 0]]
        else:
            dissolves = [[0, 0], [1, 0.1]]
            note["_customData"]["_animation"]["_dissolve"] = dissolves
            note["_customData"]["_animation"]["_dissolveArrow"] = dissolves

        notes_out.append(note)

    # Path animations for the black hole approach
    # near, far = 0, 20
    # custom_data["_customEvents"].append({
    #     "_time": 234,
    #     "_type": "AssignPathAnimation",
    #     "_data": {
    #         "_track": "bh_ap",
    #         "_duration": 36,
    #         "_position": [
    #             [0, 0, far, 0],
    #             [0, 0, 0, 0.5, "easeOutCubic"],
    #         ],
    #     }
    # })
    # custom_data["_customEvents"].append({
    #     "_time": 270,
    #     "_type": "AssignPathAnimation",
    #     "_data": {
    #         "_track": "bh_ap",
    #         "_duration": 10,
    #         "_position": [
    #             [0, 0, 0, 0],
    #         ],
    #     }
    # })

    # for s, track in [(-1, "bh_al"), (1, "bh_ar")]:
    #     custom_data["_customEvents"].append({
    #         "_time": 234,
    #         "_type": "AssignPathAnimation",
    #         "_data": {
    #             "_track": track,
    #             "_duration": 10,
    #             "_position": [
    #                 [s*20, 0, 0, 0],
    #                 [0, 0, 0, 0.4, "easeOutQuad"],
    #             ],
    #         }
    #     })
    #     custom_data["_customEvents"].append({
    #         "_time": 270,
    #         "_type": "AssignPathAnimation",
    #         "_data": {
    #             "_track": track,
    #             "_duration": 12,
    #             "_position": [
    #                 [0, 0, 0, 0],
    #             ],
    #         }
    #     })

    # Tracks for black hole fake notes
    ti, tf = 288, 327
    trt = 2
    for track in track_off.keys():
        x, y = track_off[track]
        z = 0.5
        custom_data["_customEvents"].append({
            "_time": ti,
            "_type": "AnimateTrack",
            "_data": {
                "_track": track,
                "_duration": trt,
                "_position": [
                    [0, 0, z, 0.],
                    [x, y, z, 1., "easeOutSine"]
                ],
            }
        })
        custom_data["_customEvents"].append({
            "_time": tf-trt,
            "_type": "AnimateTrack",
            "_data": {
                "_track": track,
                "_duration": trt,
                "_position": [
                    [x, y, z, 0.],
                    [10*x, 10*y, z, 1., "easeInSine"]
                ],
            }
        })
    
    # Parenting some tracks to the player track
    custom_data["_customEvents"].append({
        "_time": 230,
        "_type": "AssignTrackParent",
        "_data": {
            "_childrenTracks": [*track_off.keys()],
            "_parentTrack": "player"
        }
    })

    # ti, tf, n, x, y
    pauls = [
        (302+2/3, 304, 9, -2.5, 1),
        (302+2/3, 304, 9, 2.5, 1),
    ]
    for ti, tf, pn, x, y in pauls:
        for ip in range(pn):
            tt = lerp(ti, tf, ip / (pn - 1))
            notes_out.append({
                "_time": tt,
                "_lineIndex": 0,
                "_lineLayer": 0,
                "_type": 0,
                "_cutDirection": 1,
                "_customData": {
                    "_fake": True,
                    "_track": "player",
                    "_interactable": False,
                    "_disableSpawnEffect": True,
                    "_disableNoteGravity": True,
                    "_disableNoteLook": True,
                    "_position": [x, y],
                    "_color": [.5, .5, .5],
                    "_noteJumpMovementSpeed": 22,
                    "_animation": {
                        "_dissolve": [[0, 0]],
                        "_dissolveArrow": [[0, 0], [1, 0]],
                    }
                }
            })
    
    # Black hole slow part
    left_color = [.8, .4, .2]
    right_color = blue
    # tt, side, n_swaps, color_change
    slow_notes = [
        (331, 1, 1, .35),
        (339, 0, 2, .325),
        (347, 1, 3, .3),
        (355, 0, 4, .3),
        (363, 1, 5, .3),
        (371, 0, 6, .3),
    ]
    tracks = []
    dt = 0.025
    njs = 8
    for sni, (nt, side, n_swaps, tc) in enumerate(slow_notes):
        true_position = np.random.randint(0, 3)
        swaps = [random.sample(range(3), 2)]
        for _ in range(n_swaps - 1):
            i, j = swaps[-1]
            i2 = 3 - i - j
            j2 = random.choice([i, j])
            swaps.append([i2, j2])
        for pos in range(3):
            is_true = (pos == true_position)
            track = f"sn{sni}_{pos}"
            tracks.append(track)
            if is_true:
                color = right_color if side else left_color
                notes_out.append({
                    "_time": nt,
                    "_lineIndex": 0,
                    "_lineLayer": 0,
                    "_type": side,
                    "_cutDirection": 1,
                    "_customData": {
                        "_track": track,
                        "_disableSpawnEffect": True,
                        "_disableNoteGravity": True,
                        # "_disableNoteLook": True,
                        "_position": [0, 0],
                        "_noteJumpMovementSpeed": njs,
                        "_noteJumpStartBeatOffset": 6,
                        "_animation": {
                            "_color": [
                                [*color, 1, tc-.1],
                                [*grey, 1, tc],
                            ],
                            "_dissolve": [[0, 0], [1, 0.1]],
                            "_dissolveArrow": [[0, 0], [1, 0.1]],
                        }
                    }
                })
            else:
                notes_out.append({
                    "_time": nt,
                    "_lineIndex": 0,
                    "_lineLayer": 0,
                    "_type": 3,
                    "_cutDirection": 1,
                    "_customData": {
                        "_track": track,
                        "_disableSpawnEffect": True,
                        "_disableNoteGravity": True,
                        # "_disableNoteLook": True,
                        "_position": [0, 0],
                        "_noteJumpMovementSpeed": njs,
                        "_noteJumpStartBeatOffset": 6,
                        "_color": grey,
                        # "_scale": [0.7, 0.7, 0.7],
                        "_animation": {
                            "_dissolve": [[0, 0], [1, 0.05], [1, tc-.1], [0, tc]],
                            # "_dissolveArrow": [[0, 0], [1, 0]],
                        }
                    }
                })
                notes_out.append({
                    "_time": nt,
                    "_lineIndex": 0,
                    "_lineLayer": 0,
                    "_type": side,
                    "_cutDirection": 1,
                    "_customData": {
                        "_track": track,
                        "_fake": True,
                        "_interactable": False,
                        "_disableSpawnEffect": True,
                        "_disableNoteGravity": True,
                        # "_disableNoteLook": True,
                        "_position": [0, 0],
                        "_noteJumpMovementSpeed": njs,
                        "_noteJumpStartBeatOffset": 6,
                        "_color": grey,
                        "_animation": {
                            "_dissolve": [[0, 0], [.3, 0.1], [.3, tc-.1], [1, tc]],
                            "_dissolveArrow": [[0, 0], [1, 0.1]],
                        }
                    }
                })
            x_off = -2
            positions = [[1.5*pos+x_off, 0, 0, 0]]
            cpos = pos
            for k, (i, j) in enumerate(swaps):
                i, j = min(i, j), max(i, j)
                if cpos == i:
                    npos = j
                    ym = (j-i)/2
                elif cpos == j:
                    npos = i
                    ym = -(j-i)/2
                else:
                    continue
                xm = (i+j)/2
                tt = 0.5 * (k+1) / (n_swaps+1)
                positions += [
                    [1.5*cpos+x_off, 0, 0, tt-dt],
                    [1.5*xm+x_off, 1.5*ym, 0, tt, "splineCatmullRom"],
                    [1.5*npos+x_off, 0, 0, tt+dt, "splineCatmullRom"],
                ]
                cpos = npos
            custom_data["_customEvents"].append({
                "_time": 300,
                "_type": "AssignPathAnimation",
                "_data": {
                    "_track": track,
                    "_position": positions,
                }
            })
    custom_data["_customEvents"].append({
        "_time": 300,
        "_type": "AssignTrackParent",
        "_data": {
            "_childrenTracks": tracks,
            "_parentTrack": "player"
        }
    })

    # Bombs in last part
    bomb_ranges = [
        # (391, 394+2/3),
        # (399, 402),
        # (408, 410+2/3),
        # (415, 418+2/3),

        # (391, 394),
        # (399, 402),
        # (408, 410),
        # (415, 418),
    ]
    cx, cy = -.5, 1.3
    ri, rf = 1, 7
    # zi, zf = 20, 0
    n_steps = 12
    for t0, t1 in bomb_ranges:
        n_bombs = int((t1 - t0) / (1/6))
        for ib in range(n_bombs):
            positions = []
            theta_off = 90 - 3 * (ib / n_bombs) * 360
            for i in range(n_steps):
                tt = i / (n_steps - 1)
                r = lerp(ri, rf, tt)
                theta = np.radians(theta_off - tt * 360)
                x = cx + r * np.cos(theta)
                y = cy + r * np.sin(theta)
                positions.append([x, y, 0, tt/2])
            notes_out.append({
                "_time": lerp(t0, t1, ib/(n_bombs-1)) + BASE_HJ,
                "_lineIndex": 0,
                "_lineLayer": 0,
                "_type": 3,
                "_cutDirection": 1,
                "_customData": {
                    "_track": "player",
                    "_disableSpawnEffect": True,
                    "_disableNoteGravity": True,
                    # "_noteJumpMovementSpeed": 24,
                    # "_disableNoteLook": True,
                    "_position": [0, 0],
                    "_color": [42, 42, 42, 0],
                    "_animation": {
                        "_position": positions,
                        "_dissolve": [[0, 0], [1, 0]],
                    }
                }
            })



def add_bh_walls(walls):
    ti, tf = 288, 327
    nj_offset = (tf - ti) / 2 - BASE_HJ
    track_off_sc = [
        ("bh_left", (3, -100), (.1, 200)),
        ("bh_right", (-3, -100), (.1, 200)),
        ("bh_up", (-100, -3), (200, .1)),
        ("bh_down", (-100, 3), (200, .1)),
    ]
    for track, (x, y), (sx, sy) in track_off_sc:
        walls.append({
            "_time": ti + BASE_HJ + nj_offset,
            "_duration": 0,
            "_lineIndex": 0,
            "_type": 0,
            "_width": 0,
            "_customData": {
                "_track": track,
                "_interactable": False,
                "_noteJumpStartBeatOffset": nj_offset,
                "_position": [0, 0],
                "_scale": [sx, sy, 400],
                "_color": [.2, .2, .2, 0],
                "_animation": {
                    "_definitePosition": [[x, y, -50, 0]],
                    "_dissolve": [
                        [0, 1/(tf-ti)],
                        [1, 2/(tf-ti)],
                    ],
                }
            }
        })


def add_bh_walls_slow_part(walls):
    part_start, part_end = 326, 386
    thickness = 0.2
    n_cubes = 97
    cube_duration = 20
    n_repet = int((part_end - cube_duration - part_start) / cube_duration)
    pbar = tqdm(range(n_cubes))
    pbar.set_description("up&down walls")
    for cnt in pbar:
        cube_time = part_start + cube_duration * cnt / n_cubes
        nj_offset = (part_end - cube_duration - part_start) / 2 - BASE_HJ
        scale = [thickness, np.random.uniform(3, 5), thickness]

        definite_positions = []
        dissolves = []
        ti = 0
        dt = 1 / n_repet
        for _ in range(n_repet):
            tf = ti + dt
            direction = random.choice([-1, 1])
            x = np.random.normal(0.0, 20.0)
            x += 5 * np.sign(x)
            yi = np.random.uniform(-20, 20) + (85 if direction > 0 else 115)
            yf = yi + direction * np.random.uniform(15, 30)
            z = np.random.uniform(0, 150)
            definite_positions.extend([
                [x, yi, z, ti],
                [x, yf, z, tf],
            ])
            dissolves.extend([
                [0, ti],
                [1, lerp(ti, tf, 0.2)],
                [1, lerp(ti, tf, 0.8)],
                [0, tf],
            ])
            ti = tf

        walls.append({
            "_time": cube_time + BASE_HJ + nj_offset,
            "_duration": 0,
            "_lineIndex": 0,
            "_type": 0,
            "_width": 0,
            "_customData": {
                "_interactable": False,
                "_noteJumpStartBeatOffset": nj_offset,
                "_position": [0, 0],
                "_scale": scale,
                "_color": [.5, .5, .5],
                "_animation": {
                    "_definitePosition": definite_positions,
                    "_dissolve": dissolves,
                }
            }
        })


def get_lighting_events():
    def get_lighting_events_aux(time_, types, value, color):
        return [
            {
                "_time": time_,
                "_type": type_,
                "_value": value,
                "_customData": {
                    "_color":color,
                }
            }
            for type_ in types
        ]

    def get_lighting_gradient(t1, t2, types, value, color1, color2, steps):
        return list(itertools.chain.from_iterable([
            get_lighting_events_aux(
                lerp(t1, t2, i/(steps-1)), types, value,
                list(lerp(np.array(color1), np.array(color2), i/(steps-1))))
            for i in range(steps)
        ]))

    purple = [0.7, 0.7, 0.9, 1.]
    red = [0.84, 0.29, 0.2, 1.]
    black = [0., 0., 0., 0.]
    dark_purple = [.5, .45, .6, .4]
    grey = [1, 1, 1, 7.]
    return list(itertools.chain.from_iterable([
        get_lighting_events_aux(0, range(5), 5, purple),
        get_lighting_gradient(46.6667, 47, range(5), 5, purple, red, 10),
        get_lighting_gradient(63, 65, range(5), 5, red, dark_purple, 20),
        get_lighting_gradient(75, 79, range(5), 5, dark_purple, purple, 40),
        get_lighting_gradient(126.6667, 127, range(5), 5, purple, red, 10),
        get_lighting_gradient(136, 140, range(5), 5, red, black, 60),
        get_lighting_gradient(150, 154, [1], 5, black, grey, 40),
        get_lighting_gradient(174, 180, [1], 5, grey, black, 5),
        get_lighting_gradient(182, 185, [1], 5, black, grey, 5),
        get_lighting_gradient(198, 202, [1], 5, grey, black, 40),
        get_lighting_gradient(210, 214, [1], 5, black, grey, 40),
        get_lighting_gradient(221, 225, [1], 5, grey, black, 40),
    ]))


def add_hardcoded_tracks(custom_data):
    # Bongo cats
    duration = BONGO_END - BONGO_START
    custom_data["_customEvents"].append({
        "_time": BONGO_START,
        "_type": "AnimateTrack",
        "_data": {
            "_track": "bgg",
            "_duration": duration,
            "_position": [
                [0, 105, 30, 0],
            ],
            "_rotation": [
                [0, 160, 0, 0],
                # [0, 160, 0, 1/duration, "easeOutCubic"],
                # [0, 160, 0, 1-1/duration, "easeInCubic"],
                # [0, 170, 0, 1],
            ],
        }
    })
    custom_data["_customEvents"].append({
        "_time": BONGO_START,
        "_type": "AnimateTrack",
        "_data": {
            "_track": "bgd",
            "_duration": duration,
            "_position": [
                [0, 105, 30, 0],
            ],
            "_rotation": [
                [0, 200, 0, 0],
                # [0, 200, 0, 1/duration, "easeOutCubic"],
                # [0, 200, 0, 1-1/duration, "easeInCubic"],
                # [0, 190, 0, 1],
            ],
        }
    })


def add_squares(walls, custom_data):
    n_sq = 10
    depths = [7 + 12*i for i in range(n_sq)]
    thickness = 0.5
    radius = 7
    # x, y, sx, sy
    pos_scale = [
        (-radius-thickness/2, -radius-thickness/2, thickness, 2*radius+thickness),
        (radius-thickness/2, -radius-thickness/2, thickness, 2*radius+thickness),
        (-radius+thickness/2, -radius-thickness/2, 2*radius-thickness, thickness),
        (-radius+thickness/2, radius-thickness/2, 2*radius-thickness, thickness),
    ]
    on_off = [
        [(387, 394, 395), (395, 401, 402), (403, 410, 411), (411, 418, 419), (419, 422, 423), (423+2/3, 424+1/3, 424+1/3)],
        [(387+1/3, 394, 395), (395+1/3, 401, 402), (403+1/3, 410, 411), (411+1/3, 418, 419), (419+2/3, 422, 423), (424+1/3, 425, 425)],
        [(387+2/3, 394, 395), (395+2/3, 401, 402), (403+2/3, 410, 411), (411+2/3, 418, 419), (420+1/3, 422, 423), (425, 425+2/3, 425+2/3)],
        [(388, 394, 395), (396, 401, 402), (404, 410, 411), (412, 418, 419), (420+2/3, 422, 423), (425+2/3, 426+1/3, 426+1/3)],
        [(388+1/3, 394, 395), (396+1/3, 401, 402), (404+1/3, 410, 411), (412+1/3, 418, 419), (421, 422, 423), (426+1/3, 427, 427)],
        [(388+2/3, 394, 395), (396+2/3, 401, 402), (404+2/3, 410, 411), (412+2/3, 418, 419), (427, 427+2/3, 427+2/3)],
        [(389, 394, 395), (397, 401, 402), (405, 410, 411), (413, 418, 419), (427+2/3, 428+1/3, 428+1/3)],
        [(389+2/3, 394, 395), (397+2/3, 401, 402), (405+2/3, 410, 411), (413+2/3, 418, 419), (428+1/3, 429, 429)],
        [(398+1/3, 401, 402), (414+1/3, 418, 419), (429, 429+2/3, 429+2/3)],
        [(430+1/3, 430+2/3, 431)],
    ]
    rotations = [(392, 395), (399, 402), (408, 411), (415, 419)]
    blue = [0, 0, 1, 1]
    # black = [0, 0, 0, 0]
    for i, z in enumerate(depths):
        ti = on_off[i][0][0]
        tf = on_off[i][-1][2]
        dissolves = []
        for t0, t1, t2 in on_off[i]:
            # t2 = t2 - 1/3
            tt0 = (t0 - ti) / (tf - ti)
            tt1 = (t1 - ti) / (tf - ti)
            tt2 = (t2 - ti) / (tf - ti)
            dissolves.extend([
                [0, tt0],
                [1, tt0],
                [1, tt1],
                [0, tt2],
            ])
        nj_offset = (tf - ti) / 2 - BASE_HJ
        for x, y, sx, sy in pos_scale:
            walls.append({
                "_time": ti + BASE_HJ + nj_offset,
                "_duration": 0,
                "_lineIndex": 0,
                "_type": 0,
                "_width": 0,
                "_customData": {
                    "_track": f"cf{i}",
                    "_interactable": False,
                    "_noteJumpStartBeatOffset": nj_offset,
                    "_position": [0, 0],
                    "_scale": [sx, sy, thickness],
                    # "_color": [*lerp(blue, black, i/n_sq)],
                    "_color": [*blue],
                    "_animation": {
                        "_definitePosition": [[x, y, z, 0]],
                        "_dissolve": dissolves,
                    }
                }
            })

    # Animate tracks
    for i in range(n_sq):
        ti = on_off[i][0][0]
        tf = on_off[i][-1][2]
        local_rotations = []
        for ri, rf in rotations:
            tri = (ri+i/6 - ti) / (tf - ti)
            trf = (rf - ti) / (tf - ti)
            theta = -s * 1080/(n_sq-i)
            local_rotations.extend([
                *[[0, 0, lerp(0, theta, tt), lerp(tri, trf, tt)] for tt in frange(0, 1, 0.1)],
                [0, 0, 0, trf],
            ])
        custom_data["_customEvents"].append({
            "_time": ti,
            "_type": "AnimateTrack",
            "_data": {
                "_track": f"cf{i}",
                "_duration": tf - ti,
                "_rotation": local_rotations,
            }
        })
    
    custom_data["_customEvents"].append({
        "_time": 385,
        "_type": "AnimateTrack",
        "_data": {
            "_track": "squares",
            "_duration": 50,
            "_position": [[0, 2, 0, 0]],
        }
    })

    # Parent tracks to the player
    custom_data["_customEvents"].append({
        "_time": 380,
        "_type": "AssignTrackParent",
        "_data": {
            "_childrenTracks": [f"cf{i}" for i in range(n_sq)],
            "_parentTrack": "squares"
        }
    })
    custom_data["_customEvents"].append({
        "_time": 380,
        "_type": "AssignTrackParent",
        "_data": {
            "_childrenTracks": ["squares"],
            "_parentTrack": "player"
        }
    })


def letter_width(model):
    return max(np.matmul(transform, np.transpose([x, y, z, 1]))[1]
               for transform, _ in model
               for x, y, z in itertools.product(*[[-1., 1.]]*3))


def add_text(walls):
    letter_spacing = 0.3
    space_width = 1.5
    spawn_off = 0.1
    fade_in = 0.5
    fade_out = 1
    depth = 42
    thickness = 0.05

    # DEHMS'/.
    # 45678231

    # [word, ti], tf, height
    text = [
        ([("8omeone 5lse2s 6at", 66)], 75, 7),
        ([("by 4avid 7axim 7icic", 68)], 77, 3),
        # ([("davidmaximmicic1bandcamp1com", 72)], 77, 3),
        ([("mapped by nyri0", 430)], 445, 4),
        ([("youtube1com3c3nyri0", 432)], 447, 0),
    ]

    all_letters = set()
    for words, *_ in text:
        for letters, _ in words:
            for letter in letters:
                all_letters.add(letter)
    models = {letter: load_model_old(f"font/{letter}")
              for letter in all_letters if letter != " "}
    letter_widths = {key: letter_width(
        val) + letter_spacing for key, val in models.items()}
    models[" "] = []
    letter_widths[" "] = space_width

    for sentence, tf, height in text:
        sentence_width = sum(letter_widths[letter]
                             for word, _ in sentence for letter in word)
        offset = -sentence_width / 2

        for word, ti in sentence:
            il = 0
            for letter in word:
                il += int(letter != " ")
                model = models[letter]
                for transform, _ in model:
                    position = transform[:3, 3]
                    scale = np.array(
                        [np.linalg.norm(transform[:3, i]) for i in range(3)])
                    rotation = transform[:3, :3] / scale
                    euler = Rotation.from_matrix(
                        rotation).as_euler('xyz', degrees=True)

                    pivot_diff = np.array([0, -1, 0]) * scale
                    correction = pivot_diff - np.matmul(rotation, pivot_diff)

                    new_position = position + \
                        np.matmul(rotation, np.array(
                            [1, -1, -1]) * scale) + correction

                    scale_x = scale[1]
                    scale_y = scale[2]
                    start_x = new_position[1] + offset
                    start_y = new_position[2] + height

                    nj_offset = (tf - ti) / 2 - BASE_HJ

                    apparition_offset = spawn_off * il

                    walls.append({
                        "_time": ti + BASE_HJ + nj_offset + apparition_offset,
                        "_lineIndex": 0,
                        "_type": 0,
                        "_duration": 0,
                        "_width": 0,
                        "_customData": {
                            "_track": "player",
                            "_color": [42, 42, 42, 0],
                            "_interactable": False,
                            "_noteJumpMovementSpeed": 16,
                            "_noteJumpStartBeatOffset": nj_offset,
                            "_position": [0, 0],
                            "_scale": [2*scale_x, 2*scale_y, thickness],
                            "_localRotation": [-euler[1], -euler[2], euler[0]],
                            "_animation": {
                                "_definitePosition": [
                                    [start_x, start_y, depth, 0],
                                ],
                                "_dissolve": [
                                    [0, 0],
                                    [1, fade_in / (tf - ti)],
                                    [1, 1. - (apparition_offset +
                                              fade_out) / (tf - ti)],
                                    [0, 1. - apparition_offset / (tf - ti)],
                                ]
                            }
                        }
                    })
                offset += letter_widths[letter]


def main():
    input_json = dict()
    for filename in INPUT_FILES:
        with open("{}.dat".format(filename), "r") as json_file:
            input_json[filename] = json.load(json_file)

    # TODO: efficient regular expressions!

    notes = []
    walls = []
    custom_data = {
        "_environment": list(itertools.chain.from_iterable([
            [
                {
                    "_id": id_,
                    "_lookupMethod": "Regex",
                    "_active": False,
                }
                for id_ in [
                    "PillarTrackLaneRing.*$",
                    "DustPS.*$",
                    "StarsPS.*$",
                    "Clouds.*$",
                    # "StarHemisphere.*$",
                    "Construction.*$",
                    "PlayersPlace.*$",
                    "BottomGlow.*$",
                    "TrackMirror.*$",
                    "GlowLine[RCH].*$",
                    "RotationBase.*$",
                    # "BoxLight.*$",
                    "SmallPillarPair.*$",
                    "SideLaser.*$",
                ]
            ],
            [
                {
                    "_id": id_,
                    "_lookupMethod": "Regex",
                    "_position": [
                        0,
                        -1337,
                        0
                    ]
                }
                for id_ in [
                    "PillarPair(\\s\\(\\d\\))?$",
                    "GlowLineL$",
                    "MagicDoorSprite$",
                ]
            ],
            [
                {
                    "_id": f"MagicDoorSprite.*BloomL$",
                    "_lookupMethod": "Regex",
                    "_duplicate": 1,
                    "_position": [
                        x_pos,
                        7,
                        110
                    ],
                    "_scale": [
                        1,
                        20,
                        1
                    ],
                    "_rotation": [
                        0,
                        0,
                        90
                    ]
                }
                for x_pos in [0]
            ],
        ])),
        "_customEvents": [],
    }

    events = get_lighting_events()

    add_squares(walls, custom_data)

    for model_info in MODELS:
        model = load_model(model_info[0])
        add_model(walls, notes, custom_data, model, model_info)

    for svg_model in SVGs:
        add_image(walls, svg_model)

    add_hardcoded_tracks(custom_data)
    add_sea(walls, custom_data)
    add_black_hole(walls, custom_data, notes)
    add_chimney_smoke(walls)
    add_rocket_fire(walls)
    add_stars(walls)
    add_bh_dust(walls)
    add_bh_walls(walls)
    add_bh_walls_slow_part(walls)
    add_text(walls)

    add_notes(input_json["WIP"]["_notes"], notes, custom_data)
    add_eruption(notes)

    walls.sort(key=lambda x: x["_time"])
    notes.sort(key=lambda x: x["_time"])
    custom_data["_customEvents"].sort(key=lambda x: x["_time"])

    # Prevent MM from overwriting info.dat
    shutil.copyfile("info.json", "info.dat")

    for filename in OUTPUT_FILES:
        song_json = copy.deepcopy(input_json["template"])

        song_json["_obstacles"] = trunc(walls)
        song_json["_customData"] = trunc(custom_data)
        song_json["_events"] = trunc(events)
        song_json["_notes"] = trunc(notes)

        with open("{}.dat".format(filename), "w") as json_file:
            json.dump(song_json, json_file)


main()
