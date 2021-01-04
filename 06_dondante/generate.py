from collections import defaultdict, OrderedDict
import copy
import functools
import itertools
import json
import math
import numpy as np
from scipy.spatial.transform import Rotation
import shutil
from svg.path import parse_path, Line, CubicBezier
import xml.etree.ElementTree as ET
import imageio


OUTPUT_FILES = [("ExpertStandard", True)]
INPUT_FILES = ["template", "solo_patterns", "intro_patterns", "final_patterns"]

BPM = 138
BASE_HJ = 2
NS = {"xmlns": "http://www.collada.org/2005/11/COLLADASchema"}
SNS = {"xmlns": "http://www.w3.org/2000/svg"}
NODE_EXCLUSION = ["empty", "metarig", "armature", "camera"]
DEFAULT_COL_INFO = [0.5, 0.5, 0.5, 1, 0, 0.5]

TREE_DEPTH = 30
TREE_X = -5
TREE_SCALE = 0.6
TREE_START, TREE_END = 6, 126

TAP_START, TAP_END = 10, 126
TAP_POS = [8, 7, 12]

FIRE_START, FIRE_END = 10, 126
FIRE_POS = [-6, 8]

# name, bi, bf, px, py, pz, rx, ry, rz, ax, ay, az, sx, sy, sz, spo_axis, spo_speed
MODELS = [
    ["tree", TREE_START, TREE_END, -TREE_DEPTH, TREE_X, 0, 0, 0,
        0, 0, 0, 0, TREE_SCALE, TREE_SCALE, TREE_SCALE, 2, 40],
    ["rocks", TAP_START, TAP_END, -TAP_POS[2], TAP_POS[0],
        0, 0, 0, 0, 0, 0, 0, 1, 1, 1, None, None],
    ["firewood", FIRE_START-1, FIRE_END, -FIRE_POS[1],
        FIRE_POS[0], 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, None, None],
    ["tent", FIRE_START-1, FIRE_END, -FIRE_POS[1] - 3,
        FIRE_POS[0] - 3, 0, 0, 0, 60, 0, 0, 0, 1, 1, 1, None, None],
    ["waterman", 13, 32, -TAP_POS[2], TAP_POS[0],
        0, 0, 0, 0, 0, 0, 0, 1, 1, 1, None, None],
    ["fireman", 17, 32, -FIRE_POS[1], FIRE_POS[0], 0,
        0, 0, -20, 0, 0, 0, 1.5, 1.5, 1.5, None, None],
    ["paris_street", 126, 271, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 20],
    ["street_guitarist", 129, 268, -7, 4, 0, 0,
        0, -45, 0, 0, 0, 1.3, 1.3, 1.3, 2, 10],
    ["you", 261.8, 263.8, -10, -4.5, 3, 0, 0, 0, 0, 0, 0, 1, 1, 1, None, None],
    ["had", 263.8, 265.8, -10, 0, 3, 0, 0, 0, 0, 0, 0, 1, 1, 1, None, None],
    ["me", 265.8, 267.8, -10, 4.5, 3, 0, 0, 0, 0, 0, 0, 1, 1, 1, None, None],
    ["worried", 267.4, 272.4, -10, 0, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, None, None],
    ["skeleton_reaper", 401.4, 411, -42, 0, 0,
        0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 10],
]

for ay in [-60, -30, 0, 30, 60]:
    for y in [-3, 1, 5, 9]:
        bi = np.random.uniform(556, 578)
        bf = bi + np.random.uniform(8, 12)
        yo = np.random.normal(0, 1)
        z = np.random.normal(15, 3)
        ayo = np.random.normal(0, 8)
        model = np.random.choice(["eyes", "eyes2"])
        scale = np.random.uniform(0.8, 1.2)
        MODELS.append([model, bi, bf, -z, 0, y + yo, 0, 0, 0, 0,
                       ay + ayo, 0, scale, scale, scale, None, None])

yellow = [1, 1, 0, 0]
red = [1, 0, 0, 0]

# start, note_shape, note_pitch, note_octave, color
GUITAR_SOLO = [
    (136, "crotchet", "si", 2, red),
    (138, "crotchet", "sol", 2, yellow),
    (140, "crotchet", "re", 1, red),

    (143, "quaver", "la", 2, red),
    (143.625, "quaver", "do", 3, yellow),
    (144.625, "quaver", "do", 3, yellow),
    (146, "crotchet", "re", 3, red),

    (151, "quaver", "la", 2, red),
    (151.625, "quaver", "do", 3, yellow),
    (152.6, "quaver", "do", 3, yellow),
    (154, "crotchet", "re", 3, red),

    (156, "quaver", "la", 0, red),
    (156.625, "crotchet", "la", 0, red),
    (158.6, "quaver", "mi", 2, yellow),
    (159, "quaver", "sol", 2, red),
    (159.6, "crotchet", "la", 2, yellow),

    (165, "quaver", "si", 1, red),
    (165.625, "quaver", "mi", 2, yellow),
    (167, "quaver", "do", 2, red),
    (167.625, "quaver", "mi", 2, yellow),
    (168.1, "quaver", "re", 2, red),
    (168.625, "quaver", "do", 2, yellow),
    (169, "quaver", "la", 1, red),
    (169.67, "quaver", "la", 1, red),
    (170.625, "quaver", "do", 2, yellow),
    (171, "quaver", "la", 1, red),

    (175, "quaver", "la", 2, red),
    (175.625, "quaver", "do", 3, yellow),
    (176.625, "quaver", "do", 3, yellow),
    (178.1, "quaver", "re", 3, red),

    (180, "crotchet", "re", 1, red),
    (180.625, "quaver", "re", 1, red),
    (183.1, "quaver", "la", 2, yellow),
    (183.625, "quaver", "do", 3, red),
    (184.1, "crotchet", "re", 3, yellow),

    (188.1, "quaver", "la", 0, red),
    (188.625, "crotchet", "la", 0, red),
    (190.625, "quaver", "mi", 2, yellow),
    (191.1, "quaver", "sol", 2, red),
    (191.625, "quaver", "la", 2, yellow),
    (192.1, "crotchet", "sol", 2, red),
    (194.625, "quaver", "do", 2, yellow),
    (195.1, "quaver", "re", 2, red),
    (195.625, "quaver", "mi", 2, yellow),
    (196.625, "quaver", "do", 2, red),
    (197.1, "quaver", "la", 1, yellow),
    (197.625, "quaver", "sol", 1, red),
    (198, "quaver", "la", 1, yellow),
    (198.625, "quaver", "do", 2, red),
    (199.05, "quaver", "la", 1, yellow),
    (199.33, "quaver", "sol", 1, red),
    (199.67, "quaver", "la", 1, yellow),
    (200.625, "quaver", "do", 2, red),
    (201, "quaver", "la", 1, yellow),
    (201.33, "quaver", "sol", 1, red),
    (201.67, "quaver", "la", 1, yellow),
    (202.625, "quaver", "do", 2, red),
    (203, "quaver", "la", 1, yellow),
    (203.33, "quaver", "sol", 1, red),
    (203.67, "quaver", "la", 1, yellow),

    (204, "quaver", "re", 1, red),
    (205, "quaver", "fa", 3, yellow),
    (206, "quaver", "mi", 3, red),
    (207, "quaver", "la", 2, yellow),
    (207.5, "quaver", "do", 3, red),
    (208.625, "quaver", "la", 2, yellow),
    (209, "quaver", "do", 3, red),
    (210, "crotchet", "re", 3, yellow),

    (212, "quaver", "re", 1, red),
    (213, "quaver", "fa", 3, yellow),
    (214, "quaver", "mi", 3, red),
    (215, "quaver", "la", 2, yellow),
    (215.5, "quaver", "do", 3, red),
    (216.625, "quaver", "la", 2, yellow),
    (217, "quaver", "do", 3, red),
    (218, "crotchet", "re", 3, yellow),

    (220, "quaver", "la", 0, red),
    (221, "quaver", "fa", 3, yellow),
    (222, "quaver", "mi", 3, red),
    (223, "quaver", "la", 2, yellow),
    (223.5, "quaver", "do", 3, red),
    (224.625, "quaver", "la", 2, yellow),
    (225, "quaver", "do", 3, red),
    (226, "crotchet", "re", 3, yellow),

    (228, "quaver", "la", 0, red),
    (229, "quaver", "fa", 3, yellow),
    (230, "quaver", "mi", 3, red),
    (231, "quaver", "la", 2, yellow),
    (231.5, "quaver", "do", 3, red),
    (232.625, "quaver", "la", 2, yellow),
    (233, "quaver", "do", 3, red),
    (234, "crotchet", "re", 3, yellow),

    (236, "quaver", "re", 1, red),
    (237, "quaver", "fa", 3, yellow),
    (238, "quaver", "mi", 3, red),
    (239, "quaver", "la", 2, yellow),
    (239.625, "quaver", "do", 3, red),
    (240.625, "quaver", "la", 2, yellow),
    (241, "quaver", "do", 3, red),
    (242, "crotchet", "re", 3, yellow),

    (244, "quaver", "re", 1, red),
    (245, "quaver", "fa", 3, yellow),
    (246, "quaver", "mi", 3, red),
    (247, "quaver", "la", 2, yellow),
    (247.625, "quaver", "do", 3, red),
    (248.625, "quaver", "la", 2, yellow),
    (249, "quaver", "do", 3, red),
    (250, "quaver", "la", 2, yellow),
    (250.625, "quaver", "do", 3, red),

    (252, "quaver", "la", 0, red),
    (253, "quaver", "fa", 3, yellow),
    (254, "quaver", "mi", 3, red),
    (255.5, "quaver", "mi", 3, red),
    (256.67, "quaver", "sol", 3, yellow),
    (257.33, "quaver", "sol", 3, yellow),
    (258, "quaver", "sol", 3, yellow),
    (258.67, "quaver", "la", 3, red),
    (259.33, "quaver", "la", 3, red),
    (260, "quaver", "la", 3, red),
    (260.67, "quaver", "la", 3, red),
    (261.33, "quaver", "la", 3, red),
    (262, "crotchet", "la", 3, red),
]

# Memorize walls of the skeleton reaper for cool transition
reaper_walls = []


def beat2time(bpmchanges, beat):
    """The BPM of the song is very unstable. I kinda tried to solve it with a
    lot of BPM changes (and ended up mapping some bits by ear anyway...)
    """
    t1 = 0
    b1 = 0
    bpm = BPM
    for i in range(len(bpmchanges)):
        t0 = t1
        b0 = b1
        t1 = bpmchanges[i]["_time"]
        b1 = b0 + (bpm / BPM) * (t1 - t0)
        bpm = bpmchanges[i]["_BPM"]
        if b1 > beat:
            break
    return t0 + (BPM / bpm) * (beat - b0)


def trunc(obj, precision=4):
    if isinstance(obj, float):
        return round(obj, precision)
    elif isinstance(obj, dict):
        return dict((k, trunc(v)) for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        return list(map(trunc, obj))
    return obj


def lerp(a, b, t):
    if t <= 0:
        return a
    elif t >= 1:
        return b
    else:
        return (1 - t) * a + t * b


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


def add_model(walls, model, info, bpmchanges):
    model_name, bi, bf, px, py, pz, rx, ry, rz, ax, ay, az, sx, sy, sz, spo_axis, spo_speed = info
    ti, tf = (beat2time(bpmchanges, b) for b in (bi, bf))

    # Additional rotation applied to the object
    add_rotation = np.zeros((4, 4))
    add_rotation[:3, :3] = Rotation.from_euler(
        "xyz", [rx, ry, rz], degrees=True).as_matrix()
    add_rotation[3, 3] = 1.0

    cnt = 0
    prev = -1
    for transforms, col_infos, name in model:
        if name.lower().startswith("note"):
            continue

        progress = (100 * cnt) // len(model)
        if progress > prev:
            print(f"Generating {model_name} - {progress}%", end="\r")
        prev = progress
        cnt += 1

        definite_positions = []
        local_rotations = []
        scales = []
        colors = []
        dissolves = [[0, 0]]
        dt = 1 / max(1, len(transforms) - 1)
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

            pivot_diff = np.array([0, -1, 0]) * scale
            correction = pivot_diff - np.matmul(rotation, pivot_diff)

            new_position = position + \
                np.matmul(rotation, np.array([1, -1, -1]) * scale) + correction

            definite_position = [new_position[1],
                                 new_position[2], -new_position[0]]
            local_rotation = [-euler[1], -euler[2], euler[0]]
            double_scale = [2*scale[1], 2*scale[2], 2*scale[0]]

            col_info = col_infos[i]
            col_coef = 1 + 100 * col_info[4]
            alpha = 10 * (col_info[5] - 0.5)
            color = [
                col_coef * col_info[0],
                col_coef * col_info[1],
                col_coef * col_info[2],
                alpha
            ]
            dissolve = [col_info[3]]

            if i == 0:
                if spo_speed is None:
                    apparition_offset = 0
                    apparition_offset_pts = 0
                else:
                    apparition_offset = abs(position[spo_axis]) / spo_speed
                    apparition_offset_pts = apparition_offset / (tf - ti)

            for value, values, vlen in [(definite_position, definite_positions, 3),
                                        (local_rotation, local_rotations, 3),
                                        (double_scale, scales, 3),
                                        (color, colors, 4),
                                        (dissolve, dissolves, 1)]:
                tt = max(0, i * dt - apparition_offset_pts)
                if i < 2 or values[-1][:vlen] != value \
                   or values[-1][:vlen] != values[-2][:vlen]:
                    values.append([*value, tt])
                else:
                    values[-1] = [*value, tt]

        nj_offset = (tf - ti) / 2 - BASE_HJ

        if model_name == "skeleton_reaper":
            reaper_walls.append({
                "position": definite_positions[0][:3],
                "rotation": local_rotations[0][:3],
                "scale": scales[0][:3],
            })

        walls.append({
            "_time": ti + BASE_HJ + nj_offset + apparition_offset,
            "_duration": 0,
            "_lineIndex": 0,
            "_type": 0,
            "_width": 0,
            "_customData": {
                "_interactable": False,
                "_noteJumpStartBeatOffset": nj_offset,
                "_position": [0, 0],
                "_scale": scales[0][:3],
                "_rotation": [ax, ay, az],
                "_animation": {
                    "_definitePosition": definite_positions,
                    "_localRotation": local_rotations,
                    "_color": colors,
                    "_dissolve": dissolves,
                }
            }
        })
    print(f"Generated {model_name}       ")


def add_guitar_solo_notes(walls, bpmchanges):
    pitches = ["do", "re", "mi", "fa", "sol", "la", "si"]
    note_heights = {pitches[i]: i for i in range(7)}

    spawn_point = np.array([-6.5, 3, 1.5])
    target_point = np.array([-7, 0, 1.5])
    target_sigma = 0.1
    spawn_height_tone_offset = 0.05
    half_note_range = 12

    # explosion_time = 502
    note_lifetime = 3

    note_models = {}  # memoization of the models

    prev = -1
    cnt = 0
    for bi, note_shape, note_pitch, note_octave, color in GUITAR_SOLO:
        progress = (100 * cnt) // len(GUITAR_SOLO)
        if progress > prev:
            print(f"Generating solo notes - {progress}%", end="\r")
        prev = progress
        cnt += 1

        if note_shape not in note_models:
            note_models[note_shape] = load_model(note_shape)
        model = note_models[note_shape]

        bf = bi + note_lifetime
        ti, tf = (beat2time(bpmchanges, b) for b in (bi, bf))

        target_height_bonus = (
            (7 * note_octave + note_heights[note_pitch] - half_note_range)
            * spawn_height_tone_offset)
        target_point_final = (
            target_point + np.random.normal(0.0, target_sigma, 3)
            + np.array([0, 0, target_height_bonus]))

        waypoint_info = [
            [*spawn_point, 0],
            [*target_point_final, 1],
        ]

        for transforms, _, _ in model:
            definite_positions = []
            local_rotations = []
            for px, py, pz, tt in waypoint_info:
                transform = (
                    transforms[0] + np.array(
                        [[0, 0, 0, px],
                         [0, 0, 0, py],
                         [0, 0, 0, pz],
                         [0, 0, 0, 0]]
                    )
                )

                position = transform[: 3, 3]
                scale = np.array([np.linalg.norm(transform[:3, i])
                                  for i in range(3)])
                rotation = transform[: 3, : 3] / scale
                euler = Rotation.from_matrix(
                    rotation).as_euler('xyz', degrees=True)

                pivot_diff = np.array([0, -1, 0]) * scale
                correction = pivot_diff - np.matmul(rotation, pivot_diff)

                new_position = position + \
                    np.matmul(rotation, np.array(
                        [1, -1, -1]) * scale) + correction

                definite_positions.append(
                    [new_position[1], new_position[2], -new_position[0], tt])
                local_rotations.append(
                    [-euler[1], -euler[2], euler[0], tt])
                double_scale = [2*scale[1], 2*scale[2], 2*scale[0]]

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
                    "_scale": double_scale,
                    "_animation": {
                        "_definitePosition": definite_positions,
                        "_localRotation": local_rotations,
                        "_dissolve": [
                            [0, 0],
                            [1, 0],
                        ],
                        "_color": [
                            [*color, 0, 0],
                            [*color, 0, 0.5],
                            [0, 0, 0, 0, 1],
                        ]
                    }
                }
            })
    print("Generated solo notes       ")


def add_city_ground(walls, bpmchanges):
    # bi, bf, x, alpha, fadein, fadeout
    lines = [
        [4, 269, -2, 0.5, 0, 1],
        [4.5, 269, 2, 0.5, 0, 1],
        [7, 269, 1, 0.3, 0, 1],
        [7.5, 269, -1, 0.3, 0, 1],
        [6, 269, -8, 0.2, 0, 1.5],
        [10, 269, 8, 0.2, 0, 1.5],
        [6, 269, -14, 0.1, 0, 2],
        [10, 269, 14, 0.1, 0, 2],
    ]
    sections_alpha = [1, 0.9, 0.8, 0.7, 0.6]
    n_sections = len(sections_alpha)
    depth_start = 1
    depth_end = 61
    section_depth = (depth_end - depth_start) / n_sections
    thickness = 0.002

    for bi, bf, x, alpha, fadein, fadeout in lines:
        for i in range(n_sections):
            ai = alpha * sections_alpha[i]
            d_start = depth_start + i * section_depth
            d_end = d_start + section_depth
            ti, tf = (beat2time(bpmchanges, b) for b in (bi, bf))
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
                    "_scale": [thickness, thickness, d_end-d_start],
                    "_animation": {
                        "_definitePosition": [
                            [x - thickness / 2, 0, d_start, 0]
                        ],
                        "_color": [
                            [0, 0, 0, 0, 0],
                            [ai, ai, ai, 0, fadein / (tf - ti)],
                            [ai, ai, ai, 0, 1 - fadeout / (tf - ti)],
                            [0, 0, 0, 0, 1]
                        ],
                        "_dissolve": [
                            [0, 0],
                            [1, 0]
                        ],
                    }
                }
            })
    print("Generated city ground lines")


def add_solo_patterns(in_json, out_notes):
    in_notes = in_json["_notes"]

    clone_fields = ["_lineIndex", "_lineLayer", "_type", "_cutDirection"]
    starts = [0.3, 0.3, 0.325, 0.325, 0.35]

    colors = [[1, 0, 0], [0.8, 0.8, 0]]

    for i in range(len(in_notes)):
        in_note = in_notes[i]
        out_note = {key: in_note[key] for key in clone_fields}
        out_note["_time"] = in_note["_time"]

        if i % 4 < 2:
            out_note["_customData"] = {
                "_fake": True,
                "_interactable": False,
                "_disableSpawnEffect": True,
                "_noteJumpStartBeatOffset": 0,
                "_color": colors[in_note["_type"]],
                "_animation": {
                    "_definitePosition": [
                        [0, 0, 5, 0]
                    ],
                    "_dissolve": [
                        [0, 0]
                    ],
                    "_dissolveArrow": [
                        [0, 0.5],
                        [1, 0.5],
                        [1, 0.65],
                        [0, 0.65],
                    ]
                }
            }
        else:
            start = starts[min(i // 4, len(starts)-1)]
            out_note["_customData"] = {
                "_disableSpawnEffect": True,
                "_noteJumpStartBeatOffset": 0,
                "_color": colors[in_note["_type"]],
                "_animation": {
                    "_definitePosition": [
                        [0, 0, 5, start],
                        [0, 0, 0, 0.5]
                    ],
                    "_dissolve": [
                        [0, 0],
                    ],
                    "_dissolveArrow": [
                        [0, start],
                        [1, start],
                        [1, 0.5],
                        [0, 0.5]
                    ]
                }
            }
        out_notes.append(out_note)


def add_solo_endkicks(notes, bpmchanges):
    # beat, x, y, color
    infos = [
        [260, -1.5, 1, 0],
        [261, 0.5, 1, 1],
    ]
    thetas = [90, 210, 330]
    depth0 = 6
    depth = 0

    colors = [[1, 0, 0], [1, 1, 0]]

    for beat, x, y, color in infos:
        corrected_beat = beat2time(bpmchanges, beat)

        fake = False
        for theta in thetas:
            ux = math.cos(math.radians(theta))
            uy = math.sin(math.radians(theta))
            xi, yi = (x + 1.5 * ux, y + 1.5 * uy)
            notes.append({
                "_lineIndex": 0,
                "_lineLayer": 0,
                "_type": color,
                "_cutDirection": 8,
                "_time": corrected_beat,
                "_customData": {
                    "_fake": fake,
                    "_interactable": not fake,
                    "_disableSpawnEffect": True,
                    "_noteJumpStartBeatOffset": 0,
                    "_position": [0, 0],
                    "_color": colors[color],
                    "_animation": {
                        "_scale": [
                            [0.1, 0.1, 0.1, 0],
                            [0.7, 0.7, 0.7, 0.5],
                        ],
                        "_definitePosition": [
                            [xi, yi, depth0, 0],
                            [x, y, depth, 0.5],
                            [x, y, 2*depth - depth0, 1],
                        ],
                        "_dissolve": [
                            [0, 0],
                            [1, 0],
                        ],
                        "_dissolveArrow": [
                            [0, 0],
                        ]
                    }
                }
            })
            if fake:
                notes[-1]["_customData"]["_animation"]["_dissolve"] += [
                    [1, 0.5],
                    [0, 0.5],
                ]
            fake = True


def add_intro_patterns(in_json, notes, walls, bpmchanges):
    n_notes = len(in_json["_notes"])
    tree_model = load_model("tree")
    notes_obj = sorted([wall for wall in tree_model
                        if wall[2].lower().startswith("note")],
                       key=lambda x: x[2])[n_notes-1::-1]

    prev = -1
    for cnt in range(n_notes):
        in_note = in_json["_notes"][cnt]

        progress = (100 * cnt) // n_notes
        if progress > prev:
            print(f"Generating intro blocks - {progress}%", end="\r")
        prev = progress

        transforms, *_ = notes_obj[cnt]

        px, py, pz = -TREE_DEPTH, TREE_X, 0
        sx, sy, sz = [TREE_SCALE]*3
        ti = beat2time(bpmchanges, TREE_START)
        tf = in_note["_time"] - 4
        tm = beat2time(bpmchanges, TREE_END)
        n_rectified = int(len(transforms) * (tf - ti) / (tm - ti))

        # Add random local rotation to the notes for variation
        random_rotation = Rotation.from_euler(
            "xyz", np.random.normal(0, 20, 3), degrees=True).as_matrix()

        definite_positions = []
        local_rotations = []
        dt = 1 / n_rectified
        for i in range(n_rectified):
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
            t_mat = np.matmul(add_position, rescale)
            transform = np.matmul(t_mat, transforms[i])

            position = transform[: 3, 3]
            scale = np.array([np.linalg.norm(transform[:3, i])
                              for i in range(3)])
            rotation = transform[: 3, : 3] / scale
            rotation = np.matmul(random_rotation, rotation)
            euler = Rotation.from_matrix(
                rotation).as_euler('xyz', degrees=True)

            pivot_diff = np.array([0, -1, 0]) * scale
            correction = pivot_diff - np.matmul(rotation, pivot_diff)

            new_position = position + \
                np.matmul(rotation, np.array([1, -1, -1]) * scale) + correction

            definite_position = [new_position[1],
                                 new_position[2] - 1, -new_position[0]]
            local_rotation = [-euler[1], -euler[2], euler[0]]

            if i == 0:
                spo_speed = 40
                spo_axis = 2
                apparition_offset = abs(position[spo_axis]) / spo_speed
                apparition_offset_pts = apparition_offset / (tf - ti)

            tt = max(0, i * dt - apparition_offset_pts)
            for value, values, vlen in [(definite_position, definite_positions, 3),
                                        (local_rotation, local_rotations, 3)]:
                if i < 2 or values[-1][:vlen] != value \
                   or values[-1][:vlen] != values[-2][:vlen]:
                    values.append([*value, tt])
                else:
                    values[-1] = [*value, tt]

        # I use two different blocks to avoid a bug that seems to happen
        # because of the long lifetime of a single-block solution (and to
        # reduce the number of objects alive at the same time)
        # - The first block is for display in the tree
        # - The second block is the one that the player will actually hit

        nj_offset = (tf - ti) / 2 - BASE_HJ
        notes.append({
            "_time": ti + BASE_HJ + nj_offset + apparition_offset,
            "_lineIndex": 0,
            "_lineLayer": 0,
            "_type": in_note["_type"],
            "_cutDirection": in_note["_cutDirection"],
            "_customData": {
                "_fake": True,
                "_disableSpawnEffect": True,
                "_position": [0, 0],
                "_noteJumpStartBeatOffset": nj_offset,
                "_animation": {
                    "_definitePosition": definite_positions,
                    "_localRotation": local_rotations,
                    "_dissolve": [
                        [0, 0],
                        [1, 0],
                    ],
                    "_dissolveArrow": [
                        [0, 0],
                    ],
                }
            }
        })

        note_x, note_y = in_note["_lineIndex"] - 2, in_note["_lineLayer"]
        nj_offset = 5 - BASE_HJ
        notes.append({
            "_time": in_note["_time"],
            "_lineIndex": in_note["_lineIndex"],
            "_lineLayer": in_note["_lineLayer"],
            "_type": in_note["_type"],
            "_cutDirection": in_note["_cutDirection"],
            "_customData": {
                "_disableSpawnEffect": True,
                "_noteJumpStartBeatOffset": nj_offset,
                "_animation": {
                    "_scale": [
                        [1, 1, 1, 0],
                        [.8, .8, .8, 0.5],
                    ],
                    "_definitePosition": [
                        [definite_positions[-1][0] - note_x,
                         definite_positions[-1][1] - note_y,
                         definite_positions[-1][2], 0.1],
                        [0, 0, 20, 0.3],
                        [0, 0, 0, 0.5],
                        [0, 0, -50, 1],
                    ],
                    "_localRotation": [
                        [*local_rotations[-1][:3], 0],
                    ],
                    "_dissolve": [
                        [0, 0.1],
                        [1, 0.1],
                    ],
                    "_dissolveArrow": [
                        [0, 0],
                    ],
                }
            }
        })

    print("Generated intro blocks        ")


def add_waterfall(walls, bpmchanges):
    n_cubes = 18
    cube_duration = 4
    n_repet = int((TAP_END - cube_duration - TAP_START) / cube_duration)
    for cnt in range(n_cubes):
        progress = (100 * cnt) // n_cubes
        print(f"Generating waterfall - {progress}%", end="\r")

        cube_time = beat2time(bpmchanges, TAP_START +
                              cube_duration * cnt / n_cubes)
        nj_offset = (beat2time(bpmchanges, TAP_END - cube_duration)
                     - beat2time(bpmchanges, TAP_START)) / 2 - BASE_HJ
        scale = [np.random.uniform(0.3, 0.6)] * 3

        definite_positions = []
        local_rotations = []
        dissolves = []
        ti = 0
        dt = 1 / n_repet
        for _ in range(n_repet):
            tf = ti + dt
            loc_rot = np.random.uniform(0, 45, 3)
            pos_off = np.array([-2, -1]) + np.random.normal(0, 0.6, 2)
            definite_positions.extend([
                [*TAP_POS, ti],
                [TAP_POS[0] + pos_off[0], 0, TAP_POS[2] + pos_off[1], tf],
            ])
            local_rotations.extend([
                [loc_rot[0] / 2, loc_rot[1] / 2, loc_rot[2] / 2, ti],
                [*loc_rot, tf],
            ])
            dissolves.extend([
                [0, ti],
                [1, lerp(ti, tf, 0.07)],
                [1, lerp(ti, tf, 0.9)],
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
                "_color": [0.2, 0.8, 1, 0],
                "_scale": scale,
                "_animation": {
                    "_definitePosition": definite_positions,
                    "_localRotation": local_rotations,
                    "_dissolve": dissolves,
                }
            }
        })
    print("Generated waterfall        ")


def add_fire(walls, bpmchanges):
    n_cubes = 14
    cube_duration = 4
    n_repet = int((TAP_END - cube_duration - TAP_START) / cube_duration)
    for cnt in range(n_cubes):
        progress = (100 * cnt) // n_cubes
        print(f"Generating fire - {progress}%", end="\r")

        cube_time = beat2time(bpmchanges, FIRE_START +
                              cube_duration * cnt / n_cubes)
        nj_offset = (beat2time(bpmchanges, FIRE_END - cube_duration)
                     - beat2time(bpmchanges, FIRE_START)) / 2 - BASE_HJ
        scale = [np.random.uniform(0.25, 0.5)] * 3

        definite_positions = []
        local_rotations = []
        dissolves = []
        scales = []
        colors = []
        ti = 0
        dt = 1 / n_repet
        for _ in range(n_repet):
            tf = ti + dt
            loc_rot = np.random.uniform(0, 45, 3)
            pos_off_base = np.random.normal(0, 0.4, 2)
            pos_off_top = (0.5 * np.array([math.sin(200 * ti), math.sin(50 * ti)])
                           + np.random.normal(0, 0.2, 2))
            definite_positions.extend([
                [FIRE_POS[0] + pos_off_base[0], 0.5,
                    FIRE_POS[1] + pos_off_base[1], ti],
                [FIRE_POS[0] + pos_off_top[0], 2.5,
                    FIRE_POS[1] + pos_off_top[1], tf],
            ])
            local_rotations.extend([
                [loc_rot[0] / 2, loc_rot[1] / 2, loc_rot[2] / 2, ti],
                [*loc_rot, tf],
            ])
            dissolves.extend([
                [0, ti],
                [1, lerp(ti, tf, 0.05)],
                [1, lerp(ti, tf, 0.7)],
                [0, lerp(ti, tf, 0.95)],
            ])
            scales.extend([
                [1, 1, 1, ti],
                [0.2, 0.2, 0.2, tf],
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
    print("Generated fire        ")


def add_final_patterns(in_json, out_notes):
    colors = [[1, 0.45, 0.2, 1], [0.6, 0.1, 0.6, 1]]
    in_notes = in_json["_notes"]
    clone_fields = ["_lineIndex", "_lineLayer", "_type", "_cutDirection"]
    for in_note in in_notes:
        dissolve = lerp(1, 0.4, (in_note["_time"] - 505) / 30)
        out_note = {key: in_note[key] for key in clone_fields}
        out_note["_time"] = in_note["_time"]
        out_note["_customData"] = {
            "_disableSpawnEffect": True,
            "_noteJumpStartBeatOffset": 0,
            "_color": colors[in_note["_type"]],
            "_animation": {
                "_dissolve": [[0, 0]],
                "_dissolveArrow": [
                    [0, 0],
                    [dissolve, 0]
                ],
            }
        }
        out_notes.append(out_note)


def normalize(x, y, viewbox):
    """Normalize so that the origin is at the bottom center of the image,
    and the width and height of the image are 1
    """
    xi, yi, width, height = viewbox
    return (x - xi - width / 2) / width, (yi + height - y) / height


def load_image(filename, n_div=4):
    root = ET.parse("{}.svg".format(filename)).getroot()
    viewbox = tuple(map(float, root.attrib["viewBox"].split()))
    all_lines = []
    for line in root.findall("xmlns:line", SNS):
        x1 = float(line.attrib["x1"])
        y1 = float(line.attrib["y1"])
        x2 = float(line.attrib["x2"])
        y2 = float(line.attrib["y2"])
        sw = float(line.attrib["stroke-width"]
                   if "stroke-width" in line.attrib else 1)
        x1, y1 = normalize(x1, y1, viewbox)
        x2, y2 = normalize(x2, y2, viewbox)
        all_lines.append((x1, y1, x2, y2, sw))
    for polyline in root.findall("xmlns:polyline", SNS):
        points = [normalize(*map(float, pt.split(',')), viewbox)
                  for pt in polyline.attrib["points"].split()]
        x1, y1 = points[0]
        sw = float(polyline.attrib["stroke-width"]
                   if "stroke-width" in polyline.attrib else 1)
        for x2, y2 in points[1:]:
            all_lines.append((x1, y1, x2, y2, sw))
            x1, y1 = x2, y2
    for polygon in root.findall("xmlns:polygon", SNS):
        points = [normalize(*map(float, pt.split(',')), viewbox)
                  for pt in polygon.attrib["points"].split()]
        points.append(points[0])
        x1, y1 = points[0]
        sw = float(polygon.attrib["stroke-width"]
                   if "stroke-width" in polygon.attrib else 1)
        for x2, y2 in points[1:]:
            all_lines.append((x1, y1, x2, y2, sw))
            x1, y1 = x2, y2
    for path in root.findall("xmlns:path", SNS):
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
                # length = segment.length()
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


def add_stained_glass(walls, custom_events):
    sides = "LR"
    x_list = [-9, 9]
    z_list = [7, 21, 34, 47]
    width = 3
    start_y, end_y = -2, 8

    thickness = 0.05

    burst_times = {
        "L0": 381,
        "R0": 381.6,
        "L1": 382.2,
        "R1": 382.9,
        "L2": 383.55,
        "R2": 384.1,
        "L3": 384.9,
        "R3": 385.85,
    }
    gather_start = 386.92
    gather_end = 390.8

    # Selectors for internal use
    sel_to_tracks = defaultdict(list)
    for side in ("L", "R"):
        for row in range(len(z_list)):
            sel = f"{side}{row}"
            sel_to_tracks[sel].append(sel)
    sel_to_tracks["L"] = [f"L{row}" for row in range(len(z_list))]
    sel_to_tracks["R"] = [f"R{row}" for row in range(len(z_list))]
    sel_to_tracks["S"] = sel_to_tracks["L"] + sel_to_tracks["R"]
    sel_to_tracks["A"] = sel_to_tracks["S"]
    all_tracks = sel_to_tracks["A"]

    # Start and end per track to avoid lag spike
    # Note: using actual times here because there's smth wrong with bpm
    # changes that I couldn't fix
    start = {track: 265 + 1.5*idx/len(all_tracks)
             for idx, track in enumerate(all_tracks)}
    end = {track: 392 + 1.5*idx/len(all_tracks)
           for idx, track in enumerate(all_tracks)}

    # Parse svg files
    window_lines = load_image("window")

    # Windows
    cnt = 0
    k = 0
    for ix, iz in itertools.product(range(2), range(len(z_list))):
        progress = 100 * cnt // (2 * len(z_list))
        cnt += 1
        print(f"Generating windows - {progress}%", end="\r")

        x = x_list[ix]
        z = z_list[iz]

        for line in window_lines:
            track = f"{sides[ix]}{iz}"
            nj_offset = (end[track] - start[track]) / 2 - BASE_HJ

            z1, y1, z2, y2, _ = line
            if y1 > y2:
                z1, y1, z2, y2 = z2, y2, z1, y1
            z1, y1, z2, y2 = width * z1 + \
                z, (end_y-start_y) * y1 + start_y, width * \
                z2 + z, (end_y-start_y) * y2 + start_y

            init_scale = [thickness, math.sqrt(
                (z2 - z1)**2 + (y2 - y1) ** 2), thickness]
            init_position = [x, y1, z1]
            init_rotation = [
                90 - math.degrees(math.atan2(y2 - y1, z2 - z1)), 0, 0]

            burst_position = np.array(
                init_position) + np.array([2, 7, 5]) * np.random.uniform(-1, 1, 3)
            burst_rotation = np.array(
                init_rotation) + np.random.normal(0, 20, 3)

            wall_id = k % len(reaper_walls)
            final_position = reaper_walls[wall_id]["position"]
            final_rotation = reaper_walls[wall_id]["rotation"]
            final_scale_abs = reaper_walls[wall_id]["scale"]
            final_scale_rel = [final_scale_abs[i] / init_scale[i]
                               for i in range(3)]

            burst_t0 = (burst_times[track] - start[track]
                        ) / (end[track] - start[track])
            burst_t1 = (gather_start - start[track]
                        ) / (end[track] - start[track])
            burst_t2 = (gather_end - start[track]) / \
                (end[track] - start[track])

            walls.append({
                "_time": start[track] + BASE_HJ + nj_offset,
                "_duration": 0,
                "_lineIndex": 0,
                "_type": 0,
                "_width": 0,
                "_customData": {
                    "_track": track,
                    "_interactable": False,
                    "_noteJumpStartBeatOffset": nj_offset,
                    "_position": [0, 0],
                    "_color": [0, 0, 0, 0],
                    "_scale": [*init_scale],
                    "_animation": {
                        "_definitePosition": [
                            # [*init_position, 0],
                            [*init_position, burst_t0],
                            [*burst_position, burst_t1, "easeOutCubic"],
                            [*final_position, burst_t2, "easeInCubic"],
                        ],
                        "_scale": [
                            [1, 1, 1, burst_t1],
                            [*final_scale_rel, burst_t2, "easeInQuint"],
                        ],
                        "_localRotation": [
                            # [*init_rotation, 0],
                            [*init_rotation, burst_t0],
                            [*burst_rotation, burst_t1],
                            [*final_rotation, burst_t2, "easeInCubic"],
                        ]
                    }
                }
            })
            k += 1
    print("Generated windows        ")

    purple = [0.6, 0.1, 0.6, 1]
    orange = [1, 0.45, 0.2, 1]
    white1 = [1, 1, 1, 0]
    white2 = [80, 80, 80, 0]

    # time, selector, type, params
    window_events = [
        (267, "L", "set", (purple,)),
        (267, "R", "set", (purple,)),

        (269, "L0", "on", None),
        (269.46, "R0", "on", None),
        (270.9, "L1", "on", None),
        (270.9, "R1", "on", None),
        (271.9, "L2", "on", None),
        (272.3, "R2", "on", None),
        (273.3, "L3", "on", None),
        (273.8, "R3", "on", None),

        (282.6, "L0", "tempset", (orange, 0.4)),
        (282.7, "L1", "tempset", (orange, 0.4)),
        (282.8, "L2", "tempset", (orange, 0.4)),
        (282.9, "L3", "tempset", (orange, 0.4)),
        (284.6, "R0", "tempset", (orange, 0.4)),
        (284.7, "R1", "tempset", (orange, 0.4)),
        (284.8, "R2", "tempset", (orange, 0.4)),
        (284.9, "R3", "tempset", (orange, 0.4)),

        (292.2, "S", "tempset", (orange, 1)),

        (298.05, "L0", "tempset", (orange, 0.4)),
        (298.5, "L1", "tempset", (orange, 0.4)),
        (299, "L2", "tempset", (orange, 0.4)),
        (299.25, "L3", "tempset", (orange, 0.4)),

        (307.7, "S", "tempset", (orange, 1)),

        (313.5, "L0", "tempset", (orange, 0.4)),
        (313.6, "L1", "tempset", (orange, 0.4)),
        (313.7, "L2", "tempset", (orange, 0.4)),
        (313.8, "L3", "tempset", (orange, 0.4)),
        (315.5, "R0", "tempset", (orange, 0.4)),
        (315.6, "R1", "tempset", (orange, 0.4)),
        (315.7, "R2", "tempset", (orange, 0.4)),
        (315.8, "R3", "tempset", (orange, 0.4)),

        (323.2, "S", "tempset", (orange, 1)),

        (331.1, "L0", "tempset", (orange, 0.4)),
        (331.2, "L1", "tempset", (orange, 0.4)),
        (331.3, "L2", "tempset", (orange, 0.4)),
        (331.4, "L3", "tempset", (orange, 0.4)),

        (336.81, "L", "tempset", (orange, 1)),
        (338.87, "R", "tempset", (orange, 1)),

        (344.6, "L0", "tempset", (orange, 0.4)),
        (344.7, "L1", "tempset", (orange, 0.4)),
        (344.8, "L2", "tempset", (orange, 0.4)),
        (344.9, "L3", "tempset", (orange, 0.4)),
        (346.6, "R0", "tempset", (orange, 0.4)),
        (346.7, "R1", "tempset", (orange, 0.4)),
        (346.8, "R2", "tempset", (orange, 0.4)),
        (346.9, "R3", "tempset", (orange, 0.4)),

        (353.9, "S", "tempset", (orange, 1)),

        (359.85, "L0", "tempset", (orange, 0.4)),
        (360.3, "L1", "tempset", (orange, 0.4)),
        (360.8, "L2", "tempset", (orange, 0.4)),
        (361.1, "L3", "tempset", (orange, 0.4)),

        (364.6, "R0", "tempset", (orange, 0.4)),
        (364.7, "R1", "tempset", (orange, 0.4)),
        (364.8, "R2", "tempset", (orange, 0.4)),
        (364.9, "R3", "tempset", (orange, 0.4)),

        (369.5, "S", "tempset", (orange, 1)),

        (377.2, "L0", "tempset", (orange, 0.4)),
        (377.2, "R0", "tempset", (orange, 0.4)),
        (377.2, "L1", "tempset", (orange, 0.4)),
        (377.2, "R1", "tempset", (orange, 0.4)),
        (377.2, "L2", "tempset", (orange, 0.4)),
        (377.2, "R2", "tempset", (orange, 0.4)),
        (377.2, "L3", "tempset", (orange, 0.4)),
        (377.2, "R3", "tempset", (orange, 0.4)),

        (burst_times["L0"], "L0", "set", (purple,)),
        (burst_times["R0"], "R0", "set", (purple,)),
        (burst_times["L1"], "L1", "set", (purple,)),
        (burst_times["R1"], "R1", "set", (purple,)),
        (burst_times["L2"], "L2", "set", (purple,)),
        (burst_times["R2"], "R2", "set", (purple,)),
        (burst_times["L3"], "L3", "set", (purple,)),
        (burst_times["R3"], "R3", "set", (purple,)),

        (burst_times["L0"] + 1, "L0", "set", (white1,)),
        (burst_times["R0"] + 1, "R0", "set", (white1,)),
        (burst_times["L1"] + 1, "L1", "set", (white1,)),
        (burst_times["R1"] + 1, "R1", "set", (white1,)),
        (burst_times["L2"] + 1, "L2", "set", (white1,)),
        (burst_times["R2"] + 1, "R2", "set", (white1,)),
        (burst_times["L3"] + 1, "L3", "set", (white1,)),
        (burst_times["R3"] + 1, "R3", "set", (white1,)),

        (gather_end, "L", "smoothset", (white2,)),
        (gather_end, "R", "smoothset", (white2,)),

        (gather_end, "L", "off", None),
        (gather_end, "R", "off", None),
    ]

    # Create tracks
    colors = defaultdict(list)
    dissolves = defaultdict(list)
    cnt = 0
    for event_time, selector, event_type, params in window_events:
        progress = 100 * cnt // len(window_events)
        cnt += 1
        print(f"Generating window events - {progress}%", end="\r")

        for track in sel_to_tracks[selector]:
            time_point = (event_time - start[track]) / \
                (end[track] - start[track])

            if not colors[track]:
                colors[track].append([0, 0, 0, 0, time_point])
            if not dissolves[track]:
                dissolves[track].append([0, time_point])

            # Color events
            if event_type == "set":
                if colors[track]:
                    colors[track].append(colors[track][-1][:-1] + [time_point])
                colors[track].append([*params[0], time_point])
            elif event_type == "smoothset":
                colors[track].append([*params[0], time_point])
            elif event_type == "tempset":
                first_color = colors[track][-1][:-1]
                second_color, duration = params
                second_time_point = (
                    event_time + duration - start[track]) / (end[track] - start[track])
                colors[track].append([*first_color, time_point])
                colors[track].append([*second_color, time_point])
                colors[track].append([*first_color, second_time_point])

            # Dissolve events
            if event_type == "on":
                if dissolves[track] and dissolves[track][-1][0] != 1:
                    dissolves[track].append(
                        dissolves[track][-1][:-1] + [time_point])
                dissolves[track].append([1, time_point])
            elif event_type == "off":
                if dissolves[track] and dissolves[track][-1][0] != 0:
                    dissolves[track].append(
                        dissolves[track][-1][:-1] + [time_point])
                dissolves[track].append([0, time_point])

    print("Generated window events        ")

    for track in all_tracks:
        custom_events.append({
            "_time": start[track],
            "_type": "AssignPathAnimation",
            "_data": {
                "_track": track,
                "_color": colors[track],
                "_dissolve": dissolves[track],
            }
        })


def add_clock(walls, final_patterns):
    print("Generate clock...", end="\r")

    start, end = 274.8, 398.4
    clock_stop = 394.6
    dt = 0.2
    center = np.array([0, 10, 60])
    small_radius, big_radius = 5, 5.5
    tick_length = 0.5
    tick_thickness = 0.2
    tick_alpha = 1
    hand_lengths = 4, 2
    hand_alphas = 1, 1
    hand_offsets = 0.1, 0.05
    hand_thickness = 0.2
    BASE_PERIOD = (end-start) / 24  # beats/roation
    hand_periods = (BASE_PERIOD, 12 * BASE_PERIOD)

    # Get note timings and remove duplicates
    times = list(OrderedDict.fromkeys(
        trunc((note["_time"] for note in final_patterns["_notes"]), 1)))

    # Add contour
    for i in range(12):
        theta = 90 - 30 * i
        base_position = np.array([math.cos(math.radians(theta)),
                                  math.sin(math.radians(theta)), 0])

        rest_position = center + small_radius * base_position
        displaced_position = center + big_radius * base_position
        burst_position = center + 10 * base_position

        definite_positions = [[*rest_position, 0]]
        for nt in times[i::12]:
            if nt >= clock_stop:
                break
            t0 = (nt - start) / (end - start)
            t1 = (nt + dt - start) / (end - start)
            definite_positions.append([*rest_position, t0])
            definite_positions.append([*displaced_position, t1])
        t0 = (clock_stop - start) / (end - start)
        t1 = (clock_stop + .5 - start) / (end - start)
        definite_positions.append([*rest_position, t0])
        definite_positions.append([*burst_position, t1])

        nj_offset = (end - start) / 2 - BASE_HJ
        walls.append({
            "_time": start + BASE_HJ + nj_offset,
            "_duration": 0,
            "_lineIndex": 0,
            "_type": 0,
            "_width": 0,
            "_customData": {
                "_interactable": False,
                "_noteJumpStartBeatOffset": nj_offset,
                "_position": [0, 0],
                "_scale": [tick_length, tick_thickness, tick_thickness],
                "_localRotation": [0, 0, theta],
                "_color": [tick_alpha, tick_alpha, tick_alpha, 0],
                "_animation": {
                    "_definitePosition": definite_positions,
                    "_dissolve": [
                        [0, 0],
                        [1, 0],
                        [1, t0],
                        [0, t1, "easeInCirc"],
                    ],
                }
            }
        })

    # Add clock hands
    for i in range(2):
        angle = 0
        tt = start
        T = hand_periods[i] / 3
        rotations = []
        while tt < clock_stop:
            rotations.append([0, 0, (-angle) %
                              360, (tt - start) / (end - start)])
            angle = (angle + 120) % 360
            tt += T
        rotations.append([0, 0, (120 * (tt-clock_stop)/T - angle) %
                          360, (clock_stop - start) / (end - start)])

        nj_offset = (end - start) / 2 - BASE_HJ
        walls.append({
            "_time": start + BASE_HJ + nj_offset,
            "_duration": 0,
            "_lineIndex": 0,
            "_type": 0,
            "_width": 0,
            "_customData": {
                "_track": "clock_hands",  # for debug
                "_interactable": False,
                "_noteJumpStartBeatOffset": nj_offset,
                "_position": [0, 0],
                "_scale": [hand_thickness, hand_lengths[i], hand_thickness],
                "_color": [hand_alphas[i], hand_alphas[i], hand_alphas[i], 0],
                "_animation": {
                    "_definitePosition": [
                        [*(center - np.array([0, 0, hand_offsets[i]])), 0]
                    ],
                    "_localRotation": rotations,
                    "_dissolve": [
                        [0, 0],
                        [1, 0],
                    ],
                }
            }
        })
    print("Generated clock  ")


def add_final_lines(walls):
    lines_x1 = [-2, 2]
    lines_x2 = [-16, 16]
    start_z, end_z = 1, 42
    start = 267
    prop_start, prop_end = 398.4, 399

    for i in range(2):
        nj_offset = (prop_end - start) / 2 - BASE_HJ
        t1 = (prop_start - start) / (prop_end - start)
        walls.append({
            "_time": start + BASE_HJ + nj_offset,
            "_duration": 0,
            "_lineIndex": 0,
            "_type": 0,
            "_width": 0,
            "_customData": {
                "_interactable": False,
                "_noteJumpStartBeatOffset": nj_offset,
                "_position": [0, 0],
                "_scale": [0.05, 0.05, end_z - start_z],
                "_color": [0.7, 0.7, 0.7, 0],
                "_animation": {
                    "_definitePosition": [
                        [lines_x1[i], 0, start_z, 0],
                        [lines_x1[i], 0, start_z, t1],
                        [lines_x2[i], 0, start_z, 1],
                    ],
                    "_dissolve": [
                        [0, 0],
                        [1, 0],
                        [1, t1],
                        [0, 1, "easeInCirc"],
                    ],
                }
            }
        })


def add_cemetary_crosses(walls):
    x_right = [3, 9, 15, 21]
    x_list = x_right + list(map(lambda x: -x, x_right))
    z_list = list(range(2, 82, 10))

    prop_start, prop_end = 398.4, 399
    disa_start, disa_end = 536.8, 537.2

    wall_positions = [
        [-.05, 0, 0],
        [-.5, 1.1, 0],
    ]
    wall_scales = [
        [.1, 1.6, .1],
        [1, .1, .1],
    ]

    n_pts = 14

    cnt = 0
    for x, z in itertools.product(x_list, z_list):
        progress = 100 * cnt // (len(x_list)*len(z_list))
        cnt += 1
        print(f"Generating crosses - {progress}%", end="\r")

        start = lerp(prop_start, prop_end, abs(x) / max(x_list))

        pt_timings = np.r_[0:n_pts] / (n_pts - 1)
        pt_timings[1:-
                   1] += np.random.uniform(-0.25 / n_pts, 0.25 / n_pts, n_pts - 2)
        pt_timings[0] = 0
        pt_timings[-1] = 1
        rotations = np.random.normal(0., 4., n_pts)
        rotations[0] = 0

        for i in range(2):
            c = 1. / (1. + abs(x) / 5 + z / 20)
            nj_offset = (disa_end - start) / 2 - BASE_HJ
            positions = [
                [*(np.matmul(Rotation.from_euler("z", rotations[j],
                                                 degrees=True).as_matrix(),
                             np.array(wall_positions[i]))
                   + np.array([x, 0, z])),
                 pt_timings[j]]
                for j in range(n_pts)
            ]
            walls.append({
                "_time": start + BASE_HJ + nj_offset,
                "_duration": 0,
                "_lineIndex": 0,
                "_type": 0,
                "_width": 0,
                "_customData": {
                    "_track": f"cross{cnt}",
                    "_interactable": False,
                    "_noteJumpStartBeatOffset": nj_offset,
                    "_position": [0, 0],
                    "_scale": wall_scales[i],
                    "_color": [c, c, c, 0],
                    "_animation": {
                        "_definitePosition": positions,
                        "_localRotation": [
                            [0, 0, rotations[j], pt_timings[j]]
                            for j in range(n_pts)
                        ],
                        "_dissolve": [
                            [0, 0],
                            [1, 0],
                            [1, (disa_start - start) / (disa_end - start)],
                            [0, 1],
                        ],
                    }
                }
            })
    print("Generated crosses        ")


def add_lightning(walls, final_patterns):
    scale = 40
    thickness = 0.1
    off0 = 0.5
    off1 = 1
    pst = 0.1

    notes = final_patterns["_notes"]
    strikes = [notes[i]["_time"]
               for i in range(len(notes)-1)
               if notes[i]["_time"] > 402
               and abs(notes[i]["_time"] - notes[i+1]["_time"]) < 0.1]

    svg_models = [load_image(f"lightning{i}") for i in range(6)]
    chosen_model_ids = [0]
    for i in range(len(strikes)-1):
        candidate = chosen_model_ids[-1]
        while candidate == chosen_model_ids[-1]:
            candidate = np.random.choice(len(svg_models))
        chosen_model_ids.append(candidate)
    chosen_models = [svg_models[i] for i in chosen_model_ids]

    cnt = 0
    for tt, model in zip(strikes, chosen_models):
        progress = 100 * cnt // len(strikes)
        cnt += 1
        print(f"Generating lightning - {progress}%", end="\r")

        x = np.random.uniform(-25, 25)
        z = np.random.normal(70, 5)
        n_flicker = np.random.randint(2, 4)
        off0 = np.random.normal(0.4, 0.05)
        xs = np.random.choice([-1, 1])
        for line in model:
            x1, y1, x2, y2, w = line
            x1, x2 = xs*x1, xs*x2
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            t0 = tt - y2 * off0
            t1 = t0 + pst
            t2 = tt + off1
            t3 = t2 + (1-y2) * off0

            # could have ended the secondary arms earlier but there's a weird
            # timing bug...
            start = t0
            end = t3

            white0 = np.array([2, 2, 2, 0], dtype=float)
            black = np.array([0, 0, 0, 0], dtype=float)

            white1 = [200, 200, 200, 1]
            white2 = [1, 1, 1, 1]

            colors = [[*white0, 0]]

            if w == 1:
                colors.append([*black, (t1 - start) / (end - start)])
            else:
                t1_ = min(t1, tt)
                faded = lerp(white0, black, (tt-t0) / pst)
                colors.append([*faded, (t1_ - start) / (end - start)])

                tfd = off1 / n_flicker
                colors.append([*faded, (tt - start) / (end - start)])
                for fi in range(n_flicker):
                    tf0 = tt + fi * tfd
                    tf1 = tt + (fi + 1) * tfd
                    colors += [
                        [*white1, (tf0 - start) / (end - start)],
                        [*white2, (tf1 - start) / (end - start),
                         "easeOutCirc"],
                    ]
                colors += [
                    [*white0, (max(t2, t3-pst) - start) / (end - start)],
                    [*black, (t3 - start) / (end - start)],
                ]

            x1, y1 = x + scale * x1, scale * y1
            x2, y2 = x + scale * x2, scale * y2

            nj_offset = (end - start) / 2 - BASE_HJ
            walls.append({
                "_time": start + BASE_HJ + nj_offset,
                "_duration": 0,
                "_lineIndex": 0,
                "_type": 0,
                "_width": 0,
                "_customData": {
                    "_track": "lightning_main" if w > 1 else "lightning_aux",  # for debug
                    "_interactable": False,
                    "_noteJumpStartBeatOffset": nj_offset,
                    "_position": [0, 0],
                    "_scale": [thickness,
                               math.sqrt((x2 - x1)**2 + (y2 - y1) ** 2),
                               thickness],
                    "_color": [0, 0, 0, 0],
                    "_animation": {
                        "_definitePosition": [
                            [x1, y1, z, 0],
                        ],
                        "_localRotation": [
                            [0,
                             0,
                             math.degrees(math.atan2(y2 - y1, x2 - x1)) - 90,
                             0],
                        ],
                        "_dissolve": [
                            [0, 0],
                            [1, 0],
                        ],
                        "_color": colors,
                    }
                }
            })
            if w > 1:
                tp = (tt - start) / (end - start)
                walls[-1]["_customData"]["_animation"]["_scale"] = [
                    [1, 1, 1, tp],
                    [2, 1, 2, tp],
                ]
    print("Generated lightning        ")


def add_credits(walls):
    lh = [3, 4, 6]

    # time, list[height, text]
    texts = [
        [508, lh[1], "dondante"],
        [518, lh[2], "my_morning"],
        [519, lh[0], "jacket"],
        [528, lh[2], "mapped_by"],
        [529, lh[0], "nyri0"],
    ]

    color = [1, 1, 1, 0]
    text_dist = 35
    scale = 15
    spawn_duration = 1
    thickness = 0.1

    cnt = 0
    for tt, line_height, svg_name in texts:
        progress = 100 * cnt // 5
        cnt += 1
        print(f"Generating credits - {progress}%", end="\r")

        model = load_image(svg_name)

        for line in model:
            x1, y1, x2, y2, _ = line
            x1, x2 = x1, x2
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1

            start = tt + spawn_duration * (x1 + x2 + 1) / 2
            end = start + 8

            x1, y1 = scale * x1, line_height + scale * y1
            x2, y2 = scale * x2, line_height + scale * y2

            nj_offset = (end - start) / 2 - BASE_HJ
            walls.append({
                "_time": start + BASE_HJ + nj_offset,
                "_duration": 0,
                "_lineIndex": 0,
                "_type": 0,
                "_width": 0,
                "_customData": {
                    "_interactable": False,
                    "_noteJumpStartBeatOffset": nj_offset,
                    "_position": [0, 0],
                    "_scale": [thickness,
                               math.sqrt((x2 - x1)**2 + (y2 - y1) ** 2),
                               thickness],
                    "_color": color,
                    "_animation": {
                        "_definitePosition": [
                            [x1, y1, text_dist, 0],
                        ],
                        "_localRotation": [
                            [0,
                             0,
                             math.degrees(math.atan2(y2 - y1, x2 - x1)) - 90,
                             0],
                        ],
                        "_dissolve": [
                            [0, 0],
                            [1, 0],
                        ],
                    }
                }
            })
    print("Generated credits        ")


def main():
    input_json = dict()
    for filename in INPUT_FILES:
        with open("{}.dat".format(filename), "r") as json_file:
            input_json[filename] = json.load(json_file)

    bpmchanges = input_json["template"]["_customData"]["_BPMChanges"]

    notes = []
    walls = []
    custom_data = {
        "_BPMChanges": bpmchanges,
        "_customEvents": [],
    }
    custom_events = custom_data["_customEvents"]

    # Lights off
    events = [{"_time": 0, "_type": i, "_value": 0} for i in range(5)]

    for model_info in MODELS:
        model = load_model(model_info[0])
        add_model(walls, model, model_info, bpmchanges)

    add_city_ground(walls, bpmchanges)
    add_intro_patterns(input_json["intro_patterns"],
                       notes, walls, bpmchanges)
    add_waterfall(walls, bpmchanges)
    add_fire(walls, bpmchanges)

    add_solo_patterns(input_json["solo_patterns"], notes)
    add_guitar_solo_notes(walls, bpmchanges)
    add_solo_endkicks(notes, bpmchanges)

    add_final_patterns(input_json["final_patterns"], notes)
    add_stained_glass(walls, custom_events)
    add_clock(walls, input_json["final_patterns"])
    add_final_lines(walls)
    add_cemetary_crosses(walls)
    add_lightning(walls, input_json["final_patterns"])
    add_credits(walls)

    walls.sort(key=lambda x: x["_time"])
    notes.sort(key=lambda x: x["_time"])

    # Prevent MM from overwriting info.dat
    shutil.copyfile("info.json", "info.dat")

    for filename, has_notes in OUTPUT_FILES:
        song_json = copy.deepcopy(input_json["template"])

        song_json["_obstacles"] = trunc(walls)
        song_json["_customData"] = trunc(custom_data)
        song_json["_events"] = trunc(events)
        if has_notes:
            song_json["_notes"] = trunc(notes)

        with open("{}.dat".format(filename), "w") as json_file:
            json.dump(song_json, json_file)


main()
