from collections import defaultdict
import copy
import functools
import itertools
import json
import math
import numpy as np
from scipy.spatial.transform import Rotation
import shutil
# from svg.path import parse_path, Line, CubicBezier
import xml.etree.ElementTree as ET
import imageio


# (filename, has_notes)
OUTPUT_FILES = [("ExpertPlus", True)]
INPUT_FILES = ["template", "WIP"]

BPM = 115
BASE_HJ = 2
NS = {"xmlns": "http://www.collada.org/2005/11/COLLADASchema"}
# SNS = {"xmlns": "http://www.w3.org/2000/svg"}
NODE_EXCLUSION = ["empty", "metarig", "armature", "camera"]
DEFAULT_COL_INFO = [0.5, 0.5, 0.5, 1, 0, 0.5]

# name, bi, bf, px, py, pz, rx, ry, rz, ax, ay, az, sx, sy, sz, spo_axis,
# spo_speed, cycle_duration, fade_in, fade_out
MODELS = [
    ["muffet", 14, 208, -30, 0, 1, 0, 0, 0, 0,
        0, 0, 1.3, 1.3, 1.3, 2, 15, 2, 1, 1],
]

PURPLE = [0.54, 0.07, 0.51, 0]

# TODO: compress Muffet animation (only write cycle, don't repeat it)


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


def add_model(walls, model, info):
    model_name, ti, tf, px, py, pz, rx, ry, rz, ax, ay, az, sx, sy, sz, \
        spo_axis, spo_speed, cycle_duration, fade_in, fade_out = info

    # Additional rotation applied to the object
    add_rotation = np.zeros((4, 4))
    add_rotation[:3, :3] = Rotation.from_euler(
        "xyz", [rx, ry, rz], degrees=True).as_matrix()
    add_rotation[3, 3] = 1.0

    cnt = 0
    prev = -1
    for transforms, _, _ in model:
        progress = (100 * cnt) // len(model)
        if progress > prev:
            print(f"Generating {model_name} - {progress}%", end="\r")
        prev = progress
        cnt += 1

        positions_cycle = []
        rotations_cycle = []
        scales_cycle = []
        # colors = []
        # dissolves = [[0, 0]]
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

            if i == 0:
                if spo_speed is None:
                    apparition_offset = 0
                    apparition_offset_pts = 0
                else:
                    apparition_offset = abs(position[spo_axis]) / spo_speed
                    apparition_offset_pts = apparition_offset / (tf - ti)

            positions_cycle.append(definite_position)
            rotations_cycle.append(local_rotation)
            scales_cycle.append(double_scale)

        n_cycles = (tf - ti) // cycle_duration
        cycle_len = len(positions_cycle)

        positions = []
        rotations = []
        dt = 1. / ((n_cycles * cycle_len - 1))

        for ic in range(n_cycles):
            for i in range(cycle_len - 1):
                for cycle, out in [(positions_cycle, positions),
                                   (rotations_cycle, rotations)]:
                    tt = max(0, (ic * cycle_len + i) *
                             dt - apparition_offset_pts)
                    if len(out) < 2 or out[-1][:3] != cycle[i] \
                       or out[-1][:3] != out[-2][:3]:
                        out.append([*cycle[i], tt])
                    else:
                        out[-1] = [*cycle[i], tt]

        nj_offset = (tf - ti) / 2 - BASE_HJ

        t_fade_in = max(0, fade_in / (tf - ti) - apparition_offset_pts)
        t_fade_out = min(1, (tf - fade_out - ti) / (tf - ti) -
                         apparition_offset_pts)

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
                "_scale": scales_cycle[0][:3],
                "_color": [80, 80, 80, 0],
                # "_rotation": [ax, ay, az],
                "_animation": {
                    "_definitePosition": positions,
                    "_localRotation": rotations,
                    "_dissolve": [
                        [0, 0],
                        [1, t_fade_in],
                        [1, t_fade_out],
                        [0, 1],
                    ],
                }
            }
        })
    print(f"Generated {model_name}       ")


def add_dancing_spiders(obstacles):
    model = load_model("spidy")

    # ti, tf, a, [frame, time, duration]
    dt = 0.6
    spider_specs = [
        (80, 144, -10, [("L", 80), ("L", 81), ("R", 82), ("R", 83), ("L", 84), ("L", 85), ("R", 86), ("R", 87),
                        ("L", 88), ("L", 89), ("R", 90), ("R", 91), ("L", 92), ("L", 93), ("R", 94), ("R", 95),
                        ("L", 96), ("L", 97), ("R", 98), ("R", 99), ("L", 100), ("L", 101), ("R", 102), ("R", 103),
                        ("L", 104), ("L", 105), ("R", 106), ("R", 107), ("L", 108), ("R", 109), ("R", 111), 
                        ("L", 112), ("L", 113), ("R", 114), ("R", 115), ("L", 116), ("R", 118), ("R", 118.5), ("L", 119),
                        ("R", 123.25), ("L", 126.5),
                        ("L", 128), ("L", 129), ("R", 130), ("R", 131), ("L", 132), ("L", 133), ("R", 134), ("R", 135),
                        ("L", 136), ("L", 137), ("R", 138), ("R", 139), ("L", 140), ("R", 141), ("R", 143), ]),
        (80, 144, 10, [("R", 80), ("R", 81), ("L", 82), ("L", 83), ("R", 84), ("R", 85), ("L", 86), ("L", 87),
                       ("R", 88), ("R", 89), ("L", 90), ("L", 91), ("R", 92), ("R", 93), ("L", 94), ("L", 95),
                       ("R", 96), ("R", 97), ("L", 98), ("L", 99), ("R", 100), ("R", 101), ("L", 102), ("L", 103),
                       ("R", 104), ("R", 105), ("L", 106), ("L", 107), ("R", 108), ("L", 109), ("L", 111),
                       ("R", 112), ("R", 113), ("L", 114), ("L", 115), ("R", 116), ("L", 118), ("L", 118.5), ("R", 119),
                       ("R", 123.5), ("L", 126),
                       ("R", 128), ("R", 129), ("L", 130), ("L", 131), ("R", 132), ("R", 133), ("L", 134), ("L", 135),
                       ("R", 136), ("R", 137), ("L", 138), ("L", 139), ("R", 140), ("L", 141), ("L", 143), ]),
        (88, 144, -15, [("L", 88), ("L", 89), ("R", 90), ("R", 91), ("L", 92), ("L", 93), ("R", 94), ("R", 95),
                        ("L", 96), ("L", 97), ("R", 98), ("R", 99), ("L", 100), ("L", 101), ("R", 102), ("R", 103),
                        ("L", 104), ("L", 105), ("R", 106), ("R", 107), ("L", 108), ("R", 109), ("R", 110.5),
                        ("L", 112), ("L", 113), ("R", 114), ("R", 115), ("L", 116), ("R", 117.5), ("L", 119.5),
                        ("R", 123), ("L", 126.75),
                        ("L", 128), ("L", 129), ("R", 130), ("R", 131), ("L", 132), ("L", 133), ("R", 134), ("R", 135),
                        ("L", 136), ("L", 137), ("R", 138), ("R", 139), ("L", 140), ("R", 141), ("R", 142.5), ]),
        (88, 144, 15, [("R", 88), ("R", 89), ("L", 90), ("L", 91), ("R", 92), ("R", 93), ("L", 94), ("L", 95),
                       ("R", 96), ("R", 97), ("L", 98), ("L", 99), ("R", 100), ("R", 101), ("L", 102), ("L", 103),
                       ("R", 104), ("R", 105), ("L", 106), ("L", 107), ("R", 108), ("L", 109), ("L", 110.5),
                       ("R", 112), ("R", 113), ("L", 114), ("L", 115), ("R", 116), ("L", 117.5), ("R", 119.5),
                       ("R", 123.75), ("L", 125.5),
                       ("R", 128), ("R", 129), ("L", 130), ("L", 131), ("R", 132), ("R", 133), ("L", 134), ("L", 135),
                       ("R", 136), ("R", 137), ("L", 138), ("L", 139), ("R", 140), ("L", 141), ("L", 142.5), ]),
        (96, 144, -20, [("L", 96), ("L", 97), ("R", 98), ("R", 99), ("L", 100), ("L", 101), ("R", 102), ("R", 103),
                        ("L", 104), ("L", 105), ("R", 106), ("R", 107), ("L", 108), ("R", 109), ("R", 110),
                        ("L", 112), ("L", 113), ("R", 114), ("R", 115), ("L", 116), ("R", 117), ("L", 120),
                        ("L", 121), ("R", 122), ("L", 127),
                        ("L", 128), ("L", 129), ("R", 130), ("R", 131), ("L", 132), ("L", 133), ("R", 134), ("R", 135),
                        ("L", 136), ("L", 137), ("R", 138), ("R", 139), ("L", 140), ("R", 141), ("R", 142), ]),
        (96, 144, 20, [("R", 96), ("R", 97), ("L", 98), ("L", 99), ("R", 100), ("R", 101), ("L", 102), ("L", 103),
                       ("R", 104), ("R", 105), ("L", 106), ("L", 107), ("R", 108), ("L", 109), ("L", 110),
                       ("R", 112), ("R", 113), ("L", 114), ("L", 115), ("R", 116), ("L", 117), ("R", 120),
                       ("R", 121), ("L", 122), ("R", 124), ("L", 125),
                       ("R", 128), ("R", 129), ("L", 130), ("L", 131), ("R", 132), ("R", 133), ("L", 134), ("L", 135),
                       ("R", 136), ("R", 137), ("L", 138), ("L", 139), ("R", 140), ("L", 141), ("L", 142), ]),
    ]
    to_fid = {"L": 0, "R": 2}

    sx, sy, sz = 0.5, 0.5, 0.5
    px, py, pz = -30, 0, 0
    fade_out = 2

    cnt = 0
    prev = -1
    for ti, tf, a, frames in spider_specs:
        for transforms, _, _ in model:
            progress = (100 * cnt) // (len(spider_specs) * len(model))
            if progress > prev:
                print(f"Generating dancing spiders - {progress}%", end="\r")
            prev = progress
            cnt += 1

            positions_frames = []
            rotations_frames = []
            scales_frames = []
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
                t_mat = np.matmul(add_position, rescale)
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

                positions_frames.append(definite_position)
                rotations_frames.append(local_rotation)
                scales_frames.append(double_scale)

            positions = []
            rotations = []

            for frame, start in frames:
                duration = dt

                tt0 = (start - ti) / (tf - ti)
                tt1 = min(1, (start + duration - ti) / (tf - ti))
                fid = to_fid[frame]
                for src, dest in [(positions_frames, positions), (rotations_frames, rotations)]:
                    dest.append([*src[1], tt0])
                    dest.append([*src[fid], tt0])
                    dest.append([*src[fid], tt1])
                    dest.append([*src[1], tt1])

            nj_offset = (tf - ti) / 2 - BASE_HJ

            # t_fade_in = max(0, fade_in / (tf - ti))
            t_fade_out = min(1, (tf - fade_out) / (tf - ti))

            obstacles.append({
                "_time": ti + BASE_HJ + nj_offset,
                "_duration": 0,
                "_lineIndex": 0,
                "_type": 0,
                "_width": 0,
                "_customData": {
                    "_interactable": False,
                    "_noteJumpStartBeatOffset": nj_offset,
                    "_position": [0, 0],
                    "_scale": scales_frames[0][:3],
                    "_color": [80, 80, 80, 0],
                    "_rotation": [0, a, 0],
                    "_animation": {
                        "_definitePosition": positions,
                        "_localRotation": rotations,
                        "_dissolve": [
                            [0, 0],
                            [1, 0],
                            [1, t_fade_out],
                            [0, 1],
                        ],
                    }
                }
            })
    print(f"Generated dancing spiders       ")


def load_font():
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ',.?*"
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
        font[letter] = (7, letter_walls)
        # font[letter] = (letter_widths[k], letter_walls)

    font[" "] = (7, [])

    return (img_dim[0], font)


def add_dialogue(walls, dial_id):
    print(f"Generating text...", end="\r")

    # line, list[ti0, ti1, tf, text]
    if dial_id == 0:
        dial_text = [
            (0, [(0.8, 1.6, 2.7, "* Ahuhuhuhu...")]),
            (0, [(2.75, 4.1, 9.5, "* You think you can slice")]),
            (1, [(4.1, 5.4, 9.5, "  all these bloks with only")]),
            (2, [(5.4, 6.1, 9.5, "  two arms, "), (6.75, 7.6,
                                                9.5, "don't you, "), (8.25, 8.8, 9.5, "dearie?")]),
            (0, [(9.6, 10.2, 13.7, "* Ahuhuhu...")]),
            (1, [(10.75, 12.5, 13.7, "* I disagree with that")]),
            (2, [(12.5, 13.4, 13.7, "  notion.")]),
        ]
    else:
        dial_text = [
            (0, [(209.5, 210.25, 211, "* Ahuhuhuhu...")]),
            (0, [(211.2, 212, 216, "* That was fun.")]),
            (1, [(212.5, 213.75, 216, "* See you again, "), (215, 215.5, 216, "dearie.")]),
        ]
    line_heights = [2.2, 1.1, 0]
    x_offset = -8.6
    color = [42, 42, 42, 0]
    text_dist = 20
    letter_spacing = 1
    scale = 0.06

    _, font = load_font()

    for line, pieces in dial_text:
        line_height = line_heights[line]
        offset = 0
        for ti0, ti1, tf, text in pieces:
            for li, letter in enumerate(text):
                ti = lerp(ti0, ti1, li / len(text))
                nj_offset = (tf - ti) / 2 - BASE_HJ
                for letter_wall in font[letter][1]:
                    pos = [x_offset + scale * (offset + letter_wall[0]),
                           line_height + scale * letter_wall[1],
                           text_dist]
                    walls.append({
                        "_time": ti + BASE_HJ + nj_offset,
                        "_duration": 0,
                        "_lineIndex": 0,
                        "_type": 0,
                        "_width": 0,
                        "_customData": {
                            "_interactable": False,
                            "_position": [0, 0],
                            "_scale": [scale * letter_wall[2],
                                       scale * letter_wall[3],
                                       scale],
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

    # Frame
    if dial_id == 0:
        t0, t1 = 0.5, 0.8
        t2s = [14.69, 14.69, 14.19, 14.44]
    else:
        t0, t1 = 209, 209.5
        t2s = [217] * 4
    thickness = 0.1
    x0, x1 = 18, 9.5
    ya0, ya1, yb0, yb1 = -10, -0.4, 12.5, 3.3
    frame_p0 = [
        [-x0, ya1],
        [x0, ya1],
        [-x1, ya0],
        [-x1, yb0],
    ]
    frame_p1 = [
        [-x1 + thickness, ya1],
        [x1, ya1],
        [-x1, ya1],
        [-x1, yb1],
    ]
    scales = [
        [thickness, yb1 - ya1 + thickness],
        [thickness, yb1 - ya1 + thickness],
        [2 * x1 + 2 * thickness, thickness],
        [2 * x1 + 2 * thickness, thickness],
    ]
    for i in range(4):
        t2 = t2s[i]
        t3 = t2 + 0.3
        nj_offset = (t3 - t0) / 2 - BASE_HJ
        walls.append({
            "_time": t0 + BASE_HJ + nj_offset,
            "_duration": 0,
            "_lineIndex": 0,
            "_type": 0,
            "_width": 0,
            "_customData": {
                "_interactable": False,
                "_position": [0, 0],
                "_scale": [*scales[i], thickness],
                "_noteJumpStartBeatOffset": nj_offset,
                "_color": color,
                "_animation": {
                     "_definitePosition": [
                         [*frame_p0[i], text_dist, 0],
                         [*frame_p1[i], text_dist, (t1 - t0) / (t3 - t0)],
                         [*frame_p1[i], text_dist, (t2 - t0) / (t3 - t0)],
                         [*frame_p0[i], text_dist, 1],
                     ],
                     "_dissolve": [
                         [0, 0],
                         [1, (t1 - t0) / (t3 - t0)],
                         [1, (t2 - t0) / (t3 - t0)],
                         [0, 1],
                     ],
                     }
            }
        })
    print("Generated text    ")


def add_purple_lines(walls_out):
    lines_x = [-1.5, -0.5, 0.5, 1.5]
    alpha_steps = [1.]
    z0, z1 = 3, 20
    thickness = 0.05
    ti, tf = 14.69, 206
    anim_length = 0.3
    sec_length = (z1 - z0) / len(alpha_steps)

    lines_anim = [[(143, 0), (176, 1)] for _ in range(4)]
    for tt in [144, 145, 146, 147, 152, 153, 154, 155, 160, 161, 162, 163, 168, 169, 170, 171]:
        for i in range(4):
            lines_anim[3 - i] += [(tt + 0.25 * i, 1), (tt + 0.25 * i + 0.2, 0)]
    for tt in [148, 149, 150, 151, 156, 157, 158, 159, 164, 165, 166, 167, 172, 173, 174, 175]:
        for i in range(4):
            lines_anim[i] += [(tt + 0.25 * i, 1), (tt + 0.25 * i + 0.2, 0)]
    for i in range(4):
        lines_anim[i].sort(key = lambda x: x[0])
    
    dissolves = [[[0, 0], [1, 0]] for i in range(4)]
    for i in range(4):
        for tt, anim in lines_anim[i]:
            ttr = (tt - ti) / (tf - ti)
            prev = dissolves[i][-1][0]
            dissolves[i] += [(prev, ttr), (anim, ttr)]

    for lx, dissolve in zip(lines_x, dissolves):
        for ai, alpha in enumerate(alpha_steps):
            nj_offset = (tf - ti) / 2 - BASE_HJ
            walls_out.append({
                "_time": ti + BASE_HJ + nj_offset,
                "_duration": 0,
                "_lineIndex": 0,
                "_type": 0,
                "_width": 0,
                "_customData": {
                    "_interactable": False,
                    "_position": [0, 0],
                    "_scale": [thickness, thickness, sec_length],
                    "_noteJumpStartBeatOffset": nj_offset,
                    "_color": [alpha * p for p in PURPLE],
                    "_animation": {
                        "_definitePosition": [
                            [- thickness / 2, 0, z0 + ai * sec_length, 0],
                            [lx - thickness / 2, 0, z0 + ai * \
                                sec_length, anim_length / (tf - ti)],
                        ],
                        "_dissolve": dissolve,
                    }
                }
            })


def add_obstacles(obstacles_out, obstacles_in):
    sx, sz = 1.5, 4

    prev = -1
    cnt = 0
    for obstacle_in in obstacles_in:
        progress = (100 * cnt) // len(obstacles_in)
        if progress > prev:
            print(f"Generating walls - {progress}%", end="\r")
        prev = progress
        cnt += 1

        time_ = obstacle_in["_time"]
        line_index = obstacle_in["_lineIndex"]
        x = -2 if line_index <= 0 else 2 if line_index >= 3 else 0

        # TODO: use track
        obstacles_out.append({
            "_time": time_,
            "_duration": 0.25,
            "_lineIndex": obstacle_in["_lineIndex"],
            "_type": 0,
            "_width": 1,
            "_customData": {
                "_track": "obsw",
                "_color": [42, 42, 42, 0],
            }
        })
    print(f"Generated walls        ")


def add_bombs(notes_out, notes_in):
    prev = -1
    cnt = 0

    ds = 2
    s2 = ds / math.sqrt(2)
    xy_offsets = [(-ds, 0), (ds, 0), (0, -ds), (0, ds),
                  (-s2, -s2), (-s2, s2), (s2, -s2), (s2, s2)]

    for note_in in notes_in:
        if note_in["_type"] != 3:
            continue

        progress = (100 * cnt) // len(notes_in)
        if progress > prev:
            print(f"Generating bombs - {progress}%", end="\r")
        prev = progress
        cnt += 1

        x = note_in["_lineIndex"] -1.5
        y = note_in["_lineLayer"] + 0.5
        for xo, yo in xy_offsets:
            notes_out.append({
                "_time": note_in["_time"],
                "_lineIndex": note_in["_lineIndex"],
                "_lineLayer": note_in["_lineLayer"],
                "_type": 3,
                "_cutDirection": note_in["_cutDirection"],
                "_customData": {
                    "_position": [0, 0],
                    "_track": "bombs",
                    "_disableSpawnEffect": True,
                    "_disableNoteGravity": True,
                    "_color": [42, 42, 42, 0],
                    "_animation": {
                        "_position": [
                            [x, y, 0, 0],
                            [x + xo, y + yo, 0, 0.5, "easeOutSine"],
                        ],
                        "_dissolve": [
                            [0, 0.],
                            [1, 0.],
                        ],
                        "_scale": [
                            [0.1, 0.1, 0.1, 0.],
                            [1., 1., 1., 0.2, "easeOutSine"],
                        ]
                    }
                }
            })

    print(f"Generated bombs       ")


def add_notes(notes_out, notes_in):
    prev = -1
    cnt = 0

    lefts = {25, 27, 42, 57, 59, 74, 176, 178, 182, 186, 192, 194, 198, 202}
    rights = {26, 41, 43, 58, 73, 75, 177, 181, 185, 193, 197, 201}
    updowns = [(81.8, 87), (89.8, 95), (97.8, 103), (129.8, 135)]

    for note_in in notes_in:
        if note_in["_type"] == 3:
            continue

        progress = (100 * cnt) // len(notes_in)
        if progress > prev:
            print(f"Generating notes - {progress}%", end="\r")
        prev = progress
        cnt += 1

        notes_out.append({
            "_time": note_in["_time"],
            "_lineIndex": note_in["_lineIndex"],
            "_lineLayer": note_in["_lineLayer"],
            "_type": note_in["_type"],
            "_cutDirection": note_in["_cutDirection"],
            "_customData": {
                "_disableSpawnEffect": True,
                "_animation": {
                    "_dissolve": [
                        [0, 0.],
                        [1, 0.],
                    ],
                    "_dissolveArrow": [
                        [0, 0.],
                        [1, 0.],
                    ],
                }
            }
        })

        if note_in["_time"] >= 112 and note_in["_time"] <= 127.5:
            notes_out[-1]["_customData"]["_track"] = "sin"
        if int(note_in["_time"] + 0.5) in lefts:
            notes_out[-1]["_customData"]["_track"] = "left"
        if int(note_in["_time"] + 0.5) in rights:
            notes_out[-1]["_customData"]["_track"] = "right"
        if any(note_in["_time"] >= zi and note_in["_time"] <= zf
               for zi, zf in updowns):
            notes_out[-1]["_customData"]["_disableNoteGravity"] = True
            if note_in["_lineLayer"] == 0:
                notes_out[-1]["_customData"]["_track"] = "down"
            else:
                notes_out[-1]["_customData"]["_track"] = "up"

    print(f"Generated notes       ")


def add_tracks(custom_data):
    print("Generating tracks")
    track_events = [
        {
            "_time": 2,
            "_type": "AssignPathAnimation",
            "_data": {
                "_track": "obsw",
                "_scale": [
                    [1, 0.05, 1, 0.],
                    [1, 1, 1, 0.5],
                    [1, 2, 1, 1.],
                ],
            }
        },
        {
            "_time": 3,
            "_type": "AssignPathAnimation",
            "_data": {
                "_track": "obsw",
                "_dissolve": [
                    [0, 0.],
                    [1, 0.],
                ],
            }
        },
        {
            "_time": 12,
            "_type": "AssignPathAnimation",
            "_data": {
                "_track": "right",
                "_position": [
                    [-0.7, 0, 0, 0.3],
                    [0, 0, 0, 0.55, "easeInSine"],
                ]
            }
        },
        {
            "_time": 12,
            "_type": "AssignPathAnimation",
            "_data": {
                "_track": "left",
                "_position": [
                    [0.7, 0, 0, 0.3],
                    [0, 0, 0, 0.55, "easeInSine"],
                ]
            }
        },
        {
            "_time": 80,
            "_type": "AssignPathAnimation",
            "_data": {
                "_track": "down",
                "_position": [
                    [0, 2, 0, 0],
                    [0, 0, 0, 0.5, "easeOutSine"],
                ]
            }
        },
        {
            "_time": 80,
            "_type": "AssignPathAnimation",
            "_data": {
                "_track": "up",
                "_position": [
                    [0, -2, 0, 0],
                    [0, 0, 0, 0.5, "easeInSine"],
                ]
            }
        },
        {
            "_time": 112,
            "_type": "AnimateTrack",
            "_data": {
                "_track": "sin",
                "_duration": 15,
                "_position": [
                    [0, 0, 0, 0],
                    [-0.5, 0, 0, 0.2, "splineCatmullRom"],
                    [1, 0.2, 0, 0.4, "splineCatmullRom"],
                    [-1, 0.2, 0, 0.6, "splineCatmullRom"],
                    [1.5, 0, 0, 0.8, "splineCatmullRom"],
                    [-1, 0.2, 0, 1.0, "splineCatmullRom"],
                ]
            }
        }
    ]
    custom_data["_customEvents"] += track_events
    print(f"Generated tracks ")


def main():
    input_json = dict()
    for filename in INPUT_FILES:
        with open("{}.dat".format(filename), "r") as json_file:
            input_json[filename] = json.load(json_file)

    notes = []
    obstacles = []
    custom_data = {
        "_environment": [
            {
                "_id": id_,
                "_lookupMethod": "Contains",
                "_active": False,
            }
            for id_ in ["Floor",
                        "BackColumns",
                        "Construction",
                        "Buildings",
                        "Spectrograms",
                        "BigTrackLaneRing",
                        "BoxLight"]
        ],
        "_customEvents": [],
    }
    # custom_events = custom_data["_customEvents"]

    # Lights off
    events = [{"_time": 0, "_type": i, "_value": 0} for i in range(5)]

    for model_info in MODELS:
        model = load_model(model_info[0])
        add_model(obstacles, model, model_info)

    add_dialogue(obstacles, 0)
    add_dialogue(obstacles, 1)
    add_purple_lines(obstacles)
    add_obstacles(obstacles, input_json["WIP"]["_obstacles"])
    add_dancing_spiders(obstacles)
    add_bombs(notes, input_json["WIP"]["_notes"])
    add_notes(notes, input_json["WIP"]["_notes"])
    add_tracks(custom_data)

    print("Sorting obstacles", end="\r")
    obstacles.sort(key=lambda x: x["_time"])
    print("Sorted obstacles ")
    print("Sorting notes", end="\r")
    notes.sort(key=lambda x: x["_time"])
    print("Sorted notes ")
    print("Sorting custom events", end="\r")
    custom_data["_customEvents"].sort(key=lambda x: x["_time"])
    print("Sorted custom events ")

    # Prevent MM from overwriting info.dat
    print("Copying info file", end="\r")
    shutil.copyfile("info.json", "info.dat")
    print("Copied info file ")

    print("Truncating floats", end="\r")
    obstacles = trunc(obstacles)
    custom_data = trunc(custom_data)
    events = trunc(events)
    notes = trunc(notes)
    print("Truncated floats ")

    for filename, has_notes in OUTPUT_FILES:
        print(f"Writing {filename}", end="\r")
        song_json = copy.deepcopy(input_json["template"])

        song_json["_obstacles"] = obstacles
        song_json["_customData"] = custom_data
        song_json["_events"] = events
        if has_notes:
            song_json["_notes"] = notes

        with open("{}.dat".format(filename), "w") as json_file:
            json.dump(song_json, json_file)
        print(f"Written {filename}")


main()
