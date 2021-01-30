import functools
import json
import math
import numpy as np
import shutil
from svg.path import parse_path, Line, CubicBezier
import xml.etree.ElementTree as ET


FILES = ["HardStandard"]

BPM = 67
BASE_HJ = 1
SNS = {"xmlns": "http://www.w3.org/2000/svg"}

# scale, x, y, z, list[time, name]
MODELS = [
    [8, 0, -2, 30, [(1, "ghost_a_0"), (2, "ghost_a_1"),
                    (2.25, "ghost_a_0"), (2.5, "ghost_a_1"),
                    (2.75, "ghost_a_2"), (3, "ghost_a_3"),
                    (4, "ghost_a_4"), (4.75, "ghost_a_5"),
                    (5, "ghost_a_3"), (7, "ghost_a_6"),
                    (7.75, "ghost_a_7"), (8, "ghost_a_8"),
                    (8.75, "ghost_a_7"), (9, "ghost_a_9"),
                    (10, "ghost_a_10"), (10.75, "ghost_a_9"),
                    (11, "ghost_a_11"), (12, "ghost_a_12"),
                    (13, "ghost_a_11"), (14, "ghost_a_13"),
                    (14.3, "ghost_a_14"), (14.6, "ghost_a_15"),
                    (15, "ghost_a_16"), (16, "ghost_a_17"),
                    (17, "ghost_a_18"), (18, "ghost_a_19"),
                    (19, "ghost_a_20"), (20, "ghost_a_21"),
                    (21, "ghost_a_20"), (22, "ghost_a_22"),
                    (23, "ghost_a_23"), (25, "ghost_a_24"),
                    (26, "ghost_a_25"), (27, "ghost_a_20"),
                    (28, "ghost_a_21"), (29, "ghost_a_20"),
                    (30.3, "ghost_a_26"), (31, "ghost_a_27"),
                    (33, None)]],
    [8, 3.6, -1.5, 30, [(7, "ghost_b_0"), (8, "ghost_b_1"),
                        (8.75, "ghost_b_2"), (9, "ghost_b_3"),
                        (10, "ghost_b_3bis"), (10.5, "ghost_b_3"),
                        (11, "ghost_b_4"), (12, "ghost_b_5"),
                        (13, "ghost_b_4"), (14, "ghost_b_6"),
                        (15, "ghost_b_7"), (16, "ghost_b_8"),
                        (17, "ghost_b_9"), (18, "ghost_b_10"),
                        (19, "ghost_b_11"), (20, "ghost_b_12"),
                        (21, "ghost_b_11"), (22, "ghost_b_13"),
                        (23, "ghost_b_14"), (23.4, "ghost_b_15"),
                        (23.8, "ghost_b_16"), (24.2, "ghost_b_17"),
                        (24.7, "ghost_b_18"), (25, "ghost_b_19"),
                        (26, "ghost_b_10"), (27, "ghost_b_11"),
                        (28, "ghost_b_12"), (29, "ghost_b_11"),
                        (30, "ghost_b_20"), (31, "ghost_b_21"),
                        (33, None), ]],
    [8, -3.9, -1.8, 30, [(11, "ghost_c_0"), (12, "ghost_c_1"),
                         (13, "ghost_c_0"), (14, "ghost_c_2"),
                         (14.6, "ghost_c_3"), (15, "ghost_c_2"),
                         (16, "ghost_c_4"), (17, "ghost_c_5"),
                         (18, "ghost_c_6"), (19, "ghost_c_7"),
                         (20, "ghost_c_8"), (21, "ghost_c_7"),
                         (22, "ghost_c_9"), (23, "ghost_c_10"),
                         (23.4, "ghost_c_11"), (23.8, "ghost_c_12"),
                         (24.2, "ghost_c_13"), (24.7, "ghost_c_14"),
                         (25, "ghost_c_15"), (26, "ghost_c_9"),
                         (27, "ghost_c_7"), (28, "ghost_c_8"),
                         (29, "ghost_c_7"), (30.6, "ghost_c_16"),
                         (31, "ghost_c_17"),
                         (33, None), ]],
    [6, 4, -2, 15, [(32.75, "ghost_d_0"), (33, "ghost_d_1"),
                    (33.3, "ghost_d_2"), (34, None), ]],
    [6, 4, -2, 20, [(35, "ghost_d_3"), (35.5, "ghost_d_4"),
                    (35.75, "ghost_d_5"), (36, "ghost_d_6"),
                    (36.6, "ghost_d_5"), (37, "ghost_d_7"),
                    (37.2, "ghost_d_8"), (37.4, "ghost_d_9"),
                    (37.6, "ghost_d_10"), (37.8, "ghost_d_11"),
                    (38, "ghost_d_12"), (40, "ghost_d_13"),
                    (41, "ghost_d_14"), (41.4, "ghost_d_15"),
                    (41.8, "ghost_d_14"), (42.2, "ghost_d_15"),
                    (42.6, "ghost_d_14"), (43, "ghost_d_16"),
                    (43.75, "ghost_d_17"), (44, "ghost_d_16"),
                    (44.75, "ghost_d_17"), (45, "ghost_d_16"),
                    (48.5, "ghost_d_18"), (49, "ghost_d_19"),
                    (52.5, "ghost_d_20"), (52.75, "ghost_d_21"),
                    (53, "ghost_d_22"), (53.5, "ghost_d_23"),
                    (54, "ghost_d_24"), (54.5, "ghost_d_23"),
                    (57, "ghost_d_24_bis"), (57.3, "ghost_d_25"),
                    (57.6, "ghost_d_26"), (58, "ghost_d_27"),
                    (58.3, "ghost_d_26"), (58.6, "ghost_d_27"),
                    (59, "ghost_d_26"), (61, "ghost_d_28"),
                    (62, "ghost_d_29"), (62.25, "ghost_d_30"),
                    (62.5, "ghost_d_29"), (62.75, "ghost_d_30"),
                    (63, "ghost_d_29"), (65.1, "ghost_d_31"),
                    (67, None), ]],
    [7, -4, -2, 17, [(38, "ghost_e_0"), (38.3, "ghost_e_1"),
                     (38.6, "ghost_e_2"), (39, "ghost_e_3"),
                     (39.3, "ghost_e_4"), (39.6, "ghost_e_3"),
                     (40, "ghost_e_5"), (41, "ghost_e_6"),
                     (46, "ghost_e_7"), (46.3, "ghost_e_8"),
                     (46.7, "ghost_e_10"), (47, "ghost_e_11"),
                     (47.25, "ghost_e_11"), (47.5, "ghost_e_11"),
                     (47.75, "ghost_e_11"), (48, "ghost_e_11"),
                     (48.3, "ghost_e_13"), (50.3, "ghost_e_14"),
                     (50.6, "ghost_e_15"), (51, "ghost_e_16"),
                     (51.3, "ghost_e_17"), (54.5, "ghost_e_18"),
                     (55, "ghost_e_19"), (55.3, "ghost_e_20"),
                     (55.7, "ghost_e_21"), (56, "ghost_e_20"),
                     (56.3, "ghost_e_21"), (59, "ghost_e_22"),
                     (59.3, "ghost_e_23"), (59.7, "ghost_e_24"),
                     (60, "ghost_e_25"), (60.3, "ghost_e_24"),
                     (60.7, "ghost_e_25"), (63, "ghost_e_26"),
                     (63.3, "ghost_e_27"), (63.7, "ghost_e_28"),
                     (64, "ghost_e_29"), (64.25, "ghost_e_28"),
                     (64.5, "ghost_e_29"), (64.75, "ghost_e_28"),
                     (65.1, "ghost_e_30"), (67, None), ]],
    [6, 3, -2, 15, [(69, "ghost_f_0"), (69.4, "ghost_f_1"),
                    (69.7, "ghost_f_0"), (70, "ghost_f_1"),
                    (70.3, "ghost_f_0"), (70.6, "ghost_f_1"),
                    (70.8, "ghost_f_0"), (71, "ghost_f_1"),
                    (71.2, "ghost_f_0"), (71.4, "ghost_f_1"),
                    (71.6, "ghost_f_0"), (71.8, "ghost_f_1"),
                    (72, "ghost_f_0"), (72.2, "ghost_f_1"),
                    (72.3, "ghost_f_0"), (72.4, "ghost_f_1"),
                    (72.5, "ghost_f_2"), (72.9, "ghost_f_3"),
                    (73.2, "ghost_f_2"), (73.5, "ghost_f_3"),
                    (73.8, "ghost_f_2"), (74, "ghost_f_3"),
                    (74.2, "ghost_f_2"), (74.6, "ghost_f_3"),
                    (74.8, "ghost_f_2"), (74.9, "ghost_f_3"),
                    (75, "ghost_f_4"), (76.6, "ghost_f_5"),
                    (77.3, "ghost_f_6"), (77.7, "ghost_f_7"),
                    (78, "ghost_f_6"), (78.3, "ghost_f_7"),
                    (78.6, "ghost_f_6"), (78.8, "ghost_f_7"),
                    (79, "ghost_f_6"), (79.2, "ghost_f_7"),
                    (79.4, "ghost_f_6"), (79.6, "ghost_f_7"),
                    (79.8, "ghost_f_6"), (79.9, "ghost_f_7"),
                    (80, "ghost_f_8"), (81, "ghost_f_9"),
                    (81.4, "ghost_f_10"), (81.7, "ghost_f_9"),
                    (82, "ghost_f_10"), (82.3, "ghost_f_9"),
                    (82.6, "ghost_f_10"), (82.8, "ghost_f_9"),
                    (83, "ghost_f_10"), (83.2, "ghost_f_9"),
                    (83.4, "ghost_f_10"), (83.6, "ghost_f_9"),
                    (83.8, "ghost_f_10"), (84, "ghost_f_9"),
                    (84.25, "ghost_f_8"), (85.1, "ghost_f_2"),
                    (85.5, "ghost_f_3"), (85.8, "ghost_f_2"),
                    (86.1, "ghost_f_3"), (86.4, "ghost_f_2"),
                    (86.6, "ghost_f_3"), (86.8, "ghost_f_2"),
                    (87, "ghost_f_3"), (87.2, "ghost_f_2"),
                    (87.4, "ghost_f_3"), (87.6, "ghost_f_2"),
                    (87.8, "ghost_f_3"), (88, "ghost_f_8"),
                    (89, None)]],
    [6, 3, -2, 15, [(89, "ghost_f_11"), (89.4, "ghost_f_12"),
                    (89.7, "ghost_f_11"), (90, "ghost_f_12"),
                    (90.3, "ghost_f_11"), (90.6, "ghost_f_12"),
                    (90.8, "ghost_f_11"), (91, None), ]],
    [6, 3, -2, 15, [(91, "ghost_f_13"), (91.3, "ghost_f_14"),
                    (91.6, "ghost_f_13"), (91.8, "ghost_f_14"),
                    (92, "ghost_f_13"), (92.3, "ghost_f_15"),
                    (93.3, "ghost_f_16"), (94, "ghost_f_17"),
                    (95, "ghost_f_18"), (95.5, "ghost_f_19"),
                    (97, None), ]],
    [6, -3, -2, 15, [(69, "ghost_g_0"), (72.75, "ghost_g_1"),
                     (72.8, "ghost_g_0"), (73, "ghost_g_1"),
                     (73.05, "ghost_g_0"), (75, "ghost_g_1"),
                     (75.05, "ghost_g_0"), (76.75, "ghost_g_1"),
                     (76.8, "ghost_g_0"), (77, "ghost_g_1"),
                     (77.05, "ghost_g_0"), (80, "ghost_g_1"),
                     (80.05, "ghost_g_0"), (80.75, "ghost_g_1"),
                     (80.8, "ghost_g_0"), (81, "ghost_g_1"),
                     (81.05, "ghost_g_0"), (85, "ghost_g_1"),
                     (85.05, "ghost_g_0"), (88, "ghost_g_1"),
                     (88.05, "ghost_g_0"), (88.75, "ghost_g_1"),
                     (88.8, "ghost_g_0"),
                     (89, None), ]],
    [8, 5.2, -2, 15, [(91, "ghost_h_0"), (93.1, "ghost_h_1"),
                      (95, "ghost_h_2"), (95.4, "ghost_h_3"),
                      (95.7, "ghost_h_2"), (96.1, "ghost_h_3"),
                      (96.4, "ghost_h_2"), (96.8, None), ]],
]

# next ghost: 69


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


def normalize(x, y, viewbox):
    """Normalize so that the origin is at the bottom center of the image,
    and the width and height of the image are 1
    """
    xi, yi, width, height = viewbox
    return (x - xi - width / 2) / width, (yi + height - y) / height


@functools.lru_cache(maxsize=None)
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
    scale, x, y, z, times = image_data
    thickness = 0.005

    model0 = load_image(times[0][1])
    ti = times[0][0]
    tf = times[-1][0]

    for i in range(len(model0)):
        start = ti - 1 + i / len(model0)
        end = tf + i / len(model0)

        positions = []
        rotations = []
        scales = []

        for j in range(len(times) - 1):
            tt, name = times[j]
            ttn = times[j + 1][0]

            model = load_image(name)
            line = model[i]
            x1, y1, x2, y2, _ = line
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            x1, y1 = x + scale * x1, y + scale * y1
            x2, y2 = x + scale * x2, y + scale * y2

            pos = [x1, y1, z]
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

            # TODO: avoid redundancy! (three consecutive identical values)

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
                "_color": [1, 1, 1, 0],
                "_scale": [thickness * 2, 1, thickness * 2],
                "_animation": {
                    "_definitePosition": positions,
                    "_localRotation": rotations,
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


def add_main_title(walls):
    scale = 15
    y = 5
    z1, z2 = 50, 30
    thickness = 0.005
    t0, t1, t2, t3 = 16.75, 17, 18, 18.25
    v_angle = 5

    model = load_image("title")

    for i in range(len(model)):
        start = t0 - 1 + i / len(model)
        end = t3 + i / len(model)

        line = model[i]
        x1, y1, x2, y2, _ = line
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        x1, y1 = scale * x1, y + scale * y1
        x2, y2 = scale * x2, y + scale * y2

        pos = [x1, y1]

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
                "_color": [1, 1, 1, 0],
                "_rotation": [-v_angle, 0, 0],
                "_localRotation": [0, 0, math.degrees(math.atan2(y2 - y1, x2 - x1)) - 90],
                "_scale": [thickness, math.sqrt((x2 - x1)**2 + (y2 - y1) ** 2), thickness],
                "_animation": {
                    "_definitePosition": [
                        [*pos, z1, (t0 - start) / (end - start)],
                        [*pos, z2, (t1 - start) / (end - start)],
                        [*pos, z2, (t2 - start) / (end - start)],
                        [*pos, z1, (t3 - start) / (end - start)],
                    ],
                    "_dissolve": [
                        [0, (t0 - start) / (end - start)],
                        [1, (t1 - start) / (end - start)],
                        [1, (t2 - start) / (end - start)],
                        [0, (t3 - start) / (end - start)],
                    ],
                }
            }
        })


def add_end_title(walls):
    scale = 5
    y, z = 0, 20
    thickness = 0.005
    t0, t1, t2, t3 = 97, 97.2, 99, 100.5

    model = load_image("end")

    for i in range(len(model)):
        start = t0 - 1 + i / len(model)
        end = t3 + i / len(model)

        line = model[i]
        x1, y1, x2, y2, _ = line
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        x1, y1 = scale * x1, y + scale * y1
        x2, y2 = scale * x2, y + scale * y2

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
                "_color": [1, 1, 1, 0],
                "_localRotation": [0, 0, math.degrees(math.atan2(y2 - y1, x2 - x1)) - 90],
                "_scale": [thickness, math.sqrt((x2 - x1)**2 + (y2 - y1) ** 2), thickness],
                "_animation": {
                    "_definitePosition": [
                        [x1, y1, z, 0],
                    ],
                    "_dissolve": [
                        [0, (t0 - start) / (end - start)],
                        [1, (t1 - start) / (end - start)],
                        [1, (t2 - start) / (end - start)],
                        [0, (t3 - start) / (end - start)],
                    ],
                }
            }
        })


def process_notes(notes):
    start_z = 13
    pts_ranges = [
        (0.1, 0.3),
        (0.2, 0.3),
        (0.1, 0.3),
        (0.1, 0.2)
    ]
    n_pts = len(pts_ranges)
    for note in notes:
        position_timings = np.r_[:n_pts] * 0.5 / n_pts
        positions = []
        par = (note["_type"] % 2) * 2 - 1
        for i in range(n_pts):
            positions.append([
                0,
                par * ((i % 2) * 2 - 1) * np.random.uniform(*pts_ranges[i]),
                0,
                position_timings[i],
                "easeInOutSine",
            ])

        note["_customData"] = {
            "_disableSpawnEffect": True,
            "_noteJumpStartBeatOffset": 0.5,
            "_animation": {
                "_dissolve": [
                    [0, 0],
                ],
                "_dissolveArrow": [
                    [0, 0],
                    [1, 0.1],
                ],
                "_definitePosition": [
                    [0, 0, start_z, 0],
                    [0, 0, 0, 0.5],
                    [0, 0, -start_z, 1],
                ],
                "_position": positions,
            }
        }


def main():
    # Lights off
    events = [{"_time": 0, "_type": i, "_value": 0} for i in range(5)]

    walls = []
    for model in MODELS:
        add_image(walls, model)

    add_main_title(walls)
    add_end_title(walls)

    walls.sort(key=lambda x: x["_time"])

    # Prevent MM from overwriting info.dat
    shutil.copyfile("info.json", "info.dat")

    for filename in FILES:
        with open("{}.dat".format(filename), "r") as json_file:
            song_json = json.load(json_file)

        song_json["_obstacles"] = walls
        song_json["_events"] = events
        process_notes(song_json["_notes"])

        song_json = trunc(song_json)

        with open("{}.dat".format(filename), "w") as json_file:
            json.dump(song_json, json_file)


main()
