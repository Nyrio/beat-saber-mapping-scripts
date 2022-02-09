import itertools
import json
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
import shutil

BLENDER_FILE = "track.dae"
SCALE = 5  # Scaling factor for the x and y axes
NS = {"xmlns": "http://www.collada.org/2005/11/COLLADASchema"}
ENV_NJS = 8
ENV_OFF = 10
BASE_HJ = 2

FILES = [("ExpertWIP.dat", "ExpertStandard.dat")]

COLLECTIONS = {
    "desert": [("tiny_rock", 0.5), ("small_rock", 0.4), ("big_rock", 0.1)],
    "ocean_animals": [("turtle", 0.6), ("squid", 0.2), ("shark", 0.2)],
    "seabed": [("plant0", 0.4), ("plant1", 0.2), ("seastar", 0.4)],
    "sand": [("tiny_rock", 1)],
    "stars": [("star", 1)],
    "sat0": [("sat0", 1)],
    "sat1": [("sat1", 1)],
    "sat2": [("sat2", 1)],
    "sat3": [("sputnik", 1)],
}

# collection, nb, ranges: x, y, z, ax, ay, az, spawn angle
SPAWNS = [
    ("desert", 500, [(-173, -1), (-5, -0.75),
                     (0, 0), (0, 0), (0, 0), (0, 360), (0, 0)]),
    ("desert", 500, [(-173, -1), (0.75, 5),
                     (0, 0), (0, 0), (0, 0), (0, 360), (0, 0)]),
    ("ocean_animals", 21, [(-430, -188), (-3, -0.5),
                           (0.4, 2.0), (-10, 10), (0, 0), (0, 0), (5, 20)]),
    ("ocean_animals", 21, [(-430, -188), (0.5, 3),
                           (0.4, 2.0), (-10, 10), (0, 0), (0, 0), (-20, -5)]),
    ("seabed", 50, [(-430, -188), (-4, -0.75),
                    (0, 0), (0, 0), (0, 0), (0, 360), (0, 0)]),
    ("seabed", 50, [(-430, -188), (0.75, 4),
                    (0, 0), (0, 0), (0, 0), (0, 360), (0, 0)]),
    ("sand", 1000, [(-430, -178), (-5, 5),
                    (0, 0), (0, 0), (0, 0), (0, 360), (0, 0)]),
    ("stars", 2000, [(-828, -430), (-4, -0.2),
                     (-2, 3), (0, 0), (0, 0), (0, 360), (0, 0)]),
    ("stars", 2000, [(-828, -430), (0.2, 4),
                     (-2, 3), (0, 0), (0, 0), (0, 360), (0, 0)]),
    ("sat0", 1, [(-460, -460), (-1, -1),
                 (1.5, 1.5), (0, 0), (0, 0), (-20, -20), (20, 20)]),
    ("sat3", 1, [(-540, -540), (1, 1),
                 (1.5, 1.5), (0, 0), (0, 0), (0, 0), (-20, -20)]),
    ("sat1", 1, [(-620, -620), (1.5, 1.5),
                 (2, 2), (0, 0), (10, 10), (0, 0), (-25, -25)]),
    ("sat2", 1, [(-678, -678), (-1, -1),
                 (1, 1), (0, 0), (-10, -10), (0, 0), (10, 10)]),
    ("sat3", 1, [(-730, -730), (-1, -1),
                 (1.5, 1.5), (0, 0), (0, 0), (-20, -20), (20, 20)]),
    ("sat0", 1, [(-790, -790), (1, 1),
                 (1.5, 1.5), (0, 0), (0, 0), (-20, -20), (-20, -20)]),
]


def trunc(obj, precision=4):
    if isinstance(obj, float):
        return round(obj, precision)
    elif isinstance(obj, dict):
        return dict((k, trunc(v)) for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        return list(map(trunc, obj))
    return obj


def load_model(filename):
    root = ET.parse(filename).getroot()
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


def add_model(walls, model, offset, collection_name):
    # Additional rotation applied to the object
    add_rotation = np.zeros((4, 4))
    add_rotation[:3, :3] = Rotation.from_euler(
        "xyz", offset[3:6], degrees=True).as_matrix()

    for tr_init, name in model:
        transform = (
            np.matmul(add_rotation, tr_init) + np.array(
                [[0, 0, 0, offset[0]],
                 [0, 0, 0, offset[1]],
                 [0, 0, 0, offset[2]],
                 [0, 0, 0, 0]]
            )
        )

        position = transform[:3, 3]
        scale = np.array([np.linalg.norm(transform[:3, i]) for i in range(3)])
        rotation = transform[:3, :3] / scale
        euler = Rotation.from_matrix(rotation).as_euler('xyz', degrees=True)

        pivot_diff = np.array([0, -1, 0]) * scale
        correction = pivot_diff - np.matmul(rotation, pivot_diff)

        new_position = position + \
            np.matmul(rotation, np.array([1, -1, -1]) * scale) + correction

        scale_x = scale[1] * SCALE
        scale_y = scale[2] * SCALE
        start_x = new_position[1] * SCALE
        start_y = new_position[2] * SCALE
        tt = -new_position[0]

        walls.append({
            "_time": tt,
            "_lineIndex": 0,
            "_type": 0,
            "_duration": 2 * scale[0],
            "_width": 0,
            "_customData": {
                "_color": [42, 42, 42, 0] if name.startswith("dino") else [0, 0, 0, 1],
                "_interactable": False,
                "_noteJumpMovementSpeed": ENV_NJS,
                "_noteJumpStartBeatOffset": ENV_OFF,
                "_position": [start_x, start_y],
                "_scale": [2*scale_x, 2*scale_y],
                "_rotation": offset[6],
                "_localRotation": [-euler[1], -euler[2], euler[0]],
                "_animation": {
                    "_dissolve": [
                        [0, 0],
                        [1, 0.1],
                    ]
                }
            }
        })

        if tt > 400 and tt < 431 and collection_name != "main":
            walls[-1]["_customData"]["_track"] = "b"
        elif tt > 431 and tt < 450 and collection_name != "main":
            walls[-1]["_customData"]["_track"] = "a"


def add_windmill(walls, custom_data):
    model = load_model("windmill.dae")

    for tr_init, _ in model:
        transform = (
            tr_init + np.array(
                [[0, 0, 0, -31.18],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]]
            )
        )

        position = transform[:3, 3]
        scale = np.array([np.linalg.norm(transform[:3, i]) for i in range(3)])
        rotation = transform[:3, :3] / scale
        euler = Rotation.from_matrix(rotation).as_euler('xyz', degrees=True)

        pivot_diff = np.array([0, -1, 0]) * scale
        correction = pivot_diff - np.matmul(rotation, pivot_diff)

        new_position = position + \
            np.matmul(rotation, np.array([1, -1, -1]) * scale) + correction

        scale_x = scale[1] * SCALE
        scale_y = scale[2] * SCALE
        start_x = new_position[1] * SCALE
        start_y = new_position[2] * SCALE

        walls.append({
            "_time": -new_position[0],
            "_lineIndex": 0,
            "_type": 0,
            "_duration": 2 * scale[0],
            "_width": 0,
            "_customData": {
                "_track": "mill",
                "_color": [0, 0, 0, 1],
                "_interactable": False,
                "_noteJumpMovementSpeed": ENV_NJS,
                "_noteJumpStartBeatOffset": ENV_OFF,
                "_position": [start_x, start_y],
                "_scale": [2*scale_x, 2*scale_y],
                "_localRotation": [-euler[1], -euler[2], euler[0]],
                "_animation": {
                    "_dissolve": [
                        [0, 0],
                        [1, 0.1],
                    ],
                    "_rotation": [
                        [0, 0, 0, .0],
                        [0, 0, 120, .2],
                        [0, 0, 240, .4],
                        [0, 0, 0, .6],
                        [0, 0, 120, .8],
                        [0, 0, 240, 1.],
                    ],
                }
            }
        })

        track_events = [
            {
                "_time": 8,
                "_type": "AssignTrackParent",
                "_data": {
                    "_childrenTracks": [
                        "mill"
                    ],
                    "_parentTrack": "millpar"
                }
            },
            {
                "_time": 10,
                "_type": "AnimateTrack",
                "_data": {
                    "_track": "millpar",
                    "_position": [
                        [-SCALE*3.81, SCALE*2.804, 0, 0],
                    ],
                    "_duration": 40
                }
            },
        ]
        custom_data["_customEvents"] += track_events


def letter_width(model):
    return max(np.matmul(transform, np.transpose([x, y, z, 1]))[1]
               for transform, _ in model
               for x, y, z in itertools.product(*[[-1., 1.]]*3))


def add_text(walls):
    letter_spacing = 0.3
    space_width = 1.5
    spawn_off = 0.2
    fade_in = 0.5
    fade_out = 1
    depth = 30
    thickness = 0.05

    # [word, ti], tf, height
    text = [
        ([("send ", 75.5), ("your", 83.5)], 95, 5.5),
        ([("dreams", 91.5)], 95, 3.2),
        ([("where ", 95.8), ("no", 96.8), ("bo", 97.8), ("dy", 98.8)], 105, 5.5),
        ([("hides", 100)], 105, 3.2),
        ([("give ", 140), ("your", 147.5)], 159, 5.5),
        ([("tears", 156)], 159, 3.2),
        ([("to ", 159.8), ("the ", 160.8), ("tide", 162)], 170, 4.5),
        ([("no ", 208), ("time", 210)], 214, 4.5),
        ([("no ", 240), ("time", 242)], 246, 4.5),
        ([("there", 331.5), ("zs ", 339), ("no", 340)], 351, 5.5),
        ([("end", 348)], 351, 3.2),
        ([("there ", 351.5), ("is ", 353), ("no", 354)], 362, 5.5),
        ([("good", 355), ("bye", 356)], 362, 3.2),
        ([("di", 396), ("sap", 403.5), ("pear", 412)], 415, 4.5),
        ([("with ", 416), ("the ", 417), ("night", 418)], 427, 5),
        ([("no ", 464), ("time", 466)], 470, 4.5),
        ([("no ", 496), ("time", 498)], 502, 4.5),
        ([("no ", 528), ("time", 530)], 534, 4.5),
        ([("no ", 560), ("time", 562)], 566, 4.5),
        ([("no ", 592), ("time", 594)], 598, 4.5),
        ([("wait by x83", 806)], 812, 4.5),
        ([("mapped by nyri0", 814)], 820, 4.5),
    ]
    models = {letter: load_model(f"font/{letter}.dae")
              for letter in "abdeghimnoprstuvwxyz083"}
    letter_widths = {key: letter_width(val) + letter_spacing for key, val in models.items()}
    models[" "] = []
    letter_widths[" "] = space_width
    
    for sentence, tf, height in text:
        sentence_width = sum(letter_widths[letter] for word, _ in sentence for letter in word)
        offset = -sentence_width / 2

        for word, ti in sentence:
            il = 0
            for letter in word:
                il += int(letter != " ")
                model = models[letter]
                for transform, _ in model:
                    position = transform[:3, 3]
                    scale = np.array([np.linalg.norm(transform[:3, i]) for i in range(3)])
                    rotation = transform[:3, :3] / scale
                    euler = Rotation.from_matrix(rotation).as_euler('xyz', degrees=True)

                    pivot_diff = np.array([0, -1, 0]) * scale
                    correction = pivot_diff - np.matmul(rotation, pivot_diff)

                    new_position = position + \
                        np.matmul(rotation, np.array([1, -1, -1]) * scale) + correction

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
                                    [1, 1. - (apparition_offset + fade_out) / (tf - ti)],
                                    [0, 1. - apparition_offset / (tf - ti)],
                                ]
                            }
                        }
                    })
                offset += letter_widths[letter]


def add_tide(walls):
    ti, tf = 164, 174
    depth = 100
    width = 100
    start_height, end_height = 0.5, 10
    start_z, end_z = -20, -20
    nj_offset = (tf - ti) / 2 - BASE_HJ
    walls.append({
        "_time": ti + BASE_HJ + nj_offset,
        "_lineIndex": 0,
        "_type": 0,
        "_duration": 0,
        "_width": 0,
        "_customData": {
            "_interactable": False,
            "_noteJumpMovementSpeed": 16,
            "_noteJumpStartBeatOffset": nj_offset,
            "_position": [0, 0],
            "_scale": [width, 0.1, depth],
            "_animation": {
                "_definitePosition": [
                    [-width/2, start_height, start_z, 0.6],
                    [-width/2, end_height, end_z, 1, "easeInCubic"],
                ],
                "_color": [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, -10, .2],
                    [0, 0, 0, -10, .8],
                    [0, 0, 0, 0, 1],
                ],
                "_dissolve": [
                    [0, 0],
                    [1, .2],
                    [1, .8],
                    [0, 1.],
                ]
            }
        }
    })


def process_notes(notes):
    for note in notes:
        note["_customData"] = {
            "_disableSpawnEffect": True,
        }


def main():
    walls = []

    # Main geometry
    print("Adding main track", end="\r")
    add_model(walls, load_model(BLENDER_FILE), (0, 0, 0, 0, 0, 0, 0), "main")
    print("Added main track ")

    # Random spawns
    print("Spawning random elements", end="\r")
    for spawn in SPAWNS:
        collection = COLLECTIONS[spawn[0]]
        nb, ranges = spawn[1:]
        models = [load_model("{}.dae".format(item[0])) for item in collection]
        probas = [item[1] for item in collection]
        gen = np.random.choice(len(collection), nb, p=probas)
        offsets = [np.random.uniform(r_low, r_high, nb)
                   for r_low, r_high in ranges]
        for i in range(nb):
            model = models[gen[i]]
            offset = [offsets[j][i] for j in range(7)]
            add_model(walls, model, offset, spawn[0])
    print("Spawned random elements ")

    # Special effects
    print("Adding special effects", end="\r")
    add_tide(walls)
    print("Added special effects ")

    # Text
    print("Adding text", end="\r")
    add_text(walls)
    print("Added text ")

    # Lights off
    events = [{"_time": 0, "_type": i, "_value": 0} for i in range(5)]

    # Prevent MM from overwriting info.dat
    print("Copying info file", end="\r")
    shutil.copyfile("info.json", "info.dat")
    print("Copied info file ")

    print("Sorting and truncating", end="\r")
    walls = trunc(sorted(walls, key=lambda x: x["_time"]))
    print("Sorted and truncated  ")

    custom_data = {
        "_environment": [
            {
                "_id": id_,
                "_lookupMethod": "Regex",
                "_active": False,
            }
            for id_ in ["^.*Floor$",
                        "^.*Columns$",
                        "^.*Construction$",
                        "^.*Building.*$",
                        "^.*Spectrograms$",
                        "^.*BakedBloom$",
                        "^.*DirectionalLight$",
                        "^.*BoxLight$",
                        "^.*SaberBurnMarks.*$",
                        "^.*Mirror*$",
                        "^.*Ring.*$",
                        "^.*EnergyPanel$"]
        ],
        "_customEvents": [
            {
                "_time": 412,
                "_type": "AnimateTrack",
                "_data": {
                    "_track": "b",
                    "_dissolve": [
                        [1, 0],
                        [0, 1],
                    ],
                    "_duration": 0.5
                }
            },
            {
                "_time": 412,
                "_type": "AnimateTrack",
                "_data": {
                    "_track": "a",
                    "_dissolve": [
                        [0, 0.947],
                        [1, 1],
                    ],
                    "_duration": 19
                }
            },
        ]
    }

    # Windmill
    print("Adding windmill", end="\r")
    add_windmill(walls, custom_data)
    print("Added windmill ")

    for file_in, file_out in FILES:
        print(f"Processing {file_out}", end="\r")
        with open(file_in, "r") as json_file:
            song_json = json.load(json_file)

        process_notes(song_json["_notes"])
        song_json["_obstacles"] = walls
        song_json["_customData"] = custom_data
        song_json["_events"] = events

        with open(file_out, "w") as json_file:
            json.dump(song_json, json_file)
        print(f"Processed {file_out} ")


main()
