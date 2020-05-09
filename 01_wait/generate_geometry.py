import argparse
import json
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation

BLENDER_FILE = "track.dae"
SCALE = 5  # Scaling factor for the x and y axes
NS = {"xmlns": "http://www.collada.org/2005/11/COLLADASchema"}
HYPER_DURATION = -10

COLLECTIONS = {
    "desert": [("tiny_rock", 0.5), ("small_rock", 0.4), ("big_rock", 0.1)],
    "ocean_animals": [("turtle", 0.75), ("squid", 0.25)],
    "seabed": [("plant0", 0.4), ("plant1", 0.2), ("seastar", 0.4)],
    "sand": [("tiny_rock", 1)],
    "stars": [("star", 1)],
    "sat0": [("sat0", 1)],
    "sat1": [("sat1", 1)],
    "sat2": [("sat2", 1)],
}

# collection, nb, ranges: x, y, z, ax, ay, az, spawn angle
SPAWNS = [
    ("desert", 500, [(-173, -1), (-5, -0.75),
                     (0, 0), (0, 0), (0, 0), (0, 360), (0, 0)]),
    ("desert", 500, [(-173, -1), (0.75, 5),
                     (0, 0), (0, 0), (0, 0), (0, 360), (0, 0)]),
    ("ocean_animals", 40, [(-430, -178), (-3, -0.5),
                           (0.4, 2.0), (-10, 10), (0, 0), (0, 0), (5, 20)]),
    ("ocean_animals", 40, [(-430, -178), (0.5, 3),
                           (0.4, 2.0), (-10, 10), (0, 0), (0, 0), (-20, -5)]),
    ("seabed", 50, [(-430, -178), (-4, -0.75),
                    (0, 0), (0, 0), (0, 0), (0, 360), (0, 0)]),
    ("seabed", 50, [(-430, -178), (0.75, 4),
                    (0, 0), (0, 0), (0, 0), (0, 360), (0, 0)]),
    ("sand", 1000, [(-430, -178), (-5, 5),
                    (0, 0), (0, 0), (0, 0), (0, 360), (0, 0)]),
    ("stars", 2000, [(-828, -430), (-4, -0.2),
                     (0, 3), (0, 0), (0, 0), (0, 360), (0, 0)]),
    ("stars", 2000, [(-828, -430), (0.2, 4),
                     (0, 3), (0, 0), (0, 0), (0, 360), (0, 0)]),
    ("sat0", 1, [(-460, -460), (-1, -1),
                 (1.5, 1.5), (0, 0), (0, 0), (-20, -20), (20, 20)]),
    ("sat1", 1, [(-620, -620), (1.5, 1.5),
                 (2, 2), (0, 0), (10, 10), (0, 0), (-25, -25)]),
    ("sat2", 1, [(-678, -678), (-1, -1),
                 (1, 1), (0, 0), (-10, -10), (0, 0), (10, 10)]),
    ("sat0", 1, [(-740, -740), (1, 1),
                 (1.5, 1.5), (0, 0), (0, 0), (-20, -20), (-20, -20)]),
]


def get_args():
    parser = argparse.ArgumentParser(
        description='Generate geometry of the map')
    parser.add_argument('-o', dest='out_file', default='track.json',
                        help='Path to the output json file')
    return parser.parse_args()


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


def add_model(walls, model, offset):
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

        walls.append({
            "_time": -new_position[0],
            "_lineIndex": 0,
            "_type": 0,
            "_duration": HYPER_DURATION if name.lower().startswith("hyper") else 2 * scale[0],
            "_width": 0,
            "_customData": {
                "_position": [start_x, start_y],
                "_scale": [2*scale_x, 2*scale_y],
                "_rotation": offset[6],
                "_localRotation": [-euler[1], -euler[2], euler[0]],
            }
        })


def door_effect(walls):
    for angle in range(-80, 81, 10):
        transform = np.array([
            [0.005, 0, 0, 0],
            [0, 0.03, 0, 0],
            [0, 0, 0.005, 0],
            [0, 0, 0, 0]
        ])
        add_model(walls, [(transform, "hyper")], (-441, 0, 0, 0, 0, 0, angle))


def main():
    args = get_args()

    walls = []

    # Main geometry
    add_model(walls, load_model(BLENDER_FILE), (0, 0, 0, 0, 0, 0, 0))

    # Random spawns
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
            add_model(walls, model, offset)

    # Special effects
    door_effect(walls)

    walls.sort(key=lambda x: x["_time"])

    with open(args.out_file, "r") as json_file:
        song_json = json.load(json_file)

    song_json["_obstacles"] = walls

    with open(args.out_file, "w") as json_file:
        json.dump(song_json, json_file)


main()
