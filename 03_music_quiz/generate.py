import json
import imageio
import numpy as np

FILES = ["EasyStandard.dat"]
RATIO = 16  # length of a beat in BeatWalls units
SCALE = 0.05
THICKNESS = 0.1
SCROLL_DIST = 8
LETTER_SPACING = 2
DIV_WALL_WIDTH = 0.2
DIV_WALL_HEIGHT = 5


def load_font():
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,'#1234567890-"
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


def add_text(walls, font, text):
    text_width = sum(font[letter][0] for letter in text["content"])
    text_width += LETTER_SPACING * (len(text["content"]) - 1)

    offset = text["offset"] if "offset" in text else 0
    if text["type"] in {"ground", "air"}:
        offset -= text_width // 2

    for letter in text["content"]:
        for letter_wall in font[letter][1]:
            if text["type"] == "ground":
                pos = [SCALE * (letter_wall[0] + offset),
                        0,
                        text["time"] + SCALE * letter_wall[1] / RATIO]
                dim = [SCALE * letter_wall[2],
                        THICKNESS,
                        SCALE * letter_wall[3] / RATIO]
            elif text["type"] == "air":
                pos = [SCALE * (letter_wall[0] + offset),
                        SCALE * (letter_wall[1] + text["height"]),
                        text["time"]]
                dim = [SCALE * letter_wall[2],
                        SCALE * letter_wall[3],
                        THICKNESS / RATIO]
            elif text["type"] == "sidescroll":
                pos = [-SCROLL_DIST,
                        SCALE * (letter_wall[1] + text["height"]),
                        text["time"] + SCALE * (letter_wall[0] + offset) / RATIO]
                dim = [THICKNESS,
                        SCALE * letter_wall[3],
                        SCALE * letter_wall[2] / RATIO]
            walls.append({
                "_time": pos[2],
                "_lineIndex": 0,
                "_type": 0,
                "_duration": dim[2],
                "_width": 0,
                "_customData": {
                    "_position": pos[:2],
                    "_scale": dim[:2],
                    "_rotation": 90 if text["type"] == "sidescroll" else text["angle"] if "angle" in text else 0,
                    "_localRotation": [0, 0, 0],
                }
            })
        offset += font[letter][0] + LETTER_SPACING


def main():
    with open("text.json", encoding='raw_unicode_escape') as json_file:
        content_json = json.loads(
            json_file.read().encode('raw_unicode_escape').decode())

    _, font = load_font()

    walls = []
    notes = []

    for text in content_json["text"]:
        add_text(walls, font, text)
    
    for choice in content_json["choices"]:
        walls.append({
            "_time": choice["time"],
            "_lineIndex": 0,
            "_type": 0,
            "_duration": 2.1,
            "_width": 0,
            "_customData": {
                "_position": [-DIV_WALL_WIDTH / 2, 0],
                "_scale": [DIV_WALL_WIDTH, DIV_WALL_HEIGHT],
                "_rotation": 0,
                "_localRotation": [0, 0, 0],
            }
        })
        add_text(walls, font, {
            "type": "air",
            "content": "bravo",
            "time": choice["time"] + 2,
            "height": 60,
            "offset": -50 if choice["side"] == "left" else 50
        })
        add_text(walls, font, {
            "type": "air",
            "content": "wrong",
            "time": choice["time"] + 2,
            "height": 60,
            "offset": -50 if choice["side"] == "right" else 50
        })
        notes.append({
            "_time": choice["time"] + 2,
            "_lineIndex": 0 if choice["side"] == "left" else 3,
            "_lineLayer": 0,
            "_type": 1,
            "_cutDirection": 1
        })

    walls.sort(key=lambda x: x["_time"])

    for filename in FILES:
        with open(filename, "r") as json_file:
            song_json = json.load(json_file)

        song_json["_obstacles"] = walls
        song_json["_notes"] = notes

        with open(filename, "w") as json_file:
            json.dump(song_json, json_file)


main()
