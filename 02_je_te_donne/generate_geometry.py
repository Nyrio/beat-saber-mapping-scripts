import json
import imageio
import numpy as np

FILES = ["ExpertStandard.dat"]


def fix_time(t):
    """The change of BPM causes a mess with Noodle Extensions...
    """
    return t if t < 320 else 320 + 122 * (t - 320) / 136


def load_fonts(lyrics_json):
    fonts = {}

    for name, alphabet in lyrics_json["fonts"].items():
        font_img = imageio.imread("{}.png".format(name))
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

        fonts[name] = (img_dim[0], font)

    return fonts


def main():
    with open("lyrics.json", encoding='raw_unicode_escape') as json_file:
        lyrics_json = json.loads(
            json_file.read().encode('raw_unicode_escape').decode())

    fonts = load_fonts(lyrics_json)
    scale = lyrics_json["pixel_scale"]
    word_spacing = lyrics_json["word_spacing"]
    letter_spacing = lyrics_json["letter_spacing"]
    duration = lyrics_json["text_thickness"]
    time_offset = lyrics_json["time_offset"]

    walls = []

    for group in lyrics_json["lyrics"]:
        font_height, font = fonts[group["font"]]
        text_width = sum(font[letter][0]
                         for word in group["text"]
                         for letter in word["word"])
        text_width += word_spacing * (len(group["text"]) - 1)
        text_width += letter_spacing * sum(
            len(word["word"]) - 1 for word in group["text"])

        offset = [lyrics_json["side_offsets"][group["side"]] - text_width / 2,
                  lyrics_json["line_offsets"][group["line"]]]

        text = reversed(group["text"]) if group["side"] == 1 else group["text"]
        for word in text:
            for letter in word["word"]:
                for letter_wall in font[letter][1]:
                    walls.append({
                        "_time": fix_time(word["time"] + time_offset),
                        "_lineIndex": 0,
                        "_type": 0,
                        "_duration": duration,
                        "_width": 0,
                        "_customData": {
                            "_position": [scale * (letter_wall[0] + offset[0]),
                                          scale * (letter_wall[1] + offset[1])],
                            "_scale": [scale * letter_wall[2],
                                       scale * letter_wall[3]],
                            "_rotation": lyrics_json["side_angles"][group["side"]],
                            "_localRotation": [0, 0, 0],
                        }
                    })
                offset[0] += font[letter][0] + letter_spacing
            offset[0] += word_spacing - letter_spacing
    
    walls.sort(key=lambda x: x["_time"])

    for filename in FILES:
        with open(filename, "r") as json_file:
            song_json = json.load(json_file)

        song_json["_obstacles"] = walls

        with open(filename, "w") as json_file:
            json.dump(song_json, json_file)


main()
