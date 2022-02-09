# Beat Saber Mapping Scripts

## Intro

This is not a generic tool, just individual scripts that I made to generate my maps. You can use these scripts without any restriction (even for commissions), but all the assets (3D models, SVG drawings, etc) strictly belong to me and you cannot use them without my explicit permission. When you use my scripts or scripts derivated from them, I'd really appreciate if you mention me in the credits (either in the description on BeatSaver or in the map details that can be seen in-game).

Videos of the maps:

 - Wait: https://youtu.be/oe6ThhijPy0
 - Je te donne: https://youtu.be/NfRuHcYC620
 - Music Quiz: https://youtu.be/HKX3iOh1WPI
 - Hide & Seek: https://youtu.be/VWgx2RAqE6o
 - Star Wars: https://youtu.be/yerrcuhG8tI
 - Dondante: https://youtu.be/O2ez-ucOBHY
 - Ghost Choir: https://youtu.be/XC7RY9C_EHs
 - Spider Dance: https://youtu.be/fy8KmUor6fI
 - Wait (2021 remap): https://youtu.be/bGf5PJm_qxU
 - Someone Else's Hat: https://youtu.be/RX-payaVsKU

## FAQ

I get a lot of questions about my maps and the techniques and effects that I have developed. Here are a few pointers.

**I want to make a wall map / modchart, where do I start?**
Reading the [documentation of Heck, Chroma and Noodle Extensions](https://github.com/Aeroluna/Heck/wiki) is the main requirement. You need a good understanding of the mods. Then you can either make your own scripts or use the existing ones. [BeatWalls](https://github.com/spookyGh0st/beatwalls) and [ScuffedWalls](https://github.com/thelightdesigner/ScuffedWalls) are two fairly accessible options. My scripts are for people who have some programming knowledge and are interested in particular effects or systems i have implemented for my maps.

**How do I use Blender-to-walls / Blender-to-environment?**
There are multiple scripts for that. The first one, for *Wait*, can create walls that scroll with the map. The second one, for *Dondante*, can do static animations using Noodle Extensions 1.2+ features. The script for *Someone Else's Hat* can animate some objects from the BTS environment too.
Here are a few tips to know, but you'll have to figure out a lot of things by yourself if you decide to use my scripts (again, ScuffedWalls is a more beginner-friendly option!):
- Blender models need to be made exclusively out of standard cubes, and modified only in object mode without ever applying the modifications in a way that will modify the vertices. I advise to start with an empty scene, spawn cubes (with the default scale) and only use rotate / move / rescale / duplicate.
- Blender and NE have different coordinate systems, so the math in my code is a mess, sorry for that.
- In terms of number of walls, you can go as far as hundreds on screen at the same time but be reasonable. Also don't spawn too many simultaneously, you'd have a lag spike. Spread spawns.

**How do I make text?**
There's text in multiple maps that I've made. Some use pixel art with fonts defined as png, some use svg graphics, some use Blender models.

**How do I use svg-to-walls?**
The *Hide & Seek* script does that. The main thing to know is that lines are rendered with one wall whereas Bezier curves are approximated with many walls, so use straight lines as much as possible. And export the file with the SVG Tiny specification. If some shapes don't display, the SVG files have primitives that my script doesn't understand.
The *Ghost Choir* script is based on the same ideas but uses Noodle Extensions 1.2+ animations to animate the svg files (frame by frame). If you want to reuse the same walls for multiple frames like in this script, to reduce lag, you need to modify the same file and make sure you're not adding new points.
The *Someone Else's Hat* script uses an improved version of the *Ghost Choir* script.