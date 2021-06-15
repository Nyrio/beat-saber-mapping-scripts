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

## FAQ

I get a lot of questions about my maps and the techniques and effects that I have developed. I will try to answer the most frequently asked questions here. If you have more questions, please use the channel *extension talk* of the *Beat Saber Mapping* Discord server. I don't have time to provide systematic support but I can help a bit if you really make an effort to understand how this stuff works.

**I want to make a wall map, where do I start?**
Reading the [documentation of Noodle Extensions](https://github.com/Aeroluna/NoodleExtensions) is the main requirement. You need a good understanding of the mod. Then you can either make your own scripts or use the existing ones. [BeatWalls](https://github.com/spookyGh0st/beatwalls) and [ScuffedWalls](https://github.com/thelightdesigner/ScuffedWalls) are two fairly accessible options. My scripts are for people who have some programming knowledge and are interested in particular effects or systems i have implemented for my maps.

**Have other mappers used these scripts?**
Yes, some cool maps have been made with scripts derivated from mine, such as [Midnight Lady](https://youtu.be/pE_s9bvntA0) by Reaxt, [Gloom](https://youtu.be/b0K8UBGt3zs) by Mine Thing, [Rain](https://youtu.be/a4h04wDuB64) by Caeden117, etc.
Other tools like ScuffedWalls are reimplementing some of the techniques I developed (model to walls, text to walls, etc).

**How do I use Blender-to-walls?**
There are multiple scripts for that. The first one, for *Wait*, can create environments that scroll with the map. The second one, for *Dondante*, can do static animations using Noodle Extensions 1.2+.
Here are some basic instructions for the *Wait* script, but you'll also have to look at the code and the 3D files to understand how it works.
- Blender models need to be made exclusively out of standard cubes, and modified only in object mode without ever applying the modifications in a way that will modify the vertices. I advise to start with an empty scene, spawn cubes (with the default scale) and only use rotate / move / rescale / duplicate.
- You need to adjust the `SCALE` parameter in the code based on your map's BPM and NJS
- Blender and NE have different coordinate systems, so the math in my code is a mess, sorry for that.
- In terms of number of walls, you can go as far as hundreds on screen at the same time but be reasonable. Also don't spawn too many simultaneously, you'd have a lag spike

**How do I use text-to-walls?**
There's text in multiple maps that I've made. You can create text with Blender-to-walls and have a lot more control but it's also more effort. Reaxt has done that in a few maps. Or you could use either the *Je te donne*, *Music Quiz* or *Hide & Seek* script based on which effect you're trying to achieve. They all work a bit differently but the basic idea is to provide a font and some kind of representation of the words and their timings. The font is parsed from a pixel art png file, with a greedy algorithm that takes the longest line of pixels and makes a wall with it, then does the same for the remaining pixels etc. You can change the height, and the characters have a variable width (you just need to separate them with an empty column with a blue pixel at the top)

**How do I use svg-to-walls?**
The *Hide & Seek* script does that. The main thing to know is that lines are rendered with one wall whereas Bezier curves are approximated with many walls, so use straight lines as much as possible. And export the file with the SVG Tiny specification. If some shapes don't display, the SVG files have primitives that my script doesn't understand.
The *Ghost Choir* script is based on the same ideas but uses Noodle Extensions 1.2+ animations to animate the svg files (frame by frame). If you want to reuse the same walls for multiple frames like in this script, to reduce lag, you need to modify the same file and make sure you're not adding new points.
