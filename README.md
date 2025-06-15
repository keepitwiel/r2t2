# R2T2: a simple Ray-Tracing Terrain Renderer

![Example 1](documentation/example.gif)

This project started from the a desire to make a real-time, realistic,
top-down height map renderer (a.k.a. relief mapper).

It allows the user to render shadows cast by a directional light
onto a height map.

## Installation

```commandline
git clone https://github.com/keepitwiel/r2t2
```

Open the cloned repository in PyCharm, install a virtual environment and install the package as follows:

```commandline
pip install r2t2
```

## Getting started

### Hello World:

```python
import numpy as np
import taichi as ti
from r2t2 import Renderer

# initialize backend framework
ti.init(ti.cpu)

# create height map
mx, my = np.mgrid[-1:1:11j, -1:1:11j]
height_map = np.sin(mx * np.pi) + np.sin(my * np.pi)

# create a color array: this gives each cell an RGB color.
# in this case, each cell is white
w, h = height_map.shape
color_map = np.ones((w, h, 3))  

# initialize renderer
renderer = Renderer(height_map, canvas_shape=height_map.shape)

# render scene
renderer.render(color_map)

# get image
image = renderer.get_image()  # an 11x11 RGB float32 array
```

### Examples
The following examples are included in the package, but require
some extra packages. These can be installed as follows:

```commandline
pip install r2t2[examples]
```

```commandline
python examples/example1.py
```
This example gives you a GUI window where you can play around with
R2T2's parameters.

```commandline
python examples/example2.py
```
A simple example showing framerate (FPS) of a height map based on an image of a raccoon.
On a Macbook Air M3 (2024) with 8GB RAM, the framerate is around 90 FPS in GPU mode.

```commandline
python examples/example3.py
```
New in version 1.1: this example shows the MinimalRenderer in action.

```commandline
python examples/example_maximum_altitude.py
```
New in version 1.2: this example shows the Maximum Altitude renderer in action.
