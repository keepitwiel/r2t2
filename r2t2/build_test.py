import numpy as np
import taichi as ti

ti.init(ti.cpu)

print("Hello, world!")
print("Numpy version:", np.__version__)
print("Taichi version:", ti.__version__)
