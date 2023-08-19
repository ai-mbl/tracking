import numpy as np
import skimage
import matplotlib.pyplot as plt

plt.close("all")
x = skimage.io.imread("figures/mbl.png")

ds = 3
x = x[::ds, ::ds, :]

red_th = 100
blue_th = 150
cyan = x[..., 0] <= red_th
green = (x[..., 0] > red_th) & (x[..., 0] < 255) &  (x[..., 2] < blue_th)

cyan = skimage.transform.rotate(cyan, -90)
green = skimage.transform.rotate(green, -90)

fig, ax = plt.subplots(1,3)
ax[0].imshow(x)
ax[1].imshow(cyan)
ax[2].imshow(green)

plt.show()

cyan = np.stack(np.nonzero(cyan)).transpose()
green = np.stack(np.nonzero(green)).transpose()

np.savez("points.npz", cyan=cyan, green=green)

p = np.load("points.npz")

plt.figure()
plt.axes().set_aspect('equal')
plt.scatter(p["cyan"][:,0], p["cyan"][:, 1], c="c")
plt.scatter(p["green"][:,0], p["green"][:, 1], c="g")
plt.show()

