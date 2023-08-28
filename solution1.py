# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Exercise 1: Tracking by detection and simple frame-by-frame matching
#
# You could also run this notebook on your laptop, a GPU is not needed.
#
# ![](figures/tracking.gif "tracking")
#
# <div class="alert alert-danger">
# Set your python kernel to <code>08-tracking</code>
# </div>
#
# Here we will walk through all basic components of a tracking-by-detection algorithm.
#
# You will learn
# - to **store and visualize** tracking results with `napari` (Exercise 1.1).
# - to use a robust pretrained deep-learning-based **object detection** algorithm called *StarDist* (Exercise 1.2).
# - to implement a basic **nearest-neighbor linking algorithm** (Exercises 1.3 - 1.6).
# - to compute optimal frame-by-frame linking by setting up a **bipartite matching problem** and using a python-based solver (Exercise 1.7).
# - to compute suitable object **features** for the object linking process with `scikit-image` (Exercise 1.8).
#
# Places where you are expected to write code are marked with
# ```
# ######################
# ### YOUR CODE HERE ###
# ######################
# ```
#
# This notebook was originally written by Benjamin Gallusser.

# %% [markdown] editable=true jp-MarkdownHeadingCollapsed=true slideshow={"slide_type": ""}
# ## Import packages

# %% editable=true slideshow={"slide_type": ""} trusted=true
# Force keras to run on CPU
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Notebook at full width in the browser
from IPython.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))

import sys
from urllib.request import urlretrieve
from pathlib import Path
from collections import defaultdict
from abc import ABC, abstractmethod

import matplotlib
import matplotlib.pyplot as plt

# %matplotlib inline
matplotlib.rcParams["image.interpolation"] = "none"
matplotlib.rcParams["figure.figsize"] = (14, 10)
import numpy as np
from tifffile import imread, imwrite
from tqdm import tqdm
import skimage
import pandas as pd
import scipy

from stardist import fill_label_holes, random_label_cmap
from stardist.plot import render_label
from stardist.models import StarDist2D
from stardist import _draw_polygons
from csbdeep.utils import normalize

import napari

lbl_cmap = random_label_cmap()
# Pretty tqdm progress bars
# ! jupyter nbextension enable --py widgetsnbextension

# %% [markdown]
# Some utility functions


# %% editable=true slideshow={"slide_type": ""} trusted=true
def plot_img_label(img, lbl, img_title="image", lbl_title="label", **kwargs):
    fig, (ai, al) = plt.subplots(1, 2, gridspec_kw=dict(width_ratios=(1, 1)))
    im = ai.imshow(img, cmap="gray", clim=(0, 1))
    ai.set_title(img_title)
    ai.axis("off")
    al.imshow(render_label(lbl, img=0.3 * img, normalize_img=False, cmap=lbl_cmap))
    al.set_title(lbl_title)
    al.axis("off")
    plt.tight_layout()


def preprocess(X, Y, axis_norm=(0, 1)):
    # normalize channels independently
    X = np.stack(
        [
            normalize(x, 1, 99.8, axis=axis_norm)
            for x in tqdm(X, leave=True, desc="Normalize images")
        ]
    )
    # fill holes in labels
    Y = np.stack(
        [fill_label_holes(y) for y in tqdm(Y, leave=True, desc="Fill holes in labels")]
    )
    return X, Y


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Inspect the dataset

# %% [markdown]
# For this exercise we will be working with a fluorescence microscopy time-lapse of breast cancer cells with stained nuclei (SiR-DNA). It is similar to the dataset at https://zenodo.org/record/4034976#.YwZRCJPP1qt.

# %% trusted=true
base_path = Path("data/exercise1")

# %% [markdown]
# Load the dataset (images and tracking annotations) from disk into this notebook.

# %% trusted=true
x = np.stack(
    [imread(xi) for xi in sorted((base_path / "images").glob("*.tif"))]
)  # images
y = np.stack(
    [imread(yi) for yi in sorted((base_path / "gt_tracking").glob("*.tif"))]
)  # ground truth annotations
assert x.shape == y.shape
print(f"Number of images: {len(x)}")
print(f"Shape of images: {x[0].shape}")
x, y = preprocess(x, y)

# %% [markdown]
# Let's visualize some images (by changing `idx`).

# %% trusted=true
idx = 0
plot_img_label(x[idx], y[idx])

# %% [markdown]
# This is ok to take a glimpse, but a dynamic viewer would be much better to understand how cells move. Let's use [napari](https://napari.org/tutorials/fundamentals/getting_started.html) for this. Napari is a wonderful viewer for imaging data that you can interact with in python, even directly out of jupyter notebooks.

# %% trusted=true
viewer = napari.Viewer()
viewer.add_image(x, name="image")

# %% [markdown]
# If you've never used napari, you might want to take a few minutes to go through [this tutorial](https://napari.org/stable/tutorials/fundamentals/viewer.html).

# %% [markdown]
# <div class="alert alert-block alert-danger"><h3>Napari in a jupyter notebook:</h3>
#
# - To have napari working in a jupyter notebook, you need to use up-to-date versions of napari, pyqt and pyqt5, as is the case in the conda environments provided together with this exercise.
# - When you are coding and debugging, close the napari viewer with `viewer.close()` to avoid problems with the two event loops of napari and jupyter.
# - **If a cell is not executed (empty square brackets on the left of a cell) despite you running it, running it a second time right after will usually work.**
# </div>

# %% [markdown]
# Let's add the ground truth annotations.

# %% trusted=true
viewer.add_labels(y, name="labels")

# %% [markdown]
# Now it is easy to see that the nuclei have consistent IDs (visualized as random colors) over time.
#
# If you zoom in, you will note that the annotations are not perfect segmentations, but rather circles placed roughly in the center of each nucleus.
#
# If you look carefully, you will see that there are some cell divisions in this dataset, and the annotation color of the daughter cells does not match the color of the parent cell. This information is stored in an additional table, which we will load now.

# %% trusted=true
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()

# %% [markdown]
# Let's load in the cell divisions.

# %% trusted=true
links = np.loadtxt(base_path / "gt_tracking" / "man_track.txt", dtype=int)
links = pd.DataFrame(data=links, columns=["track_id", "from", "to", "parent_id"])
print("Links")
links[:10]


# %% [markdown]
# Each row in this table describes a daughter cell track:
# - it has a unique identifier *track_id*
# - it starts in frame *from*
# - it ends in frame *to*
# - it has a parent cell ID *parent_id*
#
# This is the standard data format of the [Cell Tracking Challenge](http://celltrackingchallenge.net) ([Ulman et al. (2017)](https://www.nature.com/articles/nmeth.4473])).

# %% [markdown]
# Here is a function to visualize the tracks of cells over time, including cell divisions. Note that the color of the track is also random and does not match the color of the corresponding spot.


# %% trusted=true
def visualize_tracks(viewer, y, links=None, name=""):
    """Utility function to visualize segmentation and tracks"""
    max_label = max(links.max(), y.max()) if links is not None else y.max()
    colorperm = np.random.default_rng(42).permutation((np.arange(1, max_label + 2)))
    tracks = []
    for t, frame in enumerate(y):
        centers = skimage.measure.regionprops(frame)
        for c in centers:
            tracks.append(
                [colorperm[c.label], t, int(c.centroid[0]), int(c.centroid[1])]
            )
    tracks = np.array(tracks)
    tracks = tracks[tracks[:, 0].argsort()]

    graph = {}
    if links is not None:
        divisions = links[links[:, 3] != 0]
        for d in divisions:
            if (
                colorperm[d[0]] not in tracks[:, 0]
                or colorperm[d[3]] not in tracks[:, 0]
            ):
                continue
            graph[colorperm[d[0]]] = [colorperm[d[3]]]

    viewer.add_labels(y, name=f"{name}_detections")
    viewer.layers[f"{name}_detections"].contour = 3
    viewer.add_tracks(tracks, name=f"{name}_tracks", graph=graph)
    return tracks


# %% trusted=true
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()
viewer = napari.Viewer()
viewer.add_image(x)
visualize_tracks(viewer, y, links.to_numpy(), "ground_truth")

# %% trusted=true
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Exercise 1.1
# <div class="alert alert-block alert-info"><h3>Exercise 1.1: Highlight the cell divisions</h3>
#
# The visualization of the ground truth tracks are useful to grasp this video, but it is still hard see the cell divisions. Given the dense annotations `y` and the track links `links`, write a function to create a new layer that highlights the pairs of daughter cells just after mitosis.
# </div>

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Expected outcome:<br>
# <figure style="display:inline-block">
#     <img src="figures/prediv.png" width="400" />
#     <figcaption>frame 6</figcaption>
# </figure>
# <figure style="display:inline-block">
#     <img src="figures/postdiv.png" width="400" />
#     <figcaption>frame 7</figcaption>
# </figure>


# %% trusted=true
def extract_divisions(img_labels, links):
    """Function to highlight divisions.

    Copies over the labels of daughter nuclei just after division into an empty numpy array.

    Args:
        img_labels: numpy array of shape (time, height, width).
            Each nucleus is marked with a unique integer ID.
        links: pandas DataFrame of tracklets with columns
            `track_id`, `from` (frame number), `to` (frame number), `parent_id`.
            If parent_id = 0, the tracklet is not associated to any mother cell.
    Returns:
        img_divisons: numpy array of shape (time, height, width).
            Only the daughter nuclei right after mitosis are marked.
    """

    img_divisions = np.zeros_like(img_labels)
    daughters = links[
        links.parent_id != 0
    ]  # identify daughters i.e. links whose parent_id is not zero

    for _, d in daughters.iterrows():
        ######################
        ### YOUR CODE HERE ###
        ######################
        print(d)

    return img_divisions


# %% editable=true slideshow={"slide_type": ""} tags=["solution"] trusted=true
# Solution Exercise 1.1
def extract_divisions(img_labels, links):
    """Function to highlight divisions.

    Copies over the labels of daughter nuclei just after division into an empty numpy array.

    Args:
        img_labels: numpy array of shape (time, height, width).
            Each nucleus is marked with a unique integer ID.
        links: pandas DataFrame of tracklets with columns
            `track_id`, `from` (frame number), `to` (frame number), `parent_id`.
            If parent_id = 0, the tracklet is not associated to any mother cell.
    Returns:
        img_divisons: numpy array of shape (time, height, width).
            Only the daughter nuclei right after mitosis are marked.
    """
    img_divisions = np.zeros_like(img_labels)
    daughters = links[links.parent_id != 0]

    for _, d in daughters.iterrows():
        frame_divs = img_divisions[d["from"]]
        frame_lbls = img_labels[d["from"]]
        daughter_id = d["track_id"]
        frame_divs[frame_lbls == daughter_id] = daughter_id

    return img_divisions


# %% [markdown]
# Test your function with this minimal example (with 1D toy "images" of shape (time, height)).


# %% editable=true slideshow={"slide_type": ""} trusted=true
def test_extract_divisions():
    y = np.array([[0, 10, 0, 0], [0, 11, 12, 13], [0, 11, 12, 13], [0, 11, 0, 13]])
    links = pd.DataFrame(
        [[11, 1, 3, 10], [12, 1, 2, 10]],
        columns=["track_id", "from", "to", "parent_id"],
    )
    divs = extract_divisions(y, links)
    expected_divs = np.array([[0, 0, 0, 0], [0, 11, 12, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    if np.all(divs == expected_divs):
        print("Success :)")
    else:
        print(f"Output\n{divs}\ndoes not match expected output\n{expected_divs}")


test_extract_divisions()

# %% [markdown]
# Visualize the output of your function:

# %% trusted=true
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()
viewer = napari.Viewer()
viewer.add_image(x)
divisions = extract_divisions(y, links)
viewer.add_labels(divisions, name="divisions")

# %% trusted=true
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Object detection using a pre-trained neural network

# %% [markdown]
# Now let's assume we have just acquired the raw video and want to track the cells in it from scratch! The first step is detecting the nuclei in each frame.
#
# We wil use **StarDist** (Schmidt et al., MICCAI 2018), a robust deep-learning based detection algorithm for cell nuclei. It represents objects as star-convex polygons, which in turn can be represented by a center point and distances along predefined rays going out from the center point.
# Please refer to the paper for details.
#
# We will load a pretrained StarDist model and directly run detection in the video at hand.
#
# [Schmidt, Uwe, et al. "Cell detection with star-convex polygons." MICCAI, 2018.](https://link.springer.com/chapter/10.1007/978-3-030-00934-2_30)

# %% trusted=true
idx = 0
model = StarDist2D.from_pretrained("2D_versatile_fluo")
(detections, details), (prob, _) = model.predict_instances(
    x[idx], scale=(1, 1), return_predict=True
)
plot_img_label(x[idx], detections, lbl_title="detections")

# %% [markdown]
# Here we visualize in detail the polygons we have detected with StarDist.

# %% editable=true slideshow={"slide_type": ""} trusted=true
coord, points, polygon_prob = details["coord"], details["points"], details["prob"]
plt.figure(figsize=(24, 12))
plt.subplot(121)
plt.title("Predicted Polygons")
_draw_polygons(coord, points, polygon_prob, show_dist=True)
plt.imshow(x[idx], cmap="gray")
plt.axis("off")

plt.subplot(122)
plt.title("Object center probability")
plt.imshow(prob, cmap="magma")
plt.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Exercise 1.2
# <div class="alert alert-block alert-info"><h3>Exercise 1.2: Explore the parameters of cell detection</h3>
#
# Explore the following aspects of the detection algorithm:
# - The `scale` parameter of the function `predict_instances` downscales (< 1) or upscales (> 1) the images by the given factor before feeding them to the neural network. How do the detections change if you adjust it?
# - Inspect false positive and false negative detections. Do you observe patterns?
# - So far we have used a StarDist model off the shelf. Luckily, we also have a StarDist model that was trained on a very similar breast cancer cell dataset (from https://zenodo.org/record/4034976#.Yv-aNPFBzao). Load it with `model = StarDist2D(None, name="stardist_breast_cancer", basedir="models")` and qualitatively observe differences.
#
# </div>

# %% [markdown]
# Detect centers and segment nuclei in all images of the time lapse.

# %% trusted=true
scale = (1.0, 1.0)
pred = [
    model.predict_instances(xi, show_tile_progress=False, scale=scale) for xi in tqdm(x)
]
detections = [xi[0] for xi in pred]
detections = np.stack(
    [skimage.segmentation.relabel_sequential(d)[0] for d in detections]
)  # ensure that label ids are contiguous and start at 1 for each frame
centers = [xi[1]["points"] for xi in pred]

# %% [markdown]
# Visualize the dense detections. Note that they are still not linked and therefore randomly colored.

# %% trusted=true
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()
viewer = napari.Viewer()
viewer.add_image(x)
viewer.add_labels(detections, name=f"detections_scale_{scale}")

# %% trusted=true
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()

# %% [markdown]
# We see that the number of detections increases over time, corresponding to the cells that insert the field of view from below during the video.

# %% editable=true slideshow={"slide_type": ""} trusted=true
plt.figure(figsize=(10, 6))
plt.bar(range(len(centers)), [len(xi) for xi in centers])
plt.title(f"Number of detections in each frame (scale={scale})")
plt.xticks(range(len(centers)))
plt.show()


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Checkpoint 1
# <div class="alert alert-block alert-success"><h3>Checkpoint 1: We have good detections, now on to the linking.</h3></div>

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Functions for greedy linking by nearest neighbor

# %% [markdown]
# The detections in each frame now need to be linked in time to form tracks. In the following exercises, we will explore different ways of doing this.
#
# We will start with the simplest algorithm: We take a pair of adjacent frames and compute a distance function between each detection $p \in P$ in frame $t$ and each detection $q \in Q$ in frame $t+1$. For example, we can calculate the euclidian distance between the two centroids of detections. This can be written as a matrix of size $|P| \times |Q|$.
#
# We want to minimize the total distance of the links we assign. A *greedy* (locally optimal) algorithm to do so is iteratively linking detections with the minimum distance that remains in the matrix.

# %% [markdown] editable=true jp-MarkdownHeadingCollapsed=true slideshow={"slide_type": ""}
# ## Exercise 1.3
# <div class="alert alert-block alert-info"><h3>Exercise 1.3: Write a function that computes pairwise euclidian distances given two lists of points, with two simple for-loops.</h3></div>


# %% editable=true slideshow={"slide_type": ""} trusted=true
def pairwise_euclidian_distance(points0, points1):
    """Computes pairwise euclidian distances of two lists of points.

    Args:
        points0, points1: Lists of point locations (x,y).
    Returns:
        dists: numpy array of size (#points0, #points1).
    """
    dists = np.zeros((len(points0), len(points1)))

    ######################
    ### YOUR CODE HERE ###
    ######################

    return dists


# %% editable=true slideshow={"slide_type": ""} tags=["solution"] trusted=true
# Solution Exercise 1.3
def pairwise_euclidian_distance(points0, points1):
    # Iterative pairwise euclidian distance
    dists = np.zeros((len(points0), len(points1)))
    for i, p0 in enumerate(points0):
        for j, p1 in enumerate(points1):
            dists[i, j] = np.sqrt(((p0 - p1) ** 2).sum())

    return dists


# def pairwise_euclidian_distance(points0, points1):
#     # Numpy-based, but still slow
#     print("Vectorized pairwise euclidian distance")
#     return np.apply_along_axis(
#         np.linalg.norm,
#         2,
#         points0[:, None, :] - points1[None, :, :]
#     )

# def pairwise_euclidian_distance(points0, points1):
#     print("Scipy pairwise euclidian distance")
#     return scipy.spatial.distance.cdist(points0, points1)


# %% [markdown] editable=true slideshow={"slide_type": ""}
# Here are two (almost random ;)) lists of points to test your function on.

# %% editable=true slideshow={"slide_type": ""} trusted=true
green_points = np.load("points.npz")["green"]
cyan_points = np.load("points.npz")["cyan"]

# %% editable=true slideshow={"slide_type": ""} trusted=true
# %time dists = pairwise_euclidian_distance(green_points, cyan_points)
assert np.allclose(dists, np.load("points.npz")["dists_green_cyan"])

# %% editable=true slideshow={"slide_type": ""} tags=["solution"] trusted=true
# You just calculated the distances between the green and the cyan color patches in the MBL logo ;)
plt.figure(figsize=(4, 4))
plt.scatter(green_points[:, 0], green_points[:, 1], c="yellowgreen", s=1)
plt.scatter(cyan_points[:, 0], cyan_points[:, 1], c="darkcyan", s=1)
plt.gca().set_aspect("equal")


# %% [markdown] editable=true jp-MarkdownHeadingCollapsed=true slideshow={"slide_type": ""}
# ## Exercise 1.4
# <div class="alert alert-block alert-info"><h3>Exercise 1.4: Complete a function that greedily extracts a nearest neighbor assignment given a cost matrix.</h3>
# </div>

# %% [markdown]
# Test your function with the cell below.
#
# Hints:
# - Make sure links do not exceed the cost threshold.
# - Once you've found the minimum in the matrix with the given code below (`row, col`), set


# %% trusted=true
def nearest_neighbor(cost_matrix, threshold=np.finfo(float).max):
    """Greedy nearest neighbor assignment.

    Each point in both sets can only be assigned once.

    Args:
        cost_matrix: m x n matrix with pairwise linking costs of two sets of points.
        threshold (int): Maximal cost of links.
    Returns:
        matches: List of tuples (from_id, to).
    """

    A = cost_matrix.copy().astype(float)
    matches = []

    for i in range(min(A.shape[0], A.shape[1])):
        # get the indices of the current minimum in the matrix
        row, col = np.unravel_index(A.argmin(), A.shape)

        ######################
        ### YOUR CODE HERE ###
        ######################
        # Hints:
        # 1) Check whether the costs of the link is below `threshold`
        # 2) Store the link
        # 3) Set the entire row and column of the found link to `threshold`

    return matches


# %% editable=true slideshow={"slide_type": ""} tags=["solution"] trusted=true
# Solution exercise 1.4


def nearest_neighbor(cost_matrix, threshold=np.finfo(float).max):
    """Greedy nearest neighbor assignment.

    Each point in both sets can only be assigned once.

    Args:
        cost_matrix: m x n matrix with pairwise linking costs of two sets of points.
        threshold (int): Maximal cost of links.
    Returns:
        matches: List of tuples (from_id, to).
    """
    A = cost_matrix.copy().astype(float)
    matches = []

    for i in range(min(A.shape[0], A.shape[1])):
        row, col = np.unravel_index(A.argmin(), A.shape)

        if A.min() > threshold:
            break
        matches.append((row, col))
        A[row, :] = threshold + 1
        A[:, col] = threshold + 1

    return matches


# %% [markdown]
# Test your implementation

# %% trusted=true
test_matrix = np.array(
    [
        [8, 2, 8],
        [9, 9, 9],
        [1, 8, 8],
        [8, 3, 8],
    ]
)
matches = nearest_neighbor(test_matrix, threshold=7)
assert np.all(matches[0] == (2, 0)), "The first match should be (2,0), at cost 1."
assert np.all(matches[1] == (0, 1)), "The second match should (0,1), at cost 2."
assert (
    len(matches) == 2
), "You should only produce two matches, since the maximal cost of links is 7."
print("Success :)")


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Linking

# %% [markdown]
# Now we provide you with the class `FrameByFrameLinker`, which is a complete class for linking detections in a video with some local frame-by-frame algorithm.
#
# This class has many methods that you won't have to worry about, such as iterating over frames, visualizing linked results as well as sanity checks of inputs.
#
# However, there are two methods which you will overwrite in the exercises 1.5 - 1.8 by making different [subclasses](https://en.wikipedia.org/wiki/Inheritance_(object-oriented_programming)#Subclasses_and_superclasses) of `FrameByFrameLinker`:
# - `linking_cost_function` takes the detections of two adjacent frames and calculates a pairwise cost matrix.
# - `_link_two_frames` takes the cost matrix for two frames and returns lists of links, appearing cells and disappearing cells.


# %% trusted=true
class FrameByFrameLinker:
    """Base class for linking detections by considering pairs of adjacent frames."""

    def linking_cost_function(self, detections0, detections1, image0=None, image1=None):
        """Calculate features for each detection and extract pairwise costs.

        Args:
            detections0: image with background 0 and detections 1, ..., m
            detections1: image with backgruond 0 and detections 1, ..., n
            image0 (optional): image corresponding to detections0
            image1 (optional): image corresponding to detections1
        Returns:
            dists: m x n cost matrix
        """

        # regionprops regions are sorted by label
        regions0 = skimage.measure.regionprops(detections0)
        points0 = [np.array(r.centroid) for r in regions0]

        regions1 = skimage.measure.regionprops(detections1)
        points1 = [np.array(r.centroid) for r in regions1]

        # Dummy cost function: Absolute difference between index numbers
        dists = np.zeros((len(points0), len(points1)))
        for i, p0 in enumerate(points0):
            for j, p1 in enumerate(points1):
                dists[i, j] = np.abs(i - j)

        return dists

    def _link_two_frames(self, cost_matrix):
        """Link two frames.

        Args:
            cost_matrix: m x n matrix
        Returns:
            links: Tuple of lists. Links from frame t to frame t+1 of form (from0, to0) are split up into two lists:
                - idgs_from: [from0, from1 , ...])
                - ids_to: [to0, to1 , ...])
            births: List of cell ids that newly appear in frame t+1.
            deaths: List of cell ids that disappear in frame t+1.

            Returned IDs are one-indexed, 0 is reserved for background.
        """
        # simply link diagonal elements in the cost matrix
        matches = [(i, i) for i in range(min(cost_matrix.shape))]

        # split links into two lists for easier handling
        ids_from = np.array([from_id for from_id, _ in matches])
        ids_to = np.array([to_id for _, to_id in matches])

        births = np.array(list(set(range(cost_matrix.shape[1])) - set(ids_to)))
        deaths = np.array(list(set(range(cost_matrix.shape[0])) - set(ids_from)))

        # Account for +1 offset dense labels, 0 = background
        ids_from += 1
        ids_to += 1
        births += 1
        deaths += 1

        links = {"links": (ids_from, ids_to), "births": births, "deaths": deaths}
        return links

    def link(self, detections, images=None):
        """Links detections in t frames.

        Args:
            detections: List of t numpy arrays of shape (x,y) with contiguous label ids. Background = 0.
            images (optional): List of t numpy arrays of shape (x,y).
        Returns:
            List of t linking dictionaries, each containing:
                "links": Tuple of lists (ids frame t, ids frame t+1),
                "births": List of ids,
                "deaths": List of ids.
            Ids are one-based, 0 is reserved for background.
        """
        if images is not None:
            assert len(images) == len(detections)
        else:
            images = [None] * len(detections)

        links = []
        for i in tqdm(range(len(images) - 1), desc="Linking"):
            detections0 = detections[i]
            detections1 = detections[i + 1]
            self._assert_relabeled(detections0)
            self._assert_relabeled(detections1)

            cost_matrix = self.linking_cost_function(
                detections0, detections1, images[i], images[i + 1]
            )
            li = self._link_two_frames(cost_matrix)
            self._assert_links(
                links=li, time=i, detections0=detections0, detections1=detections1
            )
            links.append(li)

        return links

    def relabel_detections(self, detections, links):
        """Relabel dense detections according to computed links, births and deaths.

        Args:
            detections:
                List of t numpy arrays of shape (x,y) with contiguous label ids. Background = 0.
            links:
                List of t linking dictionaries, each containing:
                    "links": Tuple of lists (ids frame t, ids frame t+1),
                    "births": List of ids,
                    "deaths": List of ids.
                Ids are one-based, 0 is reserved for background.
            Returns:
                Numpy array of shape (t,x,y) with relabeled instance ids.
        """
        detections = detections.copy()

        assert len(detections) - 1 == len(links)
        self._assert_relabeled(detections[0])
        out = [detections[0]]
        n_tracks = out[0].max()
        lookup_tables = [{i: i for i in range(1, out[0].max() + 1)}]

        for i in tqdm(range(len(links)), desc="Recoloring detections"):
            (ids_from, ids_to) = links[i]["links"]
            births = links[i]["births"]
            deaths = links[i + 1]["deaths"] if i + 1 < len(links) else []
            new_frame = np.zeros_like(detections[i + 1])
            self._assert_relabeled(detections[i + 1])

            lut = {}
            for _from, _to in zip(ids_from, ids_to):
                # Copy over ID
                new_frame[detections[i + 1] == _to] = lookup_tables[i][_from]
                lut[_to] = lookup_tables[i][_from]

            # Start new track for birth tracks
            for b in births:
                if b in deaths:
                    continue

                n_tracks += 1
                lut[b] = n_tracks
                new_frame[detections[i + 1] == b] = n_tracks

            lookup_tables.append(lut)
            out.append(new_frame)

        return np.stack(out)

    def _assert_links(self, links, time, detections0, detections1):
        if len(links["links"][0]) != len(links["links"][1]):
            raise RuntimeError("Format of links['links'] not correct.")

        if sorted([*links["links"][0], *links["deaths"]]) != list(
            range(1, len(np.unique(detections0)))
        ):
            raise RuntimeError(
                f"Some detections in frame {time} are not properly assigned as either linked or death."
            )

        if sorted([*links["links"][1], *links["births"]]) != list(
            range(1, len(np.unique(detections1)))
        ):
            raise RuntimeError(
                f"Some detections in frame {time + 1} are not properly assigned as either linked or birth."
            )

        for b in links["births"]:
            if b in links["links"][1]:
                raise RuntimeError(
                    f"Links frame {time+1}: Detection {b} marked as birth, but also linked."
                )

        for d in links["deaths"]:
            if d in links["links"][0]:
                raise RuntimeError(
                    f"Links frame {time}: Detection {d} marked as death, but also linked."
                )

    def _assert_relabeled(self, x):
        if x.min() < 0:
            raise ValueError("Negative ID in detections.")
        if x.min() == 0:
            n = x.max() + 1
        else:
            n = x.max()
        if n != len(np.unique(x)):
            raise ValueError("Detection IDs are not contiguous.")


# %% trusted=true
dummy_linker = FrameByFrameLinker()
dummy_links = dummy_linker.link(detections)
dummy_tracks = dummy_linker.relabel_detections(detections, dummy_links)

# %% [markdown]
# Visualize results.

# %% trusted=true
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()
viewer = napari.Viewer()
viewer.add_image(x)
visualize_tracks(viewer, dummy_tracks, name="dummy")

# %% [markdown]
# The extracted tracks are completely off. Let's fix that :)

# %% trusted=true
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Exercise 1.5
# <div class="alert alert-block alert-info"><h3>Exercise 1.5: Complete a thresholded nearest neighbor linker using your functions from exercises 1.3 and 1.4.</h3>
#
# You have to complete two methods:
#
# - Method 1 (`linking_cost_function`): Given dense detections in a pair of frames, extract their centroids and calculate pairwise euclidian distances between them.
# - Method 2 (`_link_two_frames`): We greedily find the nearest neighbors in two frames given the cost matrix. If the cost is below a threshold $\tau$, link the two objects.
#     - Complete the function such that it returns all unlinked detections in frame $t$ as death events and all unlinked detections in frame $t+1$ as birth events.
#     - Explore different values of threshold $\tau$.
# </div>


# %% trusted=true
class NearestNeighborLinkerEuclidian(FrameByFrameLinker):
    def __init__(self, threshold=sys.float_info.max, *args, **kwargs):
        """
        Args:
            threshold (float): Maximum euclidian distance for linking.
        """
        self.threshold = threshold
        super().__init__(*args, **kwargs)

    def linking_cost_function(self, detections0, detections1, image0=None, image1=None):
        """Get centroids from detections and compute pairwise euclidian distances.

        Args:
            detections0: image with background 0 and detections 1, ..., m
            detections1: image with backgruond 0 and detections 1, ..., n
        Returns:
            m x n cost matrix
        """
        # Extract centroids from detections. `regionprops` regions are already sorted by label id.
        regions0 = skimage.measure.regionprops(detections0)
        points0 = [np.array(r.centroid) for r in regions0]

        regions1 = skimage.measure.regionprops(detections1)
        points1 = [np.array(r.centroid) for r in regions1]

        dists = np.zeros((detections0.max(), detections1.max()))

        ######################
        ### YOUR CODE HERE ###
        ######################
        # Apply your pairwise euclidian cost function from Exercise 1.3

        return dists

    def _link_two_frames(self, cost_matrix):
        """Greedy nearest neighbor assignment.

        Returns:
            links: Tuple of lists. Links from frame t to frame t+1 of form (from0, to0) are split up into two lists:
                - idgs_from: [from0, from1 , ...])
                - ids_to: [to0, to1 , ...])
            births: List of cell ids that newly appear in frame t+1.
            deaths: List of cell ids that disappear in frame t+1.

            Returned IDs are one-indexed, 0 is reserved for background.
        """

        # Applying your nearest neighbor cost function from exercise 1.4
        matches = nearest_neighbor(cost_matrix, threshold=self.threshold)

        # split links into two lists for easier handling
        ids_from = np.array([from_id for from_id, _ in matches])
        ids_to = np.array([to_id for _, to_id in matches])

        births = np.array(
            [
                ######################
                ### YOUR CODE HERE ###
                ######################
                # All unmatched detections in frame t+1
                # Hint: use python `set` operations
            ]
        )
        deaths = np.array(
            [
                ######################
                ### YOUR CODE HERE ###
                ######################
                # All unmatched detections in frame t
                # Hint: use python `set` operations
            ]
        )

        # Account for +1 offset of the dense labels
        ids_from += 1
        ids_to += 1
        births += 1
        deaths += 1

        links = {"links": (ids_from, ids_to), "births": births, "deaths": deaths}
        return links


# %% editable=true slideshow={"slide_type": ""} tags=["solution"] trusted=true
# Solution Exercise 1.5
class NearestNeighborLinkerEuclidian(FrameByFrameLinker):
    def __init__(self, threshold=sys.float_info.max, *args, **kwargs):
        """
        Args:
            threshold (float): Maximum euclidian distance for linking.
        """
        self.threshold = threshold
        super().__init__(*args, **kwargs)

    def linking_cost_function(self, detections0, detections1, image0=None, image1=None):
        """Get centroids from detections and compute pairwise euclidian distances.

        Args:
            detections0: image with background 0 and detections 1, ..., m
            detections1: image with backgruond 0 and detections 1, ..., n
        Returns:
            m x n cost matrix
        """
        # Extract centroids from detections. `regionprops` regions are already sorted by label id.
        regions0 = skimage.measure.regionprops(detections0)
        points0 = [np.array(r.centroid) for r in regions0]

        regions1 = skimage.measure.regionprops(detections1)
        points1 = [np.array(r.centroid) for r in regions1]

        # Apply your pairwise euclidian cost function from Exercise 1.3
        dists = pairwise_euclidian_distance(points0, points1)

        return dists

    def _link_two_frames(self, cost_matrix):
        """Greedy nearest neighbor assignment.

        Returns:
            links: Tuple of lists. Links from frame t to frame t+1 of form (from0, to0) are split up into two lists:
                - idgs_from: [from0, from1 , ...])
                - ids_to: [to0, to1 , ...])
            births: List of cell ids that newly appear in frame t+1.
            deaths: List of cell ids that disappear in frame t+1.

            Returned IDs are one-indexed, 0 is reserved for background.
        """

        # Applying your nearest neighbor cost function from exercise 1.4
        matches = nearest_neighbor(cost_matrix, threshold=self.threshold)

        # split links into two lists for easier handling
        ids_from = np.array([from_id for from_id, _ in matches])
        ids_to = np.array([to_id for _, to_id in matches])

        births = np.array(list(set(range(cost_matrix.shape[1])) - set(ids_to)))
        deaths = np.array(list(set(range(cost_matrix.shape[0])) - set(ids_from)))

        # Account for +1 offset of the dense labels
        ids_from += 1
        ids_to += 1
        births += 1
        deaths += 1

        links = {"links": (ids_from, ids_to), "births": births, "deaths": deaths}
        return links


# %% trusted=true
nn_linker = NearestNeighborLinkerEuclidian(threshold=50)
nn_links = nn_linker.link(detections)
nn_tracks = nn_linker.relabel_detections(detections, nn_links)

# %% [markdown]
# Visualize results

# %% trusted=true
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()
viewer = napari.Viewer()
viewer.add_image(x)
visualize_tracks(viewer, nn_tracks, name="nn")

# %% trusted=true
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Checkpoint 2
# <div class="alert alert-block alert-success"><h3>Checkpoint 2: We built a basic tracking algorithm from scratch :).</h3></div>

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=[] jp-MarkdownHeadingCollapsed=true
# ## Exercise 1.6
# <div class="alert alert-block alert-info"><h3>Exercise 1.6: Estimate the global drift of the data</h3>
#
# We can observe that all cells move upwards with an approximately constant displacement in each timestep. Below you have a slightly modified version of `NearestNeighborLinkerEuclidian` with a modified `linking_cost_function` that models linear drift.
#
# Find values of `threshold` and `drift` that lead to an improved solution compared to exercise 1.5.
#
# </div>


# %% trusted=true
class NearestNeighborLinkerDriftCorrection(NearestNeighborLinkerEuclidian):
    def __init__(self, drift, *args, **kwargs):
        """
        Args:
            drift: tuple for drift correction per frame.
        """
        self.drift = np.array(drift)
        super().__init__(*args, **kwargs)

    def linking_cost_function(self, detections0, detections1, image0=None, image1=None):
        """Get centroids from detections and compute pairwise euclidian distances with drift correction.

        Args:

            detections0: image with background 0 and detections 1, ..., m
            detections1: image with backgruond 0 and detections 1, ..., n

        Returns:

            m x n cost matrix
        """
        # regionprops regions are sorted by label
        regions0 = skimage.measure.regionprops(detections0)
        points0 = [np.array(r.centroid) for r in regions0]

        regions1 = skimage.measure.regionprops(detections1)
        points1 = [np.array(r.centroid) for r in regions1]

        dists = np.zeros((len(points0), len(points1)))
        for i, p0 in enumerate(points0):
            for j, p1 in enumerate(points1):
                dists[i, j] = np.sqrt(((p0 + self.drift - p1) ** 2).sum())

        return dists


# %% trusted=true
######################
### YOUR CODE HERE ###
######################

drift_linker = NearestNeighborLinkerDriftCorrection(threshold=200, drift=(0, 0))
drift_links = drift_linker.link(detections)
drift_tracks = drift_linker.relabel_detections(detections, drift_links)

# %% editable=true slideshow={"slide_type": ""} tags=["solution"] trusted=true
# Solution Exercise 1.6
drift_linker = NearestNeighborLinkerDriftCorrection(
    threshold=50, drift=(-20, 0)
)  # SOLUTION params
drift_links = drift_linker.link(detections)
drift_tracks = drift_linker.relabel_detections(detections, drift_links)

# %% [markdown]
# Visualize results. You should mostly vertically moving tracks.

# %% trusted=true
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()
viewer = napari.Viewer()
viewer.add_image(x)
visualize_tracks(viewer, drift_tracks, name="drift")

# %% trusted=true
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()


# %% [markdown] editable=true jp-MarkdownHeadingCollapsed=true slideshow={"slide_type": ""} tags=[]
# ## Optimal frame-by-frame matching (*Linear assignment problem* or *Weighted bipartite matching*)

# %% [markdown]
# The nearest neighbor algorithm above will not pick the best solution in many cases. For example, it does not consider the local arrangement of a few detections to create links, something which the human visual system is very good at.
#
# We need a better optimization algorithm to minimize the total minimal linking distance between two frames. To use a classic and efficient optimization algorithm, we will represent this linking problem as a bipartite graph. Here is an example:
#
# <img src="figures/bipartite_graph.png" width="300"/>
#
# Red vertices correspond to detections in frame $t$, blue vertices to detections in frame $t+1$. Since we know that we don't want to link a detection to another one from the same frame, possible edges only connect blue vertices to red vertices, but not within each set.
#
# We can also put weights on the edges, which correspond the distanes we have calculated. The task is now to prune the edges of this graph such that each vertex has at most one incident edge. This is called a *weighted bipartite matching* or *linear assignment problem (LAP)*.

# %% [markdown] editable=true slideshow={"slide_type": ""}
# In the seminal tracking algorithm proposed by Jaqaman et al. (2008), the bipartite matching additionally includes cost for not linking a vertex. An unlinked vertex from frame $t$ corresponds to the death of a cell, an unlinked vertex from frame $t+1$ to the birth of a cell. Here is the cost matrix in detail:
#
# <img src="figures/LAP_cost_matrix.png" width="300"/>
#
#
# from [Jaqaman, Khuloud, et al. "Robust single-particle tracking in live-cell time-lapse sequences." Nature Methods (2008)](https://www.nature.com/articles/nmeth.1237)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Exercise 1.7
# <div class="alert alert-block alert-info"><h3>Exercise 1.7: Perform optimal frame-by-frame linking</h3>
#
# Set up the cost matrix following Jaqaman et al. (2008) such that you can use [`scipy.optimize.linear_sum_assignment`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) to solve the matching problem in the bipartite graph.
#
# </div>


# %% trusted=true
class BipartiteMatchingLinker(FrameByFrameLinker):
    def __init__(
        self,
        threshold=sys.float_info.max,
        drift=(0, 0),
        birth_cost_factor=1.05,
        death_cost_factor=1.05,
        *args,
        **kwargs,
    ):
        """
        Args:
            threshold (float): Maximum euclidian distance for linking.
            drift: tuple of (x,y) drift correction per frame.
            birth_cost_factor (float): Multiply factor with maximum entry in cost matrix.
            death_cost_factor (float): Multiply factor with maximum entry in cost matrix.
        """
        self.threshold = threshold
        self.drift = np.array(drift)
        self.birth_cost_factor = birth_cost_factor
        self.death_cost_factor = death_cost_factor

        super().__init__(*args, **kwargs)

    def linking_cost_function(self, detections0, detections1, image0=None, image1=None):
        """Get centroids from detections and compute pairwise euclidian distances with drift correction.

        Args:
            detections0: image with background 0 and detections 1, ..., m
            detections1: image with backgruond 0 and detections 1, ..., n
        Returns:
            m x n cost matrix
        """
        # regionprops regions are sorted by label
        regions0 = skimage.measure.regionprops(detections0)
        points0 = [np.array(r.centroid) for r in regions0]

        regions1 = skimage.measure.regionprops(detections1)
        points1 = [np.array(r.centroid) for r in regions1]

        dists = []
        for p0 in points0:
            for p1 in points1:
                dists.append(np.sqrt(((p0 + self.drift - p1) ** 2).sum()))

        dists = np.array(dists).reshape(len(points0), len(points1))

        return dists

    def _link_two_frames(self, cost_matrix):
        """Weighted bipartite matching with square matrix from Jaqaman et al (2008).

        Args:
            cost_matrix: m x n matrix.
        Returns:
            Linking dictionary:
                "links": Tuple of lists (ids frame t, ids frame t+1),
                "births": List of ids,
                "deaths": List of ids.
            Ids are one-based, 0 is reserved for background.
        """
        cost_matrix = cost_matrix.copy().astype(float)

        # Cost of birth event
        b = self.birth_cost_factor * min(self.threshold, cost_matrix.max())

        # Cost of death event
        d = self.death_cost_factor * min(self.threshold, cost_matrix.max())

        # High value to represent impossible link
        no_link = max(cost_matrix.max(), max(b, d)) * 1e9

        ### Set up the blockwise square cost matrix for the linear assignment problem ###

        # Set all linking costs exceeding `threshold` to `no_link`.
        ######################
        ### YOUR CODE HERE ###
        ######################

        # Set up the lower right block correctly, instead of all 0s.
        # It should be the transpose of the upper left block.
        ######################
        ### YOUR CODE HERE ###
        ######################
        lower_right = np.zeros((cost_matrix.shape[1], cost_matrix.shape[0]))

        # Set up the block for death costs correctly, instead of all 0s.
        ######################
        ### YOUR CODE HERE ###
        ######################
        deaths = np.full(
            shape=(cost_matrix.shape[0], cost_matrix.shape[0]), fill_value=0
        )

        # Set up the block for birth costs correctly, instead of all 0s.
        ######################
        ### YOUR CODE HERE ###
        ######################
        births = np.full(
            shape=(cost_matrix.shape[1], cost_matrix.shape[1]), fill_value=0
        )

        # Assemble blockwise cost matrix
        square_cost_matrix = np.block(
            [
                [cost_matrix, deaths],
                [births, lower_right],
            ]
        )

        ### Run linear assignment problem ###

        row_ind, col_ind = scipy.optimize.linear_sum_assignment(square_cost_matrix)

        # Write out matches, birth and death events to the class's format
        ids_from = []
        ids_to = []
        births = []
        deaths = []
        for row, col in zip(row_ind, col_ind):
            if row < cost_matrix.shape[0] and col < cost_matrix.shape[1]:
                ids_from.append(row)
                ids_to.append(col)

            if row >= cost_matrix.shape[0] and col < cost_matrix.shape[1]:
                births.append(col)
            if row < cost_matrix.shape[0] and col >= cost_matrix.shape[1]:
                deaths.append(row)

        # Account for +1 offset of the dense labels
        ids_from = np.array(ids_from) + 1
        ids_to = np.array(ids_to) + 1
        births = np.array(births) + 1
        deaths = np.array(deaths) + 1

        links = {"links": (ids_from, ids_to), "births": births, "deaths": deaths}
        return links


# %% editable=true slideshow={"slide_type": ""} tags=["solution"] trusted=true
# Solution exercise 1.7


class BipartiteMatchingLinker(FrameByFrameLinker):
    """.

    Args:
        threshold (float): Maximum euclidian distance for linking.
        drift: tuple of (x,y) drift correction per frame.
        birth_cost_factor (float): Multiply factor with maximum entry in cost matrix.
        death_cost_factor (float): Multiply factor with maximum entry in cost matrix.
    """

    def __init__(
        self,
        threshold=np.finfo(float).max,
        drift=(0, 0),
        birth_cost_factor=1.05,
        death_cost_factor=1.05,
        *args,
        **kwargs,
    ):
        self.threshold = threshold
        self.drift = np.array(drift)
        self.birth_cost_factor = birth_cost_factor
        self.death_cost_factor = death_cost_factor

        super().__init__(*args, **kwargs)

    def linking_cost_function(self, detections0, detections1, image0=None, image1=None):
        """Get centroids from detections and compute pairwise euclidian distances with drift correction.

        Args:

            detections0: image with background 0 and detections 1, ..., m
            detections1: image with backgruond 0 and detections 1, ..., n

        Returns:

            m x n cost matrix
        """
        # regionprops regions are sorted by label
        regions0 = skimage.measure.regionprops(detections0)
        points0 = [np.array(r.centroid) for r in regions0]

        regions1 = skimage.measure.regionprops(detections1)
        points1 = [np.array(r.centroid) for r in regions1]

        dists = []
        for p0 in points0:
            for p1 in points1:
                dists.append(np.sqrt(((p0 + self.drift - p1) ** 2).sum()))

        dists = np.array(dists).reshape(len(points0), len(points1))

        return dists

    def _link_two_frames(self, cost_matrix):
        """Weighted bipartite matching with square matrix from Jaqaman et al (2008).

        Args:

            cost_matrix: m x n matrix.

        Returns:
            "links":

                Tuple of lists. Links from frame t to frame t+1 of form (from0, to0) are split up into two lists:
                    - idgs_from: [from0, from1 , ...])
                    - ids_to: [to0, to1 , ...])

                "births": List of ids from frame t that are
                "deaths": List of ids.

            Ids are one-based, 0 is reserved for background.
        """

        cost_matrix = cost_matrix.copy().astype(float)

        # Cost of birth event
        b = self.birth_cost_factor * min(self.threshold, cost_matrix.max())

        # Cost of death event
        d = self.death_cost_factor * min(self.threshold, cost_matrix.max())

        # High value to represent impossible link
        no_link = max(cost_matrix.max(), max(b, d)) * 1e9

        ### Set up the blockwise square cost matrix for the linear assignment problem ###

        # Set all linking costs exceeding `threshold` to `no_link`
        cost_matrix[cost_matrix > self.threshold] = no_link

        # Set up the lower right block correctly, instead of all 0s.
        lower_right = cost_matrix.transpose()

        # Set up the block for death costs correctly, instead of all 0s.
        deaths = np.full(
            shape=(cost_matrix.shape[0], cost_matrix.shape[0]), fill_value=no_link
        )
        np.fill_diagonal(deaths, d)

        # Set up the block for birth costs correctly, instead of all 0s.
        births = np.full(
            shape=(cost_matrix.shape[1], cost_matrix.shape[1]), fill_value=no_link
        )
        np.fill_diagonal(births, b)

        # Assemble blockwise cost matrix
        square_cost_matrix = np.block(
            [
                [cost_matrix, deaths],
                [births, lower_right],
            ]
        )

        ### Run linear assignment problem ###

        row_ind, col_ind = scipy.optimize.linear_sum_assignment(square_cost_matrix)

        # Write out matches, birth and death events to the class's format
        ids_from = []
        ids_to = []
        births = []
        deaths = []
        for row, col in zip(row_ind, col_ind):
            if row < cost_matrix.shape[0] and col < cost_matrix.shape[1]:
                ids_from.append(row)
                ids_to.append(col)

            if row >= cost_matrix.shape[0] and col < cost_matrix.shape[1]:
                births.append(col)
            if row < cost_matrix.shape[0] and col >= cost_matrix.shape[1]:
                deaths.append(row)

        # Account for +1 offset of the dense labels
        ids_from = np.array(ids_from) + 1
        ids_to = np.array(ids_to) + 1
        births = np.array(births) + 1
        deaths = np.array(deaths) + 1

        links = {"links": (ids_from, ids_to), "births": births, "deaths": deaths}
        return links


# %% trusted=true
bm_linker = BipartiteMatchingLinker(
    threshold=50, drift=(-20, 0), birth_cost_factor=1.05, death_cost_factor=1.05
)
bm_links = bm_linker.link(detections)
bm_tracks = bm_linker.relabel_detections(detections, bm_links)

# %% [markdown]
# Visualize results. You should observe mostly vertically moving tracks, without ever explicitely modelling the drift. This is quite cool :)

# %% trusted=true
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()
viewer = napari.Viewer()
viewer.add_image(x)
visualize_tracks(viewer, bm_tracks, name="bm")

# %% trusted=true
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Exercise 1.8 (Bonus)
#
# <div class="alert alert-block alert-info"><h3>Exercise 1.8: Explore different features for assigment problem</h3>
#
# Explore solving the assignment problem based different features and cost functions.
# For example:
# - Different morphological properties of detections (e.g. using `skimage.measure.regionprops`).
# - Extract texture features from the images, e.g. mean intensity for each detection.
# - Pairwise *Intersection over Union (IoU)* of detections.
# - ...
#
# Feel free to share features that improved the results with the class :).
# </div>


# %% trusted=true
class YourLinker(BipartiteMatchingLinker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def linking_cost_function(self, detections0, detections1, image0=None, image1=None):
        """Your very smart cost function for frame-by-frame linking.

        Args:

            detections0: image with background 0 and detections 1, ..., m
            detections1: image with backgruond 0 and detections 1, ..., n
            image0 (optional): image corresponding to detections0
            image1 (optional): image corresponding to detections1

        Returns:

            m x n cost matrix
        """
        return np.zeros((detections0.max(), detections1.max()))


# %% trusted=true
your_linker = YourLinker()
your_links = your_linker.link(detections)
your_tracks = your_linker.relabel_detections(detections, your_links)

# %% trusted=true
