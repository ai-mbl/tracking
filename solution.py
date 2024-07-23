# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Exercise 9: Tracking-by-detection with an integer linear program (ILP)
#
# You could also run this notebook on your laptop, a GPU is not needed :).
#
# <center><img src="figures/ilp_nodiv.png" width="900"/></center>
#
# <div class="alert alert-danger">
# Set your python kernel to <code>08-ilp-tracking</code>
# </div>
#
# You will learn
# - how linking with global context can be modeled and solved efficiently as a **network flow** using `motile` ([docs here](https://funkelab.github.io/motile/)) for a small-scale problem (Exercise 2.1).
# - to adapt the previous formulation to allow for **arbitrary track starting and ending points** (Exercise 2.2).
# - to extend the ILP to properly model **cell divisions** (Exercise 2.3).
# - to tune the **hyperparameters** of the ILP (Exercise 2.4, bonus).
#
#
# Places where you are expected to write code are marked with
# ```
# ######################
# ### YOUR CODE HERE ###
# ######################
# ```
#
# TEST
#
# This notebook was originally written by Benjamin Gallusser.

# %% [markdown]
# ## Import packages

# %%
# Notebook at full width in the browser
from IPython.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))

import time
from pathlib import Path

import skimage
import pandas as pd
import numpy as np
import napari
import networkx as nx
import plotly.io as pio

pio.renderers.default = "vscode"

import motile
from motile.plot import draw_track_graph, draw_solution
from utils import InOutSymmetry, MinTrackLength

import traccuracy
from traccuracy import run_metrics
from traccuracy.metrics import CTCMetrics, DivisionMetrics
from traccuracy.matchers import CTCMatcher
import zarr

from tqdm.auto import tqdm

# %% [markdown]
# ## Load the dataset and inspect it in napari

# %% [markdown]
# For this exercise we will be working with a fluorescence microscopy time-lapse of breast cancer cells with stained nuclei (SiR-DNA). It is similar to the dataset at https://zenodo.org/record/4034976#.YwZRCJPP1qt. The raw data is saved in a zarr, and the ground truth tracks are saved 

# %%
base_path = Path("data/exercise1")

from tifffile import imread
from utils import normalize

def preprocess(X, Y, axis_norm=(0, 1)):
    # normalize channels independently
    X = np.stack(
        [
            normalize(x, 1, 99.8, axis=axis_norm)
            for x in tqdm(X, leave=True, desc="Normalize images")
        ]
    )
x = np.stack(
    [imread(xi) for xi in sorted((base_path / "images").glob("*.tif"))]
)  # images
y = np.stack(
    [imread(yi) for yi in sorted((base_path / "gt_tracking").glob("*.tif"))]
)  # ground truth annotations
assert x.shape == y.shape
print(f"Number of images: {len(x)}")
print(f"Shape of images: {x[0].shape}")




# %%
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()
viewer = napari.Viewer()
viewer.add_image(x)
viewer.add_labels(y)

# %%


data_root = zarr.open("data/breast_cancer_fluo.zarr", 'a')
data_root["raw"] = x

# %%
links_path = base_path / "gt_tracking" / "man_track.txt"
from csv import DictReader
track_to_parent_dict = {}
with open(links_path) as f:
    reader = DictReader(f, fieldnames=["track_id", "from_time", "to_time", "parent_id"], delimiter=" ")
    for row in reader:
        print(row)
        parent_time = int(row["from_time"]) - 1
        if parent_time not in track_to_parent_dict:
            track_to_parent_dict[parent_time] = {}
        track_to_parent_dict[parent_time][int(row["track_id"])] = int(row["parent_id"])
track_to_parent_dict

# %%
from skimage.measure import regionprops
from csv import DictWriter

def find_parent(nodes_by_time, time, label):
    if time in track_to_parent_dict and label in track_to_parent_dict[time]:
        parent_label = track_to_parent_dict[time][label]
        print(time, label, parent_label)
    else:
        parent_label = label
    if time in nodes_by_time:
        for node in nodes_by_time[time]:
            if node["label"] == parent_label:
                return node["id"]
    return -1
    

def write_gt_tracks_csv(labels, outfile):
    with open(outfile, 'w') as f:
        writer = DictWriter(f, fieldnames=["id", "time", "x", "y", "parent_id"], extrasaction='ignore')
        writer.writeheader()

        node_id = 0
        nodes_by_time = {}
        for _time in range(len(labels)):
            nodes_by_time[_time] = []
            for detection in regionprops(labels[_time]):
                node_dict = {
                    "id": node_id,
                    "x": detection.centroid[0],
                    "y": detection.centroid[1],
                    "label": detection.label,
                    "time": _time,
                    "parent_id": find_parent(nodes_by_time, _time - 1, detection.label)
                }
                node_id += 1
                nodes_by_time[_time].append(node_dict)
                writer.writerow(node_dict)

write_gt_tracks_csv(y, "data/breast_cancer_fluo_gt_tracks.csv")


# %%
def read_gt_tracks():
    with open("data/breast_cancer_fluo_gt_tracks.csv") as f:
        reader = DictReader(f)
        gt_tracks = nx.DiGraph()
        for row in reader:
            _id = int(row["id"])
            row["pos"] = [float(row["x"]), float(row["y"])]
            parent_id = int(row["parent_id"])
            del row["x"]
            del row["y"]
            del row["id"]
            del row["parent_id"]
            gt_tracks.add_node(_id, **row)
            if parent_id != -1:
                gt_tracks.add_edge(parent_id, _id)
    return gt_tracks

gt_tracks = read_gt_tracks()

# %%
from motile_toolbox.visualization import to_napari_tracks_layer
from napari.layers import Tracks
data, properties, edges = to_napari_tracks_layer(gt_tracks, frame_key="time", location_key="pos")
tracks_layer = Tracks(data, graph=edges, properties=properties,  name="gt_tracks")
viewer.add_layer(tracks_layer)

# %%

x, y = preprocess(x, y)

# %%
data = np.load(base_path / "detected_renumbered.npz", allow_pickle=True)
img = data["img"]
labels = data["labels"]
links = pd.DataFrame(data["links"], columns=["track_id", "from", "to", "parent_id"])
det = data["det"]
det_center_probs = data["det_center_probs"][()]  # det_center_probs is a dictionary

# %% [markdown]
# According to the `links` table, there should be two cell divisions in this video:

# %%
links


# %% [markdown]
# Let's use [napari](https://napari.org/tutorials/fundamentals/getting_started.html) to visualize the data. Napari is a wonderful viewer for imaging data that you can interact with in python, even directly out of jupyter notebooks.If you've never used napari, you might want to take a few minutes to go through [this tutorial](https://napari.org/stable/tutorials/fundamentals/viewer.html).

# %% [markdown]
# <div class="alert alert-block alert-danger"><h3>Napari in a jupyter notebook:</h3>
#
# - To have napari working in a jupyter notebook, you need to use up-to-date versions of napari, pyqt and pyqt5, as is the case in the conda environments provided together with this exercise.
# - When you are coding and debugging, close the napari viewer with `viewer.close()` to avoid problems with the two event loops of napari and jupyter.
# - **If a cell is not executed (empty square brackets on the left of a cell) despite you running it, running it a second time right after will usually work.**
# </div>

# %% [markdown]
# Here's a little convenience function to visualize the ground truth tracks.


# %%
def visualize_tracks(viewer, y, links=None, name=""):
    """Utility function to visualize segmentation and tracks

    Args:
        viewer: napari viewer
        y: labels: list of 2D arrays, each array is a label image.
        links: np.ndarray, each row is a link (parent, child, parent_frame, child_frame).

    Returns:
        tracks: np.ndarray, shape (N, 4)
    """
    max_label = max(links.max(), y.max()) if links is not None else y.max()
    colorperm = np.random.default_rng(42).permutation(np.arange(1, max_label + 2))
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


# %%
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()
viewer = napari.Viewer()
viewer.add_image(img)
visualize_tracks(viewer, labels, links.to_numpy(), "ground_truth")
viewer.add_labels(det, name="detections")
viewer.grid.enabled = True

# %% [markdown]
# Now it is easy to see that the ground truth nuclei have consistent IDs (visualized as random colors) over time.
#
# If you zoom in, you will note that the annotations are not perfect segmentations, but rather circles placed roughly in the center of each nucleus. However, our detections are full segmentations.

# %%
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()


# %% [markdown]
# ## Build the ground truth graph, as well as a candidate graph from the detections

# %% [markdown]
# We will represent a linking problem as a [directed graph](https://en.wikipedia.org/wiki/Directed_graph) that contains all possible detections (graph nodes) and links (graph edges) between them.
#
# Then we remove certain nodes and edges using discrete optimization techniques such as an integer linear program (ILP).
#
# First of all, we will build and inspect two graphs:
# - One for the ground truth data.
# - A candidate graph built from the detected cells in the video.


# %%
def build_gt_graph(labels, links=None):
    """Build a ground truth graph from a list of labels and links.

    Args:
        labels: list of 2D arrays, each array is a label image
        links: np.ndarray, each row is a link (parent, child, parent_frame, child_frame).
    Returns:
        trackgraph: motile.TrackGraph containing the ground truth graph.
        G: networkx.DiGraph containing the ground truth graph.
    """

    print("Build ground truth graph")
    G = nx.DiGraph()

    luts = []
    n_v = 0
    for t, d in enumerate(labels):
        lut = {}
        regions = skimage.measure.regionprops(d)
        positions = []
        for i, r in enumerate(regions):
            draw_pos = int(d.shape[0] - r.centroid[0])
            if draw_pos in positions:
                draw_pos += 3  # To avoid overlapping nodes
            positions.append(draw_pos)
            G.add_node(
                n_v,
                time=t,
                show=r.label,
                draw_position=draw_pos,
                y=int(r.centroid[0]),
                x=int(r.centroid[1]),
            )
            lut[r.label] = n_v
            n_v += 1
        luts.append(lut)

    n_e = 0
    for t, (d0, d1) in enumerate(zip(labels, labels[1:])):
        r0 = skimage.measure.regionprops(d0)
        c0 = [np.array(r.centroid) for r in r0]

        r1 = skimage.measure.regionprops(d1)
        c1 = [np.array(r.centroid) for r in r1]

        for _r0, _c0 in zip(r0, c0):
            for _r1, _c1 in zip(r1, c1):
                dist = np.linalg.norm(_c0 - _c1)
                if _r0.label == _r1.label:
                    G.add_edge(
                        luts[t][_r0.label],
                        luts[t + 1][_r1.label],
                        edge_id=n_e,
                        is_intertrack_edge=0,
                    )
                    n_e += 1

    if links is not None:
        divisions = links[links[:, 3] != 0]
        for d in divisions:
            if d[1] > 0 and d[1] < labels.shape[0]:
                try:
                    G.add_edge(
                        luts[d[1] - 1][d[3]],
                        luts[d[1]][d[0]],
                        edge_id=n_e,
                        show="DIV",
                        is_intertrack_edge=1,
                    )
                    n_e += 1
                except KeyError:
                    pass

    trackgraph = motile.TrackGraph(G, frame_attribute="time")

    return trackgraph, G


def build_graph(detections, max_distance, detection_probs=None, drift=(0, 0)):
    """Build a candidate graph from a list of detections.

     Args:
        detections: list of 2D arrays, each array is a label image.
            Labels are expected to be consecutive integers starting from 1, background is 0.
        max distance: maximum distance between centroids of two detections to place a candidate edge.
        detection_probs: list of arrays, corresponding to ordered ids in detections.
        drift: (y, x) tuple for drift correction in euclidian distance feature.
    Returns:
        G: motile.TrackGraph containing the candidate graph.
    """

    print("Build candidate graph")
    G = nx.DiGraph()

    for t, d in enumerate(detections):
        regions = skimage.measure.regionprops(d)
        positions = []
        for i, r in enumerate(regions):
            draw_pos = int(d.shape[0] - r.centroid[0])
            if draw_pos in positions:
                draw_pos += 3  # To avoid overlapping nodes
            positions.append(draw_pos)
            feature = (
                np.round(detection_probs[r.label], decimals=2).item()
                if detection_probs is not None
                else 1
            )
            G.add_node(
                r.label - 1,
                time=t,
                show=r.label,
                feature=feature,
                draw_position=draw_pos,
                y=int(r.centroid[0]),
                x=int(r.centroid[1]),
            )

    n_e = 0
    for t, (d0, d1) in enumerate(zip(detections, detections[1:])):
        r0 = skimage.measure.regionprops(d0)
        c0 = [np.array(r.centroid) for r in r0]

        r1 = skimage.measure.regionprops(d1)
        c1 = [np.array(r.centroid) for r in r1]

        for _r0, _c0 in zip(r0, c0):
            for _r1, _c1 in zip(r1, c1):
                dist = np.linalg.norm(_c0 + np.array(drift) - _c1)
                if dist < max_distance:
                    G.add_edge(
                        _r0.label - 1,
                        _r1.label - 1,
                        # 1 - normalized euclidian distance
                        feature=1
                        - np.round(
                            np.linalg.norm(_c0 + np.array(drift) - _c1) / max_distance,
                            decimals=3,
                        ).item(),
                        edge_id=n_e,
                        show="?",
                    )
                    n_e += 1

    G = motile.TrackGraph(G, frame_attribute="time")

    return G


# %%
gt_graph, gt_nx_graph = build_gt_graph(labels, links.to_numpy())
candidate_graph = build_graph(
    det, max_distance=50, detection_probs=det_center_probs, drift=(-4, 0)
)

# %% [markdown]
# Let's visualize the two graphs.
#
# In the ground truth graph nodes that belong to the same linear tracklet are marked with the same id. The two divisions in the dataset are marked in yellow.

# %%
gt_edge_colors = [
    (255, 140, 0) if "show" in edge else (0, 128, 0) for edge in gt_graph.edges.values()
]

fig_gt = draw_track_graph(
    gt_graph,
    position_attribute="draw_position",
    width=1000,
    height=500,
    label_attribute="show",
    node_color=(0, 128, 0),
    edge_color=gt_edge_colors,
    node_size=25,
)
fig_gt = fig_gt.update_layout(
    title={
        "text": "Ground truth",
        "y": 0.98,
        "x": 0.5,
    }
)
fig_gt.show()

# %% [markdown]
# You can hover over the nodes and edges of the candidate graph to inspect their features.
#
# In contrast to the ground truth graph above, in the candidate graph, nodes have unique IDs.
#
# The nodes' `feature` is set to their detection probability, and the edges' `feature` to 1 - normalized_detection_distance, which is also visualized as their color saturation.

# %%
fig_candidate = draw_track_graph(
    candidate_graph,
    position_attribute="draw_position",
    width=1000,
    height=500,
    label_attribute="show",
    alpha_attribute="feature",
    node_size=25,
)
fig_candidate = fig_candidate.update_layout(
    title={
        "text": "Candidate graph",
        "y": 0.98,
        "x": 0.5,
    }
)
fig_candidate.show()


# %% [markdown]
# ## Network flow

# %% [markdown]
# As hinted earlier, our goal is to prune the candidate graph. More formally we want to find a graph $\tilde{G}=(\tilde{V}, \tilde{E})$ whose vertices $\tilde{V}$ are a subset of the candidate graph vertices $V$ and whose edges $\tilde{E}$ are a subset of the candidate graph edges $E$.
#
# The first algorithm we will use to do this is a [network flow](https://en.wikipedia.org/wiki/Network_flow_problem). It tries to find as many disjunct paths from the first frame to the last frame as possible. All other vertices and edges are discarded. This specific algorithm is called maximum flow.
#
#
# Finding a good subgraph $\tilde{G}=(\tilde{V}, \tilde{E})$ can be formulated as an [integer linear program (ILP)](https://en.wikipedia.org/wiki/Integer_programming) (also, refer to the tracking lecture slides), where we assign a binary variable $x$ and a cost $c$ to each vertex and edge in $G$, and then computing $min_x c^Tx$.
#
# A set of linear constraints ensures that the solution will be a feasible cell tracking graph. For example, if an edge is part of $\tilde{G}$, both its incident nodes have to be part of $\tilde{G}$ as well.
#
# Here we want to express the network flow as an ILP using `motile` ([docs here](https://funkelab.github.io/motile/)), a convenient wrapper around linking with an ILP.

# %% [markdown]
# ## Exercise 2.1 - Network flow
# <div class="alert alert-block alert-info"><h3>Exercise 2.1: The network flow formulation below needs properly set parameters.</h3>
# </div>
#
# Try different values for `node_weight`, `edge_weight` and `max_flow` and try to get an output similar to the plot right below.
#
# Give a short explanation why your parameters work.

# %% [markdown]
# <img src="figures/network_flow.png" width="700"/>


# %% [markdown]
# Here is a utility function to gauge some statistics of a solution.


# %%
def print_solution_stats(solver, graph, gt_graph):
    """Prints the number of nodes and edges for candidate, ground truth graph, and solution graph.

    Args:
        solver: motile.Solver, after calling solver.solve()
        graph: motile.TrackGraph, candidate graph
        gt_graph: motile.TrackGraph, ground truth graph
    """
    time.sleep(0.1)  # to wait for ilpy prints
    print(
        f"\nCandidate graph\t\t{len(graph.nodes):3} nodes\t{len(graph.edges):3} edges"
    )
    print(
        f"Ground truth graph\t{len(gt_graph.nodes):3} nodes\t{len(gt_graph.edges):3} edges"
    )

    node_selected = solver.get_variables(motile.variables.NodeSelected)
    edge_selected = solver.get_variables(motile.variables.EdgeSelected)
    nodes = 0
    for node in candidate_graph.nodes:
        if solver.solution[node_selected[node]] > 0.5:
            nodes += 1
    edges = 0
    for u, v in candidate_graph.edges:
        if solver.solution[edge_selected[(u, v)]] > 0.5:
            edges += 1
    print(f"Solution graph\t\t{nodes:3} nodes\t{edges:3} edges")


# %% [markdown]
# This is the actual formulation of the network flow.
#
# First we associate costs for each node and weight to be picked, which are a product of `weight` and `attribute`.
#
# Then we add a constraint on how many parents and children each node can be connected to in the solution, and some specific constraints for the network flow.


# %%
def solve_network_flow(graph, node_weight, edge_weight, max_flow):
    """Set up and solve the network flow problem.

    Args:
        graph (motile.TrackGraph): The candidate graph.
        node_weight (float): The weighting factor of the node selection cost.
        edge_weight (float): The weighting factor of the edge selection cost.
        max_flow (int): The maximum number of incoming and outgoing edges in the solution.

    Returns:
        motile.Solver: The solver object, ready to be inspected.
    """
    solver = motile.Solver(graph)

    # Add costs
    solver.add_costs(
        motile.costs.NodeSelection(
            weight=node_weight, attribute="feature"
        )  # Adapt this weight
    )
    solver.add_costs(
        motile.costs.EdgeSelection(
            weight=edge_weight, attribute="feature"
        )  # Adapt this weight
    )

    solver.add_constraints(motile.constraints.MaxParents(max_flow))
    solver.add_constraints(motile.constraints.MaxChildren(max_flow))

    # Special contraints for network flow
    solver.add_constraints(InOutSymmetry())
    solver.add_constraints(MinTrackLength(1))

    solution = solver.solve()

    return solver


# %%
######################
### YOUR CODE HERE ###
######################

# Reminder: The optimization problem *minimizes* the cost of the solution.
node_weight = 1  # Adapt this weight
edge_weight = 1  # Adapt this weight
max_flow = 4  # Adapt this

"""
Explanation: ???
"""

# %% tags=["solution"]
# Solution

node_weight = -1
edge_weight = -1
max_flow = 1

"""
Explanation: Since the ILP formulation is a minimization problem, the total weight of each node and edge needs to be negative.
The cost of each node corresponds to its detection probability, so we can simply mulitply with `node_weight=-1`.
The cost of each edge corresponds to 1 - distance between the two nodes, so agai we can simply mulitply with `edge_weight=-1`.

Futhermore, each detection (node) should maximally be linked to one other detection in the previous and next frames, so we set `max_flow=1`.
"""


# %% [markdown]
# Here we actually run the optimization, and compare the found solution to the ground truth.
#
# <div class="alert alert-block alert-warning"><h3>Gurobi license error</h3>
# Please ignore the warning `Could not create Gurobi backend ...`.
#
#
# Our integer linear program (ILP) tries to use the proprietary solver Gurobi. You probably don't have a license, in which case the ILP will fall back to the open source solver SCIP.
# </div>

# %%
flow = solve_network_flow(candidate_graph, node_weight, edge_weight, max_flow)
print_solution_stats(flow, candidate_graph, gt_graph)

# %%
fig_gt.show()
fig_flow = draw_solution(
    candidate_graph,
    flow,
    position_attribute="draw_position",
    width=1000,
    height=500,
    label_attribute="show",
    node_size=25,
)
fig_flow = fig_flow.update_layout(
    title={
        "text": f"Network flow (no divisions) - cost: {flow.solution.get_value()}",
        "y": 0.98,
        "x": 0.5,
    }
)
fig_flow.show()


# %% [markdown]
# ### Recolor detections in napari according to solution and compare to ground truth


# %%
def solution2graph(solver, base_graph, detections, label_key="show"):
    """Convert a solver solution to a graph and corresponding dense selected detections.

    Args:
        solver: A solver instance
        base_graph: The base graph
        detections: The detections
        label_key: The key of the label in the detections
    Returns:
        track_graph: Solution as motile.TrackGraph
        graph: Solution as networkx graph
        selected_detections: Dense label array containing only selected detections
    """
    graph = nx.DiGraph()
    node_indicators = solver.get_variables(motile.variables.NodeSelected)
    edge_indicators = solver.get_variables(motile.variables.EdgeSelected)

    selected_detections = np.zeros_like(detections)

    # Build nodes
    for node, index in node_indicators.items():
        if solver.solution[index] > 0.5:
            node_features = base_graph.nodes[node]
            graph.add_node(node, **node_features)
            t = node_features[base_graph.frame_attribute]
            selected_detections[t][
                detections[t] == node_features[label_key]
            ] = node_features[label_key]

    # Build edges
    for edge, index in edge_indicators.items():
        if solver.solution[index] > 0.5:
            # print(base_graph.edges[edge])
            graph.add_edge(*edge, **base_graph.edges[edge])

    # Add cell division markers on edges for traccuracy
    for (u, v), features in graph.edges.items():
        out_edges = graph.out_edges(u)
        if len(out_edges) == 2:
            features["is_intertrack_edge"] = 1
        elif len(out_edges) == 1:
            features["is_intertrack_edge"] = 0
        else:
            raise ValueError()

    track_graph = motile.TrackGraph(graph, frame_attribute="time")

    return track_graph, graph, selected_detections


# %%
def recolor_segmentation(segmentation, graph, det_attribute="show"):
    """Recolor a segmentation based on a graph, such that each cell and its daughter cells have a unique color.

    Args:
        segmentation (np.ndarray): Predicted dense segmentation.
        graph (motile.TrackGraph): A directed graph representing the tracks.
        det_attribute (str): The attribute of the graph nodes that corresponds to ids in `segmentation`.

    Returns:
        out (np.ndarray): A recolored segmentation.
    """
    out = []
    n_tracks = 1
    color_lookup_tables = []

    for t in range(0, len(segmentation)):
        new_frame = np.zeros_like(segmentation[t])
        color_lut = {}
        for node_id in graph.nodes_by_frame(t):
            det_id = graph.nodes[node_id][det_attribute]
            if node_id not in graph.nodes:
                continue

            in_edges = []
            for u, v in graph.edges:
                if v == node_id:
                    in_edges.append((u, v))
            if not in_edges:
                new_frame[segmentation[t] == det_id] = n_tracks
                color_lut[det_id] = n_tracks
                n_tracks += 1
            else:
                for v_tm1, u_t0 in in_edges:
                    new_frame[
                        segmentation[t] == graph.nodes[u_t0][det_attribute]
                    ] = color_lookup_tables[t - 1][graph.nodes[v_tm1][det_attribute]]
                    color_lut[graph.nodes[u_t0][det_attribute]] = color_lookup_tables[
                        t - 1
                    ][graph.nodes[v_tm1][det_attribute]]

        color_lookup_tables.append(color_lut)
        out.append(new_frame)

    out = np.stack(out)
    return out


# %%
recolored_gt = recolor_segmentation(labels, gt_graph)
recolored_flow = recolor_segmentation(
    det, graph=solution2graph(flow, candidate_graph, det)[0]
)

viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()
viewer = napari.Viewer()
viewer.add_image(img)
viewer.add_labels(recolored_gt)
viewer.add_labels(recolored_flow)
viewer.grid.enabled = True

# %%
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()


# %% [markdown]
# ### Metrics
#
# We were able to understand via plotting the solution graph as well as visualizing the predicted tracks on the images that the network flow solution is far from perfect for this problem.
#
# Additionally, we would also like to quantify this. We will use the package [`traccuracy`](https://traccuracy.readthedocs.io/en/latest/) to calculate some [standard metrics for cell tracking](http://celltrackingchallenge.net/evaluation-methodology/). For example, a high-level indicator for tracking performance is called TRA.
#
# If you're interested in more detailed metrics, you can check out for example the false positive (FP) and false negative (FN) nodes, edges and division events.


# %%
def get_metrics(gt_graph, labels, pred_graph, pred_segmentation):
    """Calculate metrics for linked tracks by comparing to ground truth.

    Args:
        gt_graph (networkx.DiGraph): Ground truth graph.
        labels (np.ndarray): Ground truth detections.
        pred_graph (networkx.DiGraph): Predicted graph.
        pred_segmentation (np.ndarray): Predicted dense segmentation.

    Returns:
        results (dict): Dictionary of metric results.
    """

    gt_graph = traccuracy.TrackingGraph(
        graph=gt_graph,
        frame_key="time",
        label_key="show",
        location_keys=("x", "y"),
        segmentation=labels,
    )

    pred_graph = traccuracy.TrackingGraph(
        graph=pred_graph,
        frame_key="time",
        label_key="show",
        location_keys=("x", "y"),
        segmentation=pred_segmentation,
    )

    results = run_metrics(
        gt_data=gt_graph,
        pred_data=pred_graph,
        matcher=CTCMatcher(),
        metrics=[CTCMetrics(), DivisionMetrics()],
    )

    return results


# %%
_, flow_nx_graph, flow_det = solution2graph(flow, candidate_graph, det)
get_metrics(gt_nx_graph, labels, flow_nx_graph, flow_det)

# %% [markdown]
# ## Checkpoint 1
# <div class="alert alert-block alert-success"><h3>Checkpoint 1: We have familiarized ourselves with the formulation of linking as a graph-based optimization problem and have an solution found by an efficient network flow formulation.</h3>
#
# However, in the video at hand there are cells coming into the field of view from below. To track these, we need change the optimization problem to allow for appearing and disappearing object at any timepoint.
# </div>

# %% [markdown]
# ## Exercise 2.2 - ILP with track birth and death
# <div class="alert alert-block alert-info"><h3>Exercise 2.2: Adapt the network flow from Exercise 2.1 such that tracks can start and end at arbitrary time points.</h3>
#
# Hint: You will have to add both costs and constraints to the template below.
# </div>

# %% [markdown]
# Expected output:
#
# <img src="figures/ilp_nodiv.png" width="700"/>


# %%
def solve_ilp_birth(graph):
    """ILP allowing for appearance and disappearance of cells.

    Args:
        graph (motile.TrackGraph): The candidate graph.

    Returns:
        solver (motile.Solver): The solver.
    """

    solver = motile.Solver(graph)

    # Add costs
    # Docs: https://funkelab.github.io/motile/api.html#costs
    ######################
    ### YOUR CODE HERE ###
    ######################

    # Add constraints
    # Docs: https://funkelab.github.io/motile/api.html#constraints
    ######################
    ### YOUR CODE HERE ###
    ######################
    solver.add_constraints(motile.constraints.MaxChildren(1))

    solution = solver.solve()

    return solver


# %% tags=["solution"]
# Solution


def solve_ilp_birth(graph):
    """ILP allowing for appearance and disappearance of cells.

    Args:
        graph (motile.TrackGraph): The candidate graph.

    Returns:
        solver (motile.Solver): The solver.
    """
    solver = motile.Solver(graph)

    # Add costs
    solver.add_costs(
        motile.costs.NodeSelection(
            weight=-1,
            attribute="feature",
        )
    )
    solver.add_costs(
        motile.costs.EdgeSelection(
            weight=-1,
            attribute="feature",
        )
    )

    # Add constraints
    solver.add_constraints(motile.constraints.MaxParents(1))
    solver.add_constraints(motile.constraints.MaxChildren(1))

    solution = solver.solve()

    return solver


# %% [markdown]
# Run the optimization, and compare the found solution to the ground truth.

# %%
with_birth = solve_ilp_birth(candidate_graph)
print_solution_stats(with_birth, candidate_graph, gt_graph)

# %%
fig_gt.show()
fig_birth = draw_solution(
    candidate_graph,
    with_birth,
    position_attribute="draw_position",
    width=1000,
    height=500,
    label_attribute="show",
    node_size=25,
)
fig_birth = fig_birth.update_layout(
    title={
        "text": f"ILP formulation (no divisions) - cost: {with_birth.solution.get_value()}",
        "y": 0.98,
        "x": 0.5,
    }
)
fig_birth.show()

# %%
recolored_gt = recolor_segmentation(labels, gt_graph)
recolored_birth = recolor_segmentation(
    det, graph=solution2graph(with_birth, candidate_graph, det)[0]
)

viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()
viewer = napari.Viewer()
viewer.add_image(img)
viewer.add_labels(recolored_gt)
viewer.add_labels(recolored_birth)
viewer.grid.enabled = True

# %%
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()


# %%
_, birth_graph, birth_det = solution2graph(with_birth, candidate_graph, det)
get_metrics(gt_nx_graph, labels, birth_graph, birth_det)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## ILP model including divisions

# %% [markdown]
# ## Exercise 2.3
# <div class="alert alert-block alert-info"><h3>Exercise 2.3: Adapt the ILP formulation to include divisions.</h3>
# </div>
#
# Specifically, adapt one constraint and add costs for `Appear` and `Split` events, refer to [docs](https://funkelab.github.io/motile/api.html#costs)

# %% [markdown]
# Expected output: **Capture at least one of the two divisions**.
#
# Try to make sure that there are little or no false positive predictions.
#
# <img src="figures/ilp_div.png" width="300"/>


# %%
def solve_full_ilp(graph):
    """ILP allowing for cell division.

    Args:
        graph (motile.TrackGraph): The candidate graph.

    Returns:
        solver (motile.Solver): The solver.
    """
    solver = motile.Solver(graph)

    # Add costs
    solver.add_costs(motile.costs.NodeSelection(weight=-1, attribute="feature"))
    solver.add_costs(motile.costs.EdgeSelection(weight=-1, attribute="feature"))

    ######################
    ### YOUR CODE HERE ###
    ######################

    # Add constraints
    solver.add_constraints(motile.constraints.MaxParents(1))
    solver.add_constraints(motile.constraints.MaxChildren(1))

    solution = solver.solve()

    return solver


# %% tags=["solution"]
# Solution


def solve_full_ilp(graph):
    """ILP allowing for cell division.

    Args:
        graph (motile.TrackGraph): The candidate graph.

    Returns:
        solver (motile.Solver): The solver.
    """
    solver = motile.Solver(graph)

    # Add costs
    solver.add_costs(motile.costs.NodeSelection(weight=-1, attribute="feature"))
    solver.add_costs(motile.costs.EdgeSelection(weight=-1, attribute="feature"))
    solver.add_costs(motile.costs.Appear(constant=0.75))
    solver.add_costs(motile.costs.Split(constant=1.5))

    # Add constraints
    solver.add_constraints(motile.constraints.MaxParents(1))
    solver.add_constraints(motile.constraints.MaxChildren(2))

    solution = solver.solve()

    return solver


# %%
full_ilp = solve_full_ilp(candidate_graph)
print_solution_stats(full_ilp, candidate_graph, gt_graph)

# %%
fig_gt.show()
fig_ilp = draw_solution(
    candidate_graph,
    full_ilp,
    position_attribute="draw_position",
    width=1000,
    height=500,
    label_attribute="show",
    node_size=25,
)
fig_ilp = fig_ilp.update_layout(
    title={
        "text": f"ILP formulation with divisions - cost: {full_ilp.solution.get_value()}",
        "y": 0.98,
        "x": 0.5,
    }
)
fig_ilp.show()

# %%
recolored_gt = recolor_segmentation(labels, gt_graph)
recolored_ilp = recolor_segmentation(
    det, graph=solution2graph(full_ilp, candidate_graph, det)[0]
)

viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()
viewer = napari.Viewer()
viewer.add_image(img)
viewer.add_labels(recolored_gt)
viewer.add_labels(recolored_ilp)
viewer.grid.enabled = True

# %%
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()

# %%
_, ilp_graph, ilp_det = solution2graph(full_ilp, candidate_graph, det)
get_metrics(gt_nx_graph, labels, ilp_graph, ilp_det)

# %% [markdown]
# ## Exercise 2.4 (Bonus)
# <div class="alert alert-block alert-info"><h3>Exercise 2.4: Try to improve the ILP-based tracking from exercise 2.3</h3>
#
# For example
# - Tune the hyperparameters.
# - Better edge features than drift-corrected euclidian distance.
# - Tune the detection algorithm to avoid false negatives.
#
# </div>

# %% [markdown]
#
