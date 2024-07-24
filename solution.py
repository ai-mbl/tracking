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
import scipy

pio.renderers.default = "vscode"

import motile
from motile.plot import draw_track_graph, draw_solution
from utils import InOutSymmetry, MinTrackLength

import traccuracy
from traccuracy import run_metrics
from traccuracy.metrics import CTCMetrics, DivisionMetrics
from traccuracy.matchers import CTCMatcher
import zarr
from motile_toolbox.visualization import to_napari_tracks_layer
from napari.layers import Tracks
from csv import DictReader

from tqdm.auto import tqdm

from typing import Iterable, Any

# %% [markdown]
# ## Load the dataset and inspect it in napari

# %% [markdown]
# For this exercise we will be working with a fluorescence microscopy time-lapse of breast cancer cells with stained nuclei (SiR-DNA). It is similar to the dataset at https://zenodo.org/record/4034976#.YwZRCJPP1qt. The raw data, pre-computed segmentations, and detection probabilities are saved in a zarr, and the ground truth tracks are saved in a csv. The segmentation was generated with a pre-trained StartDist model, so there may be some segmentation errors which can affect the tracking process. The detection probabilities also come from StarDist, and are downsampled in x and y by 2 compared to the detections and raw data.

# %%
data_path = "data/breast_cancer_fluo.zarr"
data_root = zarr.open(data_path, 'r')
image_data = data_root["raw"][:]
segmentation = data_root["seg_relabeled"][:]
probabilities = data_root["probs"][:]

# %%
print(segmentation)

# %%
# %load_ext autoreload
# %autoreload 2

from motile_toolbox.utils.relabel_segmentation import ensure_unique_seg_ids

ensure_unique_seg_ids(data_path, "seg", data_path, "seg_relabeled")


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
            row["time"] = int(row["time"])
            gt_tracks.add_node(_id, **row)
            if parent_id != -1:
                gt_tracks.add_edge(parent_id, _id)
    return gt_tracks

gt_tracks = read_gt_tracks()

# %% [markdown]
# Let's use [napari](https://napari.org/tutorials/fundamentals/getting_started.html) to visualize the data. Napari is a wonderful viewer for imaging data that you can interact with in python, even directly out of jupyter notebooks.If you've never used napari, you might want to take a few minutes to go through [this tutorial](https://napari.org/stable/tutorials/fundamentals/viewer.html).

# %% [markdown]
# <div class="alert alert-block alert-danger"><h3>Napari in a jupyter notebook:</h3>
#
# - To have napari working in a jupyter notebook, you need to use up-to-date versions of napari, pyqt and pyqt5, as is the case in the conda environments provided together with this exercise.
# - When you are coding and debugging, close the napari viewer with `viewer.close()` to avoid problems with the two event loops of napari and jupyter.
# - **If a cell is not executed (empty square brackets on the left of a cell) despite you running it, running it a second time right after will usually work.**
# </div>

# %%
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()
viewer = napari.Viewer()
viewer.add_image(image_data, name="raw")
viewer.add_labels(segmentation, name="seg")

# %%
data, properties, edges = to_napari_tracks_layer(gt_tracks, frame_key="time", location_key="pos")
tracks_layer = Tracks(data, graph=edges, properties=properties,  name="gt_tracks")
viewer.add_layer(tracks_layer)

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
gt_trackgraph = motile.TrackGraph(gt_tracks, frame_attribute="time")

def nodes_from_segmentation(
    segmentation: np.ndarray, probabilities: np.ndarray
) -> tuple[nx.DiGraph, dict[int, list[Any]]]:
    """Extract candidate nodes from a segmentation. Also computes specified attributes.
    Returns a networkx graph with only nodes, and also a dictionary from frames to
    node_ids for efficient edge adding.

    Args:
        segmentation (np.ndarray): A numpy array with integer labels and dimensions
            (t, y, x), where h is the number of hypotheses.
        probabilities (np.ndarray): A numpy array with integer labels and dimensions
            (t, y, x), where h is the number of hypotheses.

    Returns:
        tuple[nx.DiGraph, dict[int, list[Any]]]: A candidate graph with only nodes,
            and a mapping from time frames to node ids.
    """
    cand_graph = nx.DiGraph()
    # also construct a dictionary from time frame to node_id for efficiency
    node_frame_dict: dict[int, list[Any]] = {}
    print("Extracting nodes from segmentation")
    for t in tqdm(range(len(segmentation))):
        segs = segmentation[t]
        nodes_in_frame = []
        props = skimage.measure.regionprops(segs)
        print(len(props))
        for regionprop in props:
            node_id = regionprop.label
            attrs = {
                "time": t,
            }
            attrs["label"] = regionprop.label
            centroid = regionprop.centroid  #  y, x
            attrs["pos"] = centroid
            probability = probabilities[t, int(centroid[0] // 2), int(centroid[1] // 2)]
            attrs["prob"] = probability
            assert node_id not in cand_graph.nodes
            cand_graph.add_node(node_id, **attrs)
            nodes_in_frame.append(node_id)
        if t not in node_frame_dict:
            node_frame_dict[t] = []
        node_frame_dict[t].extend(nodes_in_frame)
    return cand_graph, node_frame_dict


def create_kdtree(cand_graph: nx.DiGraph, node_ids: Iterable[Any]) -> scipy.spatial.KDTree:
    positions = [cand_graph.nodes[node]["pos"] for node in node_ids]
    return scipy.spatial.KDTree(positions)


def add_cand_edges(
    cand_graph: nx.DiGraph,
    max_edge_distance: float,
    node_frame_dict: dict[int, list[Any]] = None,
) -> None:
    """Add candidate edges to a candidate graph by connecting all nodes in adjacent
    frames that are closer than max_edge_distance. Also adds attributes to the edges.

    Args:
        cand_graph (nx.DiGraph): Candidate graph with only nodes populated. Will
            be modified in-place to add edges.
        max_edge_distance (float): Maximum distance that objects can travel between
            frames. All nodes within this distance in adjacent frames will by connected
            with a candidate edge.
        node_frame_dict (dict[int, list[Any]] | None, optional): A mapping from frames
            to node ids. If not provided, it will be computed from cand_graph. Defaults
            to None.
    """
    print("Extracting candidate edges")

    frames = sorted(node_frame_dict.keys())
    prev_node_ids = node_frame_dict[frames[0]]
    prev_kdtree = create_kdtree(cand_graph, prev_node_ids)
    for frame in tqdm(frames):
        if frame + 1 not in node_frame_dict:
            continue
        next_node_ids = node_frame_dict[frame + 1]
        next_kdtree = create_kdtree(cand_graph, next_node_ids)

        matched_indices = prev_kdtree.query_ball_tree(next_kdtree, max_edge_distance)

        for prev_node_id, next_node_indices in zip(prev_node_ids, matched_indices):
            for next_node_index in next_node_indices:
                next_node_id = next_node_ids[next_node_index]
                cand_graph.add_edge(prev_node_id, next_node_id)

        prev_node_ids = next_node_ids
        prev_kdtree = next_kdtree

cand_graph, node_frame_dict = nodes_from_segmentation(segmentation, probabilities)
print(cand_graph.number_of_nodes())
add_cand_edges(cand_graph, max_edge_distance=50, node_frame_dict=node_frame_dict)
cand_trackgraph = motile.TrackGraph(cand_graph, frame_attribute="time")

# %% [markdown]
# ## Setting Up the Tracking Optimization Problem

# %% [markdown]
# As hinted earlier, our goal is to prune the candidate graph. More formally we want to find a graph $\tilde{G}=(\tilde{V}, \tilde{E})$ whose vertices $\tilde{V}$ are a subset of the candidate graph vertices $V$ and whose edges $\tilde{E}$ are a subset of the candidate graph edges $E$.
#
#
# Finding a good subgraph $\tilde{G}=(\tilde{V}, \tilde{E})$ can be formulated as an [integer linear program (ILP)](https://en.wikipedia.org/wiki/Integer_programming) (also, refer to the tracking lecture slides), where we assign a binary variable $x$ and a cost $c$ to each vertex and edge in $G$, and then computing $min_x c^Tx$.
#
# A set of linear constraints ensures that the solution will be a feasible cell tracking graph. For example, if an edge is part of $\tilde{G}$, both its incident nodes have to be part of $\tilde{G}$ as well.
#
# Here we want to express the network flow as an ILP using `motile` ([docs here](https://funkelab.github.io/motile/)), a convenient wrapper around linking with an ILP.

# %% [markdown]
# ## Exercise 2.1 - Basic Tracking with Motile
# <div class="alert alert-block alert-info"><h3>Exercise 2.1: Set up a basic motile tracking pipeline</h3>
# </div>
#

# %% [markdown]
# Here is a utility function to gauge some statistics of a solution.


# %%
from motile_toolbox.candidate_graph import graph_to_nx
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
    solution = graph_to_nx(solver.get_selected_subgraph())

    print(f"Solution graph\t\t{solution.number_of_nodes()} nodes\t{solution.number_of_edges()} edges")


# %% [markdown]
# This is the actual formulation of the network flow.
#
# First we associate costs for each node and weight to be picked, which are a product of `weight` and `attribute`.
#
# Then we add a constraint on how many parents and children each node can be connected to in the solution, and some specific constraints for the network flow.


# %%
def solve_basic_optimization(graph, edge_weight, edge_constant):
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

    solver.add_costs(
        motile.costs.EdgeDistance(weight=edge_weight, constant=edge_constant, position_attribute="pos")  # Adapt this weight
    )

    solver.add_constraints(motile.constraints.MaxParents(1))
    solver.add_constraints(motile.constraints.MaxChildren(2))

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

edge_weight = 1
edge_constant=-20
solver = solve_basic_optimization(cand_trackgraph, edge_weight, edge_constant)
solution = graph_to_nx(solver.get_selected_subgraph())
print_solution_stats(solver, cand_trackgraph, gt_trackgraph)

"""
Explanation: Since the ILP formulation is a minimization problem, the total weight of each node and edge needs to be negative.
The cost of each node corresponds to its detection probability, so we can simply mulitply with `node_weight=-1`.
The cost of each edge corresponds to 1 - distance between the two nodes, so agai we can simply mulitply with `edge_weight=-1`.

Futhermore, each detection (node) should maximally be linked to one other detection in the previous and next frames, so we set `max_flow=1`.
"""


# %%
print(solution)

# %%
# viewer = napari.viewer.current_viewer()
# if viewer:
#     viewer.close()
# viewer = napari.Viewer()
# viewer.add_image(image_data, name="raw")
# viewer.add_labels(segmentation, name="seg")

data, properties, edges = to_napari_tracks_layer(solution, frame_key="time", location_key="pos")
tracks_layer = Tracks(data, graph=edges, properties=properties,  name="solution_tracks")
viewer.add_layer(tracks_layer)

# %% [markdown]
# Here we actually run the optimization, and compare the found solution to the ground truth.
#
# <div class="alert alert-block alert-warning"><h3>Gurobi license error</h3>
# Please ignore the warning `Could not create Gurobi backend ...`.
#
#
# Our integer linear program (ILP) tries to use the proprietary solver Gurobi. You probably don't have a license, in which case the ILP will fall back to the open source solver SCIP.
# </div>

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
