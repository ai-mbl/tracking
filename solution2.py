# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Exercise 2: Tracking-by-detection with an integer linear program (ILP)
#
# You could also run this notebook on your laptop, a GPU is not needed :).
#
# <div class="alert alert-danger">
# Set your python kernel to <code>08-ilp-tracking</code>
# </div>
#
# You will learn
# - how linking with global context can be modeled and solved efficiently as a **network flow** using `motile` ([docs here](https://funkelab.github.io/motile/)) for a small-scale problem (Exercise 3.1).
# - to adapt the previous formulation to allow for **arbitrary track starting and ending points** (Exercise 3.2).
# - to extend the ILP to properly model **cell divisions** (Exercise 3.3).
# - to tune the **hyperparameters** of the ILP (Exercise 3.4).
#
#
# Places where you are expected to write code are marked with
# ```
# ######################
# ### YOUR CODE HERE ###
# ######################
# ```
#
# This notebook was originally written by Benjamin Gallusser.

# %% [markdown]
# TODO remove
# 1. Obligatory conda install the following packages to solve the environment:
#     - napari
#     - jupyter-lab
#     - conda install -c conda-forge -c gurobi -c funkelab ilpy
# 1. Install motile directly from Github: `pip install git+https://github.com/funkelab/motile`

# %% [markdown]
# ## Import packages

# %%
# Notebook at full width in the browser
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# TODO remove unnecessary imports, automatically with black
import sys
import time
from urllib.request import urlretrieve
from pathlib import Path
from collections import defaultdict
from abc import ABC, abstractmethod

import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
matplotlib.rcParams["image.interpolation"] = "none"
matplotlib.rcParams['figure.figsize'] = (12, 6)
from tifffile import imread, imwrite
from tqdm.auto import tqdm
import skimage
from skimage.segmentation import relabel_sequential
import pandas as pd
import scipy
import numpy as np
import napari
import networkx as nx

import motile
from motile.plot import draw_track_graph, draw_solution

# Pretty tqdm progress bars 
# ! jupyter nbextension enable --py widgetsnbextension

# %% [markdown]
# ## Load the dataset and inspect it in napari

# %% [markdown]
# For this exercise we will work with a small excerpt of the dataset from exercise 1. We already provide you with the detections this time, let's load them and take a look. 

# %%
base_path = Path("data/exercise2")
data = np.load(base_path / "detected_renumbered.npz", allow_pickle=True)
img = data["img"]
labels = data["labels"]
links = pd.DataFrame(data["links"], columns=["track_id", "from", "to", "parent_id"])
det = data["det"]
det_prob_maps = data["det_prob_maps"]
det_centers = data["det_centers"][()] # det_centers is a dictionary
det_center_probs = data["det_center_probs"][()] # det_center_probs is a dictionary

# %% [markdown]
# According to the `links` table, there should be two cell divisions in this video: 

# %%
links


# %% [markdown]
# <div class="alert alert-block alert-danger"><h3>Napari in a jupyter notebook:</h3>
#     
# - To have napari working in a jupyter notebook, you need to use up-to-date versions of napari, pyqt and pyqt5, as is the case in the conda environments provided together with this exercise.
# - When you are coding and debugging, close the napari viewer with `viewer.close()` to avoid problems with the two event loops of napari and jupyter.
# - **If a cell is not executed (empty square brackets on the left of a cell) despite you running it, running it a second time right after will usually work.**
# </div>

# %% [markdown]
# Here's a little convenience function to visualize the ground truth tracks

# %%
def visualize_tracks(viewer, y, links=None, name=""):
    """Utility function to visualize segmentation and tracks"""
    max_label = max(links.max(), y.max()) if links is not None else y.max()
    colorperm = np.random.default_rng(42).permutation(np.arange(1, max_label + 2))
    tracks = []
    for t, frame in enumerate(y):
        centers = skimage.measure.regionprops(frame)
        for c in centers:
            tracks.append([colorperm[c.label], t, int(c.centroid[0]), int(c.centroid[1])])
    tracks = np.array(tracks)
    tracks = tracks[tracks[:, 0].argsort()]
    
    graph = {}
    if links is not None:
        divisions = links[links[:,3] != 0]
        for d in divisions:
            if colorperm[d[0]] not in tracks[:, 0] or colorperm[d[3]] not in tracks[:, 0]:
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
visualize_tracks(viewer, labels, links.to_numpy(), "ground_truth");
viewer.add_labels(det, name="detections");
viewer.grid.enabled = True

# %%
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()


# %% [markdown]
# ## Build the ground truth graph, as well as a candidate graph from the detections

# %% [markdown]
# We will represent a linking problem as a [directed graph](https://en.wikipedia.org/wiki/Directed_graph) that contains all possible detections (graph nodes) and links (graph edges) between them. Then we remove certain nodes and edges using discrete optimization techniques such as an integer linear problem.
#
# First of all, we will build and inspect two graphs:
# - One for the ground truth data
# - A candidate graph built from the detected cells in the video

# %%
def build_gt_graph(labels, links=None):
    """TODO"""    
    
    print("Build ground truth graph")
    G = nx.DiGraph()
    
    luts = []
    n_v = 0
    # for t, d in tqdm(enumerate(labels), desc="add nodes"):
    for t, d in enumerate(labels):
        lut = {}
        regions = skimage.measure.regionprops(d)
        for i, r in enumerate(regions):
            draw_pos = int(d.shape[0] - r.centroid[0])
            G.add_node(n_v, time=t, show=r.label, draw_position=draw_pos)
            lut[r.label] = n_v
            n_v += 1 
        luts.append(lut)

    n_e = 0
    # for t, (d0, d1) in tqdm(enumerate(zip(labels, labels[1:])), desc="add edges"):
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
                        luts[t+1][_r1.label],
                        edge_id = n_e,
                    )
                    n_e += 1

    if links is not None:
        divisions = links[links[:,3] != 0]
        for d in divisions:
            if d[1] > 0 and d[1] < labels.shape[0]:
                try:
                    G.add_edge(
                        luts[d[1] - 1][d[3]],
                        luts[d[1]][d[0]],
                        edge_id = n_e,
                        show="DIV",
                    )
                    n_e += 1
                except KeyError:
                    pass
                    
    G = motile.TrackGraph(G, frame_attribute="time")
    
    return G

def build_graph(detections, max_distance, detection_probs=None, drift=(0,0)):
    """TODO
    
    detection_probs: list of arrays, corresponding to ordered ids in detections."""
    
    print("Build candidate graph")
    G = nx.DiGraph()
    
    # for t, d in tqdm(enumerate(detections), desc="add nodes"):
    for t, d in enumerate(detections):
        regions = skimage.measure.regionprops(d)
        for i, r in enumerate(regions):
            draw_pos = int(d.shape[0] - r.centroid[0])
            weight = np.round(detection_probs[r.label], decimals=2).item() if detection_probs is not None else 1
            # TODO update motile plotting function to not require contiguous node ids starting from 0
            G.add_node(r.label - 1, time=t, show=r.label, weight=weight, draw_position=draw_pos)

    n_e = 0
    # for t, (d0, d1) in tqdm(enumerate(zip(detections, detections[1:])), desc="add edges"):
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
                        # TODO update motile plotting function to not require contiguous node ids starting from 0
                        _r0.label - 1,
                        _r1.label - 1,
                        # 1 - normalized euclidian distance
                        weight=1 - np.round(np.linalg.norm(_c0 + np.array(drift) - _c1) / max_distance, decimals=3).item(),
                        edge_id=n_e,
                        # score=1,
                        show="?",
                    )
                    n_e += 1

    G = motile.TrackGraph(G, frame_attribute="time")
    
    return G


# %%
gt_graph = build_gt_graph(labels, links.to_numpy())
candidate_graph = build_graph(det, max_distance=50, detection_probs=det_center_probs, drift=(-4 , 0))

# %% [markdown]
# Let's visualize the two graphs.

# %%
fig_gt = draw_track_graph(
    gt_graph,
    position_attribute="draw_position",
    width=1000,
    height=500,
    label_attribute='show',
    node_color=(0, 128, 0),
    edge_color=(0, 128, 0),
)
fig_gt = fig_gt.update_layout(
    title={
        'text': "Ground truth",
        'y':0.98,
        'x':0.5,
    }
)
fig_gt.show()

# %% [markdown]
# You can hover over the nodes and edges of the candidate graph to inspect their features.
#
# The nodes' `weight` is set to their detection probability, and the edges' `weight` to 1 - normalized_detection_distance, which is also visualized as their color saturation.

# %%
fig_candidate = draw_track_graph(
    candidate_graph,
    position_attribute="draw_position",
    width=1000,
    height=500,
    label_attribute='show',
    alpha_attribute='weight',
    
)
fig_candidate = fig_candidate.update_layout(
    title={
        'text': "Candidate graph",
        'y':0.98,
        'x':0.5,
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
# Finding a good subgraph $\tilde{G}=(\tilde{V}, \tilde{E})$ can be formulated as an [integer linear program (ILP)](https://en.wikipedia.org/wiki/Integer_programming) (also, refer to the tracking lecture slides), where we assign a binary variable $x$ and a cost $c$ to each vertex and edge in $G$, and then computing $min_x c^Tx$. A set of linear constraints ensure that the solution will be a feasible graph. For example, if an edge is part of $\tilde{G}$, both its incident nodes have to be part of $\tilde{G}$ as well.
#
# Here we want to express the network flow as an ILP using `motile` ([docs here](https://funkelab.github.io/motile/)), a convenient wrapper around solving linking with an ILP.
#
# We 

# %% [markdown]
# ## Exercise 2.1 - Network flow
# <div class="alert alert-block alert-info"><h3>Exercise 2.1: The network flow formulation below is flawed, please fix it. </h3>
# You can adapt/remove costs and constraints.
# </div>

# %% [markdown]
# Expected output:
#
# <img src="figures/network_flow.png" width="700"/>

# %%
def solve_network_flow(graph):
    solver = motile.Solver(graph)
    
    # Add costs
    solver.add_costs(
        motile.costs.NodeSelection(
            weight=1,  # Adapt this weight
            attribute="weight"))
    solver.add_costs(
        motile.costs.EdgeSelection(
            weight=1,  # Adapt this weight
            attribute="weight"))
    
    # Add constraints
    
    ######################
    ### YOUR CODE HERE ###
    ######################
    
    solver.add_constraints(motile.constraints.MaxChildren(2))
    
    # Special contraints for network flow
    solver.add_constraints(motile.constraints.InOutSymmetry())
    solver.add_constraints(motile.constraints.MinTrackLength(1))
    
    solution = solver.solve()
    
    return solver


# %% tags=["solution"]
# Solution

def solve_network_flow(graph):
    solver = motile.Solver(graph)
    
    # Add costs
    solver.add_costs(
        motile.costs.NodeSelection(
            weight=-1,
            attribute="weight"))
    solver.add_costs(
        motile.costs.EdgeSelection(
            weight=-1,
            attribute="weight"))
    
    # Add constraints
    solver.add_constraints(motile.constraints.MaxParents(1))
    solver.add_constraints(motile.constraints.MaxChildren(1))
    solver.add_constraints(motile.constraints.InOutSymmetry())
    solver.add_constraints(motile.constraints.MinTrackLength(1))
    
    solution = solver.solve()
    
    return solver


# %% [markdown]
# A utility function to gauge some statistics of a solution

# %%
def print_solution_stats(solver, graph, gt_graph):
    time.sleep(0.1) # to wait for ilpy prints
    print(f"\nCandidate graph\t\t{len(graph.nodes):3} nodes\t{len(graph.edges):3} edges")
    print(f"Ground truth graph\t{len(gt_graph.nodes):3} nodes\t{len(gt_graph.edges):3} edges")
    
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


# %%
flow = solve_network_flow(candidate_graph)
print_solution_stats(flow, candidate_graph, gt_graph)

# %%
fig_flow = draw_solution(
    candidate_graph,
    flow,
    position_attribute="draw_position",
    width=1000,
    height=500,
    label_attribute="show",
)
fig_flow = fig_flow.update_layout(
    title={
        'text': f"Network flow (no divisions) - cost: {flow.solution.get_value()}",
        'y':0.98,
        'x':0.5,
    }
)
fig_flow.show()
fig_gt.show()


# %% [markdown]
# ### Recolor detections in napari according to solution and compare to ground truth

# %%
def solution2graph(solver, base_graph):
    new_graph = nx.DiGraph()
    node_indicators = solver.get_variables(motile.variables.NodeSelected)
    edge_indicators = solver.get_variables(motile.variables.EdgeSelected)    
    
    # Build nodes
    for node, index in node_indicators.items():
        if solver.solution[index] > 0.5:
            node_features = base_graph.nodes[node]
            new_graph.add_node(node, **node_features)
    
    # Build edges
    for edge, index in edge_indicators.items():
        if solver.solution[index] > 0.5:
            # print(base_graph.edges[edge])
            new_graph.add_edge(*edge, **base_graph.edges[edge])
    
    track_graph = motile.TrackGraph(new_graph, frame_attribute="time")
    
    return track_graph


# %%
def recolor_segmentation(segmentation, graph, det_attribute="show"):
    """TODO"""    
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
            for (u,v) in graph.edges:
                if v == node_id:
                    in_edges.append((u, v))
            if not in_edges:
                new_frame[segmentation[t] == det_id] = n_tracks
                color_lut[det_id] = n_tracks
                n_tracks += 1
            else:
                for v_tm1, u_t0 in in_edges:
                    new_frame[segmentation[t] == graph.nodes[u_t0][det_attribute]] = color_lookup_tables[t-1][graph.nodes[v_tm1][det_attribute]]
                    color_lut[graph.nodes[u_t0][det_attribute]] = color_lookup_tables[t-1][graph.nodes[v_tm1][det_attribute]]
                
        color_lookup_tables.append(color_lut)
        out.append(new_frame)
        

    return np.stack(out)

# %%
recolored_gt = recolor_segmentation(labels, gt_graph)
det_flow = recolor_segmentation(det, graph=solution2graph(flow, candidate_graph))

viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()
viewer = napari.Viewer()
viewer.add_image(img)
viewer.add_labels(recolored_gt)
viewer.add_labels(det_flow)
viewer.grid.enabled = True

# %%
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()


# %% [markdown]
# ## Checkpoint 1
# <div class="alert alert-block alert-success"><h3>Checkpoint 1: We have familiarized ourselves with the formulation of linking as a graph-based optimization and have a feasible solution to a network flow.</h3></div>

# %% [markdown]
# ## Exercise 2.2 - ILP with track birth and death
# <div class="alert alert-block alert-info"><h3>Exercise 2.2: Adapt the network flow from Exercise 2.1 such that tracks can start and end at arbitrary time points</h3>
#
# Hint: you will have to remove some constraints.
# </div>

# %% [markdown]
# Expected output:
#
# <img src="figures/ilp_nodiv.png" width="700"/>

# %%
def solve_ilp_birth(graph):
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
#Solution

def solve_ilp_birth(graph):
    solver = motile.Solver(graph)
    
    # Add costs
    solver.add_costs(
        motile.costs.NodeSelection(
            weight=-1,
            attribute="weight",
        ))
    solver.add_costs(
        motile.costs.EdgeSelection(
            weight=-1,
            attribute="weight",
        ))
    
    # Add constraints
    solver.add_constraints(motile.constraints.MaxParents(1))
    solver.add_constraints(motile.constraints.MaxChildren(1))
    
    solution = solver.solve()
    
    return solver


# %%
with_birth = solve_ilp_birth(candidate_graph)
print_solution_stats(with_birth, candidate_graph, gt_graph)

# %%
fig_birth = draw_solution(
    candidate_graph,
    with_birth,
    position_attribute="draw_position",
    width=1000,
    height=500,
    label_attribute="show",
)
fig_birth = fig_birth.update_layout(
    title={
        'text': f"ILP formulation (no divisions) - cost: {with_birth.solution.get_value()}",
        'y':0.98,
        'x':0.5,
    }
)
fig_birth.show()
fig_gt.show()

# %%
recolored_gt = recolor_segmentation(labels, gt_graph)
det_birth = recolor_segmentation(det, graph=solution2graph(with_birth, candidate_graph))

viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()
viewer = napari.Viewer()
viewer.add_image(img)
viewer.add_labels(recolored_gt)
viewer.add_labels(det_birth)
viewer.grid.enabled = True

# %%
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## ILP model including divisions

# %% [markdown]
# ## Exercise 2.3
# <div class="alert alert-block alert-info"><h3>Exercise 2.3: Adapt the ILP formulation to include divisions.</h3>
# </div>
#
# Specifically, adapt one constraint and add costs for `Appear` and `Split` events, refer to [docs](https://funkelab.github.io/motile/api.html#costs)

# %% [markdown]
# Expected output: Capture at least one division.
#
# <img src="figures/ilp_div.png" width="300"/>

# %%
def solve_full_ilp(graph):
    solver = motile.Solver(graph)
    
    # Add costs
    solver.add_costs(
        motile.costs.NodeSelection(
            weight=-1,
            attribute="weight"))
    solver.add_costs(
        motile.costs.EdgeSelection(
            weight=-1,
            attribute="weight"))
    
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
    solver = motile.Solver(graph)
    
    # Add costs
    solver.add_costs(
        motile.costs.NodeSelection(
            weight=-1,
            attribute="weight"))
    solver.add_costs(
        motile.costs.EdgeSelection(
            weight=-1,
            attribute="weight"))
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
fig_ilp = draw_solution(
    candidate_graph,
    full_ilp,
    position_attribute="draw_position",
    width=1000,
    height=500,
    label_attribute="show",
)
fig_ilp = fig_ilp.update_layout(
    title={
        'text': f"ILP formulation with divisions - cost: {full_ilp.solution.get_value()}",
        'y':0.98,
        'x':0.5,
    }
)
fig_ilp.show()
fig_gt.show()

# %%
recolored_gt = recolor_segmentation(labels, gt_graph)
det_ilp = recolor_segmentation(det, graph=solution2graph(full_ilp, candidate_graph))

viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()
viewer = napari.Viewer()
viewer.add_image(img)
viewer.add_labels(recolored_gt)
viewer.add_labels(det_ilp)
viewer.grid.enabled = True

# %%
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()

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

# %%
