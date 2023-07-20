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
# - TODO
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
from motile.plot import draw_track_graph

# Pretty tqdm progress bars 
# ! jupyter nbextension enable --py widgetsnbextension

# %% [markdown]
# ## Load the dataset inspect it in napari

# %% [markdown]
# For this exercise we will work with a small excerpt of the dataset from exercise 1. We already provide you with the detections this time, let's inspect them. 

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

# %%
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()
viewer = napari.Viewer()
viewer.add_image(img, name="image");


# %% [markdown]
# <div class="alert alert-block alert-danger"><h3>Napari in a jupyter notebook:</h3>
#     
# - To have napari working in a jupyter notebook, you need to use up-to-date versions of napari, pyqt and pyqt5, as is the case in the conda environments provided together with this exercise.
# - When you are coding and debugging, close the napari viewer with `viewer.close()` to avoid problems with the two event loops of napari and jupyter.
# - **If a cell is not executed (empty square brackets on the left of a cell) despite you running it, running it a second time right after will usually work.**
# </div>

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

# %%
def build_gt_graph(labels, links=None):
    """TODO"""    
    
    print("Build ground truth graph")
    G = nx.DiGraph()
    
    luts = []
    n_v = 0
    for t, d in tqdm(enumerate(labels), desc="add nodes"):
        lut = {}
        regions = skimage.measure.regionprops(d)
        for i, r in enumerate(regions):
            draw_pos = int(d.shape[0] - r.centroid[0])
            # TODO update motile plotting function to not require contiguous node ids starting from 0
            G.add_node(n_v, time=t, show=r.label, draw_position=draw_pos)
            lut[r.label] = n_v
            n_v += 1 
        luts.append(lut)

    n_e = 0
    for t, (d0, d1) in tqdm(enumerate(zip(labels, labels[1:])), desc="add edges"):
        r0 = skimage.measure.regionprops(d0)
        c0 = [np.array(r.centroid) for r in r0]
         
        r1 = skimage.measure.regionprops(d1)
        c1 = [np.array(r.centroid) for r in r1]

        for _r0, _c0 in zip(r0, c0):
            for _r1, _c1 in zip(r1, c1):
                dist = np.linalg.norm(_c0 - _c1)
                if _r0.label == _r1.label:
                    G.add_edge(
                        # TODO update motile plotting function to not require contiguous node ids starting from 0
                        luts[t][_r0.label],
                        luts[t+1][_r1.label],
                        edge_id = n_e,
                        # show=".",
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
    
    for t, d in tqdm(enumerate(detections), desc="add nodes"):
        regions = skimage.measure.regionprops(d)
        for i, r in enumerate(regions):
            draw_pos = int(d.shape[0] - r.centroid[0])
            weight = np.round(detection_probs[r.label], decimals=2).item() if detection_probs is not None else 1
            # TODO update motile plotting function to not require contiguous node ids starting from 0
            G.add_node(r.label - 1, time=t, show=r.label, weight=weight, draw_position=draw_pos)

    n_e = 0
    for t, (d0, d1) in tqdm(enumerate(zip(detections, detections[1:])), desc="add edges"):
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
candidate_graph = build_graph(det, max_distance=30, detection_probs=det_center_probs, drift=(-6 , 0))

# %%
# TODO upgrade this a bit
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


fig_gt.show()
fig_candidate.show()


# %% [markdown]
# ## Network flow

# %% [markdown]
# ## Exercise 3.1
# <div class="alert alert-block alert-info"><h3>Exercise 3.1: TODO define task</h3></div>

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
            
    return new_graph

def print_solution_stats(solver, graph):
    time.sleep(0.1) # to wait for ilpy prints
    print(f"\nCandidate graph\t{len(graph.nodes):3} nodes\t{len(graph.edges):3} edges")
    
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
    print(f"Solution graph\t{nodes:3} nodes\t{edges:3} edges")


# %%
def solve_network_flow(graph):
    # TODO update to actual flow
    solver = motile.Solver(graph)
    
    # Add costs
    solver.add_costs(
        motile.costs.NodeSelection(
            weight=-1,
            attribute="weight"))
    solver.add_costs(
        motile.costs.EdgeSelection(
            weight=1,
            attribute="weight"))
    solver.add_costs(motile.costs.Appear(constant=10000))
    
    # Add constraints
    solver.add_constraints(motile.constraints.MaxParents(1))
    solver.add_constraints(motile.constraints.MaxChildren(1))
    
    solution = solver.solve()
    
    print_solution_stats(solver, graph)
    solution_graph = solution2graph(solver, graph)
   
    return solution_graph, solver.solution.get_value()


# %%
flow, flow_cost = solve_network_flow(candidate_graph)

# %%
fig, (ax0, ax1, ax2) = plt.subplots(1,3, figsize=(32, 12))
draw_graph(gt_graph, "Ground truth graph", ax=ax0, height=detections[0].shape[0])
draw_graph(candidate_graph, "Candidate graph", ax=ax1, height=detections[0].shape[0])
draw_graph(flow, f"Network flow (no divisions) - cost: {flow_cost}", ax=ax2, height=detections[0].shape[0])


# %%
def recolor_detections(detections, graph, node_luts):
    """."""
    assert len(detections) == len(node_luts)
    
    out = []
    n_tracks = 1
    color_lookup_tables = []
    
    for t in tqdm(range(0, len(detections)), desc="Recoloring detections"):
        new_frame = np.zeros_like(detections[t])
        color_lut = {}
        for det_id, node_id in node_luts[t].items():
            if node_id not in graph.nodes:
                continue
            edges = graph.in_edges(node_id)
            if not edges:
                new_frame[detections[t] == graph.nodes[node_id]["detection_id"]] = n_tracks
                color_lut[graph.nodes[node_id]["detection_id"]] = n_tracks
                n_tracks += 1
            else:
                for v_tm1, u_t0 in edges:
                    new_frame[detections[t] == graph.nodes[u_t0]["detection_id"]] = color_lookup_tables[t-1][graph.nodes[v_tm1]["detection_id"]]
                    color_lut[graph.nodes[u_t0]["detection_id"]] = color_lookup_tables[t-1][graph.nodes[v_tm1]["detection_id"]]
                
        color_lookup_tables.append(color_lut)
        out.append(new_frame)
        

    return np.stack(out)

# %%
recolored_gt = recolor_detections(y, gt_graph, gt_luts)
detections_ilp_flow = recolor_detections(detections=detections, graph=flow, node_luts=candidate_luts)

viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()
viewer = napari.Viewer()
viewer.add_image(x)
# visualize_tracks(viewer, y)
viewer.add_labels(recolored_gt)
viewer.add_labels(detections)
viewer.add_labels(detections_ilp_flow)
viewer.grid.enabled = True


# %% [markdown]
# ## Checkpoint 1
# <div class="alert alert-block alert-success"><h3>Checkpoint 1: We have familiarized ourselves with the formulation of an ILP for linking and and have a feasible solution to a network flow.</h3></div>

# %% [markdown]
# ## Exercise 3.2
# <div class="alert alert-block alert-info"><h3>Exercise 3.2: TODO make placeholders! Extend the network flow from Exercise 3.1 such that tracks can start and end at arbitrary time points</h3></div>

# %%
def solve_network_flow_birth_death(graph):
    solver = motile.Solver(graph)
    
    # Add costs
    solver.add_costs(
        motile.costs.NodeSelection(
            weight=-1,
            attribute="weight"))
    solver.add_costs(
        motile.costs.EdgeSelection(
            weight=1,
            attribute="weight"))
    solver.add_costs(motile.costs.Appear(constant=0.5))
    # TODO implement
    # solver.add_costs(motile.costs.Disappear(constant=0.5))
    
    # Add constraints
    solver.add_constraints(motile.constraints.MaxParents(1))
    solver.add_constraints(motile.constraints.MaxChildren(1))
    
    solution = solver.solve()
    
    print_solution_stats(solver, graph)
    solution_graph = solution2graph(solver, graph)
    
    return solution_graph, solver.solution.get_value()


# %%
birth_death, birth_death_cost = solve_network_flow_birth_death(candidate_graph)

# %%
fig, (ax0, ax1, ax2) = plt.subplots(1,3, figsize=(32, 12))
draw_graph(gt_graph, "Ground truth graph", ax=ax0, height=detections[0].shape[0])
draw_graph(candidate_graph, "Candidate graph", ax=ax1, height=detections[0].shape[0])
draw_graph(birth_death, f"ILP solution (no divisions) - cost:{birth_death_cost}", ax=ax2, height=detections[0].shape[0])

# %%
recolored_gt = recolor_detections(y, gt_graph, gt_luts)
detections_ilp_nodiv = recolor_detections(detections=detections, graph=birth_death, node_luts=candidate_luts)

viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()
viewer = napari.Viewer()
viewer.add_image(x)
# visualize_tracks(viewer, y)
viewer.add_labels(recolored_gt)
viewer.add_labels(detections)
viewer.add_labels(detections_ilp_nodiv)
viewer.grid.enabled = True


# %% [markdown]
# ## ILP model including divisions

# %% [markdown]
# ## Exercise 3.3
# <div class="alert alert-block alert-info"><h3>Exercise 3.3: TODO make placeholders! Complete yet another extension of the ILP such that it allows for cell divisions</h3></div>

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
            weight=0.4,
            attribute="weight"))
    solver.add_costs(motile.costs.Appear(constant=0.15))
    # TODO implement
    # solver.add_costs(motile.costs.Disappear(constant=0.5))
    solver.add_costs(motile.costs.Split(0.02))
    
    
    # Add constraints
    solver.add_constraints(motile.constraints.MaxParents(1))
    solver.add_constraints(motile.constraints.MaxChildren(2))
    
    solution = solver.solve()
    
    print_solution_stats(solver, graph)
    solution_graph = solution2graph(solver, graph)
    
    return solution_graph, solver.solution.get_value()

# %%
full_ilp, full_ilp_cost = solve_full_ilp(candidate_graph)

# %%
det_solved_div = recolor_detections(detections=detections, graph=full_ilp, node_luts=candidate_luts)

# %%
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()
viewer = napari.Viewer()
viewer.add_image(x)
viewer.add_labels(recolored_gt)
viewer.add_labels(detections)
viewer.add_labels(det_solved_div)
viewer.grid.enabled = True

# %%
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2, figsize=(24, 16))
draw_graph(candidate_graph, "Candidate graph", ax=ax0)
draw_graph(full_ilp, f"ILP solution (with divisions) - cost: {full_ilp_cost}", ax=ax1)
draw_graph(gt_graph, "Ground truth graph", ax=ax2)
draw_graph(birth_death, f"ILP solution (no divisions) - cost: {birth_death_cost}", ax=ax3)


# %%
