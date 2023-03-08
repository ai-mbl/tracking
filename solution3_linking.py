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
# # Exercise 3/3: Tracking with an integer linear program (ILP)
#
# You could also run this notebook on your laptop, a GPU is not needed :).

# %% [markdown]
# This notebook was written by Benjamin Gallusser and Albert Dominguez Mantes.

# %% [markdown]
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

# Pretty tqdm progress bars 
# ! jupyter nbextension enable --py widgetsnbextension

# %% [markdown]
# ## Load the dataset and the detections and inspect them in napari

# %%
base_path = Path("data/exercise3")
det = np.load("data/exercise3/detected.npz", allow_pickle=True)
x = det["x"]
y = det["y"]
links = pd.DataFrame(det["links"], columns=["track_id", "from", "to", "parent_id"])
detections = det["detections"]
centers = det["centers"]
center_probs = det["center_probs"]
prob_maps = det["prob_maps"]

# %%
viewer = napari.Viewer()
viewer.add_image(x, name="image");


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
# visualize_tracks(viewer, y, links.to_numpy(), "ground_truth");

# %%
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()
viewer = napari.Viewer()
viewer.add_image(x)
visualize_tracks(viewer, y, links.to_numpy(), "ground_truth");
viewer.add_labels(detections, name=f"detections");
viewer.add_image(prob_maps, colormap="magma", scale=(2,2));
viewer.grid.enabled = True


# %% [markdown]
# ## Build a candidate graph from the detections

# %%
def build_graph(detections, max_distance, detection_probs=None, drift=(0,0)):
    """
    
        detection_probs: list of arrays, corresponding to ordered ids in detections.
    """
    G = nx.DiGraph()
    n_v = 0
    
    luts = []
    draw_positions = {}
    
    for t, d in enumerate(detections):
        frame = skimage.segmentation.relabel_sequential(d)[0]
        regions = skimage.measure.regionprops(frame)
        lut = {}
        for i, r in enumerate(regions):
            draw_pos = np.array([t, d.shape[0] - r.centroid[0]])
            weight = detection_probs[t][i] if detection_probs is not None else 1
            G.add_node(n_v, time=t, detection_id=r.label, weight=weight, draw_position=draw_pos)
            draw_positions[n_v] = draw_pos
            lut[r.label] = n_v
            n_v += 1
        luts.append(lut)

    n_e = 0
    for t, (d0, d1) in enumerate(zip(detections, detections[1:])):
        f0 = skimage.segmentation.relabel_sequential(d0)[0]
        r0 = skimage.measure.regionprops(f0)
        c0 = [np.array(r.centroid) for r in r0]

        f1 = skimage.segmentation.relabel_sequential(d1)[0]
        r1 = skimage.measure.regionprops(f1)
        c1 = [np.array(r.centroid) for r in r1]

        for _r0, _c0 in zip(r0, c0):
            for _r1, _c1 in zip(r1, c1):
                dist = np.linalg.norm(_c0 - _c1)
                if dist < max_distance:
                    G.add_edge(
                        luts[t][_r0.label],
                        luts[t+1][_r1.label],
                        # normalized euclidian distance
                        weight = np.linalg.norm(_c0 + np.array(drift) - _c1) / max_distance,
                        edge_id = n_e,
                    )
                    n_e += 1
    
    G = motile.TrackGraph(graph_data=G, frame_attribute="time")
    
    return G, luts


# %%
def build_graph_from_tracks(detections, links=None):
    """"""
    G = nx.DiGraph()
    n_v = 0
    
    luts = []
    draw_positions = {}
    
    for t, d in enumerate(detections):
        frame = d
        regions = skimage.measure.regionprops(frame)
        lut = {}
        for r in regions:
            draw_pos = np.array([t, d.shape[0] - r.centroid[0]])
            G.add_node(n_v, time=t, detection_id=r.label, weight=1, draw_position=draw_pos)
            draw_positions[n_v] = draw_pos
            lut[r.label] = n_v
            n_v += 1
        luts.append(lut)
        
    n_e = 0
    for t, (d0, d1) in enumerate(zip(detections, detections[1:])):
        f0 = d0
        r0 = skimage.measure.regionprops(f0)
        c0 = [np.array(r.centroid) for r in r0]

        f1 = d1
        r1 = skimage.measure.regionprops(f1)
        c1 = [np.array(r.centroid) for r in r1]

        for _r0, _c0 in zip(r0, c0):
            for _r1, _c1 in zip(r1, c1):
                if _r0.label == _r1.label:
                    G.add_edge(
                        luts[t][_r0.label],
                        luts[t+1][_r1.label],
                        # normalized euclidian distance
                        weight = np.linalg.norm(_c0 - _c1),
                        edge_id = n_e,
                    )
                    n_e += 1
    
    if links is not None:
        divisions = links[links[:,3] != 0]
        for d in divisions:
            if d[1] > 0 and d[1] < detections.shape[0]:
                try:
                    G.add_edge(luts[d[1] - 1][d[3]], luts[d[1]][d[0]])
                except KeyError:
                    pass
    
    return G, luts


# %%
def draw_graph(g, title=None, ax=None, height=None):
    pos = {i: g.nodes[i]["draw_position"] for i in g.nodes}
    if ax is None:
        _, ax = plt.subplots()
    ax.set_title(title)
    nx.draw(g, pos=pos, with_labels=True, ax=ax)

    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    if height:
        ax.set_ylim(0, height)
    
    ax.set_xlabel("time")
    ax.set_ylabel("y (spatial)");


# %%
gt_graph, gt_luts = build_graph_from_tracks(y, links.to_numpy())
candidate_graph, candidate_luts = build_graph(detections, max_distance=50, detection_probs=center_probs, drift=(-6 , 0))

# %%
fig, (ax0, ax1) = plt.subplots(1,2, figsize=(24, 12))
draw_graph(gt_graph, "Ground truth graph", ax=ax0, height=detections[0].shape[0])
draw_graph(candidate_graph, "Candidate graph", ax=ax1, height=detections[0].shape[0])


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
