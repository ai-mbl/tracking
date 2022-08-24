# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Exercise 2/3: Tracking with Linear Assignment Problem (LAP)
#
# You can run this notebook on your laptop, a GPU is not needed :).
#
# Here we will use an extended version of the tracking algorithm introduced in exercise 1 which uses a linking algorithm that considers more than two frames at a time in a second optimization step.
#     
# You will learn
# - how this formulation addresses **typical challenges of tracking in bioimages**, like cell division and objects temporarily going out of focus.
# - how to use **_Trackmate_**, a versatile ready-to-go implementation of LAP tracking in ImageJ.

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## Exercise 2
#
# <div class="alert alert-block alert-info"><h3>Exercise 2: With your understanding of the challenges of the dataset from exercise 1, perform two-step LAP tracking using StarDist detections in ImageJ/Fiji with Trackmate.</h3>
#
# </div>
#
# <img src="figures/LAP_cost_matrix_2.png" width="500"/>
#
# Taken from Jaqaman et al. (2008)

# %% [markdown]
# ## Install ImageJ/Fiji, including StarDist inference

# %% [markdown]
# TODO

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## Load and prepare the dataset

# %% tags=[]
TODO

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## Extract detections

# %% [markdown]
# TODO

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## Extract features

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## Introduction to the linear assignment problem (LAP)
#
# In the previous exercise, we have been able to track individual cells over time by linking detections frame-by-frame. However, there are multiple processes that this approach is not able to capture:
# - For tracing cell lineages, we want to capture the connection between mother and daughter cells in cell divisions. To do this, we have to link one object in frame $t$ to two objects in frame $t+1$, but the matching formulation in exercise 1 only allows one-to-one links.
# - If a cell is not detected in just a single frame, its resulting track will be split apart if we only use frame-by-frame linking.
#
# To account for these processes, Jaqaman et al. (2008) have introduced a two-step linking formulation that can be solved using the convenient linear assignment problem (LAP) formulation.
#
# The first step consists of frame-by-frame linking, similar to the matching formulation in exercise 1/3. The output are track segments for individual cells. In the second step, we define costs for linking these track segments beginning to end (gap closing) and beginning to intermediate (cell division).
#
# TODO write more about second step.
#
# <!-- TODO upate!
# The goal of the linear assignment algorithm is to select a set of pairs such that the sum of the selected weights is minimized. In the framework established by Jaqaman et al, the cost matrix is divided into four quadrants. The top left and bottom right corners correspond to direct matches between objects in frame n to objects in frame $n+1$. The bottom left corner is populated by a diagonal matrix where diagonal values correspond to the probability of a cell dividing. The top right corner is filled with another diagonal matrix in which the diagonal values correspond to the probability of cell death. A diagram of the cost matrix is shown below. -->
#
# [Jaqaman et al. (2008). Robust single-particle tracking in live-cell time-lapse sequences. Nature methods, 5(8), 695-702.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2747604/)

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## Frame-by-frame linking

# %% [markdown] tags=[]
# ## Segment linking

# %% [markdown]
# TODO

# %% [markdown]
# ## Run the LAP

# %% [markdown]
# TODO

# %% [markdown] tags=[]
# ## Visualize results

# %%
