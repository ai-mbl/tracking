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
# # Exercise 2/3: Tracking with two-step Linear Assignment Problem (LAP)
#
# Here we will use an extended version of the bipartite matching algorithm we implemented in exercise 1.

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## Introduction to the two-step Linear Assignment Problem

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true tags=[]
# In the previous exercise, we have been able to track individual cells over time by matching detections one-to-one in adjacent frames. However, there are multiple phenomena that this model does not capture:
# - If a cell is not detected in some frame, for example due to going out of focus, its resulting track will be split apart. 
# - For tracing cell lineages, we want to capture the connection between mother and daughter cells in mitosis. To do this, we have to link one object in frame $t$ to two objects in frame $t+1$, but the bipartite graph matching formulation (also called *Linear Assignment Problem (LAP)*) we have implemented in exercise 1 only models one-to-one links.
#
# To account for these processes, Jaqaman et al. (2008) have introduced a second linear assignment problem that is applied to the output tracks (termed *tracklets*) of the frame-by-frame LAP from exercise 1.
#
# Here is the cost matrix or this LAP. For $N_T$ tracklets, it has shape $3N_T \times 3N_T$.
#
# <img src="figures/LAP_cost_matrix_2.png" width="500"/>
#
# [Jaqaman et al. (2008). Robust single-particle tracking in live-cell time-lapse sequences. Nature methods, 5(8), 695-702.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2747604/)

# %% [markdown]
# This LAP is run only once for an entire time-lapse, in contrast to the frame-by-frame LAPs from step 1.
#
# The costs for linking tracklets are defined in the following way:
# - Tracklets can appear (lower left block) and disappear (upper right block), just as in step 1.
# - Tracklet beginnings can be connected to tracklet ends, called gap closing (upper left block).
# - Tracklet beginnings (at time $t$) can be connected to intermediate points of tracklets at time $t$ (center left block). This allows for a division.
# - Conversely, also tracklet endings (at time $t$) can be connected to intermediate points of tracklets at time $t$ (upper middle block). This would correspond to merging cells. As we often know a priori that this is not feasible, this block of the matrix is usually set as invalid.
#
# Please refer to Jaqaman et al. (2008) for a detailed description of the matrix.

# %% [markdown]
# Instead of implementing this more involved LAP, we will use an ImageJ/Fiji implementation of it to see how it performs on the dataset from exercise 1. The implementation is part of the plugin *TrackMate* 

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true tags=[]
# ## Exercise 2.1
#
# <div class="alert alert-block alert-info"><h3>Exercise 2.1: Run LAP tracking in ImageJ/Fiji with TrackMate.</h3></div>

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## Install ImageJ/Fiji

# %% [markdown]
# Download Fiji from https://imagej.net/software/fiji/downloads and extract the zip directory:
# - on Windows and Linux: anywhere, for example in Desktop.
# - on MacOS: into the `Applications` directory.
#
# The TrackMate plugin is already included in Fiji.

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## Install StarDist inference for TrackMate

# %% [markdown]
# In exercise 1, we have seen that a deep-learning-based detector (for example StarDist) trained on a similar dataset extracts accurate detections of the nuclei. We will do this again in TrackMate. This requires the installation of some additional plugins.

# %% [markdown]
# Start up Fiji and go to `Help -> Update`, then to `Manage update sites` in the appearing window.
#
# Select `TrackMate-StarDist`, `StarDist` and `CSBDeep` and press `Close`. Finally, click `Apply changes` to start the installation. After it is done, restart Fiji.
#
# <img src="figures/trackmate/install.png" width="600"/>

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true
# ## Load the dataset

# %% [markdown] tags=[]
# We will use the same dataset as in exercise 1. It is part of the tracking exercise GitHub repository at `tracking/data/exercise1`.
#
# Drag and drop the `images` directory into Fiji and click `OK` in the appearing prompt.

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## Start TrackMate

# %% [markdown]
# You can either use the Fiji search bar or go to `Plugins -> tracking -> TrackMate`.
#
# <img src="figures/trackmate/start_trackmate.png" width="700"/>

# %% [markdown]
# TrackMate might prompt you to swap axes to internally represent the set of images as a time series, please confirm.
#
# <img src="figures/trackmate/swap_axes.png" width="300"/>.
#
# Press `next` to skip the dataset cropping.

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# ## Extract detections

# %% [markdown]
# To use the StarDist model pretrained on a similar dataset, select `StarDist detector custom model`. The model is part of the tracking exercise GitHub repository at `tracking/models/TF_SavedModel.zip` (no need to unzip).
#
# <img src="figures/trackmate/stardist_pretrained.png" width="800"/>

# %% [markdown]
# Press `next` to run StarDist. After it is done, you can skip the `Initial thresholding` and `Set filters on spots steps` by pressing `next`.
#
# You should get nice detections like these ones:
#
# <img src="figures/trackmate/detections.png" width="800"/>

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# ## Linking frame-by-frame LAP

# %% [markdown]
# First, we will run the `Simple LAP tracker`. We set
# - `Max linking distance: 50`
# - `Gap-closing max distance: 0` 
# - `Gap-closing max frame gap: 0`
#
# and run the linking.
#

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ## Exercise 2.2
# <div class="alert alert-block alert-info"><h3>Exercise 2.2: Inspect the "Simple LAP tracker" results and compare to the results from exercise 1.</h3>
# What are the differences? What are possible reasons?
#     
# </div>

# %% [markdown]
# Here are some reasonable visualization setting for this dataset (press the pliers icon to adapt).
#
# Feel free to play around to improve visualization of things you are interested in.
#
# <img src="figures/trackmate/visualization_settings.png" width="1000"/>

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## Linking with two-step LAP

# %% [markdown]
# Go one step back in the TrackMate panel and select `LAP tracker` for linking now. You will be presented with the options described at the top of this notebook.
# Using all the knowledge you have by now about this dataset, set up the options to extract a satisfying tracking.

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ## Exercise 2.3
# <div class="alert alert-block alert-info"><h3>Exercise 2.3: Using all the knowledge you have by now about this dataset to set up the LAP tracker options.</h3>    
# </div>
#
#
# Note on `feature penalties`: TrackMate uses a range of features to calculate distances between frames. By setting a penalty for a certain feature, you multiply that dimension of the distance vector. For example, if you set the penalty for `Y=100`, you will not get any vertical links.

# %% [markdown]
# <img src="figures/trackmate/results.png" width="700"/>

# %%
