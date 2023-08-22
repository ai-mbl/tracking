# Exercise 8 - Tracking

This exercise was created by Benjamin Gallusser and Albert Dominguez Mantes.

<img src="figures/tracking.gif" width="500"/><img src="figures/ilp_nodiv.png" width="500"/>

## Setup
1. Go into the folder with this repo and run
    ```
    source setup.sh
    ```
    to set up the environments for this exercise. This will take a few minutes.
   
1. Run
    ```
    jupyter lab
    ```
    , double-click on the `exercise{1|2|3}.ipynb` files and follow the instructions in the notebook.


## Overview

### 1. Tracking by detection and simple frame by frame matching

Here we will walk through all basic components of a tracking-by-detection algorithm.

You will learn
- to **store and visualize** tracking results with `napari` (Exercise 1.1).
- to use a robust pretrained deep-learning-based **object detection** algorithm called *StarDist* (Exercise 1.2).
- to implement a basic **nearest-neighbor linking algorithm** (Exercises 1.3 - 1.6).
- to compute optimal frame-by-frame linking by setting up a **bipartite matching problem** and using a python-based solver (Exercise 1.7).
- to compute suitable object **features** for the object linking process with `scikit-image` (Exercise 1.8, bonus).


### 2. Tracking with an integer linear program (ILP)

Here we will introduce a modern formulation of tracking-by-detection.

You will learn
- how linking with global context can be modeled and solved efficiently as a **network flow** using `motile` ([docs here](https://funkelab.github.io/motile/)) for a small-scale problem (Exercise 2.1).
- to adapt the previous formulation to allow for **arbitrary track starting and ending points** (Exercise 2.2).
- to extend the ILP to properly model **cell divisions** (Exercise 2.3).
- to tune the **hyperparameters** of the ILP (Exercise 2.4, bonus).


### 3. Bonus: Tracking with two-step Linear Assignment Problem (LAP)

Here we will use an extended version of the tracking algorithm introduced in exercise 1 which uses a linking algorithm that considers more than two frames at a time in a second optimization step.

You will learn
- how this formulation addresses **typical challenges of tracking in bioimages**, like cell division and objects temporarily going out of focus.
- how to use **Trackmate**, a versatile ready-to-go implementation of two-step LAP tracking in `ImageJ/Fiji`.
