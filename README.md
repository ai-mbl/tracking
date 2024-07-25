# Exercise 9 - Tracking

This exercise was created by Benjamin Gallusser and Albert Dominguez Mantes,
and updated for 2024 by Caroline Malin-Mayor.

<img src="figures/tracking.gif" width="500"/><img src="figures/ilp_nodiv.png" width="500"/>

## Setup
1. Go into the folder with this repo and run
    ```
    source setup.sh
    ```
    to set up the environment for this exercise. This will take a few minutes.
   
2. Run
    ```
    jupyter lab
    ```
    , double-click on the `exercise.ipynb` files and follow the instructions in the notebook.
    
    Alternatively, open `exercise.ipynb` in VSCode with the jupyter extension.


## Overview

### Main Exercise: Tracking by detection with an integer linear program (ILP)

Here we will introduce a modern formulation of tracking-by-detection.

You will learn:
- how to represent tracking inputs and outputs as a graph using the `networkx` library
- how to use [`motile`](https://funkelab.github.io/motile/) to solve tracking via global optimization
- how to visualize tracking inputs and outputs
- how to evaluate tracking and understand common tracking metrics
- how to add custom costs to the candidate graph and incorpate them into `motile`
- how to learn the best **hyperparameters** of the ILP using an SSVM (bonus)

### Bonus: Tracking with two-step Linear Assignment Problem (LAP)

Here we will use a two-step linking algorithm implemented in the Fiji plugin TrackMate.

You will learn
- how to use **Trackmate**, a versatile ready-to-go implementation of two-step LAP tracking in `ImageJ/Fiji`.
