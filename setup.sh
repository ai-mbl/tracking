#!/usr/bin/env -S bash -i

if [ "${BASH_SOURCE[0]}" -ef "$0" ]
then
    echo "Hey, you should source this script, not execute it!"
    echo "Try: source setup.sh"
    echo "Exiting..."
    exit 1
fi

# Create environment
conda create -y -n 09-tracking python=3.11

# Activate environment
conda activate 09-tracking

# Install dependencies
conda install -y -c conda-forge -c gurobi -c funkelab ilpy
conda install -y -c conda-forge napari pyqt

pip install numpy
pip install motile
pip install traccuracy
pip install plotly
pip install matplotlib
pip install ipywidgets
pip install nbformat
pip install pandas
pip install git+https://github.com/funkelab/motile_napari_plugin.git@track-viewer#egg=motile_plugin

# Make environment discoverable by Jupyter
pip install ipykernel

conda deactivate
