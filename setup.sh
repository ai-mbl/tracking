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
pip install numpy
pip install "motile>=0.3"
pip install "traccuracy>=0.1.1"
pip install git+https://github.com/funkelab/motile_napari_plugin.git@track-viewer#egg=motile_plugin
pip install matplotlib
pip install ipywidgets
pip install nbformat
pip install pandas

# Make environment discoverable by Jupyter
pip install ipykernel

conda deactivate

# Download data from s3
wget https://dl-at-mbl-data.s3.us-east-2.amazonaws.com/2024/09_tracking/data.zip
unzip data.zip
rm data.zip

# Alternatively, use the aws cli
# mkdir data
# aws s3 cp s3://dl-at-mbl-data/2024/09_tracking/ data/ --recursive --no-sign-request
