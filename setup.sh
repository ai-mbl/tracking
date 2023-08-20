#!/usr/bin/env -S bash -i

##################
### Exercise 1 ###
##################

# Create environment
mamba create -y -n 08-tracking python=3.9

# Activate environment
mamba activate 08-tracking

# Install dependencies
mamba install -y -c conda-forge napari

### Install tensorflow ###
# from https://www.tensorflow.org/install/pip#linux
mamba install -y -c conda-forge cudatoolkit=11.8.0
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.13.*
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# Verify install:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
###########################

pip install numpy<1.25
pip install stardist
pip install ipywidgets

# Make environment discoverable by Jupyter
pip install ipykernel

mamba deactivate

##################
### Exercise 2 ###
##################

# Create environment
mamba create -y -n 08-ilp-tracking python=3.9

# Activate environment
mamba activate 08-ilp-tracking

# Install dependencies
mamba install -y -c conda-forge napari
mamba install -y -c conda-forge -c gurobi -c funkelab ilpy
pip install git+https://github.com/funkelab/motile
pip install traccuracy
pip install plotly
pip install ipywidgets

# Make environment discoverable by Jupyter
pip install ipykernel

mamba deactivate
