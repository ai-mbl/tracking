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

# Download data from s3
mkdir data
cd data

# If needed, install the AWS CLI (https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)

# On Linux:
# curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
# unzip awscliv2.zip
# sudo ./aws/install

# On mac: brew install awscli

# If needed, set up access credentials to use the AWS CLI 

# https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sso.html#sso-configure-profile-token-auto-sso
# I used Janelia's single sign in by running `aws configure sso`

# Then actually download the data
# the `--profile dlmbl` part is specific to your cli credential setup - that is what I called my access profile
aws s3 cp --bucket s3://dl-at-mbl-data/09_tracking/ data/ --recursive --profile dlmbl
