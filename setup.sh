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

# Install additional requirements
if [[ "$CONDA_DEFAULT_ENV" == "09-tracking" ]]; then
    echo "Environment activated successfully for package installs"
    pip install numpy "motile>=0.3" "traccuracy>=0.1.1" "geff==0.5.0" "trackastra" "motile-toolbox<0.4" "zarr<3", git+https://github.com/funkelab/motile_napari_plugin.git@track-viewer#egg=motile_plugin matplotlib ipywidgets nbformat pandas ipykernel
    python -m ipykernel install --user --name "09-tracking"
else
    echo "Failed to activate environment for package installs. Dependencies not installed!"
fi

conda deactivate

# Download data from s3
wget https://dl-at-mbl-data.s3.us-east-2.amazonaws.com/2025/09_tracking/data.zip
unzip data.zip
rm data.zip

# Alternatively, use the aws cli
# mkdir data
# aws s3 cp s3://dl-at-mbl-data/2025/09_tracking/ data/ --recursive --no-sign-request
