#!/bin/bash

PathToRawData=/mnt/san_data/datasets/fluka_eu/MCS/RAW_DATA/parallel_beam
myCase=$1

# start job
echo " starting job at `date` in folder `pwd` as `whoami` on `hostname`..."

# set up python and xsuite env
source ${HOME}/local/miniforge3/bin/activate base

# actually run job
python3 job.py ${PathToRawData} input_parameters.json --module MyLib --output-root ./${myCase} --fail-if-any-error --only ${myCase}

# end of job
echo " ...ending job at `date`."
