#!/bin/bash

PathToLibs=$HOME/MCSstuff/lib

# start job
echo " starting job at `date` in folder `pwd` as `whoami` on `hostname`..."

# set up python and xsuite env
source ~/local/miniforge3/bin/activate

# actually run job
python3 job_cluster.py /home/lberetta/allenergie_lberetta/RAW_DATA input_parameters.json --module ${PathToLibs}/MyLib --output-root  $PWD/analysis_results --fail-if-any-error

# end of job
echo " ...ending job at `date`."
