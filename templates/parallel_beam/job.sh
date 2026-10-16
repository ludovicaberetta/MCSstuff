#!/bin/bash

PathToLibs=${HOME}/allenergie_lberetta/MCSstuff/lib
PathToRawData=/mnt/san_data/datasets/fluka_eu/MCS/RAW_DATA/parallel_beam
myCase="p030mm"

# start job
echo " starting job at `date` in folder `pwd` as `whoami` on `hostname`..."

# set up python and xsuite env
source ${HOME}/local/miniforge3/bin/activate

# link lib
rm -f MyLib.py ; ln -s ${PathToLibs}/MyLib.py

# actually run job
python3 job.py ${PathToRawData} input_parameters.json --module MyLib --output-root ./${myCase} --fail-if-any-error --nRows 10000 --only ${myCase}

# end of job
echo " ...ending job at `date`."
