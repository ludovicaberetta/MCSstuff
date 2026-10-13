#!/bin/bash

PathToLibs=${HOME}/allenergie_lberetta/MCSstuff/lib
PathToRawData=${HOME}/allenergie_lberetta/RAW_DATA

# start job
echo " starting job at `date` in folder `pwd` as `whoami` on `hostname`..."

# set up python and xsuite env
source ${HOME}/local/miniforge3/bin/activate

# link lib
rm -f MyLib.py ; ln -s ${PathToLibs}/MyLib.py

# actually run job
python3 job.py ${PathToRawData} input_parameters.json --module MyLib --output-root $PWD/analysis_results --fail-if-any-error

# end of job
echo " ...ending job at `date`."
