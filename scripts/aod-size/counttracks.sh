#!/bin/bash

# This script counts the number of events, bc's and tracks in a list of AO2D.root files.
# run as: counttracks.sh [file1] [file2] ...
# Remote file locations should be specified as: alien:///alice/.../AO2D.root

LIST=""
for var in "$@"
do
  LIST+="$var "
done

LIST=\"$LIST\"

root.exe -b -q -l counttracks.C\("$LIST"\)
