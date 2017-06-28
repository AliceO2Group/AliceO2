#!/bin/bash

if [ $# -lt 1 ]; then
  echo "usage: compareReadoutMode3 <fileInfo>"
fi

fileInfo=$1

cmd="root.exe -q $O2_SRC/Detectors/TPC/reconstruction/macro/addInclude.C $O2_SRC/Detectors/TPC/monitor/macro/RunCompareMode3.C+'(\"$fileInfo\")'"
echo "running: $cmd"
eval $cmd
