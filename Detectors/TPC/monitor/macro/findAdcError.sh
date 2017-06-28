#!/bin/bash

if [ $# -lt 1 ]; then
  echo "usage: findAdcError <run_min> [<run_max>]"
  exit 0
fi

RUN_MIN=$1
RUN_MAX=$1

if [ $# -ge 2 ]; then
  RUN_MAX=$2
fi

cmd="root.exe $O2_SRC/Detectors/TPC/reconstruction/macro/addInclude.C $O2_SRC/Detectors/TPC/monitor/macro/RunFindAdcError.C+'($RUN_MIN,$RUN_MAX)'"
echo "running: $cmd"
eval $cmd
