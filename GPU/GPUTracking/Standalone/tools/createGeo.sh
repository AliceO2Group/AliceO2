#!/bin/bash

PATH=$PATH:/usr/share/Modules/bin/:/home/qon/alice/alibuild
export ALIBUILD_WORK_DIR="$HOME/alice/sw"
eval "`alienv shell-helper`"
alienv load O2/latest

o2-sim -n 1

export ROOT_INCLUDE_PATH=$ROOT_INCLUDE_PATH:/home/qon/alice/GPU/Common/:/home/qon/alice/GPU/GPUTracking/Base:/home/qon/alice/GPU/GPUTracking/SliceTracker:/home/qon/alice/GPU/GPUTracking/Merger:/home/qon/alice/GPU/GPUTracking/TRDTracking
root -l -q -b createGeo.C+
