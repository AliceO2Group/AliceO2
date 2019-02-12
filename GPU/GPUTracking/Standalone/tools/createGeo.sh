#!/bin/bash

PATH=$PATH:/usr/share/Modules/bin/:/home/qon/alice/alibuild
export ALIBUILD_WORK_DIR="$HOME/alice/sw"
eval "`alienv shell-helper`"
alienv load O2/latest

o2sim -n 1

export ROOT_INCLUDE_PATH=$ROOT_INCLUDE_PATH:/home/qon/alice/AliGPU/Common/:/home/qon/alice/AliGPU/GPUTracking/Base:/home/qon/alice/AliGPU/GPUTracking/SliceTracker:/home/qon/alice/AliGPU/GPUTracking/Merger:/home/qon/alice/AliGPU/GPUTracking/TRDTracking
root -l -q -b createGeo.C+
