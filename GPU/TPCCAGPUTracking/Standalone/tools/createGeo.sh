#!/bin/bash

PATH=$PATH:/usr/share/Modules/bin/:/home/qon/alice/alibuild
export ALIBUILD_WORK_DIR="$HOME/alice/sw"
eval "`alienv shell-helper`"
alienv load O2/latest

#o2sim -n 1

export ROOT_INCLUDE_PATH=$ROOT_INCLUDE_PATH:/home/qon/alice/AliTPCCommon/Common/:/home/qon/alice/AliTPCCommon/TPCCAGPUTracking/GlobalTracker:/home/qon/alice/AliTPCCommon/TPCCAGPUTracking/SliceTracker:/home/qon/alice/AliTPCCommon/TPCCAGPUTracking/Merger:/home/qon/alice/AliTPCCommon/TPCCAGPUTracking/TRDTracking
root -l -q -b createGeo.C+
