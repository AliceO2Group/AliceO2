// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#pragma once

#include <gpucf/algorithms/steps/steps.h>

#include <CL/cl2.h>

namespace gpucf
{

class ClusterFinder
{

 public:
  ClusterFinder(ClusterFinderConfig, size_t, ClEnv);

 protected:
  cl::CommandQueue queue;

  ClusterFinderState state;

  ClusterToCPU clusterToCPU;
  CompactPeaks compactPeaks;
  ComputeCluster computeCluster;
  CountPeaks countPeaks;
  DigitsToGPU digitsToGPU;
  FillChargeMap fillChargeMap;
  FindPeaks findPeaks;
  GPUNoiseSuppression noiseSuppression;
  ResetMaps resetMaps;
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
