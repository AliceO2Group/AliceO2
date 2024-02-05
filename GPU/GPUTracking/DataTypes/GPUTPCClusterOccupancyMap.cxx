// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCClusterOccupancyMap.cxx
/// \author David Rohr

#include "GPUTPCClusterOccupancyMap.h"
#include "GPUParam.h"

using namespace GPUCA_NAMESPACE::gpu;

GPUd() unsigned int GPUTPCClusterOccupancyMapBin::getNBins(const GPUParam& param)
{
  unsigned int maxTimeBin = param.par.continuousTracking ? param.par.continuousMaxTimeBin : TPC_MAX_TIME_BIN_TRIGGERED;
  return (maxTimeBin + param.rec.tpc.occupancyMapTimeBins) / param.rec.tpc.occupancyMapTimeBins; // Not -1, since maxTimeBin is allowed
}

GPUd() unsigned int GPUTPCClusterOccupancyMapBin::getTotalSize(const GPUParam& param)
{
  return getNBins(param) * sizeof(GPUTPCClusterOccupancyMapBin);
}
