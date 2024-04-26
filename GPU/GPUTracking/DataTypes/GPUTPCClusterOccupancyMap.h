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

/// \file GPUTPCClusterOccupancyMap.h
/// \author David Rohr

#ifndef GPUTPCCLUSTEROCCUPANCYMAP_H
#define GPUTPCCLUSTEROCCUPANCYMAP_H

#include "GPUCommonDef.h"
#include "GPUDefConstantsAndSettings.h"

namespace GPUCA_NAMESPACE::gpu
{
struct GPUParam;
struct GPUTPCClusterOccupancyMapBin {
  unsigned short bin[GPUCA_NSLICES][GPUCA_ROW_COUNT];

  GPUd() static unsigned int getNBins(const GPUParam& param);
  GPUd() static unsigned int getTotalSize(const GPUParam& param);
};

} // namespace GPUCA_NAMESPACE::gpu

#endif
