// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPURawData.h
/// \author David Rohr

#ifndef O2_GPU_RAW_DATA_H
#define O2_GPU_RAW_DATA_H

// Raw data parser is not accessible from GPU, therefore we use this header to wrap direct access to the current RDH
// Since OpenCL currently doesn't support bit fields, we have to access the members directly

#include "GPUCommonDef.h"
#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/RDHUtils.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
typedef o2::header::RAWDataHeader RAWDataHeaderGPU;

class GPURawDataUtils
{
 public:
  static GPUd() unsigned int getOrbit(const RAWDataHeaderGPU* rdh);
  static GPUd() unsigned int getBC(const RAWDataHeaderGPU* rdh);
  static GPUd() unsigned int getSize(const RAWDataHeaderGPU* rdh);
};

GPUdi() unsigned int GPURawDataUtils::getOrbit(const RAWDataHeaderGPU* rdh)
{
  return o2::raw::RDHUtils::getHeartBeatOrbit(*rdh);
}

GPUdi() unsigned int GPURawDataUtils::getBC(const RAWDataHeaderGPU* rdh)
{
  return o2::raw::RDHUtils::getHeartBeatBC(*rdh);
}

GPUdi() unsigned int GPURawDataUtils::getSize(const RAWDataHeaderGPU* rdh)
{
  return o2::raw::RDHUtils::getMemorySize(*rdh);
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
