// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUCommonNumericLimits.h

#ifndef GPUCOMMONNUMERICLIMITS_H
#define GPUCOMMONNUMERICLIMITS_H

#include "GPUCommonDef.h"

#if !defined(GPUCA_GPUCODE_DEVICE) || defined(__CUDACC__) || defined(__HIPCC__) // TODO: Get rid of this once std is available in all backends
#ifndef GPUCA_GPUCODE_COMPILEKERNELS
#include <limits>
#endif
#elif defined(__METAL__) // metal::numeric_limits comes with <metal_stdlib>, which the Metal kernel source includes first
namespace std
{
using metal::numeric_limits;
} // namespace std
#else
// Not supported yet for OpenCL / other backends
#endif

#endif
