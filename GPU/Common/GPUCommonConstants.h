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

/// \file GPUCommonConstants.h
/// \author David Rohr

#ifndef GPUCOMMONCONSTANTS_H
#define GPUCOMMONCONSTANTS_H

#include "GPUCommonDef.h"

#if !defined(__OPENCL__) || defined(__OPENCLCPP__)
namespace GPUCA_NAMESPACE::gpu::gpu_common_constants
{
static CONSTEXPR const float kCLight = 0.000299792458f;
}
#endif

#endif
