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

/// \file GPUReconstructionCUDArtc.cu
/// \author David Rohr

#define GPUCA_GPUCODE_GENRTC
#define GPUCA_GPUCODE_COMPILEKERNELS
#include "GPUReconstructionCUDADef.h"
#include "GPUReconstructionIncludesDevice.h"

#ifndef GPUCA_GPUCODE_DEVICE
#error RTC Preprocessing must run on device code
#endif
