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

/// \file GPUReconstructionCUDAExternalProvider.cu
/// \author David Rohr

#include "GPUReconstructionCUDADef.h"
#include "GPUReconstructionCUDAIncludes.h"

#include "GPUReconstructionCUDA.h"
#include "GPUReconstructionCUDAInternals.h"
#include "CUDAThrustHelpers.h"

#include <stdexcept>

using namespace GPUCA_NAMESPACE::gpu;

#include "GPUConstantMem.h"

// Files needed for O2 propagator
#include "MatLayerCylSet.cxx"
#include "MatLayerCyl.cxx"
#include "Ray.cxx"
#include "TrackParametrization.cxx"
#include "TrackParametrizationWithError.cxx"
#include "Propagator.cxx"
#include "TrackLTIntegral.cxx"

#ifndef GPUCA_NO_CONSTANT_MEMORY
static GPUReconstructionDeviceBase::deviceConstantMemRegistration registerConstSymbol([]() {
  void* retVal = nullptr;
  if (cudaGetSymbolAddress(&retVal, gGPUConstantMemBuffer) != cudaSuccess) {
    throw std::runtime_error("Could not obtain GPU constant memory symbol");
  }
  return retVal;
});
#endif
