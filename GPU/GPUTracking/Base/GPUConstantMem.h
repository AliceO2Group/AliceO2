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

/// \file GPUConstantMem.h
/// \author David Rohr

#ifndef GPUCONSTANTMEM_H
#define GPUCONSTANTMEM_H

#include "GPUTPCTracker.h"
#include "GPUParam.h"
#include "GPUDataTypesIO.h"
#include "GPUErrors.h"

#include "GPUTPCGMMerger.h"
#include "GPUTRDTracker.h"

#include "GPUTPCCompression.h"
#include "GPUTPCDecompression.h"
#include "GPUTPCClusterFinder.h"
#include "GPUTrackingRefit.h"

#ifdef GPUCA_KERNEL_DEBUGGER_OUTPUT
#include "GPUKernelDebugOutput.h"
#endif

#ifdef GPUCA_HAS_ONNX
#include "GPUTPCNNClusterizer.h"
#endif

namespace o2::gpu
{
struct GPUConstantMem {
  GPUParam param;
  GPUTPCTracker tpcTrackers[GPUCA_NSECTORS];
  GPUTPCCompression tpcCompressor;
  GPUTPCDecompression tpcDecompressor;
  GPUTPCGMMerger tpcMerger;
  GPUTRDTrackerGPU trdTrackerGPU;
  GPUTRDTracker trdTrackerO2;
  GPUTPCClusterFinder tpcClusterer[GPUCA_NSECTORS];
  GPUTrackingRefitProcessor trackingRefit;
  GPUTrackingInOutPointers ioPtrs;
  GPUCalibObjectsConst calibObjects;
  GPUErrors errorCodes;
#ifdef GPUCA_KERNEL_DEBUGGER_OUTPUT
  GPUKernelDebugOutput debugOutput;
#endif
#ifdef GPUCA_HAS_ONNX
  GPUTPCNNClusterizer tpcNNClusterer[GPUCA_NSECTORS];
#endif

  template <int32_t I>
  GPUd() auto& getTRDTracker();
};

template <>
GPUdi() auto& GPUConstantMem::getTRDTracker<0>()
{
  return trdTrackerGPU;
}
template <>
GPUdi() auto& GPUConstantMem::getTRDTracker<1>()
{
  return trdTrackerO2;
}

union GPUConstantMemCopyable {
#if !defined(__OPENCL__) || defined(__OPENCL_HOST__)
  GPUh() GPUConstantMemCopyable() {}  // NOLINT: We want an empty constructor, not a default one
  GPUh() ~GPUConstantMemCopyable() {} // NOLINT: We want an empty destructor, not a default one
  GPUh() GPUConstantMemCopyable(const GPUConstantMemCopyable& o)
  {
    for (uint32_t k = 0; k < sizeof(GPUConstantMem) / sizeof(int32_t); k++) {
      ((int32_t*)&v)[k] = ((int32_t*)&o.v)[k];
    }
  }
#endif
  GPUConstantMem v;
};

#if defined(GPUCA_GPUCODE)
static constexpr size_t gGPUConstantMemBufferSize = (sizeof(GPUConstantMem) + sizeof(uint4) - 1);
#endif
} // namespace o2::gpu
#if defined(GPUCA_HAS_GLOBAL_SYMBOL_CONSTANT_MEM)
GPUconstant() o2::gpu::GPUConstantMemCopyable gGPUConstantMemBuffer; // TODO: This should go into o2::gpu namespace, but then CUDA or HIP would not find the symbol
#endif // GPUCA_HAS_GLOBAL_SYMBOL_CONSTANT_MEM
namespace o2::gpu
{

// Must be placed here, to avoid circular header dependency
GPUdi() GPUconstantref() const GPUConstantMem* GPUProcessor::GetConstantMem() const
{
#if defined(GPUCA_GPUCODE_DEVICE) && defined(GPUCA_HAS_GLOBAL_SYMBOL_CONSTANT_MEM)
  return &GPUCA_CONSMEM;
#else
  return mConstantMem;
#endif
}

GPUdi() GPUconstantref() const GPUParam& GPUProcessor::Param() const
{
  return GetConstantMem()->param;
}

GPUdi() void GPUProcessor::raiseError(uint32_t code, uint32_t param1, uint32_t param2, uint32_t param3) const
{
  GetConstantMem()->errorCodes.raiseError(code, param1, param2, param3);
}

} // namespace o2::gpu

#endif
