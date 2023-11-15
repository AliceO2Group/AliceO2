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
#include "GPUDataTypes.h"
#include "GPUErrors.h"

// Dummies for stuff not supported in legacy code (ROOT 5 / OPENCL1.2)
#if defined(GPUCA_NOCOMPAT_ALLCINT) && (!defined(GPUCA_GPULIBRARY) || !defined(GPUCA_ALIROOT_LIB))
#include "GPUTPCGMMerger.h"
#include "GPUTRDTracker.h"
#else
#include "GPUTRDDef.h"
namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUTPCGMMerger
{
};
template <class T, class P>
class GPUTRDTracker_t
{
  void SetMaxData(const GPUTrackingInOutPointers& io) {}
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE
#endif

// Dummies for stuff not suppored in legacy code, or for what requires O2 headers while not available
#if defined(GPUCA_NOCOMPAT_ALLCINT) && (!defined(GPUCA_GPULIBRARY) || !defined(GPUCA_ALIROOT_LIB)) && defined(GPUCA_HAVE_O2HEADERS)
#include "GPUTPCConvert.h"
#include "GPUTPCCompression.h"
#include "GPUITSFitter.h"
#include "GPUTPCClusterFinder.h"
#include "GPUTrackingRefit.h"
#else
#include "GPUO2FakeClasses.h"
#endif

#ifdef GPUCA_KERNEL_DEBUGGER_OUTPUT
#include "GPUKernelDebugOutput.h"
#endif

namespace GPUCA_NAMESPACE
{
namespace gpu
{
MEM_CLASS_PRE()
struct GPUConstantMem {
  MEM_CONSTANT(GPUParam)
  param;
  MEM_GLOBAL(GPUTPCTracker)
  tpcTrackers[GPUCA_NSLICES];
  GPUTPCConvert tpcConverter;
  GPUTPCCompression tpcCompressor;
  GPUTPCGMMerger tpcMerger;
  GPUTRDTrackerGPU trdTrackerGPU;
#ifdef GPUCA_HAVE_O2HEADERS
  GPUTRDTracker trdTrackerO2;
#endif
  GPUTPCClusterFinder tpcClusterer[GPUCA_NSLICES];
  GPUITSFitter itsFitter;
  GPUTrackingRefitProcessor trackingRefit;
  GPUTrackingInOutPointers ioPtrs;
  GPUCalibObjectsConst calibObjects;
  GPUErrors errorCodes;
#ifdef GPUCA_KERNEL_DEBUGGER_OUTPUT
  GPUKernelDebugOutput debugOutput;
#endif

#if defined(GPUCA_HAVE_O2HEADERS) && defined(GPUCA_NOCOMPAT)
  template <int I>
  GPUd() auto& getTRDTracker();
#else  // GPUCA_HAVE_O2HEADERS
  template <int I>
  GPUdi() GPUTRDTrackerGPU& getTRDTracker()
  {
    return trdTrackerGPU;
  }
#endif // !GPUCA_HAVE_O2HEADERS
};

#if defined(GPUCA_HAVE_O2HEADERS) && defined(GPUCA_NOCOMPAT)
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
#endif

#ifdef GPUCA_NOCOMPAT
union GPUConstantMemCopyable {
  GPUConstantMemCopyable() {}  // NOLINT: We want an empty constructor, not a default one
  ~GPUConstantMemCopyable() {} // NOLINT: We want an empty destructor, not a default one
  GPUConstantMemCopyable(const GPUConstantMemCopyable& o)
  {
    for (unsigned int k = 0; k < sizeof(GPUConstantMem) / sizeof(int); k++) {
      ((int*)&v)[k] = ((int*)&o.v)[k];
    }
  }
  GPUConstantMem v;
};
#endif

#if defined(GPUCA_GPUCODE) && defined(GPUCA_NOCOMPAT)
static constexpr size_t gGPUConstantMemBufferSize = (sizeof(GPUConstantMem) + sizeof(uint4) - 1);
#ifndef GPUCA_GPUCODE_HOSTONLY
#if defined(GPUCA_HAS_GLOBAL_SYMBOL_CONSTANT_MEM)
} // namespace gpu
} // namespace GPUCA_NAMESPACE
GPUconstant() GPUCA_NAMESPACE::gpu::GPUConstantMemCopyable gGPUConstantMemBuffer; // HIP constant memory symbol address cannot be obtained when in namespace
namespace GPUCA_NAMESPACE
{
namespace gpu
{
#endif // GPUCA_HAS_GLOBAL_SYMBOL_CONSTANT_MEM
#ifdef GPUCA_CONSTANT_AS_ARGUMENT
static GPUConstantMemCopyable gGPUConstantMemBufferHost;
#endif // GPUCA_CONSTANT_AS_ARGUMENT
#endif // !GPUCA_GPUCODE_HOSTONLY
#endif

// Must be placed here, to avoid circular header dependency
GPUdi() GPUconstantref() const MEM_CONSTANT(GPUParam) & GPUProcessor::Param() const
{
#if defined(GPUCA_GPUCODE_DEVICE) && defined(GPUCA_HAS_GLOBAL_SYMBOL_CONSTANT_MEM) && !defined(GPUCA_GPUCODE_HOSTONLY)
  return GPUCA_CONSMEM.param;
#else
  return mConstantMem->param;
#endif
}

GPUdi() GPUconstantref() const MEM_CONSTANT(GPUConstantMem) * GPUProcessor::GetConstantMem() const
{
#if defined(GPUCA_GPUCODE_DEVICE) && defined(GPUCA_HAS_GLOBAL_SYMBOL_CONSTANT_MEM) && !defined(GPUCA_GPUCODE_HOSTONLY)
  return &GPUCA_CONSMEM;
#else
  return mConstantMem;
#endif
}

GPUdi() void GPUProcessor::raiseError(unsigned int code, unsigned int param1, unsigned int param2, unsigned int param3) const
{
  GetConstantMem()->errorCodes.raiseError(code, param1, param2, param3);
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
