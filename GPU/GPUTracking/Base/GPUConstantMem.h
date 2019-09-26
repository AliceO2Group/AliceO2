// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#if defined(GPUCA_NOCOMPAT_ALLCINT) && (!defined(GPUCA_GPULIBRARY) || !defined(GPUCA_ALIROOT_LIB))
#include "GPUTPCConvert.h"
#include "GPUTPCCompression.h"
#include "GPUTPCGMMerger.h"
#include "GPUITSFitter.h"
#include "GPUTRDTracker.h"
#else
namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUTPCGMMerger
{
};
class GPUITSFitter
{
};
class GPUTRDTracker
{
  void SetMaxData() {}
};
class GPUTPCConvert
{
};
class GPUTPCCompression
{
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE
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
  GPUTRDTracker trdTracker;
  GPUITSFitter itsFitter;
  GPUTrackingInOutPointers ioPtrs;
  GPUCalibObjectsConst calibObjects;
};

// Must be placed here, to avoid circular header dependency
GPUdi() GPUconstantref() const MEM_CONSTANT(GPUParam) & GPUProcessor::Param() const { return mConstantMem->param; }

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
