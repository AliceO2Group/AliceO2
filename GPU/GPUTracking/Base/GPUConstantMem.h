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

#if (!defined(__OPENCL__) || defined(__OPENCLCPP__)) && (!defined(GPUCA_GPULIBRARY) || !defined(GPUCA_ALIROOT_LIB)) && (!defined(__CINT__) && !defined(__ROOTCINT__))
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
  GPUTPCGMMerger tpcMerger;
  GPUTRDTracker trdTracker;
  GPUITSFitter itsFitter;
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
