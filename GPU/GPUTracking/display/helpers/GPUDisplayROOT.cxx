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

/// \file GPUDisplayROOT.cxx
/// \author David Rohr

#ifndef GPUCA_NO_ROOT
#include "Rtypes.h" // Include ROOT header first, to use ROOT and disable replacements
#endif

#include "GPUDisplay.h"
using namespace GPUCA_NAMESPACE::gpu;

#ifndef GPUCA_NO_ROOT
#include "Rtypes.h" // Include ROOT header first, to use ROOT and disable replacements
#include "TROOT.h"
#include "TSystem.h"
#include "TMethodCall.h"

int GPUDisplay::buildTrackFilter()
{
  if (!mCfgH.trackFilter) {
    return 0;
  }
  if (mUpdateTrackFilter) {
    std::string name = "displayTrackFilter/";
    name += mConfig.filterMacros[mCfgH.trackFilter - 1];
    gROOT->Reset();
    if (gROOT->LoadMacro(name.c_str())) {
      GPUError("Error loading trackFilter macro %s", name.c_str());
      return 1;
    }
  }
  TMethodCall call;
  call.InitWithPrototype("gpuDisplayTrackFilter", "std::vector<bool>*, const o2::gpu::GPUTrackingInOutPointers*, const o2::gpu::GPUConstantMem*");
  const void* args[3];
  std::vector<bool>* arg0 = &mTrackFilter;
  args[0] = &arg0;
  args[1] = &mIOPtrs;
  const GPUConstantMem* arg2 = mChain ? mChain->GetProcessors() : nullptr;
  args[2] = &arg2;

  call.Execute(nullptr, args, sizeof(args) / sizeof(args[0]), nullptr);
  return 0;
}

#else

int GPUDisplay::buildTrackFilter()
{
}

#endif
