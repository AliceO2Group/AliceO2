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
#include "GPULogging.h"
#include "GPUConstantMem.h"
#include "GPUChainTracking.h"
using namespace o2::gpu;

#ifndef GPUCA_NO_ROOT
#include "Rtypes.h" // Include ROOT header first, to use ROOT and disable replacements
#include "TROOT.h"
#include "TSystem.h"
#include "TMethodCall.h"
#include "TInterpreter.h"
#include <filesystem>
#include <fstream>
#include <unistd.h>

int32_t GPUDisplay::buildTrackFilter()
{
  if (!mCfgH.trackFilter) {
    return 0;
  }
  if (mUpdateTrackFilter) {
    std::string name = trackFilterName();
    const bool custom = (uint32_t)mCfgH.trackFilter > mConfig.filterMacros.size();
    if (custom) { // Code from the Qt GUI, written to a temporary file with a new name for every change, so that ROOT loads it again
      if (mCfgO.filterMacroCustom.empty()) {
        GPUError("No custom track filter macro set");
        return 1;
      }
      static uint32_t nCustomMacro = 0;
      name = (std::filesystem::temp_directory_path() / ("o2_gpu_display_track_filter_" + std::to_string(getpid()) + "_" + std::to_string(nCustomMacro++) + ".C")).string();
      std::ofstream(name) << mCfgO.filterMacroCustom << "\n";
    }
    if (!mTrackFilterMacroLoaded.empty()) {                                    // All macros define gpuDisplayTrackFilter, the previous one must be unloaded
      const bool recreate = !std::filesystem::exists(mTrackFilterMacroLoaded); // ROOT cannot unload a macro if its file no longer exists, the content does not matter
      if (recreate) {
        std::ofstream{mTrackFilterMacroLoaded};
      }
      gInterpreter->UnloadFile(mTrackFilterMacroLoaded.c_str());
      if (recreate) {
        std::filesystem::remove(mTrackFilterMacroLoaded);
      }
      mTrackFilterMacroLoaded.clear();
    }
    gROOT->Reset();
    const int32_t loadError = gROOT->LoadMacro(name.c_str());
    if (custom) {
      std::filesystem::remove(name);
    }
    if (loadError) {
      GPUError("Error loading trackFilter macro %s", name.c_str());
      return 1;
    }
    mTrackFilterMacroLoaded = name;
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

int32_t GPUDisplay::buildTrackFilter()
{
  return 0;
}

#endif
