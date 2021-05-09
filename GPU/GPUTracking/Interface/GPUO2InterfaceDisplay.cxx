// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUO2InterfaceDisplay.cxx
/// \author David Rohr

#include "GPUParam.h"
#include "GPUDisplay.h"
#include "GPUQA.h"
#include "GPUO2InterfaceConfiguration.h"
#include "GPUO2InterfaceDisplay.h"
#include "GPUDisplayBackend.h"
#include "GPUDisplayBackendGlfw.h"
#include <unistd.h>

using namespace o2::gpu;
using namespace o2::tpc;

GPUO2InterfaceDisplay::GPUO2InterfaceDisplay(const GPUO2InterfaceConfiguration* config)
{
  mConfig.reset(new GPUO2InterfaceConfiguration(*config));
  mBackend.reset(new GPUDisplayBackendGlfw);
  mConfig->configProcessing.eventDisplay = mBackend.get();
  mConfig->configDisplay.showTPCTracksFromO2Format = true;
  mParam.reset(new GPUParam);
  mParam->SetDefaults(&config->configGRP, &config->configReconstruction, &config->configProcessing, nullptr);
  mParam->par.earlyTpcTransform = 0;
  if (mConfig->configProcessing.runMC) {
    mQA.reset(new GPUQA(nullptr, &config->configQA, mParam.get()));
    mQA->InitO2MCData();
  }
  mDisplay.reset(new GPUDisplay(mBackend.get(), nullptr, nullptr, mParam.get(), &mConfig->configCalib, &mConfig->configDisplay));
}

GPUO2InterfaceDisplay::~GPUO2InterfaceDisplay() = default;

int GPUO2InterfaceDisplay::startDisplay()
{
  int retVal = mDisplay->StartDisplay();
  if (retVal) {
    return retVal;
  }
  mDisplay->WaitForNextEvent();
  return 0;
}

int GPUO2InterfaceDisplay::show(const GPUTrackingInOutPointers* ptrs)
{
  std::unique_ptr<GPUTrackingInOutPointers> tmpPtr;
  if (mConfig->configProcessing.runMC) {
    tmpPtr = std::make_unique<GPUTrackingInOutPointers>(*ptrs);
    mQA->InitO2MCData(tmpPtr.get());
    ptrs = tmpPtr.get();
  }
  mDisplay->ShowNextEvent(ptrs);
  do {
    usleep(10000);
  } while (mBackend->mDisplayControl == 0);
  mDisplay->WaitForNextEvent();
  return 0;
}

int GPUO2InterfaceDisplay::endDisplay()
{
  mBackend->DisplayExit();
  return 0;
}
