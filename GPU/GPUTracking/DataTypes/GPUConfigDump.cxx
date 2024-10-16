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

/// \file GPUConfigDump.cxx
/// \author David Rohr

#include "GPUConfigDump.h"
#include "GPUDataTypes.h"
#include "GPUSettings.h"

#include <functional>
#include <iostream>
#include <cstdio>

#include "utils/qconfig_helpers.h"

using namespace GPUCA_NAMESPACE::gpu;

namespace
{
GPUSettingsStandalone configStandalone;
std::vector<std::function<void()>> qprint_global;
#define QCONFIG_PRINT
#include "utils/qconfig.h"
#undef QCONFIG_PRINT
} // namespace

void GPUConfigDump::dumpConfig(const GPUSettingsRec* rec, const GPUSettingsProcessing* proc, const GPUSettingsQA* qa, const GPUSettingsDisplay* display, const GPUSettingsDeviceBackend* device, const GPURecoStepConfiguration* workflow)
{
  if (rec) {
    qConfigPrint(*rec, "rec.");
  }
  if (proc) {
    qConfigPrint(*proc, "proc.");
  }
  if (qa) {
    qConfigPrint(*qa, "QA.");
  }
  if (display) {
    qConfigPrint(*display, "display.");
  }
  if (device) {
    std::cout << "\n\tGPUSettingsDeviceBackend:\n"
              << "\tdeviceType = " << (int32_t)device->deviceType << "\n"
              << "\tforceDeviceType = " << (device->forceDeviceType ? "true" : "false") << "\n"
              << "\tslave = " << (device->master ? "true" : "false") << "\n";
  }
  if (workflow) {
    printf("\n\tReconstruction steps / inputs / outputs:\n\tReco Steps = 0x%08x\n\tReco Steps GPU = 0x%08x\n\tInputs = 0x%08x\n\tOutputs = 0x%08x\n", (uint32_t)workflow->steps, (uint32_t)workflow->stepsGPUMask, (uint32_t)workflow->inputs, (uint32_t)workflow->outputs);
  }
}
