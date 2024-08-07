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

/// \file GPUConfigDump.h
/// \author David Rohr

#ifndef GPUCONFIGDUMP_H
#define GPUCONFIGDUMP_H

#include "GPUCommonDef.h"

namespace GPUCA_NAMESPACE::gpu
{
struct GPUSettingsRec;
struct GPUSettingsProcessing;
struct GPUSettingsQA;
struct GPUSettingsDisplay;
struct GPUSettingsDeviceBackend;
struct GPURecoStepConfiguration;

class GPUConfigDump
{
 public:
  static void dumpConfig(const GPUSettingsRec* rec, const GPUSettingsProcessing* proc, const GPUSettingsQA* qa, const GPUSettingsDisplay* display, const GPUSettingsDeviceBackend* device, const GPURecoStepConfiguration* workflow);
};

} // namespace GPUCA_NAMESPACE::gpu

#endif
