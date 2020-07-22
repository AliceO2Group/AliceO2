// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUSettings.cxx
/// \author David Rohr

#include "GPUSettings.h"
#include "GPUDef.h"
#include "GPUDataTypes.h"
#include <cstring>

using namespace GPUCA_NAMESPACE::gpu;

GPUSettingsDeviceBackend::GPUSettingsDeviceBackend()
{
  deviceType = GPUDataTypes::DeviceType::CPU;
  forceDeviceType = true;
  master = nullptr;
}

GPUSettingsEvent::GPUSettingsEvent()
{
  solenoidBz = -5.00668;
  constBz = 0;
  homemadeEvents = 0;
  continuousMaxTimeBin = 0;
  needsClusterer = 0;
}
