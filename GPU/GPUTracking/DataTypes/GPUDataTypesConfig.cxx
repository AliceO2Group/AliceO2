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

/// \file GPUDataTypesConfig.cxx
/// \author David Rohr

#include "GPUDataTypesConfig.h"
#include <cstring>

using namespace o2::gpu;

gpudatatypes::DeviceType gpudatatypes::GetDeviceType(const char* type)
{
  for (uint32_t i = 1; i < sizeof(DEVICE_TYPE_NAMES) / sizeof(DEVICE_TYPE_NAMES[0]); i++) {
    if (strcmp(DEVICE_TYPE_NAMES[i], type) == 0) {
      return (DeviceType)i;
    }
  }
  return DeviceType::INVALID_DEVICE;
}
