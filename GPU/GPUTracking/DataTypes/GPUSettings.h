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

/// \file GPUSettings.h
/// \author David Rohr

#ifndef GPUSETTINGS_H
#define GPUSETTINGS_H

#include "GPUCommonDef.h"
#include "GPUSettingsTypes.h"

#endif // GPUSETTINGS_H

// Outside of header guard, since might run it multiple times for different config structs
#ifdef GPUCA_GPUCODE_DEVICE
#define QCONFIG_GPU
#endif
// See GPUSettingsList.h for a list of all available settings of GPU Reconstruction
#ifndef GPUCA_GPUCODE_GENRTC
#include "utils/qconfig.h"
#endif
