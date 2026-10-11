// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionCPUDevicertc.cxx
/// \author David Rohr

#define __CPUDEVICE__              // Compile as device code, which runs on the host
#define GPUCA_DEVICE_SINGLE_THREAD // Device code with 1 thread per block
#define GPUCA_GPUCODE_GENRTC
#define GPUCA_GPUCODE_COMPILEKERNELS

#define GPUCA_RTC_SPECIAL_CODE(...) GPUCA_RTC_SPECIAL_CODE(__VA_ARGS__)
#define GPUCA_DETERMINISTIC_CODE(...) GPUCA_DETERMINISTIC_CODE(__VA_ARGS__)
#define GPUCA_RTC_CONSTEXPR GPUCA_RTC_CONSTEXPR

#include "GPUReconstructionIncludesDeviceAll.h"

#ifndef GPUCA_GPUCODE_DEVICE
#error RTC Preprocessing must run on device code
#endif
