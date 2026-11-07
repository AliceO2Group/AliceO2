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

/// \file GPUParamRTC.h
/// \author David Rohr

#ifndef GPUPARAMRTC_H
#define GPUPARAMRTC_H

#include "GPUParam.h"
#define GPUSETTINGS_ALL
#include "GPUSettings.h"
#include <string>

namespace o2::gpu
{
namespace gpu_rtc
{
#define QCONFIG_GENRTC
#define BeginNamespace(...)
#define EndNamespace(...)
#include "utils/qconfig.h"
#undef QCONFIG_GENRTC
#undef BeginNamespace
#undef EndNamespace
} // namespace gpu_rtc

struct GPUParamRTC : public internal::GPUParam_t<gpu_rtc::GPUSettingsRec, gpu_rtc::GPUSettingsParam> {
  void setFrom(const GPUParam& param);
  static std::string generateRTCCode(const GPUParam& param, bool useConstexpr);
};

} // namespace o2::gpu

#endif
