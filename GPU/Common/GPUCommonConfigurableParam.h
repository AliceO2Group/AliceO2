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

/// \file GPUCommonConfigurableParam.h
/// \author David Rohr

#ifndef GPUCOMMONCONFIGURABLEPARAM_H
#define GPUCOMMONCONFIGURABLEPARAM_H

#include "GPUCommonDef.h"

#if defined(GPUCA_STANDALONE)

namespace o2::conf
{
template <class T>
struct ConfigurableParamHelper {
  static const T& Instance()
  {
    static T instance;
    return instance;
  }
};
#define O2ParamDef(...)
} // namespace o2::conf

#else

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

#endif

#endif
