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

#ifndef GPUMANAGER_INTERFACE_PARAMS
#define GPUMANAGER_INTERFACE_PARAMS

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
namespace o2::vertexing::device
{
struct GPUInterfaceParamConfig : public o2::conf::ConfigurableParamHelper<GPUInterfaceParamConfig> {
  size_t streamPoolSize = 8;
  O2ParamDef(GPUInterfaceParamConfig, "GPUInterfaceParams")
};
} // namespace o2::vertexing::device
#endif