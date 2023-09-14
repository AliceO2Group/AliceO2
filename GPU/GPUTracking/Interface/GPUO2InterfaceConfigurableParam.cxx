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

/// \file GPUO2InterfaceConfigurableParam.cxx
/// \author David Rohr

#include "GPUO2InterfaceConfigurableParam.h"
#include "GPUO2InterfaceConfiguration.h"
#include "GPUDataTypes.h"

using namespace o2::gpu;
#define BeginNamespace(name)
#define EndNamespace()
#define AddOption(name, type, default, optname, optnameshort, help, ...)
#define AddOptionRTC(...) AddOption(__VA_ARGS__)
#define AddVariable(name, type, default)
#define AddVariableRTC(...) AddVariable(__VA_ARGS__)
#define AddOptionSet(name, type, value, optname, optnameshort, help, ...)
#define AddOptionVec(name, type, optname, optnameshort, help, ...)
#define AddOptionArray(name, type, count, default, optname, optnameshort, help, ...)
#define AddOptionArrayRTC(...) AddOptionArray(__VA_ARGS__)
#define AddSubConfig(name, instance)
#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr, o2prefix) O2ParamImpl(GPUCA_M_CAT(GPUConfigurableParam, name))
#define BeginHiddenConfig(...)
#define EndConfig()
#define AddCustomCPP(...)
#define AddHelp(...)
#define AddShortcut(...)
#include "GPUSettingsList.h"
#undef BeginNamespace
#undef EndNamespace
#undef AddOption
#undef AddOptionRTC
#undef AddVariable
#undef AddVariableRTC
#undef AddOptionSet
#undef AddOptionVec
#undef AddOptionArray
#undef AddOptionArrayRTC
#undef AddSubConfig
#undef BeginSubConfig
#undef BeginHiddenConfig
#undef EndConfig
#undef AddCustomCPP
#undef AddHelp
#undef AddShortcut

GPUSettingsO2 GPUO2InterfaceConfiguration::ReadConfigurableParam_internal()
{
#define BeginNamespace(name)
#define EndNamespace()
#define AddOption(name, type, default, optname, optnameshort, help, ...) dst.name = src.name;
#define AddOptionRTC(...) AddOption(__VA_ARGS__)
#define AddVariable(name, type, default)
#define AddVariableRTC(...) AddVariable(__VA_ARGS__)
#define AddOptionSet(name, type, value, optname, optnameshort, help, ...)
#define AddOptionVec(name, type, optname, optnameshort, help, ...)
#define AddOptionArray(name, type, count, default, optname, optnameshort, help, ...) \
  for (int i = 0; i < count; i++) {                                                  \
    dst.name[i] = src.name[i];                                                       \
  }
#define AddOptionArrayRTC(...) AddOptionArray(__VA_ARGS__)
#define AddSubConfig(name, instance) dst.instance = instance;
#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr, o2prefix) \
  name instance;                                                                             \
  {                                                                                          \
    auto& src = GPUCA_M_CAT(GPUConfigurableParam, name)::Instance();                         \
    name& dst = instance;
#define BeginHiddenConfig(name, instance) {
#define EndConfig() }
#define AddCustomCPP(...)
#define AddHelp(...)
#define AddShortcut(...)
#include "GPUSettingsList.h"
#undef BeginNamespace
#undef EndNamespace
#undef AddOption
#undef AddOptionRTC
#undef AddVariable
#undef AddVariableRTC
#undef AddOptionSet
#undef AddOptionVec
#undef AddOptionArray
#undef AddOptionArrayRTC
#undef AddSubConfig
#undef BeginSubConfig
#undef BeginHiddenConfig
#undef EndConfig
#undef AddCustomCPP
#undef AddHelp
#undef AddShortcut

  configProcessing = proc;
  configReconstruction = rec;
  configDisplay = display;
  configQA = QA;
  if (global.continuousMaxTimeBin) {
    configGRP.continuousMaxTimeBin = global.continuousMaxTimeBin;
  } else {
    configGRP.continuousMaxTimeBin = global.tpcTriggeredMode ? 0 : -1;
  }
  if (global.solenoidBz > -1e6f) {
    configGRP.solenoidBz = global.solenoidBz;
  }
  if (global.constBz) {
    configGRP.constBz = global.constBz;
  }
  if (global.gpuDisplayfilterMacro != "") {
    configDisplay.filterMacros.emplace_back(global.gpuDisplayfilterMacro);
  }
  if (configReconstruction.tpc.trackReferenceX == 1000.f) {
    configReconstruction.tpc.trackReferenceX = 83.f;
  }
  configDeviceBackend.deviceType = GPUDataTypes::GetDeviceType(global.deviceType.c_str());
  configDeviceBackend.forceDeviceType = global.forceDeviceType;
  return global;
}

#include "utils/qconfig_helpers.h"

namespace
{
GPUSettingsStandalone configStandalone;
std::vector<std::function<void()>> qprint_global;
#define QCONFIG_PRINT
#include "utils/qconfig.h"
#undef QCONFIG_PRINT
} // namepsace

void GPUO2InterfaceConfiguration::PrintParam_internal()
{
  qConfigPrint(configProcessing, "proc.");
  qConfigPrint(configReconstruction, "rec.");
}
