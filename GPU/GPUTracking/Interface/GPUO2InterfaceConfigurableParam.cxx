// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#define AddSubConfig(name, instance)
#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr) O2ParamImpl(GPUCA_M_CAT(GPUConfigurableParam, name))
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
#define AddSubConfig(name, instance)
#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr) \
  name instance;                                                                   \
  {                                                                                \
    auto& src = GPUCA_M_CAT(GPUConfigurableParam, name)::Instance();               \
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
#undef AddSubConfig
#undef BeginSubConfig
#undef BeginHiddenConfig
#undef EndConfig
#undef AddCustomCPP
#undef AddHelp
#undef AddShortcut

  configProcessing = proc;
  configReconstruction = rec;
  configDisplay = GL;
  configQA = QA;
  if (global.continuousMaxTimeBin) {
    configEvent.continuousMaxTimeBin = global.continuousMaxTimeBin;
  }
  if (global.solenoidBz > -1e6f) {
    configEvent.solenoidBz = global.solenoidBz;
  }
  if (global.constBz) {
    configEvent.constBz = global.constBz;
  }
  if (configReconstruction.TrackReferenceX == 1000.f) {
    configReconstruction.TrackReferenceX = 83.f;
  }
  configDeviceBackend.deviceType = GPUDataTypes::GetDeviceType(global.deviceType.c_str());
  configDeviceBackend.forceDeviceType = global.forceDeviceType;
  return global;
}
