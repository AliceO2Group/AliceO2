// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUO2InterfaceConfigurableParam.h
/// \author David Rohr

// This file auto-generates a ConfigurableParam object from the GPU parameter macros.
// Set via:
// --configKeyValues "GPU_global.[x]=[y]" : for global GPU run configurations, like solenoidBz, gpuType, configuration object files.
// --configKeyValues "GPU_rec.[x]=[y]" : for GPU reconstruction related settings used on the GPU, like pt threshold for track rejection.
// --configKeyValues "GPU_proc.[x]=[y]" : for processing options steering GPU reconstruction like GPU device ID, debug output level, number of CPU threads.
// Check qconfigoptions.h for all options

#ifndef GPUO2INTERFACECONFIGURABLEPARAM_H
#define GPUO2INTERFACECONFIGURABLEPARAM_H

// Some defines denoting that we are compiling for O2
#ifndef HAVE_O2HEADERS
#define HAVE_O2HEADERS
#endif
#ifndef GPUCA_TPC_GEOMETRY_O2
#define GPUCA_TPC_GEOMETRY_O2
#endif
#ifndef GPUCA_O2_INTERFACE
#define GPUCA_O2_INTERFACE
#endif

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include "GPUSettings.h"
#include "GPUDefMacros.h"
#include <vector>

#define BeginNamespace(name) \
  namespace name             \
  {
#define EndNamespace() }
#define AddOption(name, type, default, optname, optnameshort, help, ...) type name = default;
#define AddVariable(name, type, default)
#define AddOptionSet(name, type, value, optname, optnameshort, help, ...)
#define AddOptionVec(name, type, optname, optnameshort, help, ...)
#define AddOptionArray(name, type, count, default, optname, optnameshort, help, ...) type name[count] = {default};
#define AddSubConfig(name, instance)
#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr)                                                     \
  struct GPUCA_M_CAT(GPUConfigurableParam, name) : public o2::conf::ConfigurableParamHelper<GPUCA_M_CAT(GPUConfigurableParam, name)> { \
    O2ParamDef(GPUCA_M_CAT(GPUConfigurableParam, name), GPUCA_M_STR(GPUCA_M_CAT(GPU_, instance))) public:
#define EndConfig() \
  }                 \
  ;
#define AddCustomCPP(...) __VA_ARGS__
#define AddHelp(...)
#define AddShortcut(...)
#include "qconfigoptions.h"
#undef BeginNamespace
#undef EndNamespace
#undef AddOption
#undef AddVariable
#undef AddOptionSet
#undef AddOptionVec
#undef AddOptionArray
#undef AddSubConfig
#undef BeginSubConfig
#undef EndConfig
#undef AddCustomCPP
#undef AddHelp
#undef AddShortcut

#endif
