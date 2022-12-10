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
#ifndef O2_DATAINSPECTOR_H
#define O2_DATAINSPECTOR_H

#include "Framework/DeviceSpec.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/DataInspectorService.h"

#include <fairmq/FairMQDevice.h>
#include <fairmq/FairMQParts.h>

#include <cstring>

namespace o2::framework::DataInspector
{
/* Checks if a command line argument relates to the data inspector. */
inline bool isInspectorArgument(const char* argument)
{
  return std::strcmp(argument, "--inspector") == 0;
}

/* Checks if device is used by the data inspector */
inline bool isInspectorDevice(const DataProcessorSpec& spec)
{
  return spec.name == "DataInspector";
}

inline bool isInspectorDevice(const DeviceSpec& spec)
{
  return spec.name == "DataInspector";
}

inline bool isNonInternalDevice(const DeviceSpec& spec)
{
  return spec.name.find("internal") == std::string::npos;
}

void sendToProxy(DataInspectorProxyService& diProxyService, const std::vector<DataRef>& refs, const std::string& deviceName);
}

#endif //O2_DATAINSPECTOR_H
