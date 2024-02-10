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
#include "DISocket.h"

#include <fairmq/FairMQDevice.h>
#include <fairmq/FairMQParts.h>

#include <cstring>

namespace o2::framework::data_inspector
{
inline bool isNonInternalDevice(const DeviceSpec& spec)
{
  return spec.name.find("internal") == std::string::npos;
}

std::vector<DIMessage> serializeO2Messages(const std::vector<DataRef>& refs, const std::string& deviceName);
} // namespace o2::framework::data_inspector

#endif // O2_DATAINSPECTOR_H
