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
#ifndef O2_FRAMEWORK_CORE_WORKFLOWSERIALIZATIONHELPERS_H_
#define O2_FRAMEWORK_CORE_WORKFLOWSERIALIZATIONHELPERS_H_

#include "Framework/DataProcessorSpec.h"
#include "Framework/DataProcessorInfo.h"
#include "Framework/CommandInfo.h"

#include <iosfwd>
#include <vector>

namespace o2::framework
{

struct WorkflowSerializationHelpers {
  ///@return false if the previous workflow failed to generate a valid config, true otherwise
  static bool import(std::istream& s,
                     std::vector<DataProcessorSpec>& workflow,
                     std::vector<DataProcessorInfo>& metadata,
                     CommandInfo& command);
  static void dump(std::ostream& o,
                   std::vector<DataProcessorSpec> const& workflow,
                   std::vector<DataProcessorInfo> const& metadata,
                   CommandInfo const& commandInfo);
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_CORE_WORKFLOWSERIALIZATIONHELPERS_H_
