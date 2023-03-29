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
#ifndef O2_FRAMEWORK_DDSCONFIGHELPERS_H_
#define O2_FRAMEWORK_DDSCONFIGHELPERS_H_

#include "Framework/DeviceSpec.h"
#include "Framework/DeviceExecution.h"
#include "Framework/CommandInfo.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataProcessorInfo.h"
#include "Framework/ProcessingPolicies.h"
#include <vector>
#include <iosfwd>
#include <string>

namespace o2::framework
{

/// Helper to dump DDS configuration to run in a deployed
/// manner.
/// @a out is a stream where the configuration will be printed
/// @a mode is the mode in which the workflow is being run
/// @a workflow is the workflow to dump
/// @a metadata is the metadata to dump
/// @a workflowSuffix a suffix to add to all the ids in the workflow
/// @a specs is the internal representation of the dataflow topology
///          which we want to dump.
/// @a executions is the transient parameters for the afore mentioned
///          specifications
/// @a the full command being used
struct DDSConfigHelpers {
  static void dumpDeviceSpec2DDS(std::ostream& out,
                                 DriverMode mode,
                                 std::string const& workflowSuffix,
                                 std::vector<DataProcessorSpec> const& workflow,
                                 std::vector<DataProcessorInfo> const& metadata,
                                 std::vector<DeviceSpec> const& specs,
                                 std::vector<DeviceExecution> const& executions,
                                 CommandInfo const& commandInfo);
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_DDSCONFIGHELPERS_H_
