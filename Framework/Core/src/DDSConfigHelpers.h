// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_DDSCONFIGHELPERS_H_
#define O2_FRAMEWORK_DDSCONFIGHELPERS_H_

#include "Framework/DeviceSpec.h"
#include "Framework/DeviceExecution.h"
#include "Framework/CommandInfo.h"
#include <vector>
#include <iosfwd>

namespace o2::framework
{

/// Helper to dump DDS configuration to run in a deployed
/// manner.
/// @a out is a stream where the configuration will be printed
/// @a specs is the internal representation of the dataflow topology
///          which we want to dump.
/// @a executions is the transient parameters for the afore mentioned
///          specifications
/// @a the full command being used
void dumpDeviceSpec2DDS(std::ostream& out,
                        std::vector<DeviceSpec> const& specs,
                        std::vector<DeviceExecution> const& executions,
                        CommandInfo const& commandInfo);

} // namespace o2::framework
#endif // O2_FRAMEWORK_DDSCONFIGHELPERS_H_
