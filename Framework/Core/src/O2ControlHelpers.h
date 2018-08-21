// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_O2CONTROLHELPERS_H
#define FRAMEWORK_O2CONTROLHELPERS_H

#include "Framework/DeviceSpec.h"
#include "Framework/DeviceExecution.h"
#include <vector>
#include <iosfwd>

namespace o2
{
namespace framework
{

void dumpDeviceSpec2O2Control(std::ostream& out,
                              std::vector<DeviceSpec> const& specs,
                              std::vector<DeviceExecution> const& executions);

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_O2CONTROLHELPERS_H
