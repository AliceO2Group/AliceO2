// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_DRIVERCONTROL_H
#define FRAMEWORK_DRIVERCONTROL_H

#include <vector>
#include "DriverInfo.h"

namespace o2
{
namespace framework
{

/// These are the possible states for the driver controller
/// and determine what should happen of state machine transitions.
enum struct DriverControlState {
  STEP,
  PLAY,
  PAUSE
};

/// Information about the driver process (i.e.  / the one which calculates the
/// topology and actually spawns the devices )
struct DriverControl {
  std::vector<DriverState> forcedTransitions;
  DriverControlState state;
};

} // namespace framework
} // namespace o2

#endif
