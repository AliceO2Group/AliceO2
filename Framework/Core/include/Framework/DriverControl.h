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

#include <functional>
#include <vector>

#include "Framework/DriverInfo.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DeviceExecution.h"

namespace o2
{
namespace framework
{

/// These are the possible states for the driver controller
/// and determine what should happen of state machine transitions.
enum struct DriverControlState { STEP,
                                 PLAY,
                                 PAUSE };

/// Controller for the driver process (i.e. / the one which calculates the
/// topology and actually spawns the devices ). Any operation to be done by
/// the driver process should be recorded in an instance of this, so that the
/// changes can be applied at the correct moment / state.
struct DriverControl {
  using Callback = std::function<void(std::vector<DataProcessorSpec> const& workflow,
                                      std::vector<DeviceSpec> const&,
                                      std::vector<DeviceExecution> const&,
                                      std::vector<DataProcessorInfo>&)>;
  /// States to be added to the stack on next iteration
  /// of the state machine processing.
  std::vector<DriverState> forcedTransitions;
  /// Current state of the state machine player.
  DriverControlState state;
  /// Callbacks to be performed by the driver next time it
  /// goes in the "PERFORM_CALLBACK" state.
  std::vector<Callback> callbacks;
  bool defaultQuiet;
  bool defaultStopped;
};

} // namespace framework
} // namespace o2

#endif
