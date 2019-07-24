// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_DISPATCHPOLICY_H
#define FRAMEWORK_DISPATCHPOLICY_H

#include "Framework/PartRef.h"

#include <functional>
#include <string>
#include <vector>

namespace o2
{
namespace framework
{

struct DeviceSpec;
struct Output;

/// Policy to describe when to dispatch objects
/// As for now we describe this policy per device, however it can be extended
/// to match on specific outputs of the device.
struct DispatchPolicy {
  /// Action to take whenever an object in the output gets ready:
  ///
  enum struct DispatchOp {
    /// Dispatch objects when the calculation ends, this means the devices will
    /// send messages from all contextes in one bulk after computation
    AfterComputation,
    /// Dispatch the object when it becomes ready, i.e. when it goes out of the
    /// scope of the user code and no changes to the object are possible
    WhenReady,
  };

  using DeviceMatcher = std::function<bool(DeviceSpec const& device)>;
  // OutputMatcher can be a later extension, but not expected to be of high priority
  using OutputMatcher = std::function<bool(Output const&)>;

  /// Name of the policy itself.
  std::string name;
  /// Callback to be used to understand if the policy should apply
  /// to the given device.
  DeviceMatcher deviceMatcher;
  /// the action to be used for matched devices
  DispatchOp action = DispatchOp::AfterComputation;

  /// Helper to create the default configuration.
  static std::vector<DispatchPolicy> createDefaultPolicies();
};

std::ostream& operator<<(std::ostream& oss, DispatchPolicy::DispatchOp const& val);

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_DISPATCHPOLICY_H
