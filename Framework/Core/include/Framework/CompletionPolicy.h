// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_COMPLETIONPOLICY_H
#define FRAMEWORK_COMPLETIONPOLICY_H

#include "Framework/PartRef.h"

#include <functional>
#include <string>
#include <vector>

#include <gsl/span>

namespace o2
{
namespace framework
{

struct DeviceSpec;
struct InputRecord;

/// Policy to describe what to do for a matching DeviceSpec
/// whenever a new message arrives. The InputRecord being passed to
/// Callback is the partial input record received that far.
struct CompletionPolicy {
  /// Action to take with the InputRecord:
  ///
  enum struct CompletionOp {
    /// Run the ProcessCallback on the InputRecord, consuming
    /// its contents. Messages which have to be forwarded downstream
    /// will be forwarded.
    Consume,
    /// Process: run the ProcessCallback on the InputRecord. Its contents /
    /// will be kept but they will not be forwarded downstream.
    Process,
    /// Do not run the ProcessCallback. Its contents will be kept
    /// and they will be proposed again when a new message for the same
    /// record arrives. They will not be forwarded downstream.
    Wait,
    /// Do not run the ProcessCallback. Contents of the record will
    /// be forwarded to the next consumer, if any.
    Discard
  };

  using Matcher = std::function<bool(DeviceSpec const& device)>;
  using Callback = std::function<CompletionOp(gsl::span<PartRef const> const&)>;

  /// Name of the policy itself.
  std::string name;
  /// Callback to be used to understand if the policy should apply
  /// to the given device.
  Matcher matcher;
  /// Actual policy which decides what to do with a partial InputRecord.
  Callback callback;

  /// Helper to create the default configuration.
  static std::vector<CompletionPolicy> createDefaultPolicies();
};

std::ostream& operator<<(std::ostream& oss, CompletionPolicy::CompletionOp const& val);

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_COMPLETIONPOLICY_H
