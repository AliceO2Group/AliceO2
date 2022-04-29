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
#ifndef O2_FRAMEWORK_COMPLETIONPOLICY_H_
#define O2_FRAMEWORK_COMPLETIONPOLICY_H_

#include "Framework/DataRef.h"
#include "Framework/InputSpec.h"

#include <functional>
#include <string>
#include <vector>

namespace o2::framework
{

struct DeviceSpec;
struct InputRecord;
struct InputSpan;
class DataRelayer;

/// Policy to describe what to do for a matching DeviceSpec
/// whenever a new message arrives. The InputRecord being passed to
/// Callback is the partial input record received that far.
struct CompletionPolicy {
  /// Action to take with the InputRecord:
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
    Discard,
    /// ConsumeExisting: run the ProcessCallback on the InputRecord. After
    /// we are done, the processed payloads will be deallocated (but
    /// not the headers) while we wait for the record to be actually fully
    /// Consumed.
    ConsumeExisting,
    /// ConsumeAndRescan: run the ProcessCallback on the InputRecord.
    /// Messages which have to be forwarded downstream will be forwarded.
    /// Invalidate the TimesliceIndex so that all the entries are checked
    /// again.
    ConsumeAndRescan
  };

  using Matcher = std::function<bool(DeviceSpec const& device)>;
  using InputSetElement = DataRef;
  using Callback = std::function<CompletionOp(InputSpan const&)>;
  using CallbackFull = std::function<CompletionOp(InputSpan const&, std::vector<InputSpec> const&)>;
  using CallbackConfigureRelayer = std::function<void(DataRelayer&)>;

  /// Constructor
  CompletionPolicy()
    : name(), matcher(), callback() {}
  /// Constructor for emplace_back
  CompletionPolicy(std::string _name, Matcher _matcher, Callback _callback)
    : name(_name), matcher(_matcher), callback(_callback), callbackFull{nullptr} {}
  CompletionPolicy(std::string _name, Matcher _matcher, CallbackFull _callback)
    : name(_name), matcher(_matcher), callback(nullptr), callbackFull{_callback} {}

  /// Name of the policy itself.
  std::string name = "";
  /// Callback to be used to understand if the policy should apply
  /// to the given device.
  Matcher matcher = nullptr;
  /// Actual policy which decides what to do with a partial InputRecord.
  Callback callback = nullptr;
  /// Actual policy which decides what to do with a partial InputRecord, extended version
  CallbackFull callbackFull = nullptr;
  /// A callback which allows you to configure the behavior of the data relayer associated
  /// to the matching device.
  CallbackConfigureRelayer configureRelayer = nullptr;

  /// Helper to create the default configuration.
  static std::vector<CompletionPolicy> createDefaultPolicies();
};

std::ostream& operator<<(std::ostream& oss, CompletionPolicy::CompletionOp const& val);

} // namespace o2::framework

#endif // O2_FRAMEWORK_COMPLETIONPOLICY_H_
