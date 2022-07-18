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

#ifndef O2_FRAMEWORK_TIMESLICEINDEX_H_
#define O2_FRAMEWORK_TIMESLICEINDEX_H_

#include "Framework/DataDescriptorMatcher.h"
#include "Framework/CompilerBuiltins.h"
#include "Framework/RoutingIndices.h"
#include "Framework/ServiceHandle.h"
#include "Framework/TimesliceSlot.h"
#include "Framework/ChannelInfo.h"

#include <cstdint>
#include <vector>
#include <algorithm>

namespace o2::framework
{


/// This class keeps the information relative to a given slot in the cache, in
/// particular which variables are associated to it (and indirectly which
/// timeslice which is always mapped to the variable 0) and wether we should
/// consider the slot dirty (e.g. up for reprocessing by the completion
/// policy).  It also provides helpers to decide which slot to reuse in case we
/// are under overload.
class TimesliceIndex
{
 public:
  /// TimesliceIndex is threadsafe because it's accessed only by the
  /// DataRelayer.
  constexpr static ServiceKind service_kind = ServiceKind::Global;

  /// What to do when there is backpressure
  enum struct BackpressureOp {
    Wait,        // Do nothing and wait for the oldest slot to complete
    DropAncient, // Drop the message with the least recent timestamp
    DropRecent   // Drop the message with the most recent timestamp
  };

  /// The outcome for the processing of a given timeslot
  enum struct ActionTaken {
    ReplaceUnused,   /// An unused / invalid slot is used to hold the new context
    ReplaceObsolete, /// An obsolete slot is used to hold the new context and the old one is dropped
    Wait,            /// We wait for the oldest slot to complete.
    DropInvalid,     /// An invalid context is not inserted in the index and dropped
    DropObsolete     /// An obsolete context is not inserted in the index and dropped
  };

  /// The result of the replacing the LRU slot with a new one
  struct ActionSpec {
    /// The action taken
    ActionTaken action;
    /// The slot which was replaced
    TimesliceSlot slot;
  };

  /// Information about what is the oldest possible input and from which
  /// channel.
  struct OldestInputInfo {
    /// The timeslice of the the oldest input. Notice that we use zero,
    /// so any input which did not provide information will keep this
    /// to 0, effectively making the the dropping mechanism not working.
    TimesliceId timeslice = {0};
    /// The actual channel id of the oldest input.
    ChannelIndex channel = {ChannelIndex::INVALID};
  };

  struct OldestOutputInfo {
    TimesliceId timeslice = {0};
    ChannelIndex channel = {ChannelIndex::INVALID};
    TimesliceSlot slot = {(size_t)-1};
  };

  TimesliceIndex(size_t maxLanes, std::vector<InputChannelInfo>& channels);
  void resize(size_t s);
  [[nodiscard]] inline size_t size() const;
  [[nodiscard]] inline bool isValid(TimesliceSlot const& slot) const;
  [[nodiscard]] inline bool isDirty(TimesliceSlot const& slot) const;
  inline void markAsDirty(TimesliceSlot slot, bool value);
  inline void markAsInvalid(TimesliceSlot slot);
  /// Mark all the cachelines as invalid, e.g. due to an out of band event
  inline void rescan();
  /// Publish a slot to be sent via metrics.
  inline void publishSlot(TimesliceSlot slot);

  /// Associated the @a timestamp to the given @a slot. Notice that
  /// now the information about the timeslot to associate needs to be
  /// determined outside the TimesliceIndex.
  void associate(TimesliceId timestamp, TimesliceSlot slot);

  /// Given a slot, @return the VariableContext associated to it.
  /// This effectively means that the TimesliceIndex is now owner of the
  /// VariableContext.
  inline data_matcher::VariableContext& getVariablesForSlot(TimesliceSlot slot);

  /// Given a slot, @return the VariableContext associated to it.
  /// This effectively means that the TimesliceIndex is now owner of the
  /// VariableContext.
  inline data_matcher::VariableContext& getPublishedVariablesForSlot(TimesliceSlot slot);

  /// Find the LRU entry in the cache and replace it with @a newContext
  /// @a slot is filled with the slot used to hold the context, if applicable.
  /// @a timestamp must be provided to select the correct lane, in case of pipelining
  /// @return the action taken on insertion, which can be used for bookkeeping
  ///         of the messages.
  std::tuple<ActionTaken, TimesliceSlot> replaceLRUWith(data_matcher::VariableContext& newContext, TimesliceId timestamp);

  /// Set the older possible input per channel
  /// @return the updated oldest possible input. Notice that this should be
  /// used with the validateSlots below to actually discard the slots.
  [[nodiscard]] OldestInputInfo setOldestPossibleInput(TimesliceId timeslice, ChannelIndex channel);
  /// Validate that the slot @a slot is still not older than @a currentOldest
  /// @return true if the slot was not invalidated by the new currentOldest
  bool validateSlot(TimesliceSlot slot, TimesliceId currentOldest);

  /// Find the lowest value for the timeslices in this instance.
  /// This is the minimum between all the per channel oldest possible timeslices
  /// and the oldest possible timeslice in-fly which is still dirty.
  [[nodiscard]] OldestInputInfo getOldestPossibleInput() const;
  [[nodiscard]] OldestOutputInfo getOldestPossibleOutput() const;
  OldestOutputInfo updateOldestPossibleOutput();
  InputChannelInfo const& getChannelInfo(ChannelIndex channel) const;

 private:
  /// @return the oldest slot possible so that we can eventually override it.
  /// This is the timeslices for all the in flight parts.
  inline TimesliceSlot findOldestSlot(TimesliceId) const;

  /// The variables for each cacheline.
  std::vector<data_matcher::VariableContext> mVariables;

  /// The variables that are to be sent
  std::vector<data_matcher::VariableContext> mPublishedVariables;

  /// This keeps track whether or not something was relayed
  /// since last time we called getReadyToProcess()
  std::vector<bool> mDirty;

  /// This is the oldest possible timeslice for any given channel
  /// The cardinality of this vector is the number of input channels
  std::vector<InputChannelInfo>& mChannels;
  /// This is the oldest possible timeslice for this index.
  /// By default we use -1, which means that we don't have any.
  OldestInputInfo mOldestPossibleInput = {};
  OldestOutputInfo mOldestPossibleOutput = {};

  /// What to do in case of backpressure
  BackpressureOp mBackpressurePolicy = BackpressureOp::Wait;
  /// The maximum number of lanes for this timeslice index
  size_t mMaxLanes;
};

} // namespace o2::framework

#include "TimesliceIndex.inc"
#endif // FRAMEWORK_TIMESLICEINDEX_H
