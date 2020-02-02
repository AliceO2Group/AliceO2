// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef FRAMEWORK_TIMESLICEINDEX_H
#define FRAMEWORK_TIMESLICEINDEX_H

#include "Framework/DataDescriptorMatcher.h"

#include <cstdint>
#include <tuple>
#include <vector>

namespace o2
{
namespace framework
{

struct TimesliceId {
  static constexpr uint64_t INVALID = -1;
  size_t value;
  static bool isValid(TimesliceId const& timeslice);
};

struct TimesliceSlot {
  static constexpr uint64_t INVALID = -1;
  static constexpr uint64_t ANY = -2;
  size_t index;
  static bool isValid(TimesliceSlot const& slot);
  bool operator==(const TimesliceSlot that) const;
  bool operator!=(const TimesliceSlot that) const;
};

/// This class keeps the information relative to a given slot in the cache, in
/// particular which variables are associated to it (and indirectly which
/// timeslice which is always mapped to the variable 0) and wether we should
/// consider the slot dirty (e.g. up for reprocessing by the completion
/// policy).  It also provides helpers to decide which slot to reuse in case we
/// are under overload.
class TimesliceIndex
{
 public:
  /// The outcome for the processing of a given timeslot
  enum struct ActionTaken {
    ReplaceUnused,   /// An unused / invalid slot is used to hold the new context
    ReplaceObsolete, /// An obsolete slot is used to hold the new context and the old one is dropped
    DropInvalid,     /// An invalid context is not inserted in the index and dropped
    DropObsolete     /// An obsolete context is not inserted in the index and dropped
  };

  inline void resize(size_t s);
  inline size_t size() const;
  inline bool isValid(TimesliceSlot const& slot) const;
  inline bool isDirty(TimesliceSlot const& slot) const;
  inline void markAsDirty(TimesliceSlot slot, bool value);
  inline void markAsInvalid(TimesliceSlot slot);
  /// Publish a slot to be sent via metrics.
  inline void publishSlot(TimesliceSlot slot);
  /// Associated the @a timestamp to the given @a slot. Notice that
  /// now the information about the timeslot to associate needs to be
  /// determined outside the TimesliceIndex.
  inline void associate(TimesliceId timestamp, TimesliceSlot slot);
  /// Give a slot, @return the TimesliceId (i.e. the variable at position 0)
  /// associated to it. Notice that there is no unique way to
  /// determine the other way around anymore, because a single TimesliceId
  /// could correspond to different slots once we implement wildcards
  /// (e.g. if we ask for InputSpec{"*", "CLUSTERS"}).
  inline TimesliceId getTimesliceForSlot(TimesliceSlot slot) const;
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
  /// @return the action taken on insertion, which can be used for bookkeeping
  ///         of the messages.
  inline std::tuple<ActionTaken, TimesliceSlot> replaceLRUWith(data_matcher::VariableContext& newContext);

 private:
  /// @return the oldest slot possible so that we can eventually override it.
  /// This is the timeslices for all the in flight parts.
  inline TimesliceSlot findOldestSlot() const;

  /// The variables for each cacheline.
  std::vector<data_matcher::VariableContext> mVariables;

  /// The variables that are to be sent
  std::vector<data_matcher::VariableContext> mPublishedVariables;

  /// This keeps track whether or not something was relayed
  /// since last time we called getReadyToProcess()
  std::vector<bool> mDirty;
};

} // namespace framework
} // namespace o2

#include "TimesliceIndex.inc"
#endif // FRAMEWORK_TIMESLICEINDEX_H
