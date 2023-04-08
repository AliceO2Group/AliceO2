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
#ifndef O2_FRAMEWORK_DATARELAYER_H_
#define O2_FRAMEWORK_DATARELAYER_H_

#include "Framework/RootSerializationSupport.h"
#include "Framework/InputRoute.h"
#include "Framework/DataDescriptorMatcher.h"
#include "Framework/ForwardRoute.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/MessageSet.h"
#include "Framework/TimesliceIndex.h"
#include "Framework/Tracing.h"
#include "Framework/TimesliceSlot.h"
#include "Framework/ServiceRegistryRef.h"

#include <cstddef>
#include <mutex>
#include <vector>
#include <functional>

#include <fairmq/FwdDecls.h>

namespace o2::monitoring
{
class Monitoring;
}

namespace o2::framework
{

enum struct CacheEntryStatus : int {
  EMPTY,
  PENDING,
  RUNNING,
  DONE
};

class DataRelayer
{
 public:
  /// DataRelayer is thread safe because we have a lock around
  /// each method and there is no particular order in which
  /// methods need to be called.
  constexpr static ServiceKind service_kind = ServiceKind::Global;
  /// This represents what the DataRelayer did when
  /// inserting a set of messages in the cache.
  struct RelayChoice {
    enum struct Type {
      WillRelay,     /// Ownership of the data has been taken
      Invalid,       /// The incoming data was not valid and has been dropped
      Backpressured, /// The incoming data was not relayed, because we are backpressured
      Dropped        /// The incoming data was not relayed and has been dropped
    };
    /// What was the outcome of the relay operation.
    Type type;
    // The timeslice affected by the given operation.
    TimesliceId timeslice;
  };

  struct ActivityStats {
    int newSlots = 0;
    int expiredSlots = 0;
  };

  struct PruneOp {
    TimesliceSlot slot = {-1ULL};
  };

  struct RecordAction {
    TimesliceSlot slot;
    TimesliceId timeslice;
    CompletionPolicy::CompletionOp op;
  };

  DataRelayer(CompletionPolicy const&,
              std::vector<InputRoute> const& routes,
              TimesliceIndex&,
              ServiceRegistryRef);

  /// This invokes the appropriate `InputRoute::danglingChecker` on every
  /// entry in the cache and if it returns true, it creates a new
  /// cache entry by invoking the associated `InputRoute::expirationHandler`.
  /// @a createNew true if the dangling inputs are allowed to create new slots.
  /// @return true if there were expirations, false if not.
  ActivityStats processDanglingInputs(std::vector<ExpirationHandler> const&,
                                      ServiceRegistryRef context, bool createNew);

  using OnDropCallback = std::function<void(TimesliceSlot, std::vector<MessageSet>&, TimesliceIndex::OldestOutputInfo info)>;

  /// Prune all the pending entries in the cache.
  void prunePending(OnDropCallback);
  /// Prune the cache for a given slot
  void pruneCache(TimesliceSlot slot, OnDropCallback onDrop = nullptr);

  /// This is to relay a whole set of fair::mq::Messages, all which are part
  /// of the same set of split parts.
  /// @a rawHeader raw header pointer
  /// @a messages pointer to array of messages
  /// @a nMessages size of the array
  /// @a nPayloads number of payploads in the message sequence, default is 1
  ///              which is the standard header-payload message pair, in this
  ///              case nMessages / 2 pairs will be inserted and considered
  ///              separate parts
  /// @a onDrop function to be called if an message is dropped
  /// Notice that we expect that the header is an O2 Header Stack
  RelayChoice relay(void const* rawHeader,
                    std::unique_ptr<fair::mq::Message>* messages,
                    size_t nMessages,
                    size_t nPayloads = 1,
                    OnDropCallback onDrop = nullptr);

  /// This is to set the oldest possible @a timeslice this relayer can
  /// possibly see on an input channel @a channel.
  void setOldestPossibleInput(TimesliceId timeslice, ChannelIndex channel);

  /// This is to retrieve the oldest possible @a timeslice this relayer can
  /// possibly have in output.
  [[nodiscard]] TimesliceIndex::OldestOutputInfo getOldestPossibleOutput() const;

  /// @returns the actions ready to be performed.
  void getReadyToProcess(std::vector<RecordAction>& completed);

  /// Returns an input registry associated to the given timeslice and gives
  /// ownership to the caller. This is because once the inputs are out of the
  /// DataRelayer they need to be deleted once the processing is concluded.
  std::vector<MessageSet> consumeAllInputsForTimeslice(TimesliceSlot id);
  std::vector<MessageSet> consumeExistingInputsForTimeslice(TimesliceSlot id);

  /// Returns how many timeslices we can handle in parallel
  [[nodiscard]] size_t getParallelTimeslices() const;

  /// Tune the maximum number of in flight timeslices this can handle.
  void setPipelineLength(size_t s);

  /// Send metrics with the VariableContext information
  void sendContextState();
  void publishMetrics();

  /// Get timeslice associated to a given slot.
  /// Notice how this avoids exposing the timesliceIndex directly
  /// so that we can mutex on it.
  TimesliceId getTimesliceForSlot(TimesliceSlot slot);

  /// Mark a given slot as done so that the GUI
  /// can reflect that.
  void updateCacheStatus(TimesliceSlot slot, CacheEntryStatus oldStatus, CacheEntryStatus newStatus);
  /// Get the firstTForbit associate to a given slot.
  uint32_t getFirstTFOrbitForSlot(TimesliceSlot slot);
  /// Get the firstTFCounter associate to a given slot.
  uint32_t getFirstTFCounterForSlot(TimesliceSlot slot);
  /// Get the runNumber associated to a given slot
  uint32_t getRunNumberForSlot(TimesliceSlot slot);
  /// Get the creation time associated to a given slot
  uint64_t getCreationTimeForSlot(TimesliceSlot slot);
  /// Remove all pending messages
  void clear();

  /// get max number of timeslices in the queue
  static unsigned int getPipelineLength();

  /// Rescan the whole data to see if there is anything new we should do,
  /// e.g. as consequnce of an OOB event.
  void rescan() { mTimesliceIndex.rescan(); };

  [[nodiscard]] size_t getCacheSize() const { return mCache.size(); }
  [[nodiscard]] size_t getNumberOfTimeslices() const { return mTimesliceIndex.size(); }
  [[nodiscard]] size_t getNumberOfUniqueInputs() const { return mDistinctRoutesIndex.size(); }

 private:
  ServiceRegistryRef mContext;

  /// This is the actual cache of all the parts in flight.
  /// Notice that we store them as a NxM sized vector, where
  /// N is the maximum number of inflight timeslices, while
  /// M is the number of inputs which are requested.
  std::vector<MessageSet> mCache;

  /// This is the index which maps a given timestamp to the associated
  /// cacheline.
  TimesliceIndex& mTimesliceIndex;

  CompletionPolicy mCompletionPolicy;
  std::vector<size_t> mDistinctRoutesIndex;
  std::vector<InputSpec> mInputs;
  std::vector<data_matcher::DataDescriptorMatcher> mInputMatchers;
  std::vector<data_matcher::VariableContext> mVariableContextes;
  std::vector<CacheEntryStatus> mCachedStateMetrics;
  std::vector<PruneOp> mPruneOps;
  size_t mMaxLanes;

  static std::vector<std::string> sMetricsNames;
  static std::vector<std::string> sVariablesMetricsNames;
  static std::vector<std::string> sQueriesMetricsNames;

  TracyLockableN(std::recursive_mutex, mMutex, "data relayer mutex");
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_DATARELAYER_H_
