// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#include <cstddef>
#include <mutex>
#include <vector>

class FairMQMessage;

namespace o2::monitoring
{
class Monitoring;
}

namespace o2::framework
{

/// Helper struct to hold statistics about the relaying process.
struct DataRelayerStats {
  uint64_t malformedInputs = 0;         /// Malformed inputs which the user attempted to process
  uint64_t droppedComputations = 0;     /// How many computations have been dropped because one of the inputs was late
  uint64_t droppedIncomingMessages = 0; /// How many messages have been dropped (not relayed) because they were late
  uint64_t relayedMessages = 0;         /// How many messages have been successfully relayed
};

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
  enum RelayChoice {
    WillRelay,     /// Ownership of the data has been taken
    Invalid,       /// The incoming data was not valid and has been dropped
    Backpressured, /// The incoming data was not relayed, because we are backpressured
    Dropped        /// The incoming data was not relayed and has been dropped
  };

  struct ActivityStats {
    int newSlots = 0;
    int expiredSlots = 0;
  };

  struct RecordAction {
    TimesliceSlot slot;
    CompletionPolicy::CompletionOp op;
  };

  DataRelayer(CompletionPolicy const&,
              std::vector<InputRoute> const& routes,
              monitoring::Monitoring&,
              TimesliceIndex&);

  /// This invokes the appropriate `InputRoute::danglingChecker` on every
  /// entry in the cache and if it returns true, it creates a new
  /// cache entry by invoking the associated `InputRoute::expirationHandler`.
  /// @a createNew true if the dangling inputs are allowed to create new slots.
  /// @return true if there were expirations, false if not.
  ActivityStats processDanglingInputs(std::vector<ExpirationHandler> const&,
                                      ServiceRegistry& context, bool createNew);

  /// This is to relay a whole set of FairMQMessages, all which are part
  /// of the same set of split parts.
  /// @a firstHeader is the first message of such set
  /// @a restOfParts is a pointer to the rest of the messages
  /// @a restSize is how many messages are there in restOfParts
  /// is the header which is common across all subsequent elements.
  /// Notice that we expect that the header is an O2 Header Stack
  RelayChoice relay(std::unique_ptr<FairMQMessage>&& firstHeader,
                    std::unique_ptr<FairMQMessage>* restOfParts,
                    size_t restSize);

  /// This is used to ask for relaying a given (header,payload) pair.
  /// Notice that we expect that the header is an O2 Header Stack
  /// with a DataProcessingHeader inside so that we can assess time.
  RelayChoice relay(std::unique_ptr<FairMQMessage>&& header,
                    std::unique_ptr<FairMQMessage>&& payload);

  /// @returns the actions ready to be performed.
  void getReadyToProcess(std::vector<RecordAction>& completed);

  /// Returns an input registry associated to the given timeslice and gives
  /// ownership to the caller. This is because once the inputs are out of the
  /// DataRelayer they need to be deleted once the processing is concluded.
  std::vector<MessageSet> getInputsForTimeslice(TimesliceSlot id);

  /// Returns how many timeslices we can handle in parallel
  size_t getParallelTimeslices() const;

  /// Tune the maximum number of in flight timeslices this can handle.
  void setPipelineLength(size_t s);

  /// @return the current stats about the data relaying process
  DataRelayerStats const& getStats() const;

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
  /// Get the firstTFOrbit associate to a given slot.
  uint32_t getFirstTFOrbitForSlot(TimesliceSlot slot);
  /// Get the firstTFCounter associate to a given slot.
  uint32_t getFirstTFCounterForSlot(TimesliceSlot slot);
  /// Remove all pending messages
  void clear();

 private:
  monitoring::Monitoring& mMetrics;

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
  std::vector<data_matcher::DataDescriptorMatcher> mInputMatchers;
  std::vector<data_matcher::VariableContext> mVariableContextes;
  std::vector<CacheEntryStatus> mCachedStateMetrics;

  static std::vector<std::string> sMetricsNames;
  static std::vector<std::string> sVariablesMetricsNames;
  static std::vector<std::string> sQueriesMetricsNames;

  DataRelayerStats mStats;
  TracyLockableN(std::recursive_mutex, mMutex, "data relayer mutex");
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_DATARELAYER_H_
