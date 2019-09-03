// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_DATARELAYER_H
#define FRAMEWORK_DATARELAYER_H

#include "Framework/InputRoute.h"
#include "Framework/DataDescriptorMatcher.h"
#include "Framework/ForwardRoute.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/PartRef.h"
#include "Framework/TimesliceIndex.h"

#include <cstddef>
#include <vector>

class FairMQMessage;

namespace o2
{
namespace monitoring
{
class Monitoring;
}
namespace framework
{

/// Helper struct to hold statistics about the relaying process.
struct DataRelayerStats {
  uint64_t malformedInputs = 0;         /// Malformed inputs which the user attempted to process
  uint64_t droppedComputations = 0;     /// How many computations have been dropped because one of the inputs was late
  uint64_t droppedIncomingMessages = 0; /// How many messages have been dropped (not relayed) because they were late
  uint64_t relayedMessages = 0;         /// How many messages have been successfully relayed
};

class DataRelayer
{
 public:
  enum RelayChoice {
    WillRelay,
    WillNotRelay
  };

  struct RecordAction {
    TimesliceSlot slot;
    CompletionPolicy::CompletionOp op;
  };

  DataRelayer(CompletionPolicy const&,
              std::vector<InputRoute> const&,
              std::vector<ForwardRoute> const&,
              monitoring::Monitoring&,
              TimesliceIndex&);

  /// This invokes the appropriate `InputRoute::danglingChecker` on every
  /// entry in the cache and if it returns true, it creates a new
  /// cache entry by invoking the associated `InputRoute::expirationHandler`.
  void processDanglingInputs(std::vector<ExpirationHandler> const&,
                             ServiceRegistry& context);

  /// This is used to ask for relaying a given (header,payload) pair.
  /// Notice that we expect that the header is an O2 Header Stack
  /// with a DataProcessingHeader inside so that we can assess time.
  RelayChoice relay(std::unique_ptr<FairMQMessage>&& header,
                    std::unique_ptr<FairMQMessage>&& payload);

  /// @returns the actions ready to be performed.
  std::vector<RecordAction> getReadyToProcess();

  /// Returns an input registry associated to the given timeslice and gives
  /// ownership to the caller. This is because once the inputs are out of the
  /// DataRelayer they need to be deleted once the processing is concluded.
  std::vector<std::unique_ptr<FairMQMessage>>
    getInputsForTimeslice(TimesliceSlot id);

  /// Returns the index of the arguments which have to be forwarded to
  /// the next processor
  const std::vector<int>& forwardingMask();

  /// Returns how many timeslices we can handle in parallel
  size_t getParallelTimeslices() const;

  /// Tune the maximum number of in flight timeslices this can handle.
  void setPipelineLength(size_t s);

  /// @return the current stats about the data relaying process
  DataRelayerStats const& getStats() const;

  /// Send metrics with the VariableContext information
  void sendContextState();

 private:
  std::vector<InputRoute> const& mInputRoutes;
  std::vector<ForwardRoute> const& mForwardRoutes;
  monitoring::Monitoring& mMetrics;

  /// This is the actual cache of all the parts in flight.
  /// Notice that we store them as a NxM sized vector, where
  /// N is the maximum number of inflight timeslices, while
  /// M is the number of inputs which are requested.
  std::vector<PartRef> mCache;

  /// This is the index which maps a given timestamp to the associated
  /// cacheline.
  TimesliceIndex& mTimesliceIndex;

  std::vector<bool> mForwardingMask;
  CompletionPolicy mCompletionPolicy;
  std::vector<size_t> mDistinctRoutesIndex;
  std::vector<data_matcher::DataDescriptorMatcher> mInputMatchers;
  std::vector<data_matcher::VariableContext> mVariableContextes;
  std::vector<int> mCachedStateMetrics;

  static std::vector<std::string> sMetricsNames;
  static std::vector<std::string> sVariablesMetricsNames;
  static std::vector<std::string> sQueriesMetricsNames;

  DataRelayerStats mStats;
};

} // namespace framework
} // namespace o2

#endif
