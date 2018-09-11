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
#include "Framework/ForwardRoute.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/PartRef.h"

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

class DataRelayer {
public:
  enum RelayChoice {
    WillRelay,
    WillNotRelay
  };

  struct TimesliceId {
    int64_t value;
  };

  struct RecordAction {
    size_t cacheLineIdx;
    CompletionPolicy::CompletionOp op;
  };

  DataRelayer(CompletionPolicy const&,
              std::vector<InputRoute> const&,
              std::vector<ForwardRoute> const&,
              monitoring::Monitoring&);

  /// This is used to ask for relaying a given (header,payload) pair.
  /// Notice that we expect that the header is an O2 Header Stack
  /// with a DataProcessingHeader inside so that we can assess time.
  RelayChoice relay(std::unique_ptr<FairMQMessage> &&header,
                    std::unique_ptr<FairMQMessage> &&payload);

  /// @returns the actions ready to be performed.
  std::vector<RecordAction> getReadyToProcess();

  /// Returns an input registry associated to the given timeslice and gives
  /// ownership to the caller. This is because once the inputs are out of the
  /// DataRelayer they need to be deleted once the processing is concluded.
  std::vector<std::unique_ptr<FairMQMessage>>
  getInputsForTimeslice(size_t i);

  /// Returns the index of the arguments which have to be forwarded to
  /// the next processor
  const std::vector<int> &forwardingMask();

  /// Returns how many timeslices we can handle in parallel
  size_t getParallelTimeslices() const;

  /// Lookup the timeslice for the given index in the cache
  size_t getTimesliceForCacheline(size_t i) {
    assert(i < mTimeslices.size());
    return mTimeslices[i].value;
  }

  /// Tune the maximum number of in flight timeslices this can handle.
  void setPipelineLength(size_t s);
private:
  std::vector<InputRoute> mInputRoutes;
  std::vector<ForwardRoute> mForwardRoutes;
  monitoring::Monitoring& mMetrics;

  /// This is the actual cache of all the parts in flight. 
  /// Notice that we store them as a NxM sized vector, where
  /// N is the maximum number of inflight timeslices, while
  /// M is the number of inputs which are requested.
  std::vector<PartRef> mCache;

  /// This is the timeslices for all the in flight parts.
  std::vector<TimesliceId> mTimeslices;
  /// This keeps track whether or not something was relayed
  /// since last time we called getReadyToProcess()
  std::vector<bool> mDirty;

  std::vector<bool> mForwardingMask;
  CompletionPolicy mCompletionPolicy;
  std::vector<size_t> mDistinctRoutesIndex;
};

} // namespace framework
} // namespace o2

#endif
