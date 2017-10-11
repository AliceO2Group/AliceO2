// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TimeFrameBuilder/TimeFrameBuilderInput.h"
#include "TimeFrameBuilder/TimeFrameBuilderDevice.h"

#include "Common/SubTimeFrameDataModel.h"
#include "Common/SubTimeFrameVisitors.h"

#include <O2Device/O2Device.h>
#include <FairMQDevice.h>
#include <FairMQStateMachine.h>
#include <FairMQLogger.h>

#include <condition_variable>
#include <mutex>
#include <thread>
#include <chrono>

namespace o2
{
namespace DataDistribution
{

void TfBuilderInput::Start(unsigned int pNumFlp)
{
  if (!mDevice.CheckCurrentState(TfBuilderDevice::RUNNING)) {
    LOG(WARN) << "Not creating interface threads. StfBuilder is not running.";
    return;
  }

  assert(mInputThreads.size() == 0);

  // start receiver threads
  for (auto tid = 0; tid < pNumFlp; tid++) { // tid matches input channel index
    mInputThreads.emplace_back(std::thread(&TfBuilderInput::DataHandlerThread, this, tid));
  }

  // Start the merger
  {
    std::lock_guard<std::mutex> lQueueLock(mStfMergerQueueLock);
    mStfMergeQueue.clear();

    // start the merger thread
    mStfMergerThread = std::thread(&TfBuilderInput::StfMergerThread, this);
  }
}

void TfBuilderInput::Stop()
{
  assert(!mDevice.CheckCurrentState(TfBuilderDevice::RUNNING));

  // Wait for input threads to stop
  for (auto& lIdThread : mInputThreads) {
    if (lIdThread.joinable())
      lIdThread.join();
  }

  // Make sure the merger stopped
  {
    std::lock_guard<std::mutex> lQueueLock(mStfMergerQueueLock);
    mStfMergeQueue.clear();

    if (mStfMergerThread.joinable()) {
      mStfMergerThread.join();
    }
  }

  LOG(INFO) << "TfBuilderInput: Teardown complete...";
}

/// Receiving thread
void TfBuilderInput::DataHandlerThread(const std::uint32_t pFlpIndex)
{
  // Reference to the input channel
  auto& lInputChan = mDevice.GetChannel(mDevice.getInputChannelName(), pFlpIndex);

  // Deserialization object
  InterleavedHdrDataDeserializer lStfReceiver;

  while (mDevice.CheckCurrentState(TfBuilderDevice::RUNNING)) {
    // receive a STF
    std::unique_ptr<SubTimeFrame> lStf = lStfReceiver.deserialize(lInputChan);
    if (!lStf) {
      if (mDevice.CheckCurrentState(TfBuilderDevice::RUNNING)) {
        LOG(WARN) << "InputThread[" << pFlpIndex << "]: Receive failed";
      } else {
        LOG(INFO) << "InputThread[" << pFlpIndex << "](NOT RUNNING): Receive failed";
      }
      break;
    }

    const TimeFrameIdType lTfId = lStf->header().mId;

    // Push the STF into the merger queue
    std::unique_lock<std::mutex> lQueueLock(mStfMergerQueueLock);
    mStfMergeQueue.emplace(std::make_pair(lTfId, std::move(lStf)));

    // Notify the Merger if enough inputs are collected
    // NOW:  Merge STFs if exactly |FLP| chunks have been received
    //       or a next TF started arriving (STFs from previous delayed or not
    //       available)
    // TODO: Find out exactly how many STFs is arriving.
    if (mStfMergeQueue.size() >= mDevice.getFlpNodeCount())
      mStfMergerCondition.notify_one();
  }

  LOG(INFO) << "Exiting input thread[" << pFlpIndex << "]...";
}

/// STF->TF Merger thread
void TfBuilderInput::StfMergerThread()
{
  using namespace std::chrono_literals;

  while (mDevice.CheckCurrentState(TfBuilderDevice::RUNNING)) {

    std::unique_lock<std::mutex> lQueueLock(mStfMergerQueueLock);
    mStfMergerCondition.wait_for(lQueueLock, 500ms);

    // check the merge queue for partial TFs first
    const SubTimeFrameIdType lTfId = mStfMergeQueue.begin()->first;

    // Case 1: a full TF can be merged
    if (mStfMergeQueue.count(lTfId) == mDevice.getFlpNodeCount()) {

      auto lStfRange = mStfMergeQueue.equal_range(lTfId);
      assert(std::distance(lStfRange.first, lStfRange.second) == mDevice.getFlpNodeCount());

      auto lStfCount = 1UL; // start from the first element
      std::unique_ptr<SubTimeFrame> lTf = std::move(lStfRange.first->second);

      for (auto lStfIter = std::next(lStfRange.first); lStfIter != lStfRange.second; ++lStfIter) {
        // Add them all up
        lTf->mergeStf(std::move(lStfIter->second));
        lStfCount++;
      }

      if (lStfCount < mDevice.getFlpNodeCount())
        LOG(WARN) << "STF MergerThread: merging incomplete TF[" << lTf->header().mId << "]: contains "
                  << lStfCount << " instead of " << mDevice.getFlpNodeCount() << " SubTimeFrames";

      // remove consumed STFs from the merge queue
      mStfMergeQueue.erase(lStfRange.first, lStfRange.second);

      // Queue out the TF for consumption
      mDevice.queue(mOutStage, std::move(lTf));

    } else if (mStfMergeQueue.size() > (100 * mDevice.getFlpNodeCount())) {
      // FIXME: for now, discard incomplete TFs
      LOG(WARN) << "Unbounded merge queue size: " << mStfMergeQueue.size();

      const auto lDroppedStfs = mStfMergeQueue.count(lTfId);

      mStfMergeQueue.erase(lTfId);

      LOG(WARN) << "Dropping oldest incomplete TF... (" << lDroppedStfs << " STFs)";
    }
  }

  LOG(INFO) << "Exiting STF merger thread...";
}
}
} /* namespace o2::DataDistribution */
