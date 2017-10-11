// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "SubTimeFrameSender/SubTimeFrameSenderOutput.h"
#include "SubTimeFrameSender/SubTimeFrameSenderDevice.h"

#include "Common/SubTimeFrameDataModel.h"
#include "Common/SubTimeFrameVisitors.h"

#include <O2Device/O2Device.h>
#include <FairMQLogger.h>

#include <condition_variable>
#include <stdexcept>

namespace o2
{
namespace DataDistribution
{

using namespace std::chrono_literals;

void StfSenderOutput::start(std::uint32_t pEpnCnt)
{
  // create scheduler thread
  mSchedulerThread = std::thread(&StfSenderOutput::StfSchedulerThread, this);

  if (mDevice.standalone()) {
    return;
  }

  if (!mDevice.CheckCurrentState(StfSenderDevice::RUNNING)) {
    LOG(WARN) << "Not creating interface threads. StfSenderDevice is not running.";
    return;
  }

  // create thread queues
  mStfQueues.clear();
  mStfQueues = std::move(std::vector<ConcurrentFifo<std::unique_ptr<SubTimeFrame>>>(pEpnCnt));

  assert(mOutputThreads.size() == 0);

  for (auto tid = 0; tid < pEpnCnt; tid++) {
    // tid matches output channel index (EPN idx)
    mOutputThreads.emplace_back(
      std::thread(&StfSenderOutput::DataHandlerThread, this, tid));
  }
}

void StfSenderOutput::stop()
{
  // stop the scheduler
  if (mSchedulerThread.joinable()) {
    mSchedulerThread.join();
  }

  if (mDevice.standalone()) {
    return;
  }

  // stop all queues
  for (auto& lIdQueue : mStfQueues) {
    lIdQueue.stop();
  }

  // release cond variable
  mSendSlotCond.notify_all();

  // wait for threads to exit
  for (auto& lIdThread : mOutputThreads) {
    if (lIdThread.joinable()) {
      lIdThread.join();
    }
  }

  mOutputThreads.clear();
  mStfQueues.clear();
}

bool StfSenderOutput::running() const
{
  return mDevice.CheckCurrentState(StfSenderDevice::RUNNING);
}

void StfSenderOutput::StfSchedulerThread()
{
  // queue the Stf to the appropriate EPN queue
  std::unique_ptr<SubTimeFrame> lStf;

  while ((lStf = mDevice.dequeue(eSenderIn)) != nullptr) {
    const TimeFrameIdType lStfId = lStf->header().mId;

    if (mDevice.standalone()) {
      // Do not forward STFs
      continue;
    }

    // Send STF to one of the EPNs (round-robin on STF ID)
    std::unique_lock<std::mutex> lLock(mSendSlotLock);
    while (mNumSendSlots == 0 && running()) {
      // failsafe to check for exit signal
      mSendSlotCond.wait_for(lLock, 1s);
    }

    // check for the exit signal
    if (!running()) {
      break;
    }

    // use up one send slot
    mNumSendSlots--;

    const auto lTargetEpn = lStfId % mDevice.getEpnNodeCount();
    PushStf(lTargetEpn, std::move(lStf));
  }

  LOG(INFO) << "Exiting StfSchedulerOutput...";
}

/// Receiving thread
void StfSenderOutput::DataHandlerThread(const std::uint32_t pEpnIdx)
{
  auto& lOutputChan = mDevice.GetChannel(mDevice.getOutputChannelName(), pEpnIdx);

  LOG(INFO) << "StfSenderOutput[" << pEpnIdx << "]: Starting the thread";

  InterleavedHdrDataSerializer lStfSerializer(lOutputChan);

  while (mDevice.CheckCurrentState(StfSenderDevice::RUNNING)) {
    std::unique_ptr<SubTimeFrame> lStf;

    if (!mStfQueues[pEpnIdx].pop(lStf)) {
      LOG(INFO) << "StfSenderOutput[" << pEpnIdx << "]: STF queue drained. Exiting.";
      break;
    }

    const TimeFrameIdType lStfId = lStf->header().mId;

    try {
      lStfSerializer.serialize(std::move(lStf));
    } catch (std::exception& e) {
      if (mDevice.CheckCurrentState(StfSenderDevice::RUNNING))
        LOG(ERROR) << "StfSenderOutput[" << pEpnIdx << "]: exception on send: " << e.what();
      else
        LOG(INFO) << "StfSenderOutput[" << pEpnIdx << "](NOT RUNNING): exception on send: " << e.what();

      break;
    }

    // free up an slot for sending
    {
      std::unique_lock<std::mutex> lLock(mSendSlotLock);
      mNumSendSlots++;
      lLock.unlock(); // reduce contention on the lock by unlocking before notifyng
      mSendSlotCond.notify_one();
    }
  }

  LOG(INFO) << "Exiting StfSenderOutput[" << pEpnIdx << "]";
}
}
} /* o2::DataDistribution */
