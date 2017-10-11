// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_STF_SENDER_OUTPUT_H_
#define ALICEO2_STF_SENDER_OUTPUT_H_

#include "Common/SubTimeFrameDataModel.h"
#include "Common/ConcurrentQueue.h"

#include <vector>
#include <map>
#include <thread>

namespace o2
{
namespace DataDistribution
{

class StfSenderDevice;

class StfSenderOutput
{
 public:
  StfSenderOutput() = delete;
  StfSenderOutput(StfSenderDevice& pStfSenderDev)
    : mDevice(pStfSenderDev)
  {
  }

  void start(std::uint32_t pCnt);
  void stop();

  bool running() const;

  void StfSchedulerThread();
  void DataHandlerThread(const std::uint32_t pEpnIdx);

  void setMaxConcurrentSends(std::int64_t pNumSendSlots)
  {
    mNumSendSlots = pNumSendSlots <= 0 ? std::numeric_limits<std::uint64_t>::max() : pNumSendSlots;
  }

  void PushStf(const std::uint32_t pEpnIdx, std::unique_ptr<SubTimeFrame>&& pStf)
  {
    assert(pEpnIdx < mStfQueues.size());
    mStfQueues[pEpnIdx].push(std::move(pStf));
  }

 private:
  /// Ref to the main SubTimeBuilder O2 device
  StfSenderDevice& mDevice;

  /// Scheduler threads
  std::thread mSchedulerThread;

  /// Threads for output channels (to EPNs)
  std::vector<std::thread> mOutputThreads;

  /// Outstanding queues of STFs per EPN
  std::vector<ConcurrentFifo<std::unique_ptr<SubTimeFrame>>> mStfQueues;

  /// Number of Stfs on the network
  mutable std::mutex mSendSlotLock;
  std::condition_variable mSendSlotCond;
  std::uint64_t mNumSendSlots = std::numeric_limits<std::uint64_t>::max();
};
}
} /* namespace o2::DataDistribution */

#endif /* ALICEO2_STF_SENDER_OUTPUT_H_ */
