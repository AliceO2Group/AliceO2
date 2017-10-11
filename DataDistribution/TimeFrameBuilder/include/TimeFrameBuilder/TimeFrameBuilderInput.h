// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TF_BUILDER_INPUT_H_
#define ALICEO2_TF_BUILDER_INPUT_H_

#include "Common/SubTimeFrameDataModel.h"
#include "Common/ConcurrentQueue.h"

#include <vector>
#include <map>

#include <condition_variable>
#include <mutex>
#include <thread>

namespace o2
{
namespace DataDistribution
{

class TfBuilderDevice;

class TfBuilderInput
{
 public:
  TfBuilderInput() = delete;
  TfBuilderInput(TfBuilderDevice& pStfBuilderDev, unsigned pOutStage)
    : mDevice(pStfBuilderDev),
      mOutStage(pOutStage)
  {
  }

  void Start(unsigned int pNumFlp);
  void Stop();

  void DataHandlerThread(const std::uint32_t pFlpIndex);
  void StfMergerThread();

 private:
  /// Main TimeFrameBuilder O2 device
  TfBuilderDevice& mDevice;

  /// Threads for input channels (per FLP)
  std::vector<std::thread> mInputThreads;

  /// STF Merger
  std::thread mStfMergerThread;
  std::mutex mStfMergerQueueLock;
  std::condition_variable mStfMergerCondition;
  std::multimap<TimeFrameIdType, std::unique_ptr<SubTimeFrame>> mStfMergeQueue;

  /// Output pipeline stage
  unsigned mOutStage;
};
}
} /* namespace o2::DataDistribution */

#endif /* ALICEO2_TF_BUILDER_INPUT_H_ */
