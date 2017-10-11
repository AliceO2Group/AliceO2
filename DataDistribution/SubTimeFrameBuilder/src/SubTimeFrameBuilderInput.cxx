// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "SubTimeFrameBuilder/SubTimeFrameBuilderInput.h"
#include "SubTimeFrameBuilder/SubTimeFrameBuilderDevice.h"
#include "Common/SubTimeFrameVisitors.h"

#include <O2Device/O2Device.h>
#include <FairMQDevice.h>
#include <FairMQStateMachine.h>
#include <FairMQLogger.h>

#include <vector>
#include <queue>

namespace o2
{
namespace DataDistribution
{

void StfInputInterface::Start(unsigned pCnt)
{
  if (!mDevice.CheckCurrentState(StfBuilderDevice::RUNNING)) {
    LOG(WARN) << "Not creating interface threads. StfBuilder is not running.";
    return;
  }

  assert(mInputThreads.size() == 0);

  for (auto tid = 0; tid < pCnt; tid++) { // tid matches input channel index
    mInputThreads.emplace_back(std::thread(&StfInputInterface::DataHandlerThread, this, tid));
  }
}

void StfInputInterface::Stop()
{
  for (auto& lIdThread : mInputThreads)
    lIdThread.join();
}

/// Receiving thread
void StfInputInterface::DataHandlerThread(const unsigned pInputChannelIdx)
{
  int ret;
  // current TF Id
  std::int64_t lCurrentStfId = -1;
  std::vector<FairMQMessagePtr> lReadoutMsgs;
  lReadoutMsgs.reserve(1024);

  // Reference to the input channel
  auto& lInputChan = mDevice.GetChannel(mDevice.getInputChannelName(), pInputChannelIdx);
  auto& lOutputChan = mDevice.GetChannel(mDevice.getOutputChannelName());

  // Stf builder
  SubTimeFrameReadoutBuilder lStfBuilder(lOutputChan);

  try {
    while (mDevice.CheckCurrentState(StfBuilderDevice::RUNNING)) {

      // Equipment ID for the HBFrames (from the header)
      ReadoutSubTimeframeHeader lReadoutHdr;

      assert(lReadoutMsgs.empty());

      // receive readout messages
      auto lRet = lInputChan.Receive(lReadoutMsgs);
      if (lRet < 0 && mDevice.CheckCurrentState(StfBuilderDevice::RUNNING)) {
        LOG(WARNING) << "StfHeader receive failed (err = " + std::to_string(lRet) + ")";
        lReadoutMsgs.clear();
        continue;
      } else if (!mDevice.CheckCurrentState(StfBuilderDevice::RUNNING)) {
        break; // should exit?
      }

      // Copy to avoid surprises. The receiving header is not O2 compatible and can be discarded
      assert(lReadoutMsgs[0]->GetSize() == sizeof(ReadoutSubTimeframeHeader));
      std::memcpy(&lReadoutHdr, lReadoutMsgs[0]->GetData(), sizeof(ReadoutSubTimeframeHeader));

      // LOG(DEBUG) << "RECEIVED::Header::size: " << lReadoutMsgs[0]->GetSize() << ", "
      //           << "TF id: " << lReadoutHdr.timeframeId << ", "
      //           << "#HBF: " << lReadoutHdr.numberOfHBF << ", "
      //           << "EQ: " << lReadoutHdr.linkId;

      // check for the new TF marker
      if (lReadoutHdr.timeframeId != lCurrentStfId) {

        if (lCurrentStfId > 0 && lReadoutHdr.timeframeId < lCurrentStfId) {
          LOG(WARN) << "TF ID decreased! (" << lCurrentStfId << ") -> (" << lReadoutHdr.timeframeId << ")";
          // what now?
        }

        if (lCurrentStfId >= 0) {
          // Finished: queue the current STF
          std::unique_ptr<SubTimeFrame> lStf = lStfBuilder.getStf();
          if (lStf) {
            //LOG(INFO) << "Received TF[" << lStf->header().mId<< "]::size= " << lStf->getDataSize();
            mDevice.queue(eStfBuilderOut, std::move(lStf));
            // mDevice.queueStfFromReadout(std::move(lStf));
          } else {
            LOG(INFO) << "No data received? This should not happen.";
          }
        }

        // start a new STF
        lCurrentStfId = lReadoutHdr.timeframeId;
      }

      // handle HBFrames
      assert(lReadoutHdr.numberOfHBF > 0);
      assert(lReadoutHdr.numberOfHBF == lReadoutMsgs.size() - 1);

      lStfBuilder.addHbFrames(lReadoutHdr, std::move(lReadoutMsgs));

      lReadoutMsgs.clear();
    }
  } catch (std::runtime_error& e) {
    LOG(ERROR) << "Receive failed. Stopping input thread[" << pInputChannelIdx << "]...";
    return;
  }

  LOG(INFO) << "Exiting input thread[" << pInputChannelIdx << "]...";
}
}
}
