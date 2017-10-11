// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "SubTimeFrameBuilder/SubTimeFrameBuilderDevice.h"

#include "Common/SubTimeFrameVisitors.h"
#include "Common/ReadoutDataModel.h"
#include "Common/SubTimeFrameDataModel.h"
#include "Common/Utilities.h"

#include <TH1.h>

#include <options/FairMQProgOptions.h>
#include <FairMQLogger.h>

#include <chrono>
#include <thread>
#include <queue>

namespace o2 {
namespace DataDistribution {

constexpr int gStfOutputChanId = 0;

StfBuilderDevice::StfBuilderDevice()
  : O2Device{},
    mStfRootApp("StfBuilderApp", nullptr, nullptr),
    mStfBuilderCanvas("cnv", "STF Builder", 1500, 500),
    mStfSizeSamples(10000),
    mStfLinkDataSamples(10000),
    mStfDataTimeSamples(10000)
{
  mStfBuilderCanvas.Divide(3, 1);
}

StfBuilderDevice::~StfBuilderDevice()
{
  mOutputThread.join();
}

void StfBuilderDevice::InitTask()
{
  mInputChannelName = GetConfig()->GetValue<std::string>(OptionKeyInputChannelName);
  mOutputChannelName = GetConfig()->GetValue<std::string>(OptionKeyOutputChannelName);
  mCruCount = GetConfig()->GetValue<std::uint64_t>(OptionKeyCruCount);

  ChannelAllocator::get().addChannel(gStfOutputChanId, this, mOutputChannelName, 0);

  if (mCruCount < 1 || mCruCount > 32) {
    LOG(ERROR) << "CRU count parameter is not configured properly: " << mCruCount;
    exit(-1);
  }
}

void StfBuilderDevice::PreRun()
{
  mCurrentStfId.store(0);
  // start output thread
  mOutputThread = std::thread(&StfBuilderDevice::StfOutputThread, this);
  // start one thread per CRU readout process
  for (auto tid = 0; tid < mCruCount; tid++) // tid matches input channel index
    mInputThreads.emplace_back(std::thread(&StfBuilderDevice::RawDataInputThread, this, tid));
  // gui thread
  mGuiThread = std::thread(&StfBuilderDevice::GuiThread, this);
}

void StfBuilderDevice::PostRun()
{
  mOutputThread.join(); // stopped by CheckCurrentState(RUNNING)
  for (auto tid = 0; tid < mCruCount; tid++)
    mInputThreads[tid].join();

  mGuiThread.join();
}

bool StfBuilderDevice::ConditionalRun()
{
  thread_local static std::queue<O2SubTimeFrame> sStfInBuilding;

  if (sStfInBuilding.size() == 0)
    sStfInBuilding.emplace(O2SubTimeFrame{ gStfOutputChanId, mCurrentStfId });
  else
    assert(sStfInBuilding.size() == 1);

  O2SubTimeFrame& lCurrentStf = sStfInBuilding.front();
  std::uint64_t lCurrentStfId = lCurrentStf.Header().mStfId;

  while (CheckCurrentState(RUNNING)) { // loop here for 1 STF, then release

    O2SubTimeFrameLinkData lLinkData;
    if (!mReadoutLinkDataQueue.pop(lLinkData)) {
      LOG(WARN) << "Stopping StfSendThread (ConditionalRun)";
      return false;
    }

    // check if should stop the current STF and start building another
    if (lCurrentStfId < lLinkData.mCruLinkHeader->mStfId) {
      // assert(lCurrentStfId == lLinkData.Header().mStfId - 1);

      if (mBuildHistograms)
        mStfSizeSamples.Fill(lCurrentStf.getRawDataSize());

      mStfQueue.push(std::move(lCurrentStf));
      sStfInBuilding.pop();

      // queue the data chunk into the next STF
      assert(sStfInBuilding.size() == 0);
      sStfInBuilding.emplace(O2SubTimeFrame{ gStfOutputChanId, lLinkData.mCruLinkHeader->mStfId });
      sStfInBuilding.front().addCruLinkData(gStfOutputChanId, std::move(lLinkData));
      return true; // TODO: move into a real thread to avoid this strangeness
    }

    // add the data to the current STF
    lCurrentStf.addCruLinkData(gStfOutputChanId, std::move(lLinkData));
  }

  return false;
}

void StfBuilderDevice::StfOutputThread()
{
  while (CheckCurrentState(RUNNING)) {
    O2SubTimeFrame lStf;

    if (!mStfQueue.pop(lStf)) {
      LOG(WARN) << "Stopping StfOutputThread...";
      return;
    }

    const auto lStartTime = std::chrono::high_resolution_clock::now();

#if STF_SERIALIZATION == 1
    InterleavedHdrDataSerializer lStfSerializer;
    lStfSerializer.serialize(lStf, *this, mOutputChannelName, 0);
#elif STF_SERIALIZATION == 2
    HdrDataSerializer lStfSerializer;
    lStfSerializer.serialize(lStf, *this, mOutputChannelName, 0);
#else
    #error "Unknown STF_SERIALIZATION type"
#endif

    double lTimeMs =
      std::chrono::duration<double, std::micro>(std::chrono::high_resolution_clock::now() - lStartTime).count();
    mStfDataTimeSamples.Fill(lTimeMs / 1000.0);
  }
}

void StfBuilderDevice::RawDataInputThread(unsigned pCruId)
{
  while (CheckCurrentState(RUNNING)) {
    try
    {
      // receive channel object info
      ReadoutStfBuilderObjectInfo lObjectInfo(*this, mInputChannelName, pCruId);
      switch (lObjectInfo.mObjectType) {
        case eStfStart: {
          // TODO: readout interface
          // simply check if we have seen that STF ID already

          // Test and test-and-set approach
          std::uint64_t lCurrentStfId = mCurrentStfId;
          if (lCurrentStfId < lObjectInfo.mStfId)
            mCurrentStfId.compare_exchange_weak(lCurrentStfId, lObjectInfo.mStfId);
          continue; // don't expect a 'data' object
          break;
        }
        case eReadoutData: {
          try
          {
            const auto lStartTime = std::chrono::high_resolution_clock::now();
            O2SubTimeFrameLinkData lLinkData(*this, mInputChannelName, pCruId);

            if (mBuildHistograms)
              mStfLinkDataSamples.Fill(std::chrono::duration<double, std::micro>(
                std::chrono::high_resolution_clock::now() - lStartTime).count());

            lLinkData.mCruLinkHeader->mStfId = mCurrentStfId;
            // Done! Queue it up.
            mReadoutLinkDataQueue.push(std::move(lLinkData));
          }
          catch (std::runtime_error& e)
          {
            LOG(ERROR) << "CRU Link Data receive failed. Stopping RawDataInputThread[" << pCruId << "]...";
            return;
          }
          break;
        }
        default: {
          LOG(ERROR) << "Unknown object type on the input channel RawDataInputThread[" << pCruId << "]...";
          return; // change device state to stop?
        }
      }
    }
    catch (std::runtime_error& e)
    {
      LOG(ERROR) << "ObjectInfo Receive failed. Stopping RawDataInputThread[" << pCruId << "]...";
      return;
    }
  }
}

void StfBuilderDevice::GuiThread()
{
  while (CheckCurrentState(RUNNING)) {
    LOG(INFO) << "Updating histograms...";

    TH1F lStfSizeHist("StfSizeH", "Readout data size per STF", 64, 0.0, 700e+6);
    lStfSizeHist.GetXaxis()->SetTitle("Size [B]");
    for (const auto v : mStfSizeSamples)
      lStfSizeHist.Fill(v);

    mStfBuilderCanvas.cd(1);
    lStfSizeHist.Draw();

    TH1F lStfLinkDataTimeHist("SuperpageChanTimeH", "LinkData on-channel time", 100, 0.0, 3000);
    lStfLinkDataTimeHist.GetXaxis()->SetTitle("Time [us]");
    for (const auto v : mStfLinkDataSamples)
      lStfLinkDataTimeHist.Fill(v);

    mStfBuilderCanvas.cd(2);
    lStfLinkDataTimeHist.Draw();

    TH1F lStfDataTimeHist("StfChanTimeH", "STF on-channel time", 100, 0.0, 30.0);
    lStfDataTimeHist.GetXaxis()->SetTitle("Time [ms]");
    for (const auto v : mStfDataTimeSamples)
      lStfDataTimeHist.Fill(v);

    mStfBuilderCanvas.cd(3);
    lStfDataTimeHist.Draw();

    mStfBuilderCanvas.Modified();
    mStfBuilderCanvas.Update();

    using namespace std::chrono_literals;
    std::this_thread::sleep_for(5s);
  }
}
}
} /* namespace o2::DataDistribution */
