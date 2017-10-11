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

#include "Common/SubTimeFrameUtils.h"
#include "Common/SubTimeFrameVisitors.h"
#include "Common/ReadoutDataModel.h"
#include "Common/SubTimeFrameDataModel.h"
#include "Common/SubTimeFrameDPL.h"
#include "Common/Utilities.h"

#include <TH1.h>

#include <options/FairMQProgOptions.h>
#include <FairMQLogger.h>

#include <chrono>
#include <thread>
#include <queue>

namespace o2
{
namespace DataDistribution
{

using namespace std::chrono_literals;

constexpr int StfBuilderDevice::gStfOutputChanId;

StfBuilderDevice::StfBuilderDevice()
  : O2Device(),
    IFifoPipeline(eStfPipelineSize),
    mReadoutInterface(*this),
    mFileSink(*this, *this, eStfFileSinkIn, eStfFileSinkOut),
    mGui(nullptr),
    mStfSizeSamples(1024),
    mStfFreqSamples(1024),
    mStfDataTimeSamples(1024)
{
}

StfBuilderDevice::~StfBuilderDevice()
{
}

void StfBuilderDevice::InitTask()
{
  mInputChannelName = GetConfig()->GetValue<std::string>(OptionKeyInputChannelName);
  mOutputChannelName = GetConfig()->GetValue<std::string>(OptionKeyOutputChannelName);
  mDplChannelName = GetConfig()->GetValue<std::string>(OptionKeyDplChannelName);
  mStandalone = GetConfig()->GetValue<bool>(OptionKeyStandalone);
  mMaxStfsInPipeline = GetConfig()->GetValue<std::int64_t>(OptionKeyMaxBufferedStfs);
  mCruCount = GetConfig()->GetValue<std::uint64_t>(OptionKeyCruCount);
  mBuildHistograms = GetConfig()->GetValue<bool>(OptionKeyGui);
  mDplEnabled = GetConfig()->GetValue<bool>(OptionKeyDpl);

  // Buffering limitation
  if (mMaxStfsInPipeline > 0) {
    if (mMaxStfsInPipeline < 4) {
      mMaxStfsInPipeline = 4;
      LOG(WARN) << "Configuration: max buffered SubTimeFrames limit increased to: " << mMaxStfsInPipeline;
    }
    mPipelineLimit = true;
    LOG(WARN) << "Configuration: Max buffered SubTimeFrames limit is set to " << mMaxStfsInPipeline
              << ". Consider increasing it if data loss occurs.";
  } else {
    mPipelineLimit = false;
    LOG(INFO) << "Not imposing limits on number of buffered SubTimeFrames. "
                 "Possibility of creating back-pressure.";
  }

  // File sink
  if (!mFileSink.loadVerifyConfig(*(this->GetConfig())))
    exit(-1);

  ChannelAllocator::get().addChannel(gStfOutputChanId, GetChannel(mOutputChannelName, 0));

  if (mCruCount < 1 || mCruCount > 32) {
    LOG(ERROR) << "CRU count parameter is not configured properly: " << mCruCount;
    exit(-1);
  }

  // check if output enabled
  if (mStandalone && !mFileSink.enabled()) {
    LOG(WARNING) << "Running in standalone mode and with STF file sink disabled. "
                    "Data will be lost.";
  }
}

void StfBuilderDevice::PreRun()
{
  // start output thread
  mOutputThread = std::thread(&StfBuilderDevice::StfOutputThread, this);
  // start file sink
  mFileSink.start();
  // start one thread per readout process
  mReadoutInterface.Start(mCruCount);

  // gui thread
  if (mBuildHistograms) {
    mGui = std::make_unique<RootGui>("STFBuilder", "STF Builder", 1200, 400);
    mGui->Canvas().Divide(3, 1);
    mGuiThread = std::thread(&StfBuilderDevice::GuiThread, this);
  }
}

void StfBuilderDevice::PostRun()
{
  // Stop the pipeline
  stopPipeline();
  // wait for readout interface threads
  mReadoutInterface.Stop();
  // signal and wait for the output thread
  mFileSink.stop();
  // stop the output
  if (mOutputThread.joinable()) {
    mOutputThread.join();
  }

  // wait for the gui thread
  if (mBuildHistograms && mGuiThread.joinable()) {
    mGuiThread.join();
  }

  LOG(INFO) << "PostRun() done... ";
}

void StfBuilderDevice::StfOutputThread()
{
  auto& lOutputChan = GetChannel(getOutputChannelName(), 0);
  InterleavedHdrDataSerializer lStfSerializer(lOutputChan);

  auto& lDplChan = GetChannel(getDplChannelName(), 0);
  StfDplAdapter lStfDplAdapter(lDplChan);

  using hres_clock = std::chrono::high_resolution_clock;

  while (CheckCurrentState(RUNNING)) {

    const auto lFreqStartTime = hres_clock::now();

    // Get a STF readu for sending
    std::unique_ptr<SubTimeFrame> lStf = dequeue(eStfSendIn);
    if (!lStf)
      break;

    // decrement the stf counter
    mNumStfs--;

    if (mBuildHistograms) {
      mStfFreqSamples.Fill(
        1.0 / std::chrono::duration<double>(hres_clock::now() - lFreqStartTime).count());
    }

    // Send the STF
    const auto lSendStartTime = hres_clock::now();

#ifdef STF_FILTER_EXAMPLE
    // EXAMPLE:
    // split one SubTimeFrame into per detector SubTimeFrames (this is unlikely situation for a STF
    // but still... The TimeFrame structure should be similar)
    static const DataIdentifier cTPCDataIdentifier(gDataDescriptionAny, gDataOriginTPC);
    static const DataIdentifier cITSDataIdentifier(gDataDescriptionAny, gDataOriginITS);

    DataIdentifierSplitter lStfDetectorSplitter;
    SubTimeFrame lStfTPC = lStfDetectorSplitter.split(lStf, cTPCDataIdentifier, gStfOutputChanId);
    SubTimeFrame lStfITS = lStfDetectorSplitter.split(lStf, cITSDataIdentifier, gStfOutputChanId);

    if (mBuildHistograms) {
      mStfSizeSamples.Fill(lStfTPC.getDataSize());
      mStfSizeSamples.Fill(lStfITS.getDataSize());
    }

    if (!mStandalone) {
      // Send filtered data as two objects
      try {
        if (!mDplEnabled) {
          lStfSerializer.serialize(std::move(lStfTPC));
          lStfSerializer.serialize(std::move(lStfITS));
        } else {
          lStfDplAdapter.sendToDpl(std::move(lStfTPC));
          lStfDplAdapter.sendToDpl(std::move(lStfITS));
        }
      } catch (std::exception& e) {
        if (CheckCurrentState(RUNNING))
          LOG(ERROR) << "StfOutputThread: exception on send: " << e.what();
        else
          LOG(INFO) << "StfOutputThread(NOT_RUNNING): exception on send: " << e.what();

        break;
      }
    }
#else

    if (mBuildHistograms) {
      mStfSizeSamples.Fill(lStf->getDataSize());
    }

    if (!mStandalone) {
      try {
        if (!mDplEnabled) {
          lStfSerializer.serialize(std::move(lStf));
        } else {

          // DPL Channel

          static thread_local unsigned long lThrottle = 0;
          if (++lThrottle % 88 == 0) {
            LOG(DEBUG) << "Sending STF to DPL: id:" << lStf->header().mId
                       << " data size: " << lStf->getDataSize()
                       << " unique equipments: " << lStf->getEquipmentIdentifiers().size();
          }

          // Send to DPL bridge
          lStfDplAdapter.sendToDpl(std::move(lStf));
        }
      } catch (std::exception& e) {
        if (CheckCurrentState(RUNNING)) {
          LOG(ERROR) << "StfOutputThread: exception on send: " << e.what();
        } else {
          LOG(INFO) << "StfOutputThread(NOT_RUNNING): shutting down: " << e.what();
        }
        break;
      }
    }
#endif

    if (!mStandalone && mBuildHistograms) {
      double lTimeMs = std::chrono::duration<double, std::milli>(hres_clock::now() - lSendStartTime).count();
      mStfDataTimeSamples.Fill(lTimeMs);
    }
  }

  LOG(INFO) << "Exiting StfOutputThread...";
}

void StfBuilderDevice::GuiThread()
{
  while (CheckCurrentState(RUNNING)) {
    LOG(INFO) << "Updating histograms...";

    TH1F lStfSizeHist("StfSizeH", "Readout data size per STF", 100, 0.0, 400e+6);
    lStfSizeHist.GetXaxis()->SetTitle("Size [B]");
    for (const auto v : mStfSizeSamples) {
      lStfSizeHist.Fill(v);
    }

    mGui->Canvas().cd(1);
    lStfSizeHist.Draw();

    TH1F lStfFreqHist("STFFreq", "SubTimeFrame frequency", 100, 0.0, 100.0);
    lStfFreqHist.GetXaxis()->SetTitle("Frequency [Hz]");
    for (const auto v : mStfFreqSamples) {
      lStfFreqHist.Fill(v);
    }

    mGui->Canvas().cd(2);
    lStfFreqHist.Draw();

    TH1F lStfDataTimeHist("StfChanTimeH", "STF on-channel time", 100, 0.0, 20.0);
    lStfDataTimeHist.GetXaxis()->SetTitle("Time [ms]");
    for (const auto v : mStfDataTimeSamples) {
      lStfDataTimeHist.Fill(v);
    }

    mGui->Canvas().cd(3);
    lStfDataTimeHist.Draw();

    mGui->Canvas().Modified();
    mGui->Canvas().Update();

    std::this_thread::sleep_for(5s);
  }
  LOG(INFO) << "Exiting GUI thread...";
}

bool StfBuilderDevice::ConditionalRun()
{
  // nothing to do here sleep for awhile
  std::this_thread::sleep_for(1s);
  return true;
}
}
} /* namespace o2::DataDistribution */
