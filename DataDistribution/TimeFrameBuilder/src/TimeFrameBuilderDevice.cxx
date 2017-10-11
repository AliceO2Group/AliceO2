// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TimeFrameBuilder/TimeFrameBuilderDevice.h"
#include "Common/SubTimeFrameDataModel.h"
#include "Common/SubTimeFrameVisitors.h"

#include <options/FairMQProgOptions.h>
#include <FairMQLogger.h>

#include <TH1.h>

#include <chrono>
#include <thread>

namespace o2
{
namespace DataDistribution
{

using namespace std::chrono_literals;

TfBuilderDevice::TfBuilderDevice()
  : O2Device(),
    IFifoPipeline(eTfPipelineSize),
    mFlpInputHandler(*this, eTfBuilderOut),
    mFileSink(*this, *this, eTfFileSinkIn, eTfFileSinkOut),
    mGui(nullptr),
    mTfSizeSamples(1000),
    mTfFreqSamples(1000)
{
}

TfBuilderDevice::~TfBuilderDevice()
{
}

void TfBuilderDevice::InitTask()
{
  mInputChannelName = GetConfig()->GetValue<std::string>(OptionKeyInputChannelName);
  mStandalone = GetConfig()->GetValue<bool>(OptionKeyStandalone);
  mFlpNodeCount = GetConfig()->GetValue<std::uint32_t>(OptionKeyFlpNodeCount);
  mBuildHistograms = GetConfig()->GetValue<bool>(OptionKeyGui);

  // File sink
  if (!mFileSink.loadVerifyConfig(*(this->GetConfig())))
    exit(-1);
}

void TfBuilderDevice::PreRun()
{
  // start TF forwarding thread
  mTfFwdThread = std::thread(&TfBuilderDevice::TfForwardThread, this);
  // start file sink
  mFileSink.start();
  // Start output handlers
  mFlpInputHandler.Start(mFlpNodeCount);

  // start the gui thread
  if (mBuildHistograms) {
    mGui = std::make_unique<RootGui>("TFBuilder", "TF Builder", 1000, 400);
    mGui->Canvas().Divide(2, 1);
    mGuiThread = std::thread(&TfBuilderDevice::GuiThread, this);
  }
}

void TfBuilderDevice::PostRun()
{
  LOG(INFO) << "PostRun() start... ";
  // Stop the pipeline
  stopPipeline();
  // stop output handlers
  mFlpInputHandler.Stop();
  // signal and wait for the output thread
  mFileSink.stop();
  // join on fwd thread
  if (mTfFwdThread.joinable()) {
    mTfFwdThread.join();
  }

  //wait for the gui thread
  if (mBuildHistograms && mGuiThread.joinable()) {
    mGuiThread.join();
  }

  LOG(INFO) << "PostRun() done... ";
}

bool TfBuilderDevice::ConditionalRun()
{
  // nothing to do here sleep for awhile
  std::this_thread::sleep_for(500ms);

  // NOTE: Not using Run or ConditionalRun lets us put teardown in PostRun()
  return true;
}

void TfBuilderDevice::TfForwardThread()
{
  while (CheckCurrentState(RUNNING)) {
    static auto lFreqStartTime = std::chrono::high_resolution_clock::now();

    std::unique_ptr<SubTimeFrame> lTf = dequeue(eTfFwdIn);
    if (!lTf) {
      LOG(WARNING) << "ConditionalRun(): Exiting... ";
      break;
    }

    // record frequency and size of TFs
    if (mBuildHistograms) {
      mTfFreqSamples.Fill(
        1.0 / std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - lFreqStartTime)
                .count());

      lFreqStartTime = std::chrono::high_resolution_clock::now();

      // size histogram
      mTfSizeSamples.Fill(lTf->getDataSize());
    }

    // TODO: Do something with the TF
    {
      // is there a ratelimited LOG?
      static unsigned long floodgate = 0;
      if (++floodgate % 44 == 1)
        LOG(DEBUG) << "TF[" << lTf->header().mId << "] size: " << lTf->getDataSize();
    }
  }

  LOG(INFO) << "Exiting TF forwarding thread... ";
}

void TfBuilderDevice::GuiThread()
{
  while (CheckCurrentState(RUNNING)) {
    LOG(INFO) << "Updating histograms...";

    TH1F lTfSizeHist("TfSizeH", "Size of TF", 100, 0.0, float(1UL << 30));
    lTfSizeHist.GetXaxis()->SetTitle("Size [B]");
    for (const auto v : mTfSizeSamples)
      lTfSizeHist.Fill(v);

    mGui->Canvas().cd(1);
    lTfSizeHist.Draw();

    TH1F lTfFreqHist("TfFreq", "TimeFrame frequency", 200, 0.0, 100.0);
    lTfFreqHist.GetXaxis()->SetTitle("Frequency [Hz]");
    for (const auto v : mTfFreqSamples)
      lTfFreqHist.Fill(v);

    mGui->Canvas().cd(2);
    lTfFreqHist.Draw();

    mGui->Canvas().Modified();
    mGui->Canvas().Update();

    using namespace std::chrono_literals;
    std::this_thread::sleep_for(5s);
  }

  LOG(INFO) << "Exiting GUI thread...";
}
}
} /* namespace o2::DataDistribution */
