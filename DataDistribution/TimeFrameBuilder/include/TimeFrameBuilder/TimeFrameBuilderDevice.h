// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TF_BUILDER_DEVICE_H_
#define ALICEO2_TF_BUILDER_DEVICE_H_

#include "TimeFrameBuilder/TimeFrameBuilderInput.h"
#include "Common/SubTimeFrameDataModel.h"
#include "Common/SubTimeFrameFileSink.h"
#include "Common/ConcurrentQueue.h"
#include "Common/Utilities.h"

#include "O2Device/O2Device.h"

#include <TApplication.h>
#include <TCanvas.h>
#include <TH1.h>

#include <deque>
#include <mutex>
#include <memory>
#include <condition_variable>

namespace o2
{
namespace DataDistribution
{

enum TfBuilderPipeline {
  eTfBuilderOut = 0,

  eTfFileSinkIn = 0,
  eTfFileSinkOut = 1,

  eTfFwdIn = 1,

  eTfPipelineSize = 2,
  eTfInvalidStage = -1,
};

class TfBuilderDevice : public Base::O2Device,
                        public IFifoPipeline<std::unique_ptr<SubTimeFrame>>
{
 public:
  static constexpr const char* OptionKeyInputChannelName = "input-channel-name";
  static constexpr const char* OptionKeyStandalone = "stand-alone";
  static constexpr const char* OptionKeyFlpNodeCount = "flp-count";
  static constexpr const char* OptionKeyGui = "gui";

  /// Default constructor
  TfBuilderDevice();

  /// Default destructor
  ~TfBuilderDevice() override;

  void InitTask() final;

  const std::string& getInputChannelName() const { return mInputChannelName; }
  const std::uint32_t getFlpNodeCount() const { return mFlpNodeCount; }

 protected:
  void PreRun() final;
  void PostRun() final;
  bool ConditionalRun() final;

  // Run the TFBuilder pipeline
  unsigned getNextPipelineStage(unsigned pStage) final
  {
    TfBuilderPipeline lNextStage = eTfInvalidStage;

    switch (pStage) {
      case eTfBuilderOut:
        lNextStage = mFileSink.enabled() ? eTfFileSinkIn : eTfFwdIn;
        break;
      case eTfFileSinkOut:
        lNextStage = eTfFwdIn;
        break;
      default:
        throw std::runtime_error("pipeline error");
    }

    assert(lNextStage >= eTfFileSinkIn && lNextStage <= eTfFwdIn);

    return lNextStage;
  }

  void TfForwardThread();

  void GuiThread();

  /// Configuration
  std::string mInputChannelName;
  bool mStandalone;
  std::uint32_t mFlpNodeCount;

  /// Input Interface handler
  TfBuilderInput mFlpInputHandler;

  /// File sink
  SubTimeFrameFileSink mFileSink;

  /// TF forwarding thread
  std::thread mTfFwdThread;

  /// Root stuff
  bool mBuildHistograms = true;
  std::unique_ptr<RootGui> mGui;
  std::thread mGuiThread;

  RunningSamples<uint64_t> mTfSizeSamples;
  RunningSamples<float> mTfFreqSamples;
};
}
} /* namespace o2::DataDistribution */

#endif /* ALICEO2_TF_BUILDER_DEVICE_H_ */
