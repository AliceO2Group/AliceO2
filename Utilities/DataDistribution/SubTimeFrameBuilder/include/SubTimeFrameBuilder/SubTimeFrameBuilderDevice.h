// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_STFBUILDER_DEVICE_H_
#define ALICEO2_STFBUILDER_DEVICE_H_

#include "Common/ReadoutDataModel.h"
#include "Common/SubTimeFrameDataModel.h"
#include "Common/ConcurrentQueue.h"
#include "Common/Utilities.h"

#include <O2Device/O2Device.h>

#include <TApplication.h>
#include <TCanvas.h>
#include <TH1.h>

#include <deque>
#include <mutex>
#include <condition_variable>

#include <atomic>

namespace o2 {
namespace DataDistribution {

// class TH1F;

class StfBuilderDevice : public Base::O2Device {
public:
  static constexpr const char* OptionKeyInputChannelName = "input-channel-name";
  static constexpr const char* OptionKeyOutputChannelName = "output-channel-name";

  static constexpr const char* OptionKeyCruCount = "cru-count";

  /// Default constructor
  StfBuilderDevice();

  /// Default destructor
  ~StfBuilderDevice() override;

  void InitTask() final;

protected:
  void PreRun() final;
  void PostRun() final;
  bool ConditionalRun() final;

  void RawDataInputThread(unsigned pCruId);
  void StfOutputThread();

  void GuiThread();

  /// config
  std::string mInputChannelName;
  std::string mOutputChannelName;
  std::uint64_t mCruCount;

  /// Internal queues
  ConcurrentFifo<O2SubTimeFrameLinkData> mReadoutLinkDataQueue;
  ConcurrentFifo<O2SubTimeFrame> mStfQueue;

  /// New STF signal
  /// TODO: this is part of O2 Readout interface
  std::atomic<std::uint64_t> mCurrentStfId;

  /// Internal threads
  std::vector<std::thread> mInputThreads;
  std::thread mOutputThread;

  /// Root stuff
  bool mBuildHistograms = true;
  TApplication mStfRootApp; // !?
  TCanvas mStfBuilderCanvas;
  std::thread mGuiThread;

  RunningSamples<uint64_t> mStfSizeSamples;
  RunningSamples<float> mStfLinkDataSamples;
  RunningSamples<float> mStfDataTimeSamples;
};
}
} /* namespace o2::DataDistribution */

#endif /* ALICEO2_STFBUILDER_DEVICE_H_ */
