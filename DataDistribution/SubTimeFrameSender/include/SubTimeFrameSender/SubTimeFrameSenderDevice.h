// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_STF_SENDER_DEVICE_H_
#define ALICEO2_STF_SENDER_DEVICE_H_

#include "SubTimeFrameSender/SubTimeFrameSenderOutput.h"
#include "Common/SubTimeFrameFileSink.h"
#include "Common/Utilities.h"

#include "O2Device/O2Device.h"

#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>

namespace o2
{
namespace DataDistribution
{

enum StfSenderPipeline {
  eReceiverOut = 0,

  eFileSinkIn = 0,
  eFileSinkOut = 1,

  eSenderIn = 1,

  eNullIn = 2, // delete/drop
  ePipelineSize = 2,
  eInvalidStage = -1,
};

class StfSenderDevice : public Base::O2Device,
                        public IFifoPipeline<std::unique_ptr<SubTimeFrame>>
{
 public:
  static constexpr const char* OptionKeyInputChannelName = "input-channel-name";
  static constexpr const char* OptionKeyStandalone = "stand-alone";
  static constexpr const char* OptionKeyMaxBufferedStfs = "max-buffered-stfs";
  static constexpr const char* OptionKeyMaxConcurrentSends = "max-concurrent-sends";
  static constexpr const char* OptionKeyOutputChannelName = "output-channel-name";
  static constexpr const char* OptionKeyEpnNodeCount = "epn-count";

  /// Default constructor
  StfSenderDevice();

  /// Default destructor
  ~StfSenderDevice() override;

  void InitTask() final;

  const std::string& getOutputChannelName() const { return mOutputChannelName; }

  std::uint64_t stfCountIncFetch() { return ++mNumStfs; }
  std::uint64_t stfCountDecFetch() { return --mNumStfs; }
  std::uint64_t stfCountFetch() const { return mNumStfs; }

  bool standalone() const { return mStandalone; }
  std::uint32_t getEpnNodeCount() const { return mEpnNodeCount; }

 protected:
  void PreRun() final;
  void PostRun() final;
  bool ConditionalRun() final;

  void StfReceiverThread();

  unsigned getNextPipelineStage(unsigned pStage) final
  {
    StfSenderPipeline lNextStage = eInvalidStage;

    switch (pStage) {
      case eReceiverOut: {
        auto lNumStfs = stfCountIncFetch();
        if (mPipelineLimit && (lNumStfs > mMaxStfsInPipeline)) {
          stfCountDecFetch();
          lNextStage = eNullIn;
        } else {
          lNextStage = mFileSink.enabled() ? eFileSinkIn : eSenderIn;
        }
        break;
      }
      case eFileSinkOut:
        lNextStage = eSenderIn;
        break;
      default:
        throw std::runtime_error("pipeline error");
    }

    assert(lNextStage >= eFileSinkIn && lNextStage <= eSenderIn);
    return lNextStage;
  }

  /// Configuration
  std::string mInputChannelName;
  bool mStandalone;
  std::string mOutputChannelName;
  std::uint32_t mEpnNodeCount;
  std::int64_t mMaxStfsInPipeline;
  std::uint32_t mMaxConcurrentSends;
  bool mPipelineLimit;

  /// Receiver threads
  std::thread mReceiverThread;

  /// File sink
  SubTimeFrameFileSink mFileSink;

  /// Output stage handler
  StfSenderOutput mOutputHandler;

  /// number of STFs in the process
  std::atomic_uint64_t mNumStfs{ 0 };
};
}
} /* namespace o2::DataDistribution */

#endif /* ALICEO2_STF_SENDER_DEVICE_H_ */
