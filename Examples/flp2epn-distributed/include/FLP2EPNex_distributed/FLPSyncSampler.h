/**
 * FLPSyncSampler.h
 *
 * @since 2013-04-23
 * @author D. Klein, A. Rybalchenko
 */

#ifndef ALICEO2_DEVICES_FLPSYNCSAMPLER_H_
#define ALICEO2_DEVICES_FLPSYNCSAMPLER_H_

#include <string>
#include <cstdint> // UINT64_MAX

#include <thread>
#include <atomic>
#include <chrono>

#include <FairMQDevice.h>

namespace o2 {
namespace Devices {

/// Stores measurment for roundtrip time of a timeframe

struct timeframeDuration
{
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;
};

/// Publishes timeframes IDs for flpSenders (used only in test mode)

class FLPSyncSampler : public FairMQDevice
{
  public:
    /// Default constructor
    FLPSyncSampler();

    /// Default destructor
    ~FLPSyncSampler() override;

    /// Controls the send rate of the timeframe IDs
    void ResetEventCounter();

    /// Listens for acknowledgements from the epnReceivers when they collected full timeframe
    void ListenForAcks();

  protected:
    /// Overloads the InitTask() method of FairMQDevice
    void InitTask() override;

    /// Overloads the Run() method of FairMQDevice
    bool ConditionalRun() override;
    void PreRun() override;
    void PostRun() override;

    std::array<timeframeDuration, UINT16_MAX> mTimeframeRTT; ///< Container for the roundtrip values per timeframe ID
    int mEventRate; ///< Publishing rate of the timeframe IDs
    int mMaxEvents; ///< Maximum number of events to send (0 - unlimited)
    int mStoreRTTinFile; ///< Store round trip time measurements in a file.
    int mEventCounter; ///< Controls the send rate of the timeframe IDs
    uint16_t mTimeFrameId;
    std::thread mAckListener;
    std::thread mResetEventCounter;
    std::atomic<bool> mLeaving;

    std::string mAckChannelName;
    std::string mOutChannelName;
};

} // namespace Devices
} // namespace AliceO2

#endif
