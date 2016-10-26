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

#include "FairMQDevice.h"

namespace AliceO2 {
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
    virtual ~FLPSyncSampler();

    /// Controls the send rate of the timeframe IDs
    void ResetEventCounter();

    /// Listens for acknowledgements from the epnReceivers when they collected full timeframe
    void ListenForAcks();

  protected:
    /// Overloads the InitTask() method of FairMQDevice
    virtual void InitTask();

    /// Overloads the Run() method of FairMQDevice
    virtual bool ConditionalRun();
    virtual void PreRun();
    virtual void PostRun();

    std::array<timeframeDuration, UINT16_MAX> fTimeframeRTT; ///< Container for the roundtrip values per timeframe ID
    int fEventRate; ///< Publishing rate of the timeframe IDs
    int fMaxEvents; ///< Maximum number of events to send (0 - unlimited)
    int fStoreRTTinFile; ///< Store round trip time measurements in a file.
    int fEventCounter; ///< Controls the send rate of the timeframe IDs
    uint16_t fTimeFrameId;
    std::thread fAckListener;
    std::thread fResetEventCounter;
    std::atomic<bool> fLeaving;

    std::string fAckChannelName;
    std::string fOutChannelName;
};

} // namespace Devices
} // namespace AliceO2

#endif
