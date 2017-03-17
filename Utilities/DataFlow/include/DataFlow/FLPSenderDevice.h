#ifndef ALICEO2_DEVICES_FLPSENDER_H_
#define ALICEO2_DEVICES_FLPSENDER_H_

#include <string>
#include <queue>
#include <unordered_map>
#include <chrono>

#include "FairMQDevice.h"

namespace AliceO2 {
namespace Devices {

/// Sends sub-timframes to epnReceivers
///
/// Sub-timeframes are received from the previous step (or generated in test-mode)
/// and are sent to epnReceivers. Target epnReceiver is determined from the timeframe ID:
/// targetEpnReceiver = timeframeId % numEPNs (numEPNs is same for every flpSender, although some may be inactive).

class FLPSenderDevice : public FairMQDevice
{
  public:
    /// Default constructor
    FLPSenderDevice() = default;

    /// Default destructor
    virtual ~FLPSenderDevice() final = default;

  protected:
    /// Overloads the InitTask() method of FairMQDevice
    virtual void InitTask() final;

    /// Overloads the Run() method of FairMQDevice
    virtual void Run() final;

  private:
    /// Sends the "oldest" element from the sub-timeframe container
    void sendFrontData();

    std::queue<FairMQParts> fSTFBuffer; ///< Buffer for sub-timeframes
    std::queue<std::chrono::steady_clock::time_point> fArrivalTime; ///< Stores arrival times of sub-timeframes

    int fNumEPNs = 0; ///< Number of epnReceivers
    unsigned int fIndex = 0; ///< Index of the flpSender among other flpSenders
    unsigned int fSendOffset = 0; ///< Offset for staggering output
    unsigned int fSendDelay = 8; ///< Delay for staggering output

    int fEventSize = 10000; ///< Size of the sub-timeframe body (only for test mode)
    int fTestMode = false; ///< Run the device in test mode (only syncSampler+flpSender+epnReceiver)
    uint16_t fTimeFrameId;

    std::string fInChannelName = "";
    std::string fOutChannelName = "";
    int mLastTimeframeId = -1;
};

} // namespace Devices
} // namespace AliceO2

#endif
