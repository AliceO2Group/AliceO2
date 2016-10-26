/**
 * FLPSender.h
 *
 * @since 2014-02-24
 * @author D. Klein, A. Rybalchenko, M. Al-Turany, C. Kouzinopoulos
 */

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

class FLPSender : public FairMQDevice
{
  public:
    /// Default constructor
    FLPSender();

    /// Default destructor
    virtual ~FLPSender();

  protected:
    /// Overloads the InitTask() method of FairMQDevice
    virtual void InitTask();

    /// Overloads the Run() method of FairMQDevice
    virtual void Run();

  private:
    /// Sends the "oldest" element from the sub-timeframe container
    void sendFrontData();

    std::queue<FairMQParts> fSTFBuffer; ///< Buffer for sub-timeframes
    std::queue<std::chrono::steady_clock::time_point> fArrivalTime; ///< Stores arrival times of sub-timeframes

    int fNumEPNs; ///< Number of epnReceivers
    unsigned int fIndex; ///< Index of the flpSender among other flpSenders
    unsigned int fSendOffset; ///< Offset for staggering output
    unsigned int fSendDelay; ///< Delay for staggering output

    int fEventSize; ///< Size of the sub-timeframe body (only for test mode)
    int fTestMode; ///< Run the device in test mode (only syncSampler+flpSender+epnReceiver)
    uint16_t fTimeFrameId;

    std::string fInChannelName;
    std::string fOutChannelName;
};

} // namespace Devices
} // namespace AliceO2

#endif
