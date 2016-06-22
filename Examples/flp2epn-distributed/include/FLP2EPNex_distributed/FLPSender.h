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

#include <boost/date_time/posix_time/posix_time.hpp>

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
    /// Device properties
    enum
    {
        Index = FairMQDevice::Last, ///< Index of the flpSender amond other flpSenders
        SendOffset, ///< Offset for staggering output
        SendDelay, ///< Delay for staggering output
        HeartbeatTimeoutInMs, ///< Heartbeat timeout for epnReceivers
        EventSize, ///< Size of the sub-timeframe body (only for test mode)
        TestMode, ///< Run the device in test mode (only syncSampler+flpSender+epnReceiver)
        Last
    };

    /// Default constructor
    FLPSender();

    /// Default destructor
    virtual ~FLPSender();

    /// Set Device properties stored as strings
    /// @param key      Property key
    /// @param value    Property value
    virtual void SetProperty(const int key, const std::string &value);

    /// Get Device properties stored as strings
    /// @param key      Property key
    /// @param default_ not used
    /// @return         Property value
    virtual std::string GetProperty(const int key, const std::string &default_ = "");

    /// Set Device properties stored as integers
    /// @param key      Property key
    /// @param value    Property value
    virtual void SetProperty(const int key, const int value);

    /// Get Device properties stored as integers
    /// @param key      Property key
    /// @param default_ not used
    /// @return         Property value
    virtual int GetProperty(const int key, const int default_ = 0);

  protected:
    /// Overloads the InitTask() method of FairMQDevice
    virtual void InitTask();

    /// Overloads the Run() method of FairMQDevice
    virtual void Run();

  private:
    /// Receives heartbeats from epnReceivers
    void receiveHeartbeats();

    /// Sends the "oldest" element from the sub-timeframe container
    void sendFrontData();

    std::queue<std::unique_ptr<FairMQMessage>> fHeaderBuffer; ///< Stores sub-timeframe headers
    std::queue<std::unique_ptr<FairMQMessage>> fDataBuffer; ///< Stores sub-timeframe bodies
    std::queue<boost::posix_time::ptime> fArrivalTime; ///< Stores arrival times of sub-timeframes

    int fNumEPNs; ///< Number of epnReceivers
    unsigned int fIndex; ///< Index of the flpSender among other flpSenders
    unsigned int fSendOffset; ///< Offset for staggering output
    unsigned int fSendDelay; ///< Delay for staggering output

    int fEventSize; ///< Size of the sub-timeframe body (only for test mode)
    int fTestMode; ///< Run the device in test mode (only syncSampler+flpSender+epnReceiver)

    int fHeartbeatTimeoutInMs; ///< Heartbeat timeout for epnReceivers
    std::unordered_map<std::string, boost::posix_time::ptime> fHeartbeats; ///< Stores heartbeats from epnReceiver
    boost::shared_mutex fHeartbeatMutex; ///< Mutex for heartbeat synchronization
};

} // namespace Devices
} // namespace AliceO2

#endif
