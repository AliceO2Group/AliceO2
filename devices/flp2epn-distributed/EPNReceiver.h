/**
 * EPNReceiver.h
 *
 * @since 2013-01-09
 * @author D. Klein, A. Rybalchenko, M. Al-Turany, C. Kouzinopoulos
 */

#ifndef ALICEO2_DEVICES_EPNRECEIVER_H_
#define ALICEO2_DEVICES_EPNRECEIVER_H_

#include <string>
#include <unordered_map>
#include <unordered_set>

#include <boost/date_time/posix_time/posix_time.hpp>

#include "FairMQDevice.h"

namespace AliceO2 {
namespace Devices {

/// Container for (sub-)timeframes

struct TFBuffer
{
  std::vector<FairMQMessage*> parts;
  boost::posix_time::ptime startTime;
  boost::posix_time::ptime endTime;
};

/// Receives sub-timeframes from the flpSenders and merges these into full timeframes.

class EPNReceiver : public FairMQDevice
{
  public:
    /// Device properties
    enum {
      NumFLPs = FairMQDevice::Last, ///< Number of flpSenders
      BufferTimeoutInMs, ///< Time after which incomplete timeframes are dropped
      TestMode, ///< Run the device in test mode
      HeartbeatIntervalInMs, ///< Interval for sending heartbeats
      Last
    };

    /// Default constructor
    EPNReceiver();

    /// Default destructor
    virtual ~EPNReceiver();

    /// Prints the contents of the timeframe container
    void PrintBuffer(const std::unordered_map<uint16_t, TFBuffer>& buffer) const;
    /// Discared incomplete timeframes after \p fBufferTimeoutInMs.
    void DiscardIncompleteTimeframes();

    /// Set device properties stored as strings
    /// @param key      Property key
    /// @param value    Property value
    virtual void SetProperty(const int key, const std::string& value);
    /// Get device properties stored as strings
    /// @param key      Property key
    /// @param default_ not used
    /// @return         Property value
    virtual std::string GetProperty(const int key, const std::string& default_ = "");
    /// Set device properties stored as integers
    /// @param key      Property key
    /// @param value    Property value
    virtual void SetProperty(const int key, const int value);
    /// Get device properties stored as integers
    /// @param key      Property key
    /// @param default_ not used
    /// @return         Property value
    virtual int GetProperty(const int key, const int default_ = 0);

  protected:
    /// Overloads the Run() method of FairMQDevice
    virtual void Run();
    /// Sends heartbeats to flpSenders
    void sendHeartbeats();

    std::unordered_map<uint16_t, TFBuffer> fTimeframeBuffer; ///< Stores (sub-)timeframes
    std::unordered_set<uint16_t> fDiscardedSet; ///< Set containing IDs of dropped timeframes

    int fNumFLPs; ///< Number of flpSenders
    int fBufferTimeoutInMs; ///< Time after which incomplete timeframes are dropped
    int fTestMode; ///< Run the device in test mode (only syncSampler+flpSender+epnReceiver)
    int fHeartbeatIntervalInMs; ///< Interval for sending heartbeats
};

} // namespace Devices
} // namespace AliceO2

#endif
