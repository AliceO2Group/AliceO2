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

#include <boost/date_time/posix_time/posix_time.hpp>

#include "FairMQDevice.h"

namespace AliceO2 {
namespace Devices {

/// Stores measurment for roundtrip time of a timeframe

struct timeframeDuration
{
    boost::posix_time::ptime start;
    boost::posix_time::ptime end;
};

/// Publishes timeframes IDs for flpSenders (used only in test mode)

class FLPSyncSampler : public FairMQDevice
{
  public:
    enum
    {
        EventRate = FairMQDevice::Last, ///< Publishing rate of the timeframe IDs
        MaxEvents, ///< Maximum number of events to send (0 - unlimited)
        StoreRTTinFile, ///< Store round trip time measurements in a file
        Last
    };

    /// Default constructor
    FLPSyncSampler();

    /// Default destructor
    virtual ~FLPSyncSampler();

    /// Controls the send rate of the timeframe IDs
    void ResetEventCounter();

    /// Listens for acknowledgements from the epnReceivers when they collected full timeframe
    void ListenForAcks();

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

    std::array<timeframeDuration, UINT16_MAX> fTimeframeRTT; ///< Container for the roundtrip values per timeframe ID
    int fEventRate; ///< Publishing rate of the timeframe IDs
    int fMaxEvents; ///< Maximum number of events to send (0 - unlimited)
    int fStoreRTTinFile; ///< Store round trip time measurements in a file.
    int fEventCounter; ///< Controls the send rate of the timeframe IDs
};

} // namespace Devices
} // namespace AliceO2

#endif
