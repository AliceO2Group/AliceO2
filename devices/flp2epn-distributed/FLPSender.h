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

#include "FairMQDevice.h"                        // for FairMQDevice, etc

namespace AliceO2 {
namespace Devices {

class FLPSender : public FairMQDevice
{
  public:
    enum {
      HeartbeatTimeoutInMs = FairMQDevice::Last,
      Index,
      SendOffset,
      SendDelay,
      EventSize,
      TestMode,
      Last
    };

    FLPSender();
    virtual ~FLPSender();

    virtual void SetProperty(const int key, const std::string& value);
    virtual std::string GetProperty(const int key, const std::string& default_ = "");
    virtual void SetProperty(const int key, const int value);
    virtual int GetProperty(const int key, const int default_ = 0);

  protected:
    virtual void InitTask();
    virtual void Run();

  private:
    void receiveHeartbeats();
    void sendFrontData();

    unsigned int fIndex;
    unsigned int fSendOffset;
    unsigned int fSendDelay;
    std::queue<FairMQMessage*> fHeaderBuffer;
    std::queue<FairMQMessage*> fDataBuffer;
    std::queue<boost::posix_time::ptime> fArrivalTime;

    int fSndMoreFlag; // flag for multipart sending
    int fNoBlockFlag; // flag for sending without blocking

    int fNumEPNs;

    int fHeartbeatTimeoutInMs;
    std::unordered_map<std::string,boost::posix_time::ptime> fHeartbeats;
    boost::shared_mutex fHeartbeatMutex;

    int fEventSize;
    int fTestMode; // run in test mode
};

} // namespace Devices
} // namespace AliceO2

#endif
