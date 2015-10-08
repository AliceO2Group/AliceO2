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
#include "FairMQDevice.h"                        // for FairMQDevice, etc
#include "boost/date_time/posix_time/ptime.hpp"  // for ptime

namespace AliceO2 {
namespace Devices {

class FLPSender : public FairMQDevice
{
  public:
    enum {
      OutputHeartbeat = FairMQDevice::Last,
      HeartbeatTimeoutInMs,
      Index,
      SendOffset,
      SendDelay,
      EventSize,
      TestMode,
      Last
    };

    FLPSender();
    virtual ~FLPSender();

    void ResetEventCounter();

    virtual void SetProperty(const int key, const std::string& value);
    virtual std::string GetProperty(const int key, const std::string& default_ = "");
    virtual void SetProperty(const int key, const int value);
    virtual int GetProperty(const int key, const int default_ = 0);

    void SetProperty(const int key, const boost::posix_time::ptime value, const int slot = 0);
    boost::posix_time::ptime GetProperty(const int key, const boost::posix_time::ptime value, const int slot = 0);

  protected:
    virtual void Init();
    virtual void Run();

  private:
    bool updateIPHeartbeat(std::string str);
    void sendFrontData();

    int fHeartbeatTimeoutInMs;
    std::vector<boost::posix_time::ptime> fOutputHeartbeat;

    unsigned int fIndex;
    unsigned int fSendOffset;
    unsigned int fSendDelay;
    std::queue<FairMQMessage*> fHeaderBuffer;
    std::queue<FairMQMessage*> fDataBuffer;
    std::queue<boost::posix_time::ptime> fArrivalTime;

    std::unordered_map<int,boost::posix_time::ptime> fRTTimes;

    int fEventSize;
    int fTestMode; // run in test mode
};

} // namespace Devices
} // namespace AliceO2

#endif
