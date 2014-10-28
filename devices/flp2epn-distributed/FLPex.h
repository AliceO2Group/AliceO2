/**
 * FLPex.h
 *
 * @since 2014-02-24
 * @author D. Klein, A. Rybalchenko, M. Al-Turany, C. Kouzinopoulos
 */

#ifndef ALICEO2_DEVICES_FLPEX_H_
#define ALICEO2_DEVICES_FLPEX_H_

#include <string>
#include <queue>

#include <boost/date_time/posix_time/posix_time.hpp>

#include "FairMQDevice.h"

namespace AliceO2 {
namespace Devices {

class FLPex : public FairMQDevice
{
  public:
    enum {
      OutputHeartbeat  = FairMQDevice::Last,
      HeartbeatTimeoutInMs,
      NumFLPs,
      SendOffset,
      Last
    };

    FLPex();
    virtual ~FLPex();

    virtual void SetProperty(const int key, const std::string& value, const int slot = 0);
    virtual std::string GetProperty(const int key, const std::string& default_ = "", const int slot = 0);
    virtual void SetProperty(const int key, const int value, const int slot = 0);
    virtual int GetProperty(const int key, const int default_ = 0, const int slot = 0);
    virtual void SetProperty(const int key, const boost::posix_time::ptime value, const int slot = 0);
    virtual boost::posix_time::ptime GetProperty(const int key, const boost::posix_time::ptime value, const int slot = 0);

  protected:
    virtual void Init();
    virtual void Run();

  private:
    bool updateIPHeartbeat(std::string str);

    int fHeartbeatTimeoutInMs;
    int fSendOffset;
    std::queue<FairMQMessage*> fIdBuffer;
    std::queue<FairMQMessage*> fDataBuffer;
    vector<boost::posix_time::ptime> fOutputHeartbeat;
};

} // namespace Devices
} // namespace AliceO2

#endif
