/**
 * EPNex.h
 *
 * @since 2013-01-09
 * @author D. Klein, A. Rybalchenko, M.Al-Turany, C. Kouzinopoulos
 */

#ifndef ALICEO2_DEVICES_EPNEX_H_
#define ALICEO2_DEVICES_EPNEX_H_

#include <string>

#include "FairMQDevice.h"

namespace AliceO2 {
namespace Devices {

class EPNex : public FairMQDevice
{
  public:
    enum {
      HeartbeatIntervalInMs = FairMQDevice::Last,
      Last
    };
    EPNex();
    virtual ~EPNex();

    virtual void SetProperty(const int key, const std::string& value, const int slot = 0);
    virtual std::string GetProperty(const int key, const std::string& default_ = "", const int slot = 0);
    virtual void SetProperty(const int key, const int value, const int slot = 0);
    virtual int GetProperty(const int key, const int default_ = 0, const int slot = 0);

  protected:
    virtual void Run();
    void sendHeartbeats();

    int fHeartbeatIntervalInMs;
};

} // namespace Devices
} // namespace AliceO2

#endif
