/**
 * FLPexSampler.h
 *
 * @since 2013-04-23
 * @author D. Klein, A. Rybalchenko
 */

#ifndef ALICEO2_DEVICES_FLPEXSAMPLER_H_
#define ALICEO2_DEVICES_FLPEXSAMPLER_H_

#include <string>

#include "FairMQDevice.h"

namespace AliceO2 {
namespace Devices {

class FLPexSampler : public FairMQDevice
{
  public:
    enum {
      EventRate = FairMQDevice::Last,
      EventSize,
      Last
    };

    FLPexSampler();
    virtual ~FLPexSampler();

    void ResetEventCounter();
    virtual void SetProperty(const int key, const std::string& value, const int slot = 0);
    virtual std::string GetProperty(const int key, const std::string& default_ = "", const int slot = 0);
    virtual void SetProperty(const int key, const int value, const int slot = 0);
    virtual int GetProperty(const int key, const int default_ = 0, const int slot = 0);

  protected:
    virtual void Run();

    int fEventSize;
    int fEventRate;
    int fEventCounter;
};

} // namespace Devices
} // namespace AliceO2

#endif
