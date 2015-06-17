/**
 * FLPSyncSampler.h
 *
 * @since 2013-04-23
 * @author D. Klein, A. Rybalchenko
 */

#ifndef ALICEO2_DEVICES_FLPSYNCSAMPLER_H_
#define ALICEO2_DEVICES_FLPSYNCSAMPLER_H_

#include <string>

#include <boost/date_time/posix_time/posix_time.hpp>

#include "FairMQDevice.h"

namespace AliceO2 {
namespace Devices {

struct timeframeDuration
{
  boost::posix_time::ptime start;
  boost::posix_time::ptime end;
};

class FLPSyncSampler : public FairMQDevice
{
  public:
    enum {
      EventRate = FairMQDevice::Last,
      EventSize,
      Last
    };

    FLPSyncSampler();
    virtual ~FLPSyncSampler();

    void ResetEventCounter();
    void ListenForAcks();

    virtual void SetProperty(const int key, const std::string& value);
    virtual std::string GetProperty(const int key, const std::string& default_ = "");
    virtual void SetProperty(const int key, const int value);
    virtual int GetProperty(const int key, const int default_ = 0);

  protected:
    virtual void Run();

    int fEventRate;
    int fEventCounter;

    std::map<uint64_t,timeframeDuration> fTimeframeRTT;
};

} // namespace Devices
} // namespace AliceO2

#endif
