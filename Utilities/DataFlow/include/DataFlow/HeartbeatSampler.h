//-*- Mode: C++ -*-

#ifndef HEARTBEATSAMPLER_H
#define HEARTBEATSAMPLER_H

#include "Headers/DataHeader.h"
#include "Headers/HeartbeatFrame.h"
#include "O2Device/O2Device.h"

namespace AliceO2 {
namespace DataFlow {
class HeartbeatSampler : public Base::O2Device
{
public:
  /// Default constructor
  HeartbeatSampler();

  /// Default destructor
  virtual ~HeartbeatSampler();

protected:
  /// overloading the InitTask() method of FairMQDevice
  virtual void InitTask();

  /// overloading ConditionalRun method of FairMQDevice
  virtual bool ConditionalRun();

private:
};

}; // namespace DataFlow
}; // namespace AliceO2
#endif
