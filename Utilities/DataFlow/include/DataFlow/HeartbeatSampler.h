//-*- Mode: C++ -*-

#ifndef HEARTBEATSAMPLER_H
#define HEARTBEATSAMPLER_H

// @file   HeartbeatSampler.h
// @author Matthias Richter
// @since  2017-02-03
// @brief  Heartbeat sampler device

#include "Headers/DataHeader.h"
#include "Headers/HeartbeatFrame.h"
#include "O2Device/O2Device.h"
#include <string>

namespace o2 {
namespace DataFlow {

/// @class HeartbeatSampler
/// @brief A sampler for heartbeat triggers
///
/// The device is going to be used in an emulation of the reader
/// processes on the FLP
/// The heartbeat triggers are sent out with constant frequency, the
/// period in nano seconds can be configured by option --period
///
/// TODO: the class can evolve to a general clock sampler device with
/// configurable period, even randomly distributed
class HeartbeatSampler : public Base::O2Device
{
public:
  typedef o2::Base::O2Message O2Message;

  static constexpr const char* OptionKeyOutputChannelName = "out-chan-name";
  static constexpr const char* OptionKeyPeriod = "period";

  HeartbeatSampler() = default;
  ~HeartbeatSampler() final = default;

protected:
  /// overloading the InitTask() method of FairMQDevice
  void InitTask() final;

  /// overloading ConditionalRun method of FairMQDevice
  bool ConditionalRun() final;

private:
  /// publishing period (configurable)
  uint32_t mPeriod = 1000000000;
  /// name of the (configurable)
  std::string mOutputChannelName = "output";
  /// number of elapsed periods
  int mCount = 0;
};

}; // namespace DataFlow
}; // namespace AliceO2
#endif
