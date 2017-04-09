#ifndef ALICEO2_TIMEFRAMEVALIDATOR_H_
#define ALICEO2_TIMEFRAMEVALIDATOR_H_

#include "O2Device/O2Device.h"

namespace o2 {
namespace DataFlow {

/// A validating device for time frame data (coming from EPN)
class TimeframeValidatorDevice : public Base::O2Device
{
public:
    static constexpr const char* OptionKeyInputChannelName = "input-channel-name";

    /// Default constructor
    TimeframeValidatorDevice();

    /// Default destructor
    virtual ~TimeframeValidatorDevice() = default;

    virtual void InitTask() final;

  protected:
    /// Overloads the Run() method of FairMQDevice
    virtual void Run() final;

    std::string mInChannelName;
};

} // namespace Devices
} // namespace AliceO2

#endif
