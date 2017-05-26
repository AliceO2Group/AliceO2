#ifndef ALICEO2_FAKE_TIMEFRAME_GENERATOR_H_
#define ALICEO2_FAKE_TIMEFRAME_GENERATOR_H_

#include "O2Device/O2Device.h"

namespace o2 {
namespace DataFlow {

/// A device which writes to file the timeframes.
class FakeTimeframeGeneratorDevice : public Base::O2Device
{
public:
    static constexpr const char* OptionKeyOutputChannelName = "output-channel-name";
    static constexpr const char* OptionKeyMaxTimeframes = "max-timeframes";

    /// Default constructor
    FakeTimeframeGeneratorDevice();

    /// Default destructor
    ~FakeTimeframeGeneratorDevice() override = default;

    void InitTask() final;

  protected:
    /// Overloads the ConditionalRun() method of FairMQDevice
    bool ConditionalRun() final;

    std::string      mOutChannelName;
    size_t           mMaxTimeframes;
    size_t           mTimeframeCount;
};

} // namespace DataFlow
} // namespace o2

#endif
