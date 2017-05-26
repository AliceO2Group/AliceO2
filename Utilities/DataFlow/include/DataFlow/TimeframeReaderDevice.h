#ifndef ALICEO2_TIMEFRAME_READER_H_
#define ALICEO2_TIMEFRAME_READER_H_

#include "O2Device/O2Device.h"
#include <fstream>

namespace o2 {
namespace DataFlow {

/// A device which writes to file the timeframes.
class TimeframeReaderDevice : public Base::O2Device
{
public:
    static constexpr const char* OptionKeyOutputChannelName = "output-channel-name";
    static constexpr const char* OptionKeyInputFileName = "input-file";

    /// Default constructor
    TimeframeReaderDevice();

    /// Default destructor
    ~TimeframeReaderDevice() override = default;

    void InitTask() final;

  protected:
    /// Overloads the ConditionalRun() method of FairMQDevice
    bool ConditionalRun() final;

    std::string      mOutChannelName;
    std::string      mInFileName;
    std::fstream     mFile;
    std::vector<std::string> mSeen;
};

} // namespace DataFlow
} // namespace o2

#endif
