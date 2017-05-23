#include "runFairMQDevice.h"
#include "DataFlow/TimeframeReaderDevice.h"

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options)
{
  options.add_options()
    (o2::DataFlow::TimeframeReaderDevice::OptionKeyOutputChannelName,
     bpo::value<std::string>()->default_value("output"),
     "Name of the output channel");
  options.add_options()
    (o2::DataFlow::TimeframeReaderDevice::OptionKeyInputFileName,
     bpo::value<std::string>()->default_value("data.o2tf"),
     "Name of the input file");
}

FairMQDevicePtr getDevice(const FairMQProgOptions& /*config*/)
{
  return new o2::DataFlow::TimeframeReaderDevice();
}
