#include "runFairMQDevice.h"
#include "DataFlow/TimeframeValidatorDevice.h"

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options)
{
  options.add_options()
    (o2::DataFlow::TimeframeValidatorDevice::OptionKeyInputChannelName,
     bpo::value<std::string>()->default_value("input"),
     "Name of the input channel");
}

FairMQDevicePtr getDevice(const FairMQProgOptions& /*config*/)
{
  return new o2::DataFlow::TimeframeValidatorDevice();
}
