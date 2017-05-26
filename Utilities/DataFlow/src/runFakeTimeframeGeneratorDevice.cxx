#include "runFairMQDevice.h"
#include "DataFlow/FakeTimeframeGeneratorDevice.h"
#include <vector>

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options)
{
  options.add_options()
    (o2::DataFlow::FakeTimeframeGeneratorDevice::OptionKeyOutputChannelName,
     bpo::value<std::string>()->default_value("output"),
     "Name of the output channel");
  options.add_options()
    (o2::DataFlow::FakeTimeframeGeneratorDevice::OptionKeyMaxTimeframes,
     bpo::value<std::string>()->default_value("1"),
     "Number of timeframes to generate");
}

FairMQDevicePtr getDevice(const FairMQProgOptions& /*config*/)
{
  return new o2::DataFlow::FakeTimeframeGeneratorDevice();
}
