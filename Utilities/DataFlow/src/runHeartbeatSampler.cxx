#include "runFairMQDevice.h"
#include "DataFlow/HeartbeatSampler.h"

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options)
{
  options.add_options()
    (AliceO2::DataFlow::HeartbeatSampler::OptionKeyPeriod,
     bpo::value<uint32_t>()->default_value(1000000000),
     "sampling period")
    (AliceO2::DataFlow::HeartbeatSampler::OptionKeyOutputChannelName,
     bpo::value<std::string>()->default_value("output"),
     "Name of the output channel");
}

FairMQDevicePtr getDevice(const FairMQProgOptions& /*config*/)
{
  return new AliceO2::DataFlow::HeartbeatSampler();
}
