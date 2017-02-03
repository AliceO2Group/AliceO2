#include "runFairMQDevice.h"
#include "DataFlow/HeartbeatSampler.h"

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options)
{
  options.add_options()
    ("verbosity",bpo::value<int>()->default_value(0), "verbosity level");
}

FairMQDevicePtr getDevice(const FairMQProgOptions& /*config*/)
{
  return new AliceO2::DataFlow::HeartbeatSampler();
}
