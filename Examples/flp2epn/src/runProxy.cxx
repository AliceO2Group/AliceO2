#include "runFairMQDevice.h"
#include <devices/FairMQProxy.h>

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options)
{
  options.add_options()
    ("num-content", bpo::value<int>()->default_value(1000), "Number of data entries in one message");
}

FairMQDevice* getDevice(const FairMQProgOptions& config)
{
  return new FairMQProxy();
}
