#include "runFairMQDevice.h"
#include "flp2epn/O2EPNex.h"

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options)
{
}

FairMQDevice* getDevice(const FairMQProgOptions& config)
{
  return new O2EPNex();
}
