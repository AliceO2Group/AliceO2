#include "runFairMQDevice.h"

#include "MFTReconstruction/devices/Merger.h"

using namespace o2::MFT;

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options)
{

}

FairMQDevicePtr getDevice(const FairMQProgOptions& config)
{
  return new Merger();
}
