#include "runFairMQDevice.h"
#include "Publishers/DataPublisherDevice.h"

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options)
{
  options.add_options()
    (AliceO2::Utilities::DataPublisherDevice::OptionKeyDataDescription,
     bpo::value<std::string>()->default_value("unspecified"),
     "default data description")
    (AliceO2::Utilities::DataPublisherDevice::OptionKeyDataOrigin,
     bpo::value<std::string>()->default_value("void"),
     "default data origin")
    (AliceO2::Utilities::DataPublisherDevice::OptionKeySubspecification,
     bpo::value<AliceO2::Utilities::DataPublisherDevice::SubSpecificationT>()->default_value(~(AliceO2::Utilities::DataPublisherDevice::SubSpecificationT)0),
     "default sub specification")
    (AliceO2::Utilities::DataPublisherDevice::OptionKeyInputChannelName,
     bpo::value<std::string>()->default_value("input"),
     "Name of the input channel")
    (AliceO2::Utilities::DataPublisherDevice::OptionKeyOutputChannelName,
     bpo::value<std::string>()->default_value("output"),
     "Name of the output channel");
}

FairMQDevicePtr getDevice(const FairMQProgOptions& /*config*/)
{
  return new AliceO2::Utilities::DataPublisherDevice();
}
