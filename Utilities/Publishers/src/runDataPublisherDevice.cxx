#include "runFairMQDevice.h"
#include "Publishers/DataPublisherDevice.h"

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options)
{
  options.add_options()
    (o2::Utilities::DataPublisherDevice::OptionKeyDataDescription,
     bpo::value<std::string>()->default_value("unspecified"),
     "default data description")
    (o2::Utilities::DataPublisherDevice::OptionKeyDataOrigin,
     bpo::value<std::string>()->default_value("void"),
     "default data origin")
    (o2::Utilities::DataPublisherDevice::OptionKeySubspecification,
     bpo::value<o2::Utilities::DataPublisherDevice::SubSpecificationT>()->default_value(~(o2::Utilities::DataPublisherDevice::SubSpecificationT)0),
     "default sub specification")
    (o2::Utilities::DataPublisherDevice::OptionKeyFileName,
     bpo::value<std::string>()->default_value(""),
     "File name")
    (o2::Utilities::DataPublisherDevice::OptionKeyInputChannelName,
     bpo::value<std::string>()->default_value("input"),
     "Name of the input channel")
    (o2::Utilities::DataPublisherDevice::OptionKeyOutputChannelName,
     bpo::value<std::string>()->default_value("output"),
     "Name of the output channel");
}

FairMQDevicePtr getDevice(const FairMQProgOptions& /*config*/)
{
  return new o2::Utilities::DataPublisherDevice();
}
