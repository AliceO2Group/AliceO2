#include "runFairMQDevice.h"
#include "DataFlow/TimeframeWriterDevice.h"

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options)
{
  options.add_options()
    (o2::DataFlow::TimeframeWriterDevice::OptionKeyInputChannelName,
     bpo::value<std::string>()->default_value("input"),
     "Name of the input channel");
  options.add_options()
    (o2::DataFlow::TimeframeWriterDevice::OptionKeyOutputFileName,
     bpo::value<std::string>()->default_value("data.o2tf"),
     "Name of the input channel");
  options.add_options()
    (o2::DataFlow::TimeframeWriterDevice::OptionKeyMaxFiles,
     bpo::value<size_t>()->default_value(1),
     "Maximum number of files to write");
  options.add_options()
    (o2::DataFlow::TimeframeWriterDevice::OptionKeyMaxTimeframesPerFile,
     bpo::value<size_t>()->default_value(1),
     "Maximum number of timeframes per file");
  options.add_options()
    (o2::DataFlow::TimeframeWriterDevice::OptionKeyMaxFileSize,
     bpo::value<size_t>()->default_value(-1),
     "Maximum size per file");
}

FairMQDevicePtr getDevice(const FairMQProgOptions& /*config*/)
{
  return new o2::DataFlow::TimeframeWriterDevice();
}
