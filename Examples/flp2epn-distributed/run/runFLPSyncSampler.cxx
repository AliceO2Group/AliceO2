/**
 * runFLPSyncSampler.cxx
 *
 * @since 2013-01-21
 * @author D. Klein, A. Rybalchenko, M. Al-Turany, C. Kouzinopoulos
 */

#include "runFairMQDevice.h"
#include "FLP2EPNex_distributed/FLPSyncSampler.h"

#include <string>

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options)
{
  options.add_options()
    ("event-rate", bpo::value<int>()->default_value(0), "Event rate limit in maximum number of events per second")
    ("max-events", bpo::value<int>()->default_value(0), "Maximum number of events to send (0 - unlimited)")
    ("store-rtt-in-file", bpo::value<int>()->default_value(0), "Store round trip time measurements in a file (1/0)")
    ("ack-chan-name", bpo::value<std::string>()->default_value("ack"), "Name of the acknowledgement channel")
    ("out-chan-name", bpo::value<std::string>()->default_value("stf1"), "Name of the output channel (sub-time frames)");
}

FairMQDevice* getDevice(const FairMQProgOptions& config)
{
  return new o2::Devices::FLPSyncSampler();
}
