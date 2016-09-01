/**
 * runFLPSender.cxx
 *
 * @since 2013-01-21
 * @author D. Klein, A. Rybalchenko, M. Al-Turany, C. Kouzinopoulos
 */

#include "runFairMQDevice.h"
#include "FLP2EPNex_distributed/FLPSender.h"

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options)
{
  options.add_options()
    ("flp-index", bpo::value<int>()->default_value(0), "FLP Index (for debugging in test mode)")
    ("event-size", bpo::value<int>()->default_value(1000), "Event size in bytes (test mode)")
    ("num-epns", bpo::value<int>()->required(), "Number of EPNs")
    ("heartbeat-timeout", bpo::value<int>()->default_value(20000), "Heartbeat timeout in milliseconds")
    ("test-mode", bpo::value<int>()->default_value(0), "Run in test mode")
    ("send-offset", bpo::value<int>()->default_value(0), "Offset for staggered sending")
    ("send-delay", bpo::value<int>()->default_value(8), "Delay for staggered sending");
}

FairMQDevice* getDevice(const FairMQProgOptions& config)
{
  return new AliceO2::Devices::FLPSender();
}
