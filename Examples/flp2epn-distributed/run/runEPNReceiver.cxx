/**
 * runEPNReceiver.cxx
 *
 * @since 2013-01-21
 * @author D. Klein, A. Rybalchenko, M. Al-Turany, C. Kouzinopoulos
 */

#include "runFairMQDevice.h"
#include "FLP2EPNex_distributed/EPNReceiver.h"

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options)
{
  options.add_options()
    ("heartbeat-interval", bpo::value<int>()->default_value(5000), "Heartbeat interval in milliseconds")
    ("buffer-timeout", bpo::value<int>()->default_value(1000), "Buffer timeout in milliseconds")
    ("num-flps", bpo::value<int>()->required(), "Number of FLPs")
    ("test-mode", bpo::value<int>()->default_value(0), "Run in test mode");
}

FairMQDevice* getDevice(const FairMQProgOptions& config)
{
  return new AliceO2::Devices::EPNReceiver();
}
