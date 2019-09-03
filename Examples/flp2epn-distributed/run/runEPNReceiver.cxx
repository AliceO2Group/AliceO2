// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/**
 * runEPNReceiver.cxx
 *
 * @since 2013-01-21
 * @author D. Klein, A. Rybalchenko, M. Al-Turany, C. Kouzinopoulos
 */

#include "runFairMQDevice.h"
#include "FLP2EPNex_distributed/EPNReceiver.h"

#include <string>

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options)
{
  // clang-format off
  options.add_options()
    ("buffer-timeout", bpo::value<int>()->default_value(1000), "Buffer timeout in milliseconds")
    ("num-flps", bpo::value<int>()->required(), "Number of FLPs")
    ("test-mode", bpo::value<int>()->default_value(0), "Run in test mode")
    ("in-chan-name", bpo::value<std::string>()->default_value("stf2"), "Name of the input channel (sub-time frames)")
    ("out-chan-name", bpo::value<std::string>()->default_value("tf"), "Name of the output channel (time frames)")
    ("ack-chan-name", bpo::value<std::string>()->default_value("ack"), "Name of the acknowledgement channel");
  // clang-format on
}

FairMQDevice* getDevice(const FairMQProgOptions& config)
{
  return new o2::devices::EPNReceiver();
}
