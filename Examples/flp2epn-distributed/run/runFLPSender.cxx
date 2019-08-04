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
 * runFLPSender.cxx
 *
 * @since 2013-01-21
 * @author D. Klein, A. Rybalchenko, M. Al-Turany, C. Kouzinopoulos
 */

#include "runFairMQDevice.h"
#include "FLP2EPNex_distributed/FLPSender.h"

#include <string>

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options)
{
  // clang-format off
  options.add_options()
    ("flp-index", bpo::value<int>()->default_value(0), "FLP Index (for debugging in test mode)")
    ("event-size", bpo::value<int>()->default_value(1000), "Event size in bytes (test mode)")
    ("num-epns", bpo::value<int>()->required(), "Number of EPNs")
    ("test-mode", bpo::value<int>()->default_value(0), "Run in test mode")
    ("send-offset", bpo::value<int>()->default_value(0), "Offset for staggered sending")
    ("send-delay", bpo::value<int>()->default_value(8), "Delay for staggered sending")
    ("in-chan-name", bpo::value<std::string>()->default_value("stf1"), "Name of the input channel (sub-time frames)")
    ("out-chan-name", bpo::value<std::string>()->default_value("stf2"), "Name of the output channel (sub-time frames)");
  // clang-format on
}

FairMQDevice* getDevice(const FairMQProgOptions& config)
{
  return new o2::devices::FLPSender();
}
