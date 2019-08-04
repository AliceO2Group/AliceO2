// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "runFairMQDevice.h"
#include "DataFlow/TimeframeValidatorDevice.h"

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options)
{
  // clang-format off
  options.add_options()
    (o2::data_flow::TimeframeValidatorDevice::OptionKeyInputChannelName,
     bpo::value<std::string>()->default_value("input"),
     "Name of the input channel");
  // clang-format on
}

FairMQDevicePtr getDevice(const FairMQProgOptions& /*config*/)
{
  return new o2::data_flow::TimeframeValidatorDevice();
}
