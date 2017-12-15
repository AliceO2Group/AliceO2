// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "SubTimeFrameBuilder/SubTimeFrameBuilderDevice.h"
#include <options/FairMQProgOptions.h>

#include "runFairMQDevice.h"

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options)
{
  options.add_options()(o2::DataDistribution::StfBuilderDevice::OptionKeyInputChannelName,
                        bpo::value<std::string>()->default_value("readout"), "Name of the readout channel (input)")(
    o2::DataDistribution::StfBuilderDevice::OptionKeyOutputChannelName,
    bpo::value<std::string>()->default_value("stf-channel"),
    "Name of the output channel")(o2::DataDistribution::StfBuilderDevice::OptionKeyCruCount,
                                  bpo::value<std::uint64_t>()->default_value(1), "Number of CRU Readout processes");
}

FairMQDevicePtr getDevice(const FairMQProgOptions& /*config*/)
{
  return new o2::DataDistribution::StfBuilderDevice();
}
