// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TimeFrameBuilder/TimeFrameBuilderDevice.h"
#include <options/FairMQProgOptions.h>

#include "Common/SubTimeFrameFileSink.h"

#include "runFairMQDevice.h"

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options)
{
  options.add_options()(
    o2::DataDistribution::TfBuilderDevice::OptionKeyInputChannelName,
    bpo::value<std::string>()->default_value("sender-stf-channel"),
    "Name of the stf channel (input).")(
    o2::DataDistribution::TfBuilderDevice::OptionKeyStandalone,
    bpo::bool_switch()->default_value(true),
    "Standalone operation. TimeFrames will not be forwarded to other processes.")(
    o2::DataDistribution::TfBuilderDevice::OptionKeyFlpNodeCount,
    bpo::value<std::uint32_t>()->default_value(std::uint32_t(0)),
    "Number of FLP nodes.")(
    o2::DataDistribution::TfBuilderDevice::OptionKeyGui,
    bpo::bool_switch()->default_value(false),
    "Enable GUI.");

  // Add options for TF file sink
  options.add(o2::DataDistribution::SubTimeFrameFileSink::getProgramOptions());
}

FairMQDevicePtr getDevice(const FairMQProgOptions& /*config*/)
{
  return new o2::DataDistribution::TfBuilderDevice();
}
