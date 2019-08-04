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
#include "DataFlow/SubframeBuilderDevice.h"

namespace bpo = boost::program_options;

constexpr uint32_t o2::data_flow::SubframeBuilderDevice::DefaultOrbitDuration;
constexpr uint32_t o2::data_flow::SubframeBuilderDevice::DefaultOrbitsPerTimeframe;

void addCustomOptions(bpo::options_description& options)
{
  // clang-format off
  options.add_options()
    (o2::data_flow::SubframeBuilderDevice::OptionKeyOrbitDuration,
     bpo::value<uint32_t>()->default_value(o2::data_flow::SubframeBuilderDevice::DefaultOrbitDuration),
     "Orbit duration")
    (o2::data_flow::SubframeBuilderDevice::OptionKeyOrbitsPerTimeframe,
     bpo::value<uint32_t>()->default_value(o2::data_flow::SubframeBuilderDevice::DefaultOrbitsPerTimeframe),
     "Orbits per timeframe")
    (o2::data_flow::SubframeBuilderDevice::OptionKeyInputChannelName,
     bpo::value<std::string>()->default_value("input"),
     "Name of the input channel")
    (o2::data_flow::SubframeBuilderDevice::OptionKeyOutputChannelName,
     bpo::value<std::string>()->default_value("output"),
     "Name of the output channel")
    (o2::data_flow::SubframeBuilderDevice::OptionKeyDetector,
     bpo::value<std::string>()->default_value("TPC"),
     "Name of detector as data source")
    (o2::data_flow::SubframeBuilderDevice::OptionKeyFLPId,
     bpo::value<size_t>()->default_value(0),
     "ID of the FLP used as data source")
    (o2::data_flow::SubframeBuilderDevice::OptionKeyStripHBF,
     bpo::bool_switch()->default_value(false),
     "Strip HBH & HBT from each HBF");
  // clang-format on
}

FairMQDevicePtr getDevice(const FairMQProgOptions& /*config*/)
{
  return new o2::data_flow::SubframeBuilderDevice();
}
