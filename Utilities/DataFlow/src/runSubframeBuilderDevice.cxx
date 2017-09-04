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

constexpr uint32_t o2::DataFlow::SubframeBuilderDevice::DefaultOrbitDuration;
constexpr uint32_t o2::DataFlow::SubframeBuilderDevice::DefaultOrbitsPerTimeframe;

void addCustomOptions(bpo::options_description& options)
{
  options.add_options()
    (o2::DataFlow::SubframeBuilderDevice::OptionKeyOrbitDuration,
     bpo::value<uint32_t>()->default_value(o2::DataFlow::SubframeBuilderDevice::DefaultOrbitDuration),
     "Orbit duration")
    (o2::DataFlow::SubframeBuilderDevice::OptionKeyOrbitsPerTimeframe,
     bpo::value<uint32_t>()->default_value(o2::DataFlow::SubframeBuilderDevice::DefaultOrbitsPerTimeframe),
     "Orbits per timeframe")
    (o2::DataFlow::SubframeBuilderDevice::OptionKeyInputChannelName,
     bpo::value<std::string>()->default_value("input"),
     "Name of the input channel")
    (o2::DataFlow::SubframeBuilderDevice::OptionKeyOutputChannelName,
     bpo::value<std::string>()->default_value("output"),
     "Name of the output channel")
    (o2::DataFlow::SubframeBuilderDevice::OptionKeyDetector,
     bpo::value<std::string>()->default_value("TPC"),
     "Name of detector as data source")
    (o2::DataFlow::SubframeBuilderDevice::OptionKeyFLPId,
     bpo::value<size_t>()->default_value(0),
     "ID of the FLP used as data source")
    (o2::DataFlow::SubframeBuilderDevice::OptionKeyStripHBF,
     bpo::bool_switch()->default_value(false),
     "Strip HBH & HBT from each HBF");
}

FairMQDevicePtr getDevice(const FairMQProgOptions& /*config*/)
{
  return new o2::DataFlow::SubframeBuilderDevice();
}
