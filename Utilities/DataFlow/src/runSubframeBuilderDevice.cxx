// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "runFairMQDevice.h"
#include "DataFlow/SubframeBuilderDevice.h"

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options)
{
  options.add_options()
//    (AliceO2::DataFlow::SubframeBuilderDevice::OptionKeyDuration,
//     bpo::value<uint32_t>()->default_value(10000),
//     "Time frame duration")
    (o2::DataFlow::SubframeBuilderDevice::OptionKeySelfTriggered,
     bpo::bool_switch()->default_value(false),
     "Time frame duration")
    (o2::DataFlow::SubframeBuilderDevice::OptionKeyInputChannelName,
     bpo::value<std::string>()->default_value("input"),
     "Name of the input channel")
    (o2::DataFlow::SubframeBuilderDevice::OptionKeyOutputChannelName,
     bpo::value<std::string>()->default_value("output"),
     "Name of the output channel")
    (o2::DataFlow::SubframeBuilderDevice::OptionKeyDetector,
     bpo::value<std::string>()->default_value("TPC"),
     "Name of detector as data source");
}

FairMQDevicePtr getDevice(const FairMQProgOptions& /*config*/)
{
  return new o2::DataFlow::SubframeBuilderDevice();
}
