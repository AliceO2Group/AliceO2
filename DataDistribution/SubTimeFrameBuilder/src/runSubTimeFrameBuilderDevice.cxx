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

#include "Common/SubTimeFrameFileSink.h"

#include <runFairMQDevice.h>

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options)
{

  bpo::options_description lStfBuilderOptions("SubTimeFrameBuilder options", 120);

  lStfBuilderOptions.add_options()(
    o2::DataDistribution::StfBuilderDevice::OptionKeyInputChannelName,
    bpo::value<std::string>()->default_value("readout"),
    "Name of the readout channel (input).")(
    o2::DataDistribution::StfBuilderDevice::OptionKeyStandalone,
    bpo::bool_switch()->default_value(false),
    "Standalone operation. SubTimeFrames will not be forwarded to other processes.")(
    o2::DataDistribution::StfBuilderDevice::OptionKeyMaxBufferedStfs,
    bpo::value<std::int64_t>()->default_value(-1),
    "Maximum number of buffered SubTimeFrames before starting to drop data. "
    "Unlimited: -1.")(
    o2::DataDistribution::StfBuilderDevice::OptionKeyOutputChannelName,
    bpo::value<std::string>()->default_value("builder-stf-channel"),
    "Name of the output channel.")(
    o2::DataDistribution::StfBuilderDevice::OptionKeyCruCount,
    bpo::value<std::uint64_t>()->default_value(1),
    "Number of CRU Readout processes (connect to channel 'readout'[0..N-1]).")(
    o2::DataDistribution::StfBuilderDevice::OptionKeyGui,
    bpo::bool_switch()->default_value(false),
    "Enable GUI.");

  bpo::options_description lStfBuilderDplOptions("SubTimeFrameBuilder DPL options", 120);
  lStfBuilderDplOptions.add_options()(
    o2::DataDistribution::StfBuilderDevice::OptionKeyDpl,
    bpo::bool_switch()->default_value(false),
    "Enable DPL.")(
    o2::DataDistribution::StfBuilderDevice::OptionKeyDplChannelName,
    bpo::value<std::string>()->default_value("dpl-stf-channel"),
    "Name of the dpl output channel.");

  options.add(lStfBuilderOptions);
  options.add(lStfBuilderDplOptions);

  // Add options for STF file sink
  options.add(o2::DataDistribution::SubTimeFrameFileSink::getProgramOptions());
}

FairMQDevicePtr getDevice(const FairMQProgOptions& /*config*/)
{
  return new o2::DataDistribution::StfBuilderDevice();
}
