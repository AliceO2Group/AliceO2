// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ReadoutEmulator/ReadoutDevice.h"
#include <options/FairMQProgOptions.h>

#include "runFairMQDevice.h"

namespace bpo = boost::program_options;

void addCustomOptions(bpo::options_description& options)
{
  options.add_options()(o2::DataDistribution::ReadoutDevice::OptionKeyOutputChannelName,
                        bpo::value<std::string>()->default_value("readout"), "Name of the readout output channel")(
    o2::DataDistribution::ReadoutDevice::OptionKeyReadoutDataRegionSize,
    bpo::value<std::size_t>()->default_value(1ULL << 30 /* 1GiB */),
    "Size of the data shm segment")(o2::DataDistribution::ReadoutDevice::OptionKeyCruId,
                                    bpo::value<size_t>()->default_value(0), "CRU ID within FLP. Starts at 0.")(
    o2::DataDistribution::ReadoutDevice::OptionKeyCruSuperpageSize,
    bpo::value<size_t>()->default_value(1ULL << 20 /* 1MiB */),
    "CRU DMA superpage size")(o2::DataDistribution::ReadoutDevice::OptionKeyCruDmaChunkSize,
                              bpo::value<size_t>()->default_value(8ULL << 10 /* 8kiB */), "CRU DMA chunk size")(
    o2::DataDistribution::ReadoutDevice::OptionKeyCruLinkCount, bpo::value<uint64_t>()->default_value(24),
    "Number of CRU links to emulate")(o2::DataDistribution::ReadoutDevice::OptionKeyCruLinkBitsPerS,
                                      bpo::value<uint64_t>()->default_value(1000000000),
                                      "Input throughput per link (bits per second)");
}

FairMQDevicePtr getDevice(const FairMQProgOptions& /*config*/)
{
  return new o2::DataDistribution::ReadoutDevice();
}
