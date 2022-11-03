// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   ZDCRecoWriterDPLSpec.cxx

#include <vector>

#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DataFormatsZDC/BCRecData.h"
#include "DataFormatsZDC/ZDCEnergy.h"
#include "DataFormatsZDC/ZDCTDCData.h"
#include "DataFormatsZDC/ZDCWaveform.h"
#include "ZDCWorkflow/ZDCRecoWriterDPLSpec.h"

using namespace o2::framework;

namespace o2
{
namespace zdc
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
// Unused arguments: bool mctruth, bool simVersion
DataProcessorSpec getZDCRecoWriterDPLSpec(std::string fname)
{
  std::string writerName = "ZDCRecoWriter";

  using InputSpec = framework::InputSpec;
  using MakeRootTreeWriterSpec = framework::MakeRootTreeWriterSpec;
  return MakeRootTreeWriterSpec(writerName.data(),
                                fname.data(),
                                "o2rec",
                                BranchDefinition<std::vector<o2::zdc::BCRecData>>{InputSpec{"bcrec", "ZDC", "BCREC"}, "ZDCRecBC"},
                                BranchDefinition<std::vector<o2::zdc::ZDCEnergy>>{InputSpec{"energy", "ZDC", "ENERGY"}, "ZDCRecE"},
                                BranchDefinition<std::vector<o2::zdc::ZDCTDCData>>{InputSpec{"tdcdata", "ZDC", "TDCDATA"}, "ZDCRecTDC"},
                                BranchDefinition<std::vector<uint16_t>>{InputSpec{"info", "ZDC", "INFO"}, "ZDCRecInfo"},
                                BranchDefinition<std::vector<o2::zdc::ZDCWaveform>>{InputSpec{"wave", "ZDC", "WAVE"}, "ZDCWaveform"})();
}

} // namespace zdc
} // namespace o2
