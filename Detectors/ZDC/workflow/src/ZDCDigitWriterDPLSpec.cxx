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

/// @file   FT0DigitWriterSpec.cxx

#include <vector>

#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DataFormatsZDC/ChannelData.h"
#include "DataFormatsZDC/BCData.h"
#include "DataFormatsZDC/OrbitData.h"

#include "ZDCWorkflow/ZDCDigitWriterDPLSpec.h"
using namespace o2::framework;

namespace o2
{
namespace zdc
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
DataProcessorSpec getZDCDigitWriterDPLSpec(bool mctruth, bool simVersion)
{
  std::string writerName = simVersion ? "ZDCDigitWriterSim" : "ZDCDigitWriterDec";
  std::string fnameDef = simVersion ? "zdcdigits.root" : "o2digit_zdc.root";

  using InputSpec = framework::InputSpec;
  using MakeRootTreeWriterSpec = framework::MakeRootTreeWriterSpec;
  return MakeRootTreeWriterSpec(writerName.data(),
                                fnameDef.data(),
                                "o2sim",
                                BranchDefinition<std::vector<o2::zdc::BCData>>{InputSpec{"digitBCinput", "ZDC", "DIGITSBC"}, "ZDCDigitBC"},
                                BranchDefinition<std::vector<o2::zdc::ChannelData>>{InputSpec{"digitChinput", "ZDC", "DIGITSCH"}, "ZDCDigitCh"},
                                BranchDefinition<std::vector<o2::zdc::OrbitData>>{InputSpec{"digitPDinput", "ZDC", "DIGITSPD"}, "ZDCDigitOrbit"},
                                BranchDefinition<o2::dataformats::MCTruthContainer<o2::zdc::MCLabel>>{InputSpec{"labelinput", "ZDC", "DIGITSLBL"}, "ZDCDigitLabels", mctruth ? 1 : 0})();
}

} // namespace zdc
} // namespace o2
