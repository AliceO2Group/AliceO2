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

/// @file   FT0DataProcessDPLSpec.cxx

#include "FT0Workflow/FT0DataProcessDPLSpec.h"

using namespace o2::framework;

namespace o2
{
namespace ft0
{
using namespace std;
void FT0DataProcessDPLSpec::init(InitContext& ic)
{
}

void FT0DataProcessDPLSpec::run(ProcessingContext& pc)
{
  LOG(INFO) << "FT0DataProcessDPLSpec running...";
  auto vecDigits = pc.inputs().get<std::vector<Digit>>("digits");
  auto vecChannelData = pc.inputs().get<std::vector<ChannelData>>("digch");
  if (mDumpEventBlocks) {
    DigitBlockFT0::print(vecDigits, vecChannelData);
  }
}

DataProcessorSpec getFT0DataProcessDPLSpec(bool dumpProcessor)
{
  std::vector<InputSpec> inputSpec;
  inputSpec.emplace_back("digits", o2::header::gDataOriginFT0, "DIGITSBC", 0, Lifetime::Timeframe);
  inputSpec.emplace_back("digch", o2::header::gDataOriginFT0, "DIGITSCH", 0, Lifetime::Timeframe);
  LOG(INFO) << "DataProcessorSpec getFT0DataProcessDPLSpec";
  return DataProcessorSpec{
    "ft0-dataprocess-dpl-flp",
    inputSpec,
    Outputs{},
    AlgorithmSpec{adaptFromTask<FT0DataProcessDPLSpec>(dumpProcessor)},
    Options{}};
}

} // namespace ft0
} // namespace o2
