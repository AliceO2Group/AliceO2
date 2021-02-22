// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   RawDataProcessSpec.cxx

#include "FDDWorkflow/RawDataProcessSpec.h"

using namespace o2::framework;

namespace o2
{
namespace fdd
{
using namespace std;
void RawDataProcessSpec::init(InitContext& ic)
{
}

void RawDataProcessSpec::run(ProcessingContext& pc)
{
  LOG(INFO) << "RawDataProcessSpec running...";
  auto vecDigits = pc.inputs().get<std::vector<Digit>>("digits");
  auto vecChannelData = pc.inputs().get<std::vector<ChannelData>>("digch");
  if (mDumpEventBlocks) {
    DigitBlockFDD::print(vecDigits, vecChannelData);
  }
}

DataProcessorSpec getFDDRawDataProcessSpec(bool dumpProcessor)
{
  std::vector<InputSpec> inputSpec;
  inputSpec.emplace_back("digits", o2::header::gDataOriginFDD, "DIGITSBC", 0, Lifetime::Timeframe);
  inputSpec.emplace_back("digch", o2::header::gDataOriginFDD, "DIGITSCH", 0, Lifetime::Timeframe);
  LOG(INFO) << "DataProcessorSpec getRawDataProcessSpec";
  return DataProcessorSpec{
    "fdd-dataprocess-dpl-flp",
    inputSpec,
    Outputs{},
    AlgorithmSpec{adaptFromTask<RawDataProcessSpec>(dumpProcessor)},
    Options{}};
}

} // namespace fdd
} // namespace o2
