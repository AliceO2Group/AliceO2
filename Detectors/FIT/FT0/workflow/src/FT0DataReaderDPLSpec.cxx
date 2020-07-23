// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   FT0DataReaderDPLSpec.cxx

#include "FT0Workflow/FT0DataReaderDPLSpec.h"

using namespace o2::framework;

namespace o2
{
namespace ft0
{
using namespace std;
template <bool IsExtendedMode>
void FT0DataReaderDPLSpec<IsExtendedMode>::init(InitContext& ic)

{
}
template <bool IsExtendedMode>
void FT0DataReaderDPLSpec<IsExtendedMode>::run(ProcessingContext& pc)

{
  DPLRawParser parser(pc.inputs());
  mVecDigits.clear();
  mVecChannelData.clear();
  LOG(INFO) << "FT0DataReaderDPLSpec";
  uint64_t count = 0;
  for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
    //Proccessing each page
    count++;
    auto rdhPtr = it.get_if<o2::header::RAWDataHeader>();
    gsl::span<const uint8_t> payload(it.data(), it.size());
    mRawReaderFT0.proccess(rdhPtr->linkID, payload);
  }

  mRawReaderFT0.popDigits(mVecDigits, mVecChannelData);
  LOG(INFO) << "Number of Digits: " << mVecDigits.size();
  LOG(INFO) << "Number of ChannelData: " << mVecChannelData.size();
  if (mDumpEventBlocks)
    DigitBlockFT0::print(mVecDigits, mVecChannelData);
  pc.outputs().snapshot(Output{o2::header::gDataOriginFT0, "DIGITSBC", 0, Lifetime::Timeframe}, mVecDigits);
  pc.outputs().snapshot(Output{o2::header::gDataOriginFT0, "DIGITSCH", 0, Lifetime::Timeframe}, mVecChannelData);
}
AlgorithmSpec getAlgorithmSpec(bool dumpReader, bool isExtendedMode)
{
  if (isExtendedMode) {
    LOG(INFO) << "TCM mode: extended, additional TCM data blocks(TCMdataExtended) will be in payload from TCM!";
    return adaptFromTask<FT0DataReaderDPLSpec<true>>(dumpReader, isExtendedMode);
  }
  LOG(INFO) << "TCM mode: normal, only TCMdata will be in payload from TCM.";
  return adaptFromTask<FT0DataReaderDPLSpec<false>>(dumpReader, isExtendedMode);
}

DataProcessorSpec getFT0DataReaderDPLSpec(bool dumpReader, bool isExtendedMode)
{
  std::vector<OutputSpec> outputSpec;
  outputSpec.emplace_back(o2::header::gDataOriginFT0, "DIGITSBC", 0, Lifetime::Timeframe);
  outputSpec.emplace_back(o2::header::gDataOriginFT0, "DIGITSCH", 0, Lifetime::Timeframe);
  LOG(INFO) << "DataProcessorSpec getFT0DataReaderDPLSpec";
  return DataProcessorSpec{
    "ft0-datareader-dpl-flp",
    o2::framework::select("TF:FT0/RAWDATA"),
    outputSpec,
    getAlgorithmSpec(dumpReader, isExtendedMode),
    Options{}};
}

} // namespace ft0
} // namespace o2
