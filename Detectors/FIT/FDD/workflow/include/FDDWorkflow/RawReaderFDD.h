// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
//file RawReaderFDD.h class  for RAW data reading
//
// Artur.Furs
// afurs@cern.ch
//
//Main purpuse is to decode FDD data blocks and push them to DigitBlockFDD for proccess
//TODO: prepare wrappers for containers with digits and combine classes below into one template class?
#ifndef ALICEO2_FDD_RAWREADERFDD_H_
#define ALICEO2_FDD_RAWREADERFDD_H_
#include <iostream>
#include <vector>
#include <Rtypes.h>
#include "FDDRaw/RawReaderFDDBase.h"

#include "DataFormatsFDD/Digit.h"
#include "DataFormatsFDD/ChannelData.h"

#include "Framework/ProcessingContext.h"
#include "Framework/DataAllocator.h"
#include "Framework/OutputSpec.h"
#include <gsl/span>

namespace o2
{
namespace fdd
{
//Normal TCM mode
class RawReaderFDD : public RawReaderFDDBaseNorm
{
 public:
  RawReaderFDD(bool dumpData) : mDumpData(dumpData) {}
  RawReaderFDD(const RawReaderFDD&) = default;

  RawReaderFDD() = default;
  ~RawReaderFDD() = default;
  void clear()
  {
    mVecDigits.clear();
    mVecChannelData.clear();
  }
  void accumulateDigits()
  {
    getDigits(mVecDigits, mVecChannelData);
    LOG(INFO) << "Number of Digits: " << mVecDigits.size();
    LOG(INFO) << "Number of ChannelData: " << mVecChannelData.size();
    if (mDumpData) {
      DigitBlockFDD::print(mVecDigits, mVecChannelData);
    }
  }
  static void prepareOutputSpec(std::vector<o2::framework::OutputSpec>& outputSpec)
  {
    outputSpec.emplace_back(o2::header::gDataOriginFDD, "DIGITSBC", 0, o2::framework::Lifetime::Timeframe);
    outputSpec.emplace_back(o2::header::gDataOriginFDD, "DIGITSCH", 0, o2::framework::Lifetime::Timeframe);
  }
  void makeSnapshot(o2::framework::ProcessingContext& pc)
  {
    pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginFDD, "DIGITSBC", 0, o2::framework::Lifetime::Timeframe}, mVecDigits);
    pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginFDD, "DIGITSCH", 0, o2::framework::Lifetime::Timeframe}, mVecChannelData);
  }
  bool mDumpData;
  std::vector<Digit> mVecDigits;
  std::vector<ChannelData> mVecChannelData;
};

} // namespace fdd
} // namespace o2

#endif
