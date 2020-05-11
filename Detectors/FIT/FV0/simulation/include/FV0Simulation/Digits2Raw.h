// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Digits2Raw.h
/// \brief converts digits to raw format
// Alla.Maevskaya@cern.ch

#ifndef ALICEO2_FV0_DIGITS2RAW_H_
#define ALICEO2_FV0_DIGITS2RAW_H_

#include "Headers/RAWDataHeader.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsFV0/RawEventData.h"
#include "DataFormatsFV0/LookUpTable.h"
#include "DataFormatsFV0/ChannelData.h"
#include "DataFormatsFV0/BCData.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsRaw/RawFileWriter.h"
#include <TStopwatch.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <array>
#include <string>
#include <bitset>
#include <vector>
#include <gsl/span>

namespace o2
{
namespace fv0
{
class Digits2Raw
{
  static constexpr int LinkTCM = 4;
  static constexpr int GBTWordSize = 128; // with padding
  static constexpr int Max_Page_size = 8 * 1024;

 public:
  Digits2Raw() = default;
  Digits2Raw(const std::string& outDir, const std::string& fileDigitsName);
  void readDigits(const std::string& outDir, const std::string& fileDigitsName);
  void convertDigits(o2::fv0::BCData bcdigits,
                     gsl::span<const ChannelData> pmchannels,
                     const o2::fv0::LookUpTable& lut,
                     const o2::InteractionRecord& mIntRecord);

  static o2::fv0::LookUpTable linear()
  {
    LOG(INFO) << "<============Digits2Raw:linear ===============>" << std::endl;
    std::vector<o2::fv0::Topo> lut_data(N_PM_CHANNELS * (N_PMS - 1));
    for (int link = 0; link < N_PMS - 1; ++link)
      for (int mcp = 0; mcp < N_PM_CHANNELS; ++mcp) {
        lut_data[link * N_PM_CHANNELS + mcp] = o2::fv0::Topo{link, mcp};
      }

    return o2::fv0::LookUpTable{lut_data};
  }
  o2::raw::RawFileWriter& getWriter() { return mWriter; }
  void setFilePerLink(bool v) { mOutputPerLink = v; }
  bool getFilePerLink() const { return mOutputPerLink; }

  void setVerbosity(int v) { mVerbosity = v; }
  int getVerbosity() const { return mVerbosity; }

 private:
  EventHeader makeGBTHeader(int link, o2::InteractionRecord const& mIntRecord);
  RawEventData mRawEventData;
  const o2::raw::HBFUtils& mSampler = o2::raw::HBFUtils::Instance();
  o2::raw::RawFileWriter mWriter{"FV0"};
  bool mOutputPerLink = false;
  int mVerbosity = 0;
  uint32_t mLinkID = 0;
  uint16_t mCruID = 0;
  uint32_t mEndPointID = 0;
  uint64_t mFeeID = 0;
  /////////////////////////////////////////////////

  ClassDefNV(Digits2Raw, 1);
};

} // namespace fv0
} // namespace o2
#endif
