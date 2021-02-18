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

#ifndef ALICEO2_FT0_DIGITS2RAW_H_
#define ALICEO2_FT0_DIGITS2RAW_H_

#include "Headers/RAWDataHeader.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsFT0/RawEventData.h"
#include "DataFormatsFT0/LookUpTable.h"
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/ChannelData.h"
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
namespace ft0
{
class Digits2Raw
{

  static constexpr int Nchannels_FT0 = 208;
  static constexpr int Nchannels_PM = 12;
  static constexpr int LinkTCM = 18;
  static constexpr int NPMs = 19;
  static constexpr int GBTWordSize = 128; // with padding
  static constexpr int Max_Page_size = 8192;

 public:
  Digits2Raw() = default;
  Digits2Raw(const std::string fileRawName, std::string fileDigitsName);
  void readDigits(const std::string fileRawName, const std::string fileDigitsName);
  void convertDigits(o2::ft0::Digit bcdigits,
                     gsl::span<const ChannelData> pmchannels,
                     const o2::ft0::LookUpTable& lut,
                     o2::InteractionRecord const& mIntRecord);
  void close();
  static o2::ft0::LookUpTable linear()
  {
    std::vector<o2::ft0::Topo> lut_data(Nchannels_PM * (NPMs - 1));
    for (int link = 0; link < NPMs - 1; ++link)
      for (int mcp = 0; mcp < Nchannels_PM; ++mcp)
        lut_data[link * Nchannels_PM + mcp] = o2::ft0::Topo{link, mcp};

    return o2::ft0::LookUpTable{lut_data};
  }
  o2::raw::RawFileWriter& getWriter() { return mWriter; }
  void setRDH(o2::header::RAWDataHeader& rdh, int nlink, o2::InteractionRecord rdhIR);

 private:
  std::ofstream mFileDest;
  o2::ft0::RawEventData mRawEventData;
  const o2::raw::HBFUtils& mSampler = o2::raw::HBFUtils::Instance();
  o2::ft0::Triggers mTriggers;
  o2::raw::RawFileWriter mWriter;
  /////////////////////////////////////////////////

  ClassDefNV(Digits2Raw, 1);
};

} // namespace ft0
} // namespace o2
#endif
