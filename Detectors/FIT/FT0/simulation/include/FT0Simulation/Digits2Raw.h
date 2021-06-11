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
#include "FT0Base/Geometry.h"
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

  static constexpr int Nchannels_PM = 12;
  static constexpr int NPMs = 20;
  static constexpr int GBTWordSize = 128; // with padding
  static constexpr int Max_Page_size = 8192;
  static constexpr int Nchannels_FT0 = o2::ft0::Geometry::Nchannels;

 public:
  Digits2Raw() = default;
  Digits2Raw(const std::string& outDir, const std::string& fileDigitsName);
  void readDigits(const std::string& outDir, const std::string& fileDigitsName);
  void convertDigits(o2::ft0::Digit bcdigits,
                     gsl::span<const ChannelData> pmchannels,
                     const o2::ft0::LookUpTable& lut,
                     o2::InteractionRecord const& mIntRecord);

  o2::raw::RawFileWriter& getWriter() { return mWriter; }

  void setFilePerLink(bool v) { mOutputPerLink = v; }
  bool getFilePerLink() const { return mOutputPerLink; }

  void setVerbosity(int v) { mVerbosity = v; }
  int getVerbosity() const { return mVerbosity; }
  int carryOverMethod(const header::RDHAny* rdh, const gsl::span<char> data,
                      const char* ptr, int maxSize, int splitID,
                      std::vector<char>& trailer, std::vector<char>& header) const;

 private:
  EventHeader makeGBTHeader(int link, o2::InteractionRecord const& mIntRecord);

  o2::ft0::RawEventData mRawEventData;
  const o2::raw::HBFUtils& mSampler = o2::raw::HBFUtils::Instance();
  o2::ft0::Triggers mTriggers;
  o2::raw::RawFileWriter mWriter{"FT0"};
  bool mOutputPerLink = false;
  int mVerbosity = 0;
  uint32_t mLinkID = 0;
  uint16_t mCruID = 0;
  uint32_t mEndPointID = 0;
  uint64_t mFeeID = 0;
  int mLinkTCM;

  /////////////////////////////////////////////////

  ClassDefNV(Digits2Raw, 2);
};

} // namespace ft0
} // namespace o2
#endif
