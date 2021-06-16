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

/// \file Digits2Raw.h
/// \brief converts digits to raw format
/// \author Maciej.Slupecki@cern.ch
// based on FT0

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
#include <FairLogger.h>
#include <TStopwatch.h>
#include <iostream>
#include <string>
#include <vector>
#include <gsl/span>

namespace o2
{
namespace fv0
{
class Digits2Raw
{
 public:
  Digits2Raw() = default;
  void readDigits(const std::string& outDir, const std::string& fileDigitsName);
  void convertDigits(o2::fv0::BCData bcdigits,
                     gsl::span<const ChannelData> pmchannels,
                     const o2::fv0::LookUpTable& lut);

  o2::raw::RawFileWriter& getWriter() { return mWriter; }
  void setFilePerLink(bool v) { mOutputPerLink = v; }
  bool getFilePerLink() const { return mOutputPerLink; }

 private:
  static constexpr uint32_t sTcmLink = 4;
  static constexpr uint16_t sCruId = 0;
  static constexpr uint32_t sEndPointId = sCruId;

  void makeGBTHeader(EventHeader& eventHeader, int link, o2::InteractionRecord const& mIntRecord);
  void fillSecondHalfWordAndAddData(int iChannelPerLink, int prevPmLink, const o2::InteractionRecord& ir);
  RawEventData mRawEventData;
  o2::raw::RawFileWriter mWriter{"FV0"};
  bool mOutputPerLink = false;
  /////////////////////////////////////////////////

  ClassDefNV(Digits2Raw, 1);
};

} // namespace fv0
} // namespace o2
#endif
