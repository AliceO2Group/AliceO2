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
//
// Based on the FT0 file RawEventData.h
//  Alla.Maevskaya@cern.ch
//  with Artur.Furs
//
#ifndef ALICEO2_FV0_RAWEVENTDATA_H_
#define ALICEO2_FV0_RAWEVENTDATA_H_

#include "FV0Base/Constants.h"
#include "DataFormatsFIT/RawEventData.h"
#include <Framework/Logger.h>
#include "Rtypes.h"

namespace o2
{
namespace fv0
{
using EventHeader = o2::fit::EventHeader;
using EventData = o2::fit::EventData;
using TCMdata = o2::fit::TCMdata;
using TCMdataExtended = o2::fit::TCMdataExtended;
class RawEventData
{
 public:
  RawEventData() = default;
  void print() const;
  void printHexEventHeader() const;
  void printHexEventData(uint64_t i) const;
  const static int gStartDescriptor = 0x0000000f;
  static const size_t sPayloadSizeSecondWord = 11;
  static const size_t sPayloadSizeFirstWord = 5;
  static constexpr size_t sPayloadSize = 16;
  int size() const
  {
    return 1 + mEventHeader.nGBTWords; // EventHeader + EventData size
  }

  std::vector<char> to_vector(bool tcm)
  {
    constexpr int CRUWordSize = 16;

    std::vector<char> result(size() * CRUWordSize);
    char* out = result.data();
    if (!tcm) {
      std::memcpy(out, &mEventHeader, sPayloadSize);
      out += sPayloadSize;
      LOG(debug) << "write PM header words " << (int)mEventHeader.nGBTWords << "  orbit: " << int(mEventHeader.orbit) << " bc " << int(mEventHeader.bc) << " out " << result.size();
      printHexEventHeader();
      out += CRUWordSize - sPayloadSize;

      for (int i = 0; i < mEventHeader.nGBTWords; ++i) {
        std::memcpy(out, &mEventData[2 * i], sPayloadSizeFirstWord);
        LOG(debug) << " 1st word " << mEventData[2 * i].channelID << " charge " << mEventData[2 * i].charge << " time " << mEventData[2 * i].time << " out " << result.size();
        out += sPayloadSizeFirstWord;
        std::memcpy(out, &mEventData[2 * i + 1], sPayloadSizeSecondWord);
        out += sPayloadSizeSecondWord;
        LOG(debug) << " 2nd word " << mEventData[2 * i + 1].channelID << " charge " << mEventData[2 * i + 1].charge << " time " << mEventData[2 * i + 1].time << " out " << result.size();
        out += CRUWordSize - sPayloadSizeSecondWord - sPayloadSizeFirstWord;
        printHexEventData(i);
      }
    } else {
      // TCM data
      std::memcpy(out, &mEventHeader, sPayloadSize);
      out += sPayloadSize;
      LOG(debug) << "write TCM header words " << (int)mEventHeader.nGBTWords << " orbit " << int(mEventHeader.orbit) << " bc " << int(mEventHeader.bc) << " out " << result.size();
      std::memcpy(out, &mTCMdata, sizeof(TCMdata));
      out += sizeof(TCMdata);
      LOG(debug) << "write TCM words " << sizeof(mTCMdata) << " orbit " << int(mEventHeader.orbit) << " bc " << int(mEventHeader.bc) << " out " << result.size() << " sum time A " << mTCMdata.timeA;
    }

    return result;
  }

 public:
  EventHeader mEventHeader;                        //!
  EventData mEventData[Constants::nChannelsPerPm]; //!
  TCMdata mTCMdata;                                //!

  ClassDefNV(RawEventData, 1);
};
} // namespace fv0
} // namespace o2
#endif
