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
// Based on the FT0 file RawEventData.h (almost identical except additional printing) by
//  Alla.Maevskaya@cern.ch
//  with Artur.Furs
//
#ifndef ALICEO2_FDD_RAWEVENTDATA_H_
#define ALICEO2_FDD_RAWEVENTDATA_H_

#include "FDDBase/Constants.h"
#include "Headers/RAWDataHeader.h"
#include "DataFormatsFIT/RawEventData.h"
#include <CommonDataFormat/InteractionRecord.h>
#include <Framework/Logger.h>
#include <cstring>
#include <iomanip>
#include "Rtypes.h"

namespace o2
{
namespace fdd
{

using EventHeader = o2::fit::EventHeader;
using EventData = o2::fit::EventData;
using TCMdata = o2::fit::TCMdata;
using TCMdataExtended = o2::fit::TCMdataExtended;
class RawEventData
{
 public:
  RawEventData() = default;
  EventHeader* getEventHeaderPtr() { return &mEventHeader; }
  EventData* getEventDataPtr() { return mEventData; }
  void print() const;
  void printHexEventHeader() const;
  void printHexEventData(uint64_t i) const;
  enum EEventDataBit { kNumberADC,
                       kIsDoubleEvent,
                       kIs1TimeLostEvent,
                       kIs2TimeLostEvent,
                       kIsADCinGate,
                       kIsTimeInfoLate,
                       kIsAmpHigh,
                       kIsEventInTVDC,
                       kIsTimeInfoLost };
  const static int gStartDescriptor = 0x0000000f;
  static const size_t sPayloadSizeSecondWord = 11;
  static const size_t sPayloadSizeFirstWord = 5;
  static const size_t sPayloadSize = 16;
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
      LOG(DEBUG) << "      Write PM header: nWords: " << (int)mEventHeader.nGBTWords
                 << "  orbit: " << int(mEventHeader.orbit)
                 << "  BC: " << int(mEventHeader.bc)
                 << "  size: " << result.size();
      printHexEventHeader();
      out += CRUWordSize - sPayloadSize; // Padding enabled

      for (uint64_t i = 0; i < mEventHeader.nGBTWords; ++i) {
        std::memcpy(out, &mEventData[2 * i], sPayloadSizeFirstWord);
        out += sPayloadSizeFirstWord;
        LOG(DEBUG) << "        1st word: Ch: " << std::setw(2) << mEventData[2 * i].channelID
                   << "  charge: " << std::setw(4) << mEventData[2 * i].charge
                   << "  time: " << std::setw(4) << mEventData[2 * i].time;
        std::memcpy(out, &mEventData[2 * i + 1], sPayloadSizeSecondWord);
        out += sPayloadSizeSecondWord;
        LOG(DEBUG) << "        2nd word: Ch: " << std::setw(2) << mEventData[2 * i + 1].channelID
                   << "  charge: " << std::setw(4) << mEventData[2 * i + 1].charge
                   << "  time: " << std::setw(4) << mEventData[2 * i + 1].time;

        out += CRUWordSize - sPayloadSizeSecondWord - sPayloadSizeFirstWord;
        printHexEventData(i);
      }
    } else {
      // TCM data
      std::memcpy(out, &mEventHeader, sPayloadSize);
      out += sPayloadSize;
      LOG(DEBUG) << "      Write TCM header: nWords: " << (int)mEventHeader.nGBTWords
                 << "  orbit: " << int(mEventHeader.orbit)
                 << "  BC: " << int(mEventHeader.bc)
                 << "  size: " << result.size();
      std::memcpy(out, &mTCMdata, sizeof(TCMdata));
      out += sizeof(TCMdata);
      // TODO: No TCM payload printing until the meaning of trigger bits and other flags is clarified
    }
    return result;
  }

 public:
  EventHeader mEventHeader;        //!
  EventData mEventData[NChPerMod]; //!
  TCMdata mTCMdata;                //!

  ClassDefNV(RawEventData, 1);
};

} // namespace fdd
} // namespace o2
#endif
