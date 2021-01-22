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
#include "DataFormatsFDD/Digit.h"
#include "DataFormatsFDD/ChannelData.h"
#include "Headers/RAWDataHeader.h"
#include "DataFormatsFDD/LookUpTable.h"
#include <CommonDataFormat/InteractionRecord.h>
#include <Framework/Logger.h>
#include <iostream>
#include <iomanip>
#include <cstring>
#include "Rtypes.h"

namespace o2
{
namespace fdd
{
struct EventHeader {
  static constexpr size_t PayloadSize = 16;       //should be equal to 10
  static constexpr size_t PayloadPerGBTword = 16; //should be equal to 10
  static constexpr int MinNelements = 1;
  static constexpr int MaxNelements = 1;
  union {
    uint64_t word[2] = {};
    struct {
      uint64_t bc : 12;
      uint64_t orbit : 32;
      uint64_t phase : 3;
      uint64_t errorPhase : 1;
      uint64_t reservedField1 : 16;
      uint64_t reservedField2 : 8;
      uint64_t nGBTWords : 4;
      uint64_t startDescriptor : 4;
      uint64_t reservedField3 : 48;
    };
  };
  InteractionRecord getIntRec() const { return InteractionRecord{(uint16_t)bc, (uint32_t)orbit}; }
};
struct EventData {
  static constexpr size_t PayloadSize = 5;
  static constexpr size_t PayloadPerGBTword = 10;
  static constexpr int MinNelements = 1; //additional static field
  static constexpr int MaxNelements = 12;
  //
  static constexpr int BitFlagPos = 25; // position of first bit flag(numberADC)

  union {
    uint64_t word = {0};
    struct {
      int64_t time : 12;
      int64_t charge : 13;
      uint64_t numberADC : 1,
        isDoubleEvent : 1,
        is1TimeLostEvent : 1,
        is2TimeLostEvent : 1,
        isADCinGate : 1,
        isTimeInfoLate : 1,
        isAmpHigh : 1,
        isEventInTVDC : 1,
        isTimeInfoLost : 1,
        reservedField : 2,
        channelID : 4;
    };
  };
  uint64_t word_zeros = 0x0;
  static const size_t PayloadSizeSecondWord = 11;
  static const size_t PayloadSizeFirstWord = 5;
  void generateFlags()
  {
    numberADC = std::rand() % 2;
    isDoubleEvent = 0;
    is1TimeLostEvent = 0;
    is2TimeLostEvent = 1;
    isADCinGate = 0;
    isTimeInfoLate = 0;
    isAmpHigh = 0;
    isEventInTVDC = 1;
    isTimeInfoLost = 0;
  }
  uint8_t getFlagWord() const
  {
    return uint8_t(word >> BitFlagPos);
  }
};

struct TCMdata {
  static constexpr size_t PayloadSize = 16;       //should be equal to 10
  static constexpr size_t PayloadPerGBTword = 16; //should be equal to 10
  static constexpr int MinNelements = 1;
  static constexpr int MaxNelements = 1;
  union {
    uint64_t word[2] = {0};
    struct {
      uint64_t orC : 1,
        orA : 1,
        sCen : 1,
        cen : 1,
        vertex : 1,
        nChanA : 7,
        nChanC : 7;
      int64_t amplA : 18,
        amplC : 18,
        reservedField1 : 1, //56B,  PayloadSize1stWord 6
        timeA : 9,
        timeC : 9,
        reservedField2 : 46;
    };
  };
  void fillTrigger(Triggers& trg)
  {
    trg.triggersignals = ((bool)orA << Triggers::bitA) |
                         ((bool)orC << Triggers::bitC) |
                         ((bool)vertex << Triggers::bitVertex) |
                         ((bool)cen << Triggers::bitCen) |
                         ((bool)sCen << Triggers::bitSCen);
    trg.nChanA = (int8_t)nChanA;
    trg.nChanC = (int8_t)nChanC;
    trg.amplA = (int32_t)amplA;
    trg.amplC = (int32_t)amplC;
    trg.timeA = (int16_t)timeA;
    trg.timeC = (int16_t)timeC;
  }
};

class RawEventData
{
 public:
  RawEventData() = default;
  EventHeader* getEventHeaderPtr() { return &mEventHeader; }
  EventData* getEventDataPtr() { return mEventData; }
  void print();
  void printHexEventHeader();
  void printHexEventData(uint64_t i);
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
      std::memcpy(out, &mEventHeader, EventHeader::PayloadSize);
      out += EventHeader::PayloadSize;
      LOG(DEBUG) << "      Write PM header: nWords: " << (int)mEventHeader.nGBTWords
                 << "  orbit: " << int(mEventHeader.orbit)
                 << "  BC: " << int(mEventHeader.bc)
                 << "  size: " << result.size();
      printHexEventHeader();
      out += CRUWordSize - EventHeader::PayloadSize; // Padding enabled

      for (uint64_t i = 0; i < mEventHeader.nGBTWords; ++i) {
        std::memcpy(out, &mEventData[2 * i], EventData::PayloadSizeFirstWord);
        out += EventData::PayloadSizeFirstWord;
        LOG(DEBUG) << "        1st word: Ch: " << std::setw(2) << mEventData[2 * i].channelID
                   << "  charge: " << std::setw(4) << mEventData[2 * i].charge
                   << "  time: " << std::setw(4) << mEventData[2 * i].time;
        std::memcpy(out, &mEventData[2 * i + 1], EventData::PayloadSizeSecondWord);
        out += EventData::PayloadSizeSecondWord;
        LOG(DEBUG) << "        2nd word: Ch: " << std::setw(2) << mEventData[2 * i + 1].channelID
                   << "  charge: " << std::setw(4) << mEventData[2 * i + 1].charge
                   << "  time: " << std::setw(4) << mEventData[2 * i + 1].time;

        out += CRUWordSize - EventData::PayloadSizeSecondWord - EventData::PayloadSizeFirstWord;
        printHexEventData(i);
      }
    } else {
      // TCM data
      std::memcpy(out, &mEventHeader, EventHeader::PayloadSize);
      out += EventHeader::PayloadSize;
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
  EventHeader mEventHeader;
  EventData mEventData[NChPerMod];
  TCMdata mTCMdata;

  ClassDefNV(RawEventData, 1);
};

} // namespace fdd
} // namespace o2
#endif
