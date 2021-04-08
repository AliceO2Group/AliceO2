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
//file RawEventData.h class  for RAW data format
//Alla.Maevskaya@cern.ch
// with Artur.Furs
//
#ifndef ALICEO2_FIT_RAWEVENTDATA_H_
#define ALICEO2_FIT_RAWEVENTDATA_H_

#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/ChannelData.h"

#include "Headers/RAWDataHeader.h"
#include "DataFormatsFT0/LookUpTable.h"
#include <CommonDataFormat/InteractionRecord.h>
#include <Framework/Logger.h>
#include <utility>
#include <cstring>
#include "Rtypes.h"

namespace o2
{
namespace ft0
{
constexpr int Nchannels_FT0 = 208;
constexpr int Nchannels_PM = 12;
constexpr int NPMs = 20;
constexpr size_t sizeWord = 16;

struct EventHeader {
  static constexpr size_t PayloadSize = 16;       //should be equal to 10
  static constexpr size_t PayloadPerGBTword = 16; //should be equal to 10
  static constexpr size_t MinNelements = 1;
  static constexpr size_t MaxNelements = 1;
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

  void print() const;
};

struct EventData {
  static constexpr size_t PayloadSize = 5;
  static constexpr size_t PayloadPerGBTword = 10;
  static constexpr size_t MinNelements = 1; //additional static field
  static constexpr size_t MaxNelements = 12;
  //
  static constexpr int BitFlagPos = 25; // position of first bit flag(numberADC)

  union {
    uint64_t word = {0};
    struct {
      int64_t time : 12;
      int64_t charge : 13;
      uint64_t numberADC : 1, //25 bit
        isDoubleEvent : 1,
        isTimeInfoNOTvalid : 1,
        isCFDinADCgate : 1,
        isTimeInfoLate : 1,
        isAmpHigh : 1,
        isEventInTVDC : 1,
        isTimeInfoLost : 1,
        reservedField : 3,
        channelID : 4;
    };
  };
  void generateFlags()
  {
    numberADC = std::rand() % 2;
    isDoubleEvent = 0;
    isTimeInfoNOTvalid = 0;
    isCFDinADCgate = 1;
    isTimeInfoLate = 0;
    isAmpHigh = 0;
    isEventInTVDC = 1;
    isTimeInfoLost = 0;
  }
  uint8_t getFlagWord() const
  {
    return uint8_t(word >> BitFlagPos);
  }
  void print() const;

  //temporary, this method should be in ChannelData struct, TODO
  void fillChannelData(ChannelData& channelData) const
  {
    channelData.ChainQTC = getFlagWord();
  }

  uint64_t word_zeros = 0x0;                      //to remove
  static const size_t PayloadSizeSecondWord = 11; //to remove
  static const size_t PayloadSizeFirstWord = 5;   //to remove
};

struct TCMdata {
  static constexpr size_t PayloadSize = 16;       //should be equal to 10
  static constexpr size_t PayloadPerGBTword = 16; //should be equal to 10
  static constexpr size_t MinNelements = 1;
  static constexpr size_t MaxNelements = 1;
  uint64_t orA : 1,     // 0 bit (0 byte)
    orC : 1,            //1 bit
    sCen : 1,           //2 bit
    cen : 1,            //3 bit
    vertex : 1,         //4 bit
    laser : 1,          //5 bit
    reservedField1 : 2, //6 bit
    nChanA : 7,         //8 bit(1 byte)
    reservedField2 : 1, //15 bit
    nChanC : 7,         //16 bit(2 byte)
    reservedField3 : 1; // 23 bit
  int64_t amplA : 17,   //24 bit (3 byte)
    reservedField4 : 1, //41 bit
    amplC : 17,         //42 bit.
    reservedField5 : 1, //59 bit.
    //in standard case(without __atribute__((packed)) macros, or packing by using union)
    //here will be empty 4 bits, end next field("timeA") will start from 64 bit.
    timeA : 9,           //60 bit
    reservedField6 : 1,  //69 bit
    timeC : 9,           //70 bit
    reservedField7 : 1,  //79 bit
    reservedField8 : 48; //80 bit

  void print() const;

  //temporary, this method should be in Triggers struct, TODO
  void fillTrigger(Triggers& trg)
  {
    trg.triggersignals = ((bool)orA << Triggers::bitA) |
                         ((bool)orC << Triggers::bitC) |
                         ((bool)vertex << Triggers::bitVertex) |
                         ((bool)cen << Triggers::bitCen) |
                         ((bool)sCen << Triggers::bitSCen) |
                         ((bool)laser << Triggers::bitLaser);
    trg.nChanA = (int8_t)nChanA;
    trg.nChanC = (int8_t)nChanC;
    trg.amplA = (int32_t)amplA;
    trg.amplC = (int32_t)amplC;
    trg.timeA = (int16_t)timeA;
    trg.timeC = (int16_t)timeC;
  }
} __attribute__((__packed__));

struct TCMdataExtended {
  static constexpr size_t PayloadSize = 4;
  static constexpr size_t PayloadPerGBTword = 10;
  static constexpr size_t MinNelements = 0;
  static constexpr size_t MaxNelements = 20;
  union {
    uint32_t word[1] = {};
    uint32_t triggerWord;
  };

  void print() const;
};

class RawEventData
{
 public:
  RawEventData() = default;
  void print() const;
  const static int gStartDescriptor = 0x0000000f;

  int size() const
  {
    return 1                         // EventHeader
           + mEventHeader.nGBTWords; // EventData
  }

  std::vector<char> to_vector(bool tcm)
  {
    constexpr int CRUWordSize = 16;
    const char padding[CRUWordSize] = {0};

    std::vector<char> result(size() * CRUWordSize);
    char* out = result.data();
    if (!tcm) {
      std::memcpy(out, &mEventHeader, EventHeader::PayloadSize);
      out += EventHeader::PayloadSize;
      LOG(DEBUG) << "write header words " << (int)mEventHeader.nGBTWords << " orbit " << int(mEventHeader.orbit) << " bc " << int(mEventHeader.bc) << " out " << result.size() ;
      if (mIsPadded) {
        out += CRUWordSize - EventHeader::PayloadSize;
      }
      for (int i = 0; i < mEventHeader.nGBTWords; ++i) {
        std::memcpy(out, &mEventData[2 * i], EventData::PayloadSizeFirstWord);
        LOG(DEBUG) << " 1st word " << mEventData[2 * i].channelID << " charge " << mEventData[2 * i].charge << " time " << mEventData[2 * i].time << " out " << result.size();
        out += EventData::PayloadSizeFirstWord;
        std::memcpy(out, &mEventData[2 * i + 1], EventData::PayloadSizeSecondWord);
        out += EventData::PayloadSizeSecondWord;
        LOG(DEBUG) << " 2nd word " << mEventData[2 * i + 1].channelID << " charge " << mEventData[2 * i + 1].charge << " time " << mEventData[2 * i + 1].time << " out " << result.size();
        if (mIsPadded) {
          out += CRUWordSize - EventData::PayloadSizeSecondWord - EventData::PayloadSizeFirstWord;
        }
      }
    } else {
      // TCM data
      std::memcpy(out, &mEventHeader, EventHeader::PayloadSize);
      out += EventHeader::PayloadSize;
      LOG(DEBUG) << "write TCM header words " << (int)mEventHeader.nGBTWords << " orbit " << int(mEventHeader.orbit) << " bc " << int(mEventHeader.bc) << " out " << result.size();
      std::memcpy(out, &mTCMdata, sizeof(TCMdata));
      out += sizeof(TCMdata);
      LOG(DEBUG) << "write TCM words " << sizeof(mTCMdata) << " orbit " << int(mEventHeader.orbit) << " bc " << int(mEventHeader.bc) << " out " << result.size() << " sum time A " << mTCMdata.timeA;
    }

    return result;
  }
  void setIsPadded(bool isPadding128)
  {
    mIsPadded = isPadding128;
  }

 public:
  EventHeader mEventHeader;
  EventData mEventData[Nchannels_PM];
  TCMdata mTCMdata;
  bool mIsPadded = true;
  /////////////////////////////////////////////////
  ClassDefNV(RawEventData, 2);
};
std::ostream& operator<<(std::ostream& stream, const RawEventData& data);

class DataPageWriter
{
  std::vector<char> mBuffer;
  int mNpacketsInBuffer = 0;
  std::vector<std::vector<char>> mPages;
  std::vector<int> mNpackets;
  static constexpr int MAX_Page_size = 8192;

 public:
  o2::header::RAWDataHeader mRDH;
  void flush(std::ostream& str)
  {
    writePage();
    mRDH.stop = 0;
    for (int page = 0; page < int(mPages.size()); ++page) {
      mRDH.memorySize = mPages[page].size() + mRDH.headerSize;
      mRDH.offsetToNext = mRDH.memorySize;
      mRDH.packetCounter = mNpackets[page];
      str.write(reinterpret_cast<const char*>(&mRDH), sizeof(mRDH));
      str.write(mPages[page].data(), mPages[page].size());
      mRDH.pageCnt++;
      LOG(INFO)<<" header "<<mRDH.linkID<<" end "<<mRDH.endPointID;
    }
    if (!mPages.empty()) {
      mRDH.memorySize = mRDH.headerSize;
      mRDH.offsetToNext = mRDH.memorySize;
      mRDH.stop = 1;
      mRDH.pageCnt++;
      str.write(reinterpret_cast<const char*>(&mRDH), sizeof(mRDH));
      mPages.clear();
      mNpackets.clear();
    }
  }

  void writePage()
  {
    if (mBuffer.size() == 0) {
      return;
    }
    mPages.emplace_back(std::move(mBuffer));
    LOG(DEBUG) << " writePage " << mBuffer.size();
    mNpackets.push_back(mNpacketsInBuffer);
    mNpacketsInBuffer = 0;
    mBuffer.clear();
  }

  void write(std::vector<char> const& new_data)
  {
    if (mBuffer.size() + new_data.size() + mRDH.headerSize > MAX_Page_size) {
      LOG(DEBUG) << " write rest " << mBuffer.size() << " " << new_data.size() << " " << mRDH.headerSize;
      writePage();
    }
    LOG(DEBUG) << "  write vector " << new_data.size() << " buffer " << mBuffer.size() << " RDH " << mRDH.headerSize << " new data " << new_data.data();
    mBuffer.insert(mBuffer.end(), new_data.begin(), new_data.end());
    mNpacketsInBuffer++;
    LOG(DEBUG) << "  write vector end mBuffer.size " << mBuffer.size() << " mNpacketsInBuffer " << mNpacketsInBuffer << " newdtata " << new_data.size();
  }
};
} // namespace ft0
} // namespace o2
#endif
