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
#include "Headers/RAWDataHeader.h"
#include "DataFormatsFT0/LookUpTable.h"
#include <CommonDataFormat/InteractionRecord.h>
#include <Framework/Logger.h>
#include <iostream>
#include <cstring>
#include "Rtypes.h"
namespace o2
{
namespace ft0
{
constexpr int Nchannels_FT0 = 208;
constexpr int Nchannels_PM = 12;
constexpr int NPMs = 19;

struct EventHeader {
  static constexpr int PayloadSize = 16;
  union {
    uint64_t word[2] = {};
    struct {
      uint64_t bc : 12;
      uint64_t orbit : 32;
      uint64_t reservedField1 : 20;
      uint64_t reservedField2 : 8;
      uint64_t nGBTWords : 4;
      uint64_t startDescriptor : 4;
      uint64_t reservedField3 : 48;
    };
  };
  ClassDefNV(EventHeader, 1);
};
struct EventData {
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

  ClassDefNV(EventData, 1);
};

struct TCMdata {
  static constexpr int PayloadSize = 16;
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
  // ClassDefNV(TCMdata, 1);
};

class RawEventData
{
 public:
  RawEventData() = default;
  void generateHeader(int nChannels);
  void generateData();
  void generateRandomHeader(int nChannels);
  void generateRandomData();
  void generateRandomEvent(int nChannels);
  EventHeader* getEventHeaderPtr() { return &mEventHeader; }
  EventData* getEventDataPtr() { return mEventData; }
  void print();
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
    return 1                         // EventHeader
           + mEventHeader.nGBTWords; // EventData
  }

  static void printRDH(const o2::header::RAWDataHeader* h)
  {
    {
      if (!h) {
        printf("Provided RDH pointer is null\n");
        return;
      }
      printf("RDH| Ver:%2u Hsz:%2u Blgt:%4u FEEId:0x%04x PBit:%u\n",
             uint32_t(h->version), uint32_t(h->headerSize), uint32_t(h->blockLength), uint32_t(h->feeId), uint32_t(h->priority));
      printf("RDH|[CRU: Offs:%5u Msz:%4u LnkId:0x%02x Packet:%3u CRUId:0x%04x]\n",
             uint32_t(h->offsetToNext), uint32_t(h->memorySize), uint32_t(h->linkID), uint32_t(h->packetCounter), uint32_t(h->cruID));
      printf("RDH| TrgOrb:%9u HBOrb:%9u TrgBC:%4u HBBC:%4u TrgType:%u\n",
             uint32_t(h->triggerOrbit), uint32_t(h->heartbeatOrbit), uint32_t(h->triggerBC), uint32_t(h->heartbeatBC),
             uint32_t(h->triggerType));
      printf("RDH| DetField:0x%05x Par:0x%04x Stop:0x%04x PageCnt:%5u\n",
             uint32_t(h->detectorField), uint32_t(h->par), uint32_t(h->stop), uint32_t(h->pageCnt));
    }
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
      LOG(DEBUG) << "write header words " << (int)mEventHeader.nGBTWords << " orbit " << int(mEventHeader.orbit) << " bc " << int(mEventHeader.bc) << " out " << result.size();
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
  ClassDefNV(RawEventData, 1);
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
      RawEventData::printRDH(&mRDH);
      str.write(reinterpret_cast<const char*>(&mRDH), sizeof(mRDH));
      str.write(mPages[page].data(), mPages[page].size());
      mRDH.pageCnt++;
    }
    if (!mPages.empty()) {
      mRDH.memorySize = mRDH.headerSize;
      mRDH.offsetToNext = mRDH.memorySize;
      mRDH.stop = 1;
      mRDH.pageCnt++;
      RawEventData::printRDH(&mRDH);
      str.write(reinterpret_cast<const char*>(&mRDH), sizeof(mRDH));
      mPages.clear();
      mNpackets.clear();
    }
  }

  void writePage()
  {
    if (mBuffer.size() == 0)
      return;
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
