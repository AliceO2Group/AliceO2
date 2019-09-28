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
constexpr int NPMs = 18;
//constexpr int NBITS_EVENTDATA = 9;
//static constexpt int PayloadSize = 10; // size in bytes = 1GBT word

struct EventHeader {
  static constexpr int PayloadSize = 10; // size in bytes = 1GBT word
  union {
    uint32_t w[3] = {0};
    struct {
      uint16_t startDescriptor : 4;
      uint16_t nGBTWords : 4;
      uint32_t reservedField : 28;
      uint32_t orbit : 32;
      uint32_t bc : 12;
    };
  };

  ClassDefNV(EventHeader, 1);
};

struct EventData {
  static constexpr int PayloadSize = 5; // size in bytes = 1/2 GBT word
  union {
    uint64_t w = 0;
    struct {
      int16_t time : 12;
      int16_t charge : 12;
      bool numberADC : 1;
      bool isDoubleEvent : 1;
      bool is1TimeLostEvent : 1;
      bool is2TimeLostEvent : 1;
      bool isADCinGate : 1;
      bool isTimeInfoLate : 1;
      bool isAmpHigh : 1;
      bool isEventInTVDC : 1;
      bool isTimeInfoLost : 1;
      uint channelID : 4;
    };
  };
  ClassDefNV(EventData, 1);
};
class RawEventData
{
 public:
  RawEventData() = default;
  //virtual ~RawEventData();
  void GenerateHeader(int nChannels);
  void GenerateData();
  void GenerateRandomHeader(int nChannels);
  void GenerateRandomData();
  void GenerateRandomEvent(int nChannels);
  EventHeader* GetEventHeaderPtr() { return &mEventHeader; }
  EventData* GetEventDataPtr() { return mEventData; }
  void Print(bool doPrintData = false);
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
  std::vector<char> to_vector()
  {
    constexpr int CRUWordSize = 16;
    const char padding[CRUWordSize] = {0};

    std::vector<char> result(size() * CRUWordSize);
    char* out = result.data();
    //   str.write(reinterpret_cast<const char*>(&mRDH), sizeof(mRDH));
    //   printRDH(&mRDH);
    //  LOG(INFO)<<"orbit "<<mRDH.orbit<<" BC "<<mRDH.BC<<" link "<<mRDH.linkID;
    //str.write(reinterpret_cast<const char*>(&mEventHeader), EventHeader::PayloadSize);
    std::memcpy(out, &mEventHeader, EventHeader::PayloadSize);
    out += EventHeader::PayloadSize;
    LOG(INFO) << " !!@@@write header for " << (int)mEventHeader.nGBTWords << " words";
    if (mIsPadded) {
      out += CRUWordSize - EventHeader::PayloadSize;
      LOG(INFO) << " !!@@@ padding header";
    }
    for (int i = 0; i < mEventHeader.nGBTWords; ++i) {
      std::memcpy(out, &mEventData[2 * i], EventData::PayloadSize);
      out += EventData::PayloadSize;
      LOG(INFO) << " !!@@@ write 1st word channel " << mEventData[2 * i].channelID;
      std::memcpy(out, &mEventData[2 * i + 1], EventData::PayloadSize);
      out += EventData::PayloadSize;
      LOG(INFO) << " !!@@@ write 2nd word channel " << mEventData[2 * i + 1].channelID;
      if (mIsPadded) {
        out += CRUWordSize - 2 * EventData::PayloadSize;
        LOG(INFO) << " !!@@@ padding data";
      }
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
  bool mIsPadded = true;
  /////////////////////////////////////////////////
  ClassDefNV(RawEventData, 1);
};
std::ostream& operator<<(std::ostream& stream, const RawEventData& data);

class DataPageWriter
{
  std::vector<char> mBuffer;
  static constexpr int MAX_PAGE_SIZE = 8192;

 public:
  o2::header::RAWDataHeader mRDH;
  void flush(std::ostream& str)
  {
    size_t pos = 0;
    mRDH.stop = 0;
    while (pos < mBuffer.size()) {
      mRDH.memorySize = std::min(mBuffer.size() + mRDH.headerSize, (size_t)8192);
      mRDH.offsetToNext = mRDH.memorySize;
      RawEventData::printRDH(&mRDH);
      str.write(reinterpret_cast<const char*>(&mRDH), sizeof(mRDH));
      str.write(mBuffer.data() + pos, mRDH.memorySize - mRDH.headerSize);
      pos += mRDH.memorySize - mRDH.headerSize;
      mRDH.pageCnt++;
    }
    if (!mBuffer.empty()) {
      mRDH.memorySize = mRDH.headerSize;
      mRDH.offsetToNext = mRDH.memorySize;
      mRDH.stop = 1;
      RawEventData::printRDH(&mRDH);
      str.write(reinterpret_cast<const char*>(&mRDH), sizeof(mRDH));
      mRDH.pageCnt++;
      mBuffer.clear();
    }
  }

  void write(std::vector<char> const& new_data)
  {
    mBuffer.insert(mBuffer.end(), new_data.begin(), new_data.end());
  }
};
} // namespace ft0
} // namespace o2
#endif
