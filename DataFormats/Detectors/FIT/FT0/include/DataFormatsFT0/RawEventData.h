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
#include <iostream>
#include <bitset>
#include "Rtypes.h"
namespace o2
{
namespace ft0
{
constexpr int NCHANNELS_FT0 = 208;
constexpr int NCHANNELS_PM = 12;
constexpr int NPMs = 18;
constexpr int NBITS_EVENTDATA = 9;

struct EventHeader {
  ushort startDescriptor : 4;
  uint reservedField : 28;
  uint orbit : 32;
  short bc : 12;
  short nGBTWords : 4;
  ClassDefNV(EventHeader, 1);
};
struct EventData {
  short int time : 12;
  short int charge : 12;
  unsigned short int numberADC : 1;
  bool isDoubleEvent : 1;
  bool is1TimeLostEvent : 1;
  bool is2TimeLostEvent : 1;
  bool isADCinGate : 1;
  bool isTimeInfoLate : 1;
  bool isAmpHigh : 1;
  bool isEventInTVDC : 1;
  bool isTimeInfoLost : 1;
  uint channelID : 4;
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
    return 4                         // RAWDataHeader
           + 1                       // EventHeader
           + mEventHeader.nGBTWords; // EventData
  }

  void write(std::ostream& str)
  {
    str.write(reinterpret_cast<const char*>(&mRDH), sizeof(mRDH));
    str.write(reinterpret_cast<const char*>(&mEventHeader), sizeof(mEventHeader));
    char padding[16 - sizeof(mEventHeader)] = {};
    if (mIsPadded)
      str.write(padding, sizeof(padding));
    //  static_assert(sizeof(mEventHeader) == 2 * sizeof(mEventData[0]), "This code assumes that pairs of data and header require the same padding");
    for (int i = 0; i < mEventHeader.nGBTWords; ++i) {
      str.write(reinterpret_cast<const char*>(&mEventData[2 * i]), sizeof(mEventData[0]) * 2);
      if (mIsPadded)
        str.write(padding, sizeof(padding));
    }
  }

  void setIsPadded(bool isPadding128)
  {
    mIsPadded = isPadding128;
  }

 public:
  o2::header::RAWDataHeader mRDH;
  EventHeader mEventHeader;
  EventData mEventData[NCHANNELS_PM];
  bool mIsPadded = true;
  /////////////////////////////////////////////////
  ClassDefNV(RawEventData, 1);
};
std::ostream& operator<<(std::ostream& stream, const RawEventData& data);

class DataPageWriter
{
  std::vector<RawEventData> mBuffer;
  int numWords{0};

 public:
  void flush(std::ostream& str, int& nPages)
  {
    if (!mBuffer.empty()) {
      mBuffer.back().mRDH.stop = 1;
      for (RawEventData& data : mBuffer) {
        data.mRDH.pageCnt = nPages;
        data.write(str);
      }
      nPages++;
    }
    mBuffer.clear();
    numWords = 0;
  }

  void add(std::ostream& str, RawEventData const& data, int& nPages)
  {
    if (numWords + data.size() > 512)
      flush(str, nPages);
    numWords += data.size();
    mBuffer.emplace_back(data);
  }
};
} // namespace ft0
} // namespace o2
#endif
