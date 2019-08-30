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
// Artur.Furs
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
  uint startDescriptor : 4;
  uint Nchannels : 4;
  uint reservedField : 32;
  uint orbit : 32;
  uint bc : 12;
  ClassDefNV(EventHeader, 1);
};
struct EventData {
  int time : 12;
  int charge : 12;
  uint numberADC : 1;
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

 protected:
  int mTime[NCHANNELS_PM];
  int mCharge[NCHANNELS_PM];
  int mNchannels;
  o2::InteractionRecord mIntRecord;

  EventHeader mEventHeader;
  EventData mEventData[NCHANNELS_FT0];

  /////////////////////////////////////////////////
  ClassDefNV(RawEventData, 1);
};
std::ostream& operator<<(std::ostream& stream, const RawEventData& data);
} // namespace ft0
} // namespace o2
#endif
