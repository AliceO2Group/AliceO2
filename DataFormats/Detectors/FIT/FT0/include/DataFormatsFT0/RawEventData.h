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
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
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
  unsigned int startDescriptor : 4;
  unsigned int Nchannels : 4;
  unsigned int reservedField : 32;
  unsigned int orbit : 32;
  unsigned int bc : 12;
  ClassDefNV(EventHeader, 1);
};
struct EventData {
  unsigned int time : 12;
  unsigned int charge : 12;
  unsigned int numberADC : 1;
  unsigned int isDoubleEvent : 1;
  unsigned int is1TimeLostEvent : 1;
  unsigned int is2TimeLostEvent : 1;
  unsigned int isADCinGate : 1;
  unsigned int isTimeInfoLate : 1;
  unsigned int isAmpHigh : 1;
  unsigned int isEventInTVDC : 1;
  unsigned int isTimeInfoLost : 1;
  unsigned int channelID : 4;
  ClassDefNV(EventData, 1);
};
class RawEventData
{
 public:
  RawEventData();
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

  std::bitset<NBITS_EVENTDATA> fEventDataBits[NCHANNELS_PM];

  EventHeader mEventHeader;
  EventData mEventData[NCHANNELS_FT0];

  /////////////////////////////////////////////////
  ClassDefNV(RawEventData, 1);
};
std::ostream& operator<<(std::ostream& stream, const RawEventData& data);
} // namespace ft0
} // namespace o2
#endif
