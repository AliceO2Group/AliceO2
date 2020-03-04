// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsFT0/RawEventData.h"
#include <CommonDataFormat/InteractionRecord.h>
#include <Framework/Logger.h>
#include <iostream>

using namespace o2::ft0;

using namespace std;

ClassImp(RawEventData);

void RawEventData::generateData()
{
  for (int iCh = 0; iCh < mEventHeader.nGBTWords * 2; iCh++) {
    mEventData[iCh].channelID = iCh;
    mEventData[iCh].charge = 1000;
    mEventData[iCh].time = 500;
    mEventData[iCh].is1TimeLostEvent = 1;
    mEventData[iCh].is2TimeLostEvent = 1;
    mEventData[iCh].isADCinGate = 1;
    mEventData[iCh].isAmpHigh = 1;
    mEventData[iCh].isDoubleEvent = 1;
    mEventData[iCh].isEventInTVDC = 1;
    mEventData[iCh].isTimeInfoLate = 1;
    mEventData[iCh].isTimeInfoLost = 1;
    mEventData[iCh].numberADC = 1;
  }
}

void RawEventData::generateHeader(int nChannels)
{
  mEventHeader.startDescriptor = 15;
  mEventHeader.nGBTWords = (nChannels + 1) / 2;
  mEventHeader.reservedField1 = 0;
  mEventHeader.reservedField2 = 0;
  mEventHeader.bc = 200;
  mEventHeader.orbit = 100;
}

void RawEventData::generateRandomHeader(int nChannels)
{
  mEventHeader.startDescriptor = 0x0000000f;
  if (nChannels > 0 && nChannels < 13)
    mEventHeader.nGBTWords = (nChannels + 1) / 2;
  else
    mEventHeader.nGBTWords = 1;
  mEventHeader.bc = std::rand() % 2000; // 1999-max bc
  mEventHeader.orbit = std::rand() % 100;
}

void RawEventData::generateRandomData()
{
  for (int iCh = 0; iCh < mEventHeader.nGBTWords * 2; iCh++) {
    mEventData[iCh].channelID = std::rand() % 208 + 1;
    mEventData[iCh].charge = std::rand() % 1000;
    mEventData[iCh].time = std::rand() % 500;
    mEventData[iCh].is1TimeLostEvent = std::rand() % 2;
    mEventData[iCh].is2TimeLostEvent = std::rand() % 2;
    mEventData[iCh].isADCinGate = std::rand() % 2;
    mEventData[iCh].isAmpHigh = std::rand() % 2;
    mEventData[iCh].isDoubleEvent = std::rand() % 2;
    mEventData[iCh].isEventInTVDC = std::rand() % 2;
    mEventData[iCh].isTimeInfoLate = std::rand() % 2;
    mEventData[iCh].isTimeInfoLost = std::rand() % 2;
    mEventData[iCh].numberADC = std::rand() % 2;
  }
}

void RawEventData::generateRandomEvent(int nChannels)
{
  generateRandomHeader(nChannels);
  generateRandomData();
}

void RawEventData::print()
{
  std::cout << "==================Raw event data==================" << endl;
  std::cout << "##################Header##################" << endl;
  std::cout << "startDescriptor: " << mEventHeader.startDescriptor << endl;
  std::cout << "Nchannels: " << mEventHeader.nGBTWords * 2 << endl;
  std::cout << "BC: " << mEventHeader.bc << endl;
  std::cout << "Orbit: " << mEventHeader.orbit << endl;
  std::cout << "##########################################" << endl;
  std::cout << "###################DATA###################" << endl;
  for (int iCh = 0; iCh < mEventHeader.nGBTWords * 2; iCh++) {
    std::cout << "------------Channel " << mEventData[iCh].channelID << "------------" << endl;
    std::cout << "Charge: " << mEventData[iCh].charge << endl;
    std::cout << "Time: " << mEventData[iCh].time << endl;
    std::cout << "1TimeLostEvent: " << mEventData[iCh].is1TimeLostEvent << endl;
    std::cout << "2TimeLostEvent: " << mEventData[iCh].is2TimeLostEvent << endl;
    std::cout << "ADCinGate: " << mEventData[iCh].isADCinGate << endl;
    std::cout << "AmpHigh: " << mEventData[iCh].isAmpHigh << endl;
    std::cout << "DoubleEvent: " << mEventData[iCh].isDoubleEvent << endl;
    std::cout << "EventInTVDC: " << mEventData[iCh].isEventInTVDC << endl;
    std::cout << "TimeInfoLate: " << mEventData[iCh].isTimeInfoLate << endl;
    std::cout << "TimeInfoLost: " << mEventData[iCh].isTimeInfoLost << endl;
    std::cout << "numberADC: " << mEventData[iCh].numberADC << endl;
  }
  std::cout << "##########################################" << endl;
}
