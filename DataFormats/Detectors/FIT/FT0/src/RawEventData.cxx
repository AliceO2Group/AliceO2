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
#include <iostream>

using namespace o2::ft0;

using namespace std;

ClassImp(RawEventData);

/*
RawEventData::RawEventData()
{
  cout << "\n////////////////////////////////////////////////////////////////";
  cout << "\n/Initializating object RawEventData...";
  cout << "\n////////////////////////////////////////////////////////////////\n";
}

*/
/*******************************************************************************************************************/
void RawEventData::GenerateData()
{
  for (int iCh = 0; iCh < mEventHeader.Nchannels; iCh++) {
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
/*******************************************************************************************************************/
void RawEventData::GenerateHeader(int nChannels)
{
  mEventHeader.startDescriptor = 15;
  mEventHeader.Nchannels = nChannels;
  mEventHeader.reservedField = 0;
  mEventHeader.bc = 200;
  mEventHeader.orbit = 100;
}
/*******************************************************************************************************************/
void RawEventData::GenerateRandomHeader(int nChannels)
{
  mEventHeader.startDescriptor = 0x0000000f;
  if (nChannels > 0 && nChannels < 13)
    mEventHeader.Nchannels = nChannels;
  else
    mEventHeader.Nchannels = 1;
  mEventHeader.bc = std::rand() % 2000; // 1999-max bc
  mEventHeader.orbit = std::rand() % 100;
}
/*******************************************************************************************************************/
void RawEventData::GenerateRandomData()
{
  for (int iCh = 0; iCh < mEventHeader.Nchannels; iCh++) {
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
/*******************************************************************************************************************/
void RawEventData::GenerateRandomEvent(int nChannels)
{
  GenerateRandomHeader(nChannels);
  GenerateRandomData();
}
/*******************************************************************************************************************/
void RawEventData::Print(bool doPrintData)
{
  cout << endl
       << "==================Raw event data==================" << endl;
  cout << "##################Header##################" << endl;
  cout << "startDescriptor: " << mEventHeader.startDescriptor << endl;
  cout << "Nchannels: " << mEventHeader.Nchannels << endl;
  cout << "BC: " << mEventHeader.bc << endl;
  cout << "Orbit: " << mEventHeader.orbit << endl;
  cout << "##########################################" << endl;
  if (!doPrintData)
    return;
  cout << "###################DATA###################" << endl;
  for (int iCh = 0; iCh < mEventHeader.Nchannels; iCh++) {
    cout << "------------Channel " << mEventData[iCh].channelID << "------------" << endl;
    cout << "Charge: " << mEventData[iCh].charge << endl;
    cout << "Time: " << mEventData[iCh].time << endl;
    cout << "1TimeLostEvent: " << mEventData[iCh].is1TimeLostEvent << endl;
    cout << "2TimeLostEvent: " << mEventData[iCh].is2TimeLostEvent << endl;
    cout << "ADCinGate: " << mEventData[iCh].isADCinGate << endl;
    cout << "AmpHigh: " << mEventData[iCh].isAmpHigh << endl;
    cout << "DoubleEvent: " << mEventData[iCh].isDoubleEvent << endl;
    cout << "EventInTVDC: " << mEventData[iCh].isEventInTVDC << endl;
    cout << "TimeInfoLate: " << mEventData[iCh].isTimeInfoLate << endl;
    cout << "TimeInfoLost: " << mEventData[iCh].isTimeInfoLost << endl;
    cout << "numberADC: " << mEventData[iCh].numberADC << endl;
  }
  cout << "##########################################" << endl;
}
