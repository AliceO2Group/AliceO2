// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsFV0/RawEventData.h"
#include <sstream>
#include <iostream>

using namespace o2::fv0;

ClassImp(RawEventData);

void EventHeader::print() const
{
  LOG(INFO) << std::hex;
  LOG(INFO) << "################EventHeader###############";
  LOG(INFO) << "startDescriptor: " << startDescriptor;
  LOG(INFO) << "nGBTWords: " << nGBTWords;
  LOG(INFO) << "BC: " << bc;
  LOG(INFO) << "Orbit: " << orbit;
  LOG(INFO) << "##########################################";
  LOG(INFO) << std::dec;
}

void RawEventData::print() const
{
  LOG(INFO) << "==================Raw event data==================";
  LOG(INFO) << "##################Header##################";
  LOG(INFO) << "startDescriptor: " << mEventHeader.startDescriptor;
  LOG(INFO) << "Nchannels: " << mEventHeader.nGBTWords * 2;
  LOG(INFO) << "BC: " << mEventHeader.bc;
  LOG(INFO) << "Orbit: " << mEventHeader.orbit;
  LOG(INFO) << "##########################################";
  LOG(INFO) << "###################DATA###################";
  for (int iCh = 0; iCh < mEventHeader.nGBTWords * 2; iCh++) {
    LOG(INFO) << "------------Channel " << mEventData[iCh].channelID << "------------";
    LOG(INFO) << "Charge: " << mEventData[iCh].charge;
    LOG(INFO) << "Time: " << mEventData[iCh].time;
    LOG(INFO) << "1TimeLostEvent: " << mEventData[iCh].is1TimeLostEvent;
    LOG(INFO) << "2TimeLostEvent: " << mEventData[iCh].is2TimeLostEvent;
    LOG(INFO) << "ADCinGate: " << mEventData[iCh].isADCinGate;
    LOG(INFO) << "AmpHigh: " << mEventData[iCh].isAmpHigh;
    LOG(INFO) << "DoubleEvent: " << mEventData[iCh].isDoubleEvent;
    LOG(INFO) << "EventInTVDC: " << mEventData[iCh].isEventInTVDC;
    LOG(INFO) << "TimeInfoLate: " << mEventData[iCh].isTimeInfoLate;
    LOG(INFO) << "TimeInfoLost: " << mEventData[iCh].isTimeInfoLost;
    LOG(INFO) << "numberADC: " << mEventData[iCh].numberADC;
  }
  LOG(INFO) << "##########################################";
}

void RawEventData::printHexEventHeader() const
{
  std::stringstream ssheader;
  ssheader << std::setfill('0') << std::setw(16) << std::hex << mEventHeader.word[0] << " " << std::setw(16) << mEventHeader.word[1] << "\n       ";
  ssheader << std::setw(3) << (0x00000fff & mEventHeader.bc) << " "
           << std::setw(8) << (0xffffffff & mEventHeader.orbit) << " "
           << std::setw(5) << (0x000fffff & mEventHeader.reservedField1) << " "
           << std::setw(2) << (0x000000ff & mEventHeader.reservedField2) << " "
           << std::setw(1) << (0x0000000f & mEventHeader.nGBTWords) << " "
           << std::setw(1) << (0x0000000f & mEventHeader.startDescriptor) << " "
           << std::setw(12) << (0xffffffffffff & mEventHeader.reservedField3);
  LOG(DEBUG) << ssheader.str();
}

void RawEventData::printHexEventData(uint64_t i) const
{
  std::stringstream ssdata;
  ssdata << "D0:0x ";
  ssdata << std::setfill('0') << std::hex << std::setw(16) << mEventData[2 * i].word << "\n                   ";
  ssdata << std::setw(3) << (0x0fff & mEventData[2 * i].time) << " "
         << std::setw(8) << (0x1fff & mEventData[2 * i].charge) << "\n             ";
  ssdata << "D1:0x ";
  ssdata << std::setfill('0') << std::hex << std::setw(16) << mEventData[2 * i + 1].word << "\n                   ";
  ssdata << std::setw(3) << (0x0fff & mEventData[2 * i + 1].time) << " "
         << std::setw(8) << (0x1fff & mEventData[2 * i + 1].charge);
  LOG(DEBUG) << "    | " << ssdata.str();
}
