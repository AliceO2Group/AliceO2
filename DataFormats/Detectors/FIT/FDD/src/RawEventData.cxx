// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsFDD/RawEventData.h"
#include <sstream>

using namespace o2::fdd;

ClassImp(RawEventData);

void RawEventData::print()
{
  std::cout << "==================Raw event data==================" << std::endl;
  std::cout << "##################Header##################" << std::endl;
  std::cout << "startDescriptor: " << mEventHeader.startDescriptor << std::endl;
  std::cout << "Nchannels: " << mEventHeader.nGBTWords * 2 << std::endl;
  std::cout << "BC: " << mEventHeader.bc << std::endl;
  std::cout << "Orbit: " << mEventHeader.orbit << std::endl;
  std::cout << "##########################################" << std::endl;
  std::cout << "###################DATA###################" << std::endl;
  for (int iCh = 0; iCh < mEventHeader.nGBTWords * 2; iCh++) {
    std::cout << "------------Channel " << mEventData[iCh].channelID << "------------" << std::endl;
    std::cout << "Charge: " << mEventData[iCh].charge << std::endl;
    std::cout << "Time: " << mEventData[iCh].time << std::endl;
    std::cout << "1TimeLostEvent: " << mEventData[iCh].is1TimeLostEvent << std::endl;
    std::cout << "2TimeLostEvent: " << mEventData[iCh].is2TimeLostEvent << std::endl;
    std::cout << "ADCinGate: " << mEventData[iCh].isADCinGate << std::endl;
    std::cout << "AmpHigh: " << mEventData[iCh].isAmpHigh << std::endl;
    std::cout << "DoubleEvent: " << mEventData[iCh].isDoubleEvent << std::endl;
    std::cout << "EventInTVDC: " << mEventData[iCh].isEventInTVDC << std::endl;
    std::cout << "TimeInfoLate: " << mEventData[iCh].isTimeInfoLate << std::endl;
    std::cout << "TimeInfoLost: " << mEventData[iCh].isTimeInfoLost << std::endl;
    std::cout << "numberADC: " << mEventData[iCh].numberADC << std::endl;
  }
  std::cout << "##########################################" << std::endl;
}

void RawEventData::printHexEventHeader()
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

void RawEventData::printHexEventData(uint64_t i)
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
