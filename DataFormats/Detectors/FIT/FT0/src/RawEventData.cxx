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

void EventHeader::print() const
{
  std::cout << std::hex;
  std::cout << "################EventHeader###############" << std::endl;
  std::cout << "startDescriptor: " << startDescriptor << std::endl;
  std::cout << "nGBTWords: " << nGBTWords << std::endl;
  std::cout << "BC: " << bc << std::endl;
  std::cout << "Orbit: " << orbit << std::endl;
  std::cout << "##########################################" << std::endl;
  std::cout << std::dec;
}

void EventData::print() const
{
  std::cout << std::hex;
  std::cout << "###############EventData(PM)##############" << std::endl;
  std::cout << "------------Channel " << channelID << "------------" << std::endl;
  std::cout << "Charge: " << charge << std::endl;
  std::cout << "Time: " << time << std::endl;
  std::cout << "numberADC: " << numberADC << std::endl;
  std::cout << "isDoubleEvent: " << isDoubleEvent << std::endl;
  std::cout << "isTimeInfoNOTvalid: " << isTimeInfoNOTvalid << std::endl;
  std::cout << "isCFDinADCgate: " << isCFDinADCgate << std::endl;
  std::cout << "isTimeInfoLate: " << isTimeInfoLate << std::endl;
  std::cout << "isAmpHigh: " << isAmpHigh << std::endl;
  std::cout << "isEventInTVDC: " << isEventInTVDC << std::endl;
  std::cout << "isTimeInfoLost: " << isTimeInfoLost << std::endl;
  std::cout << "##########################################" << std::endl;
  std::cout << std::dec;
}

void TCMdata::print() const
{
  std::cout << std::hex;
  std::cout << "################TCMdata###################" << std::endl;
  std::cout << "orC: " << orC << std::endl;
  std::cout << "orA: " << orA << std::endl;
  std::cout << "sCen: " << sCen << std::endl;
  std::cout << "cen: " << cen << std::endl;
  std::cout << "vertex: " << vertex << std::endl;
  std::cout << "nChanA: " << nChanA << std::endl;
  std::cout << "nChanC: " << nChanC << std::endl;
  std::cout << "amplA: " << amplA << std::endl;
  std::cout << "amplC: " << amplC << std::endl;
  std::cout << "timeA: " << timeA << std::endl;
  std::cout << "timeC: " << timeC << std::endl;
  std::cout << "##########################################" << std::endl;
  std::cout << std::dec;
}

void TCMdataExtended::print() const
{
  std::cout << std::hex;
  std::cout << "############TCMdataExtended###############" << std::endl;
  std::cout << "triggerWord: " << triggerWord << std::endl;
  std::cout << "##########################################" << std::endl;
  std::cout << std::dec;
}
