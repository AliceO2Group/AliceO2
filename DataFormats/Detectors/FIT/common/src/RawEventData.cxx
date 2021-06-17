// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsFIT/RawEventData.h"

using namespace o2::fit;

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

void EventData::print() const
{
  LOG(INFO) << std::hex;
  LOG(INFO) << "###############EventData(PM)##############";
  LOG(INFO) << "------------Channel " << channelID << "------------";
  LOG(INFO) << "Charge: " << charge;
  LOG(INFO) << "Time: " << time;
  LOG(INFO) << "numberADC: " << numberADC;
  LOG(INFO) << "isDoubleEvent: " << isDoubleEvent;
  LOG(INFO) << "isTimeInfoNOTvalid: " << isTimeInfoNOTvalid;
  LOG(INFO) << "isCFDinADCgate: " << isCFDinADCgate;
  LOG(INFO) << "isTimeInfoLate: " << isTimeInfoLate;
  LOG(INFO) << "isAmpHigh: " << isAmpHigh;
  LOG(INFO) << "isEventInTVDC: " << isEventInTVDC;
  LOG(INFO) << "isTimeInfoLost: " << isTimeInfoLost;
  LOG(INFO) << "##########################################";
  LOG(INFO) << std::dec;
}

void TCMdata::print() const
{
  LOG(INFO) << std::hex;
  LOG(INFO) << "################TCMdata###################";
  LOG(INFO) << "orC: " << orC;
  LOG(INFO) << "orA: " << orA;
  LOG(INFO) << "sCen: " << sCen;
  LOG(INFO) << "cen: " << cen;
  LOG(INFO) << "vertex: " << vertex;
  LOG(INFO) << "nChanA: " << nChanA;
  LOG(INFO) << "nChanC: " << nChanC;
  LOG(INFO) << "amplA: " << amplA;
  LOG(INFO) << "amplC: " << amplC;
  LOG(INFO) << "timeA: " << timeA;
  LOG(INFO) << "timeC: " << timeC;
  LOG(INFO) << "##########################################";
  LOG(INFO) << std::dec;
}

void TCMdataExtended::print() const
{
  LOG(INFO) << std::hex;
  LOG(INFO) << "############TCMdataExtended###############";
  LOG(INFO) << "triggerWord: " << triggerWord;
  LOG(INFO) << "##########################################";
  LOG(INFO) << std::dec;
}
