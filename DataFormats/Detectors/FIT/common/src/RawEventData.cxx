// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsFIT/RawEventData.h"

using namespace o2::fit;

void EventHeader::print() const
{
  LOG(info) << std::hex;
  LOG(info) << "################EventHeader###############";
  LOG(info) << "startDescriptor: " << startDescriptor;
  LOG(info) << "nGBTWords: " << nGBTWords;
  LOG(info) << "BC: " << bc;
  LOG(info) << "Orbit: " << orbit;
  LOG(info) << "##########################################";
  LOG(info) << std::dec;
}

void EventData::print() const
{
  LOG(info) << std::hex;
  LOG(info) << "###############EventData(PM)##############";
  LOG(info) << "------------Channel " << channelID << "------------";
  LOG(info) << "Charge: " << charge;
  LOG(info) << "Time: " << time;
  LOG(info) << "##########################################";
  LOG(info) << std::dec;
}

void TCMdata::print() const
{
  LOG(info) << std::hex;
  LOG(info) << "################TCMdata###################";
  LOG(info) << "orC: " << orC;
  LOG(info) << "orA: " << orA;
  LOG(info) << "sCen: " << sCen;
  LOG(info) << "cen: " << cen;
  LOG(info) << "vertex: " << vertex;
  LOG(info) << "laser: " << laser;
  LOG(info) << "outputsAreBlocked: " << outputsAreBlocked;
  LOG(info) << "dataIsValid: " << dataIsValid;
  LOG(info) << "nChanA: " << nChanA;
  LOG(info) << "nChanC: " << nChanC;
  LOG(info) << "amplA: " << amplA;
  LOG(info) << "amplC: " << amplC;
  LOG(info) << "timeA: " << timeA;
  LOG(info) << "timeC: " << timeC;
  LOG(info) << "##########################################";
  LOG(info) << std::dec;
}

void TCMdataExtended::print() const
{
  LOG(info) << std::hex;
  LOG(info) << "############TCMdataExtended###############";
  LOG(info) << "triggerWord: " << triggerWord;
  LOG(info) << "##########################################";
  LOG(info) << std::dec;
}
