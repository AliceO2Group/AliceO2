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

#include "DataFormatsFIT/Triggers.h"
#include <Framework/Logger.h>
#include <sstream>

using namespace o2::fit;

std::string Triggers::print() const
{
  std::stringstream ss;
  ss << " chA " << nChanA << " chC " << nChanC << " A amp " << amplA << "  C amp " << amplC << " time A " << timeA << " time C " << timeC << " signals 0x" << std::hex << int(triggersignals) << std::dec;
  return ss.str();
}

void Triggers::print(std::ostream& stream) const
{
  stream << print() << std::endl;
}

void Triggers::printLog() const
{
  LOG(info) << "mTrigger: " << static_cast<uint16_t>(triggersignals);
  LOG(info) << "nChanA: " << static_cast<uint16_t>(nChanA) << " | nChanC: " << static_cast<uint16_t>(nChanC);
  LOG(info) << "amplA: " << amplA << " | amplC: " << amplC;
  LOG(info) << "timeA: " << timeA << " | timeC: " << timeC;
}
