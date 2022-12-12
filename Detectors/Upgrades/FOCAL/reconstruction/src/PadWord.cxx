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
#include <bitset>
#include <iomanip>
#include <iostream>
#include "FOCALReconstruction/PadWord.h"

std::ostream& o2::focal::operator<<(std::ostream& stream, const ASICChannel& channel)
{
  stream << "(ADC) " << channel.getADC() << ", (TOA) " << channel.getTOA() << ", (TOT) " << channel.getTOT();
  return stream;
}

std::ostream& o2::focal::operator<<(std::ostream& stream, const ASICHeader& header)
{
  stream << "(HEADER) 0x" << std::hex << header.getHeader() << std::dec << "(BCID) " << header.getBCID() << ", WADD " << header.getWadd() << ", (FOURBIT) " << std::bitset<4>(header.getFourbit()) << ", (TRAILER) 0x" << std::hex << header.getTrailer() << std::dec;
  return stream;
}

std::ostream& o2::focal::operator<<(std::ostream& stream, const TriggerWord& trigger)
{
  stream << "(HEADER) 0x" << std::hex << trigger.mHeader << std::dec << ": " << trigger.mTrigger0 << ", " << trigger.mTrigger1 << ", " << trigger.mTrigger2 << ", " << trigger.mTrigger3 << ", " << trigger.mTrigger4 << ", " << trigger.mTrigger5 << ", " << trigger.mTrigger6 << ", " << trigger.mTrigger7;
  return stream;
}