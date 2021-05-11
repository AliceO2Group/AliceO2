// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsFV0/ChannelData.h"

using namespace o2::fv0;

void ChannelData::print() const
{
  printf("  Pmt=%2d  |  time =%4d  |  charge =%6d\n", pmtNumber, time, chargeAdc);
}

void ChannelData::printLog() const
{
  LOG(INFO) << "ChId: " << static_cast<uint16_t>(pmtNumber) /*<< " |  FEE bits:" << static_cast<uint16_t>(ChainQTC)*/ << " | Time: " << time << " | Charge: " << chargeAdc;
}