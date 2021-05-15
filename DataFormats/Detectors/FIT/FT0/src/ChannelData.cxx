// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsFT0/ChannelData.h"
#include <Framework/Logger.h>
#include <iostream>

using namespace o2::ft0;

void ChannelData::print() const
{
  printf("  ChID%d | CFDtime=%d | QTCampl=%d QTC chain %d\n", ChId, CFDTime, QTCAmpl, ChainQTC);
}
void ChannelData::printLog() const
{
  LOG(INFO) << "ChId: " << static_cast<uint16_t>(ChId) << " |  FEE bits:" << static_cast<uint16_t>(ChainQTC) << " | Time: " << CFDTime << " | Charge: " << QTCAmpl;
}
