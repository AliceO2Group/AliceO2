// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ZDCSimulation/SimCondition.h"

using namespace o2::zdc;

void ChannelSimCondition::print() const
{
  printf("Shape min at bin %d, Pedestal:{%+e,%+e,%+e}, Gain:%.4f, TimeJitter: %.4f TimePos: %.2f\n",
         ampMinID, pedestal, pedestalNoise, pedestalFluct, gain, timeJitter, timePosition);
}

void SimCondition::print() const
{
  for (int i = 0; i < NChannels; i++) {
    printf("%s ", channelName(i));
    channels[i].print();
  }
}
