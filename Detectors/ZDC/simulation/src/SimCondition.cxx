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

#include "ZDCSimulation/SimCondition.h"

using namespace o2::zdc;

void ChannelSimCondition::print() const
{
  printf("Shape min at bin %d, Pedestal:{%+e,%+e,%+e}, Gain:%.4f, GainInSum:%.4f, TimeJitter: %.4f TimePos: %.2f\n",
         ampMinID, pedestal, pedestalNoise, pedestalFluct, gain, gainInSum, timeJitter, timePosition);
}

void SimCondition::print() const
{
  for (int i = 0; i < NChannels; i++) {
    printf("%s ", channelName(i));
    channels[i].print();
  }
}
