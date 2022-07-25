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

#include "DataFormatsZDC/OrbitData.h"
#include "ZDCBase/Constants.h"

using namespace o2::zdc;

void OrbitData::print() const
{
  // N.B. print encoded baseline because decoding requires ModuleConfig object
  printf("Orbit %9u bc %4u\n", ir.orbit, ir.bc);
  for (int i = 0; i < NChannels; i++) {
    printf("%2d %s: %6d cnt: %4u\n", i, ChannelNames[i].data(), data[i], scaler[i]);
  }
}
