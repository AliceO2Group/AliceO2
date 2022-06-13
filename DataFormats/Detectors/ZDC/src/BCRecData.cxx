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

#include "DataFormatsZDC/BCRecData.h"
#include "DataFormatsZDC/ChannelData.h"
#include <bitset>

using namespace o2::zdc;

void BCRecData::print() const
{
  printf("Orbit %9u bc %4u nch=%2d pos %d ntdc=%2d pos %d nmsg=%2d pos %d nwav=%d pos %d\n", ir.orbit, ir.bc,
         refe.getEntries(), refe.getFirstEntry(),
         reft.getEntries(), reft.getFirstEntry(),
         refi.getEntries(), refi.getFirstEntry(),
         refw.getEntries(), refw.getFirstEntry());
}
