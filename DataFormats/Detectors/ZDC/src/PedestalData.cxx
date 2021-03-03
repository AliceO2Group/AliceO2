// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsZDC/PedestalData.h"
#include "ZDCBase/Constants.h"

using namespace o2::zdc;

void PedestalData::print() const
{
  printf("Orbit %9u bc %4u\n", ir.orbit, ir.bc);
  for (int i = 0; i < NChannels; i++) {
    printf("%2d %s: %9.3f cnt: %4u\n", i, ChannelNames[i].data(), asFloat(i), scaler[i]);
  }
}
