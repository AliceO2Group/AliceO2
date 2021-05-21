// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  printf("Orbit %9u bc %4u nch %2d pos %d ntdc %2d pos %d nmsg %2d pos %d\n", ir.orbit, ir.bc,
         refe.getEntries(), refe.getFirstEntry(),
         reft.getEntries(), reft.getFirstEntry(),
         refi.getEntries(), refi.getFirstEntry());
}
