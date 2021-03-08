// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsZDC/BCData.h"
#include "DataFormatsZDC/ChannelData.h"
#include <bitset>

using namespace o2::zdc;

void BCData::print() const
{
  printf("Orbit %9u bc %4u nch %2d pos %d\n", ir.orbit, ir.bc, ref.getEntries(), ref.getFirstEntry());
  printf("Read:");
  for (int ic = 0; ic < NDigiChannels; ic++) {
    if (ic % NChPerModule == 0) {
      if (ic == 0) {
        printf(" %d[", ic / NChPerModule);
      } else {
        printf("] %d[", ic / NChPerModule);
      }
    }
    if (channels & (0x1 << ic)) {
      printf("R");
    } else {
      printf(" ");
    }
  }
  printf("]\nTrigs:");
  for (int i = 0; i < NChannels; i++) {
    std::bitset<10> bb(moduleTriggers[i]);
    printf("[%2d: %s]", i, bb.to_string().c_str());
    if (i % (NChannels / 3) == 0 && i) {
      printf("\n");
    }
  }

  printf("]\nHits:");
  for (int ic = 0; ic < NDigiChannels; ic++) {
    if (ic % NChPerModule == 0) {
      if (ic == 0) {
        printf(" %d[", ic / NChPerModule);
      } else {
        printf("] %d[", ic / NChPerModule);
      }
    }
    if (triggers & (0x1 << ic)) {
      printf("H");
    } else {
      printf(" ");
    }
  }
  printf("]\n");
}

gsl::span<const ChannelData> BCData::getBunchChannelData(const gsl::span<const ChannelData> tfdata) const
{
  // extract the span of channel data for this bunch from the whole TF data
  return ref.getEntries() ? gsl::span<const ChannelData>(&tfdata[ref.getFirstEntry()], ref.getEntries()) : gsl::span<const ChannelData>();
}
