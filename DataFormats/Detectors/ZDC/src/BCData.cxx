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
  ir.print();
  printf("%d channels starting from %d\n", ref.getEntries(), ref.getFirstEntry());
  printf("Read:");
  for (int ic = 0; ic < NDigiChannels; ic++) {
    if (ic % NChPerModule == 0) {
      if (ic == 0)
        printf(" %d[", ic / NChPerModule);
      else
        printf("] %d[", ic / NChPerModule);
    }
    if (channels & (0x1 << ic)) {
      printf("R");
    } else {
      printf(" ");
    }
  }
  printf("]\nHits:");
  for (int ic = 0; ic < NDigiChannels; ic++) {
    if (ic % NChPerModule == 0) {
      if (ic == 0)
        printf(" %d[", ic / NChPerModule);
      else
        printf("] %d[", ic / NChPerModule);
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
