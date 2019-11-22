// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsZDC/ChannelData.h"

using namespace o2::zdc;

void ChannelData::print() const
{
  printf("Ch%2d | ", id);
  for (int i = 0; i < NTimeBinsPerBC; i++) {
    printf("%+5d ", data[i]);
  }
  printf(" (%s)\n", channelName(id));
}
