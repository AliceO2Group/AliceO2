// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsFV0/BCData.h"
#include "DataFormatsFV0/ChannelData.h"
#include <bitset>

using namespace o2::fv0;

void BCData::print() const
{
  ir.print();
  printf("]\n");
}

gsl::span<const ChannelData> BCData::getBunchChannelData(const gsl::span<const ChannelData> tfdata) const
{
  // extract the span of channel data for this bunch from the whole TF data
  return gsl::span<const ChannelData>(&tfdata[ref.getFirstEntry()], ref.getEntries());
}
