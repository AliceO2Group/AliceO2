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

#include "PHOSBase/Geometry.h"
#include "DataFormatsPHOS/BadChannelsMap.h"

#include "FairLogger.h"

#include <TH2.h>

#include <iostream>

using namespace o2::phos;

BadChannelsMap::BadChannelsMap(int /*dummy*/)
{

  // Mark few channels as bad for test peurposes
  for (short i = 0; i < 56; i++) {
    // module 2
    short channelID = 3584 + i * 57;
    mBadCells.set(channelID - OFFSET);
    channelID = 3640 + i * 55;
    mBadCells.set(channelID - OFFSET);
  }

  for (short i = 0; i < 16; i++) {
    // module 3
    int channelID = 8972 + i * 57;
    mBadCells.set(channelID - OFFSET);
    channelID = 8092 + i * 57;
    mBadCells.set(channelID - OFFSET);
    channelID = 8147 + i * 55;
    mBadCells.set(channelID - OFFSET);
    channelID = 9059 + i * 55;
    mBadCells.set(channelID - OFFSET);
  }
}

void BadChannelsMap::getHistogramRepresentation(char module, TH2* h) const
{
  const char MAXX = 64,
             MAXZ = 56;
  if (module < 1 || module > 4) {
    LOG(error) << "module " << module << "does not exist";
    return;
  }
  if (!h) {
    LOG(error) << "provide histogram to be filled";
  }
  if (h->GetNbinsX() != MAXX || h->GetNbinsY() != MAXZ) {
    LOG(error) << "Wrong dimentions of input histogram:" << h->GetNbinsX() << "," << h->GetNbinsY() << " instead of " << MAXX << "," << MAXZ;
    return;
  }

  h->Reset();
  char relid[3] = {module, 1, 1};
  short absId;
  char xmin = 1;
  if (module == 1) {
    xmin = 33;
  }
  for (char ix = xmin; ix <= MAXX; ix++) {
    relid[1] = ix;
    for (char iz = 1; iz <= MAXZ; iz++) {
      relid[2] = iz;
      if (o2::phos::Geometry::relToAbsNumbering(relid, absId)) {
        if (!isChannelGood(absId)) {
          h->SetBinContent(ix, iz, 1);
        }
      }
    }
  }
}

void BadChannelsMap::PrintStream(std::ostream& stream) const
{
  // first sort bad channel IDs
  stream << "Number of bad cells:  " << mBadCells.count() << "\n";
  for (std::size_t cellID = 0; cellID < mBadCells.size(); cellID++) {
    if (mBadCells.test(cellID)) {
      stream << cellID + OFFSET << "\n";
    }
  }
}

std::ostream& o2::phos::operator<<(std::ostream& stream, const BadChannelsMap& bcm)
{
  bcm.PrintStream(stream);
  return stream;
}
