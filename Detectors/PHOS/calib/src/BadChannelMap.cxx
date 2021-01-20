// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "PHOSBase/Geometry.h"
#include "PHOSCalib/BadChannelMap.h"

#include "FairLogger.h"

#include <TH2.h>

#include <iostream>

using namespace o2::phos;

BadChannelMap::BadChannelMap(int /*dummy*/)
{

  //Mark few channels as bad for test peurposes
  for (short i = 0; i < 56; i++) {
    //module 2
    short channelID = 3584 + i * 57;
    mBadCells.set(channelID - OFFSET);
    channelID = 3640 + i * 55;
    mBadCells.set(channelID - OFFSET);
  }

  for (short i = 0; i < 16; i++) {
    //module 3
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

void BadChannelMap::getHistogramRepresentation(char module, TH2* h) const
{
  const char MAXX = 64,
             MAXZ = 56;
  if (module < 1 || module > 4) {
    LOG(ERROR) << "module " << module << "does not exist";
    return;
  }
  if (!h) {
    LOG(ERROR) << "provide histogram to be filled";
  }
  if (h->GetNbinsX() != MAXX || h->GetNbinsY() != MAXZ) {
    LOG(ERROR) << "Wrong dimentions of input histogram:" << h->GetNbinsX() << "," << h->GetNbinsY() << " instead of " << MAXX << "," << MAXZ;
    return;
  }

  h->Reset();
  char relid[3] = {module, 1, 1};
  short absId;
  for (char ix = 1; ix <= MAXX; ix++) {
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

void BadChannelMap::PrintStream(std::ostream& stream) const
{
  // first sort bad channel IDs
  stream << "Number of bad cells:  " << mBadCells.count() << "\n";
  for (int cellID = 0; cellID < mBadCells.size(); cellID++) {
    if (mBadCells.test(cellID)) {
      stream << cellID + OFFSET << "\n";
    }
  }
}

std::ostream& o2::phos::operator<<(std::ostream& stream, const BadChannelMap& bcm)
{
  bcm.PrintStream(stream);
  return stream;
}
