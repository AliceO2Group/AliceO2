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

void BadChannelMap::getHistogramRepresentation(int module, TH2* h) const
{
  if (!h) {
    LOG(ERROR) << "provide histogram to be filled";
  }

  const int MAXX = 64,
            MAXZ = 56;
  if (h->GetNbinsX() != MAXX || h->GetNbinsY() != MAXZ) {
    LOG(ERROR) << "Wrong dimentions of input histogram:" << h->GetNbinsX() << "," << h->GetNbinsY() << " instead of " << MAXX << "," << MAXZ;
    return;
  }

  h->Reset();
  auto geo = Geometry::GetInstance();
  if (!geo) {
    LOG(ERROR) << "Geometry needs to be initialized";
    return;
  }

  int relid[3] = {module, 1, 1};
  int absId;
  for (int ix = 1; ix <= MAXX; ix++) {
    relid[1] = ix;
    for (int iz = 1; iz <= MAXZ; iz++) {
      relid[2] = iz;
      if (geo->RelToAbsNumbering(relid, absId)) {
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
    if (mBadCells.test(cellID))
      stream << cellID << "\n";
  }
}

std::ostream& o2::phos::operator<<(std::ostream& stream, const BadChannelMap& bcm)
{
  bcm.PrintStream(stream);
  return stream;
}
