// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CPVCalib/CalibParams.h"
#include "CPVBase/Geometry.h"

#include "FairLogger.h"

#include <TH2.h>

#include <iostream>

using namespace o2::cpv;

CalibParams::CalibParams(short /*dummy*/)
{
  //produce reasonable objest for test purposes
  mGainCalib.fill(0.01);
}

bool CalibParams::setGain(TH2* h, short module)
{
  const short MAXX = 128,
              MAXZ = 56;
  if (!h) {
    LOG(ERROR) << "no input histogam";
    return false;
  }

  if (h->GetNbinsX() != MAXX || h->GetNbinsY() != MAXZ) {
    LOG(ERROR) << "Wrong dimentions of input histogram:" << h->GetNbinsX() << "," << h->GetNbinsY() << " instead of " << MAXX << "," << MAXZ;
    return false;
  }

  short relid[3] = {module, 1, 1};
  unsigned short absId;
  for (short ix = 1; ix <= MAXX; ix++) {
    relid[1] = ix;
    for (short iz = 1; iz <= MAXZ; iz++) {
      relid[2] = iz;

      if (Geometry::relToAbsNumbering(relid, absId)) {
        mGainCalib[absId] = h->GetBinContent(ix, iz);
      }
    }
  }
  return true;
}
