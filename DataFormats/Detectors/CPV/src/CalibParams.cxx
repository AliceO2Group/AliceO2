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

#include "DataFormatsCPV/CalibParams.h"
#include "CPVBase/Geometry.h"

#include <fairlogger/Logger.h>

#include <TH2.h>

#include <iostream>

using namespace o2::cpv;

CalibParams::CalibParams(short /*dummy*/)
{
  //produce reasonable objest for test purposes
  mGainCalib.fill(1.);
}

bool CalibParams::setGain(TH2* h, short module)
{
  const short MAXX = 128,
              MAXZ = 60;
  if (!h) {
    LOG(error) << "no input histogam";
    return false;
  }

  if (h->GetNbinsX() != MAXX || h->GetNbinsY() != MAXZ) {
    LOG(error) << "Wrong dimentions of input histogram:" << h->GetNbinsX() << "," << h->GetNbinsY() << " instead of " << MAXX << "," << MAXZ;
    return false;
  }

  short relid[3] = {module, 0, 0};
  unsigned short absId;
  for (short ix = 1; ix <= MAXX; ix++) {
    relid[1] = ix - 1;
    for (short iz = 1; iz <= MAXZ; iz++) {
      relid[2] = iz - 1;

      if (Geometry::relToAbsNumbering(relid, absId)) {
        mGainCalib[absId] = h->GetBinContent(ix, iz);
      }
    }
  }
  return true;
}
