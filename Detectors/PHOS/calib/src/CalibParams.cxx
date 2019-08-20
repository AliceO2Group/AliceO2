// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "PHOSCalib/CalibParams.h"
#include "PHOSBase/Geometry.h"

#include "FairLogger.h"

#include <TH2.h>

#include <iostream>

using namespace o2::phos;

bool CalibParams::setGain(TH2* h, int module)
{
  const int MAXX = 64,
            MAXZ = 56;
  if (!h) {
    LOG(ERROR) << "no input histogam";
    return false;
  }

  if (h->GetNbinsX() != MAXX || h->GetNbinsY() != MAXZ) {
    LOG(ERROR) << "Wrong dimentions of input histogram:" << h->GetNbinsX() << "," << h->GetNbinsY() << " instead of " << MAXX << "," << MAXZ;
    return false;
  }

  auto geo = Geometry::GetInstance();
  if (!geo) {
    LOG(ERROR) << "Geometry needs to be initialized";
    return false;
  }

  int relid[3] = {module, 1, 1};
  int absId;
  for (int ix = 1; ix <= MAXX; ix++) {
    relid[1] = ix;
    for (int iz = 1; iz <= MAXZ; iz++) {
      relid[2] = iz;

      if (geo->RelToAbsNumbering(relid, absId)) {
        mGainCalib[absId] = h->GetBinContent(ix, iz);
      }
    }
  }
  return true;
}

bool CalibParams::setHGLGRatio(TH2* h, int module)
{
  const int MAXX = 64,
            MAXZ = 56;
  if (!h) {
    LOG(ERROR) << "no input histogam";
    return false;
  }

  if (h->GetNbinsX() != MAXX || h->GetNbinsY() != MAXZ) {
    LOG(ERROR) << "Wrong dimentions of input histogram:" << h->GetNbinsX() << "," << h->GetNbinsY() << " instead of " << MAXX << "," << MAXZ;
    return false;
  }

  auto geo = Geometry::GetInstance();
  if (!geo) {
    LOG(ERROR) << "Geometry needs to be initialized";
    return false;
  }

  int relid[3] = {module, 1, 1};
  int absId;
  for (int ix = 1; ix <= MAXX; ix++) {
    relid[1] = ix;
    for (int iz = 1; iz <= MAXZ; iz++) {
      relid[2] = iz;

      if (geo->RelToAbsNumbering(relid, absId)) {
        mHGLGRatio[absId] = h->GetBinContent(ix, iz);
      }
    }
  }
  return true;
}

bool CalibParams::setHGTimeCalib(TH2* h, int module)
{
  const int MAXX = 64,
            MAXZ = 56;
  if (!h) {
    LOG(ERROR) << "no input histogam";
    return false;
  }

  if (h->GetNbinsX() != MAXX || h->GetNbinsY() != MAXZ) {
    LOG(ERROR) << "Wrong dimentions of input histogram:" << h->GetNbinsX() << "," << h->GetNbinsY() << " instead of " << MAXX << "," << MAXZ;
    return false;
  }

  auto geo = Geometry::GetInstance();
  if (!geo) {
    LOG(ERROR) << "Geometry needs to be initialized";
    return false;
  }

  int relid[3] = {module, 1, 1};
  int absId;
  for (int ix = 1; ix <= MAXX; ix++) {
    relid[1] = ix;
    for (int iz = 1; iz <= MAXZ; iz++) {
      relid[2] = iz;

      if (geo->RelToAbsNumbering(relid, absId)) {
        mHGTimeCalib[absId] = h->GetBinContent(ix, iz);
      }
    }
  }
  return true;
}

bool CalibParams::setLGTimeCalib(TH2* h, int module)
{
  const int MAXX = 64,
            MAXZ = 56;
  if (!h) {
    LOG(ERROR) << "no input histogam";
    return false;
  }

  if (h->GetNbinsX() != MAXX || h->GetNbinsY() != MAXZ) {
    LOG(ERROR) << "Wrong dimentions of input histogram:" << h->GetNbinsX() << "," << h->GetNbinsY() << " instead of " << MAXX << "," << MAXZ;
    return false;
  }

  auto geo = Geometry::GetInstance();
  if (!geo) {
    LOG(ERROR) << "Geometry needs to be initialized";
    return false;
  }

  int relid[3] = {module, 1, 1};
  int absId;
  for (int ix = 1; ix <= MAXX; ix++) {
    relid[1] = ix;
    for (int iz = 1; iz <= MAXZ; iz++) {
      relid[2] = iz;

      if (geo->RelToAbsNumbering(relid, absId)) {
        mLGTimeCalib[absId] = h->GetBinContent(ix, iz);
      }
    }
  }
  return true;
}
