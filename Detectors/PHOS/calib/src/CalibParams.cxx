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

CalibParams::CalibParams(int /*dummy*/)
{
  //produce reasonable objest for test purposes
  mGainCalib.fill(0.005);
  mHGLGRatio.fill(16.);
  mHGTimeCalib.fill(0.);
  mLGTimeCalib.fill(0.);
}

bool CalibParams::setGain(TH2* h, char module)
{
  const char MAXX = 64,
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

  char relid[3] = {module, 1, 1};
  short absId;
  for (char ix = 1; ix <= MAXX; ix++) {
    relid[1] = ix;
    for (char iz = 1; iz <= MAXZ; iz++) {
      relid[2] = iz;

      if (geo->relToAbsNumbering(relid, absId)) {
        mGainCalib[absId] = h->GetBinContent(ix, iz);
      }
    }
  }
  return true;
}

bool CalibParams::setHGLGRatio(TH2* h, char module)
{
  const char MAXX = 64,
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

  char relid[3] = {module, 1, 1};
  short absId;
  for (char ix = 1; ix <= MAXX; ix++) {
    relid[1] = ix;
    for (char iz = 1; iz <= MAXZ; iz++) {
      relid[2] = iz;

      if (geo->relToAbsNumbering(relid, absId)) {
        mHGLGRatio[absId] = h->GetBinContent(ix, iz);
      }
    }
  }
  return true;
}

bool CalibParams::setHGTimeCalib(TH2* h, char module)
{
  const char MAXX = 64,
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

  char relid[3] = {module, 1, 1};
  short absId;
  for (char ix = 1; ix <= MAXX; ix++) {
    relid[1] = ix;
    for (char iz = 1; iz <= MAXZ; iz++) {
      relid[2] = iz;

      if (geo->relToAbsNumbering(relid, absId)) {
        mHGTimeCalib[absId] = h->GetBinContent(ix, iz);
      }
    }
  }
  return true;
}

bool CalibParams::setLGTimeCalib(TH2* h, char module)
{
  const char MAXX = 64,
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

  char relid[3] = {module, 1, 1};
  short absId;
  for (char ix = 1; ix <= MAXX; ix++) {
    relid[1] = ix;
    for (char iz = 1; iz <= MAXZ; iz++) {
      relid[2] = iz;

      if (geo->relToAbsNumbering(relid, absId)) {
        mLGTimeCalib[absId] = h->GetBinContent(ix, iz);
      }
    }
  }
  return true;
}
