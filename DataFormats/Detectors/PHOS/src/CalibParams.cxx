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

#include "DataFormatsPHOS/CalibParams.h"
#include "PHOSBase/Geometry.h"

#include <fairlogger/Logger.h>

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
  if (module < 1 || module > 4) {
    LOG(error) << "module " << module << "does not exist";
    return false;
  }

  if (!h) {
    LOG(error) << "no input histogam";
    return false;
  }

  if (h->GetNbinsX() != MAXX || h->GetNbinsY() != MAXZ) {
    LOG(error) << "Wrong dimentions of input histogram:" << h->GetNbinsX() << "," << h->GetNbinsY() << " instead of " << MAXX << "," << MAXZ;
    return false;
  }

  char relid[3] = {module, 1, 1};
  short absId;
  for (char ix = 1; ix <= MAXX; ix++) {
    relid[1] = ix;
    for (char iz = 1; iz <= MAXZ; iz++) {
      relid[2] = iz;

      if (o2::phos::Geometry::relToAbsNumbering(relid, absId)) {
        if (absId - OFFSET < 0) { //non-existing part of a module 1
          continue;
        }
        mGainCalib[absId - OFFSET] = h->GetBinContent(ix, iz);
      }
    }
  }
  return true;
}

bool CalibParams::setHGLGRatio(TH2* h, char module)
{
  const char MAXX = 64,
             MAXZ = 56;
  if (module < 1 || module > 4) {
    LOG(error) << "module " << module << "does not exist";
    return false;
  }
  if (!h) {
    LOG(error) << "no input histogam";
    return false;
  }

  if (h->GetNbinsX() != MAXX || h->GetNbinsY() != MAXZ) {
    LOG(error) << "Wrong dimentions of input histogram:" << h->GetNbinsX() << "," << h->GetNbinsY() << " instead of " << MAXX << "," << MAXZ;
    return false;
  }

  char relid[3] = {module, 1, 1};
  short absId;
  for (char ix = 1; ix <= MAXX; ix++) {
    relid[1] = ix;
    for (char iz = 1; iz <= MAXZ; iz++) {
      relid[2] = iz;

      if (o2::phos::Geometry::relToAbsNumbering(relid, absId)) {
        if (absId - OFFSET < 0) { //non-existing part of a module 1
          continue;
        }
        mHGLGRatio[absId - OFFSET] = h->GetBinContent(ix, iz);
      }
    }
  }
  return true;
}

bool CalibParams::setHGTimeCalib(TH2* h, char module)
{
  const char MAXX = 64,
             MAXZ = 56;
  if (module < 1 || module > 4) {
    LOG(error) << "module " << module << "does not exist";
    return false;
  }
  if (!h) {
    LOG(error) << "no input histogam";
    return false;
  }

  if (h->GetNbinsX() != MAXX || h->GetNbinsY() != MAXZ) {
    LOG(error) << "Wrong dimentions of input histogram:" << h->GetNbinsX() << "," << h->GetNbinsY() << " instead of " << MAXX << "," << MAXZ;
    return false;
  }

  char relid[3] = {module, 1, 1};
  short absId;
  for (char ix = 1; ix <= MAXX; ix++) {
    relid[1] = ix;
    for (char iz = 1; iz <= MAXZ; iz++) {
      relid[2] = iz;

      if (o2::phos::Geometry::relToAbsNumbering(relid, absId)) {
        if (absId - OFFSET < 0) { //non-existing part of a module 1
          continue;
        }
        mHGTimeCalib[absId - OFFSET] = h->GetBinContent(ix, iz);
      }
    }
  }
  return true;
}

bool CalibParams::setLGTimeCalib(TH2* h, char module)
{
  const char MAXX = 64,
             MAXZ = 56;
  if (module < 1 || module > 4) {
    LOG(error) << "module " << module << "does not exist";
    return false;
  }
  if (!h) {
    LOG(error) << "no input histogam";
    return false;
  }

  if (h->GetNbinsX() != MAXX || h->GetNbinsY() != MAXZ) {
    LOG(error) << "Wrong dimentions of input histogram:" << h->GetNbinsX() << "," << h->GetNbinsY() << " instead of " << MAXX << "," << MAXZ;
    return false;
  }

  char relid[3] = {module, 1, 1};
  short absId;
  for (char ix = 1; ix <= MAXX; ix++) {
    relid[1] = ix;
    for (char iz = 1; iz <= MAXZ; iz++) {
      relid[2] = iz;

      if (o2::phos::Geometry::relToAbsNumbering(relid, absId)) {
        if (absId - OFFSET < 0) { //non-existing part of a module 1
          continue;
        }
        mLGTimeCalib[absId - OFFSET] = h->GetBinContent(ix, iz);
      }
    }
  }
  return true;
}
