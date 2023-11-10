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

/// \file TrackITS.cxx
/// \brief Implementation of the ITS cooked track
/// \author iouri.belikov@cern.ch

#include <TMath.h>

#include "CommonConstants/MathConstants.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITS/TrackITS.h"

using namespace o2::itsmft;
using namespace o2::its;
using namespace o2::constants::math;
using namespace o2::track;

bool TrackITS::operator<(const TrackITS& o) const
{
  //-----------------------------------------------------------------
  // This function compares tracks according to the their curvature
  //-----------------------------------------------------------------
  float co = TMath::Abs(o.getPt());
  float c = TMath::Abs(getPt());
  // float co=o.GetSigmaY2()*o.GetSigmaZ2();
  // float c =GetSigmaY2()*GetSigmaZ2();

  return (c > co);
}

void TrackITS::getImpactParams(float x, float y, float z, float bz, float ip[2]) const
{
  //------------------------------------------------------------------
  // This function calculates the transverse and longitudinal impact parameters
  // with respect to a point with global coordinates (x,y,0)
  // in the magnetic field "bz" (kG)
  //------------------------------------------------------------------
  float f1 = getSnp(), r1 = TMath::Sqrt((1. - f1) * (1. + f1));
  float xt = getX(), yt = getY();
  float sn = TMath::Sin(getAlpha()), cs = TMath::Cos(getAlpha());
  float a = x * cs + y * sn;
  y = -x * sn + y * cs;
  x = a;
  xt -= x;
  yt -= y;

  float rp4 = getCurvature(bz);
  if ((TMath::Abs(bz) < Almost0) || (TMath::Abs(rp4) < Almost0)) {
    ip[0] = -(xt * f1 - yt * r1);
    ip[1] = getZ() + (ip[0] * f1 - xt) / r1 * getTgl() - z;
    return;
  }

  sn = rp4 * xt - f1;
  cs = rp4 * yt + r1;
  a = 2 * (xt * f1 - yt * r1) - rp4 * (xt * xt + yt * yt);
  float rr = TMath::Sqrt(sn * sn + cs * cs);
  ip[0] = -a / (1 + rr);
  float f2 = -sn / rr, r2 = TMath::Sqrt((1. - f2) * (1. + f2));
  ip[1] = getZ() + getTgl() / rp4 * TMath::ASin(f2 * r1 - f1 * r2) - z;
}

bool TrackITS::propagate(float alpha, float x, float bz)
{
  if (rotate(alpha)) {
    if (propagateTo(x, bz)) {
      return true;
    }
  }
  return false;
}

bool TrackITS::update(const Cluster& c, float chi2)
{
  //--------------------------------------------------------------------
  // Update track params
  //--------------------------------------------------------------------
  if (!o2::track::TrackParCov::update(static_cast<const o2::BaseCluster<float>&>(c))) {
    return false;
  }
  mChi2 += chi2;
  return true;
}

bool TrackITS::isBetter(const TrackITS& best, float maxChi2) const
{
  int ncl = getNumberOfClusters();
  int nclb = best.getNumberOfClusters();

  if (ncl >= nclb) {
    float chi2 = getChi2();
    if (chi2 < maxChi2) {
      if (ncl > nclb || chi2 < best.getChi2()) {
        return true;
      }
    }
  }
  return false;
}

int TrackITS::getNFakeClusters()
{
  int nFake{0};
  int firstClus = getFirstClusterLayer();
  int lastClus = firstClus + getNClusters();
  for (int iCl{firstClus}; iCl < lastClus; ++iCl) {
    if (hasHitOnLayer(iCl) && isFakeOnLayer(iCl)) {
      ++nFake;
    }
  }
  return nFake;
}