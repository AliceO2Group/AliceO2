// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  Float_t co = TMath::Abs(o.getPt());
  Float_t c = TMath::Abs(getPt());
  // Float_t co=o.GetSigmaY2()*o.GetSigmaZ2();
  // Float_t c =GetSigmaY2()*GetSigmaZ2();

  return (c > co);
}

void TrackITS::getImpactParams(Float_t x, Float_t y, Float_t z, Float_t bz, Float_t ip[2]) const
{
  //------------------------------------------------------------------
  // This function calculates the transverse and longitudinal impact parameters
  // with respect to a point with global coordinates (x,y,0)
  // in the magnetic field "bz" (kG)
  //------------------------------------------------------------------
  Float_t f1 = getSnp(), r1 = TMath::Sqrt((1. - f1) * (1. + f1));
  Float_t xt = getX(), yt = getY();
  Float_t sn = TMath::Sin(getAlpha()), cs = TMath::Cos(getAlpha());
  Float_t a = x * cs + y * sn;
  y = -x * sn + y * cs;
  x = a;
  xt -= x;
  yt -= y;

  Float_t rp4 = getCurvature(bz);
  if ((TMath::Abs(bz) < Almost0) || (TMath::Abs(rp4) < Almost0)) {
    ip[0] = -(xt * f1 - yt * r1);
    ip[1] = getZ() + (ip[0] * f1 - xt) / r1 * getTgl() - z;
    return;
  }

  sn = rp4 * xt - f1;
  cs = rp4 * yt + r1;
  a = 2 * (xt * f1 - yt * r1) - rp4 * (xt * xt + yt * yt);
  Float_t rr = TMath::Sqrt(sn * sn + cs * cs);
  ip[0] = -a / (1 + rr);
  Float_t f2 = -sn / rr, r2 = TMath::Sqrt((1. - f2) * (1. + f2));
  ip[1] = getZ() + getTgl() / rp4 * TMath::ASin(f2 * r1 - f1 * r2) - z;
}

Bool_t TrackITS::propagate(Float_t alpha, Float_t x, Float_t bz)
{
  if (rotate(alpha))
    if (propagateTo(x, bz))
      return kTRUE;

  return kFALSE;
}

Bool_t TrackITS::update(const Cluster& c, Float_t chi2)
{
  //--------------------------------------------------------------------
  // Update track params
  //--------------------------------------------------------------------
  if (!o2::track::TrackParCov::update(static_cast<const o2::BaseCluster<float>&>(c))) {
    return kFALSE;
  }
  mChi2 += chi2;
  return kTRUE;
}

Bool_t TrackITS::isBetter(const TrackITS& best, Float_t maxChi2) const
{
  Int_t ncl = getNumberOfClusters();
  Int_t nclb = best.getNumberOfClusters();

  if (ncl >= nclb) {
    Float_t chi2 = getChi2();
    if (chi2 < maxChi2) {
      if (ncl > nclb || chi2 < best.getChi2()) {
        return kTRUE;
      }
    }
  }
  return kFALSE;
}
