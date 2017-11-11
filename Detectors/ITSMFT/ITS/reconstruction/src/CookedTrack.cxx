// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CookedTrack.cxx
/// \brief Implementation of the ITS cooked track
/// \author iouri.belikov@cern.ch

#include <TMath.h>

#include "DetectorsBase/Constants.h"
#include "ITSMFTReconstruction/Cluster.h"
#include "ITSReconstruction/CookedTrack.h"

ClassImp(o2::ITS::CookedTrack)

using namespace o2::ITSMFT;
using namespace o2::ITS;
using namespace o2::Base::Constants;
using namespace o2::Base::Track;

bool CookedTrack::operator<(const CookedTrack& o) const
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

void CookedTrack::getImpactParams(Float_t x, Float_t y, Float_t z, Float_t bz, Float_t ip[2]) const
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
  if ((TMath::Abs(bz) < kAlmost0) || (TMath::Abs(rp4) < kAlmost0)) {
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

void CookedTrack::resetClusters()
{
  //------------------------------------------------------------------
  // Reset the array of attached clusters.
  //------------------------------------------------------------------
  mChi2 = -1.;
  mNClusters = 0;
}

void CookedTrack::setClusterIndex(Int_t l, Int_t i)
{
  //--------------------------------------------------------------------
  // Set the cluster index
  //--------------------------------------------------------------------
  Int_t idx = (l << 28) + i;
  mIndex[mNClusters++] = idx;
}

void CookedTrack::setExternalClusterIndex(Int_t layer, Int_t idx)
{
  //--------------------------------------------------------------------
  // Set the cluster index within an external cluster array 
  //--------------------------------------------------------------------
  mIndex[layer]=idx;
}

Bool_t CookedTrack::propagate(Float_t alpha, Float_t x, Float_t bz)
{
  if (rotate(alpha))
    if (propagateTo(x, bz))
      return kTRUE;

  return kFALSE;
}

Bool_t CookedTrack::update(const Cluster& c, Float_t chi2, Int_t idx)
{
  //--------------------------------------------------------------------
  // Update track params
  //--------------------------------------------------------------------
  if (!o2::Base::Track::TrackParCov::update(static_cast<const o2::Base::BaseCluster<float>&>(c)))
    return kFALSE;

  mChi2 += chi2;
  mIndex[mNClusters++] = idx;

  return kTRUE;
}

/*
Bool_t CookedTrack::getPhiZat(Float_t r, Float_t &phi, Float_t &z) const
{
  //------------------------------------------------------------------
  // This function returns the global cylindrical (phi,z) of the track
  // position estimated at the radius r.
  // The track curvature is neglected.
  //------------------------------------------------------------------
  Float_t d=GetD(0., 0., GetBz());
  if (TMath::Abs(d) > r) {
    if (r>1e-1) return kFALSE;
    r = TMath::Abs(d);
  }

  Float_t rcurr=TMath::Sqrt(GetX()*GetX() + GetY()*GetY());
  if (TMath::Abs(d) > rcurr) return kFALSE;
  Float_t globXYZcurr[3]; GetXYZ(globXYZcurr);
  Float_t phicurr=TMath::ATan2(globXYZcurr[1],globXYZcurr[0]);

  if (GetX()>=0.) {
    phi=phicurr+TMath::ASin(d/r)-TMath::ASin(d/rcurr);
  } else {
    phi=phicurr+TMath::ASin(d/r)+TMath::ASin(d/rcurr)-TMath::Pi();
  }

  // return a phi in [0,2pi[
  if (phi<0.) phi+=2.*TMath::Pi();
  else if (phi>=2.*TMath::Pi()) phi-=2.*TMath::Pi();
  z=GetZ()+GetTgl()*(TMath::Sqrt((r-d)*(r+d))-TMath::Sqrt((rcurr-d)*(rcurr+d)));

  return kTRUE;
}
*/

Bool_t CookedTrack::isBetter(const CookedTrack& best, Float_t maxChi2) const
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
