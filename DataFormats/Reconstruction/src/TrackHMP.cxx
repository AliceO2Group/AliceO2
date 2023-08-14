// Copyright 2020-2022 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ReconstructionDataFormats/TrackParametrizationWithError.h"
#include "ReconstructionDataFormats/TrackParametrization.h"
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/TrackLTIntegral.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/TrackHMP.h"
#include "Field/MagneticField.h"
// #include "DetectorsBase/Propagator.h"

// #include "Field/MagFieldFast.h"

#include <TVector3.h>
#include <TGeoGlobalMagField.h>

#include <Math/Vector3D.h> //fields
// #include "Math/Vector3D.h" //fields

const Double_t kB2C = -0.299792458e-3;

ClassImp(o2::dataformats::TrackHMP)

  using XYZVector = ROOT::Math::XYZVector;

namespace o2
{
namespace dataformats
{

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
TrackHMP::TrackHMP() = default;
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
TrackHMP& TrackHMP::operator=(const o2::track::TrackParCov& t)
{
  // ass. op.

  return *this;
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
bool TrackHMP::intersect(Double_t pnt[3], Double_t norm[3], double bz) const
{
  //+++++++++++++++++++++++++++++++++++++++++
  // Origin: K. Shileev (Kirill.Shileev@cern.ch)
  // Finds point of intersection (if exists) of the helix with the plane.
  // Stores result in fX and fP.
  // Arguments: planePoint,planeNorm - the plane defined by any plane's point
  // and vector, normal to the plane
  // Returns: kTrue if helix intersects the plane, kFALSE otherwise.
  //+++++++++++++++++++++++++++++++++++++++++

  std::array<float, 3> x0;
  getXYZGlo(x0); // get track position in MARS

  // estimates initial helix length up to plane
  Double_t s = (pnt[0] - x0[0]) * norm[0] + (pnt[1] - x0[1]) * norm[1] + (pnt[2] - x0[2]) * norm[2];

  Double_t dist = 99999, distPrev = dist;
  // Double_t p[3];

  std::array<float, 3> x, p;

  while (TMath::Abs(dist) > 0.00001) {

    // calculates helix at the distance s from x0 ALONG the helix

    propagate(s, x, p, bz);
    // distance between current helix position and plane

    dist = (x[0] - pnt[0]) * norm[0] + (x[1] - pnt[1]) * norm[1] + (x[2] - pnt[2]) * norm[2];
    if (TMath::Abs(dist) >= TMath::Abs(distPrev)) { /*Printf("***********************dist > distPrev******************");*/
      return kFALSE;
    }
    distPrev = dist;
    s -= dist;
  }
  // on exit pnt is intersection point,norm is track vector at that point,
  // all in MARS
  for (Int_t i = 0; i < 3; i++) {
    pnt[i] = x.at(i);
    norm[i] = p.at(i);
  }

  return kTRUE;
} // Intersect()
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void TrackHMP::propagate(Double_t len, std::array<float, 3>& x, std::array<float, 3>& p, double bz) const
{
  //+++++++++++++++++++++++++++++++++++++++++
  // Origin: K. Shileev (Kirill.Shileev@cern.ch)
  // Extrapolate track along simple helix in magnetic field
  // Arguments: len -distance alogn helix, [cm]
  //            bz  - mag field, [kGaus]
  // Returns: x and p contain extrapolated positon and momentum
  // The momentum returned for straight-line tracks is meaningless !
  //+++++++++++++++++++++++++++++++++++++++++

  std::array<float, 3> x0;
  getXYZGlo(x0); // get track position in MARS

  x.at(0) = x0.at(0);
  x.at(1) = x0.at(1);
  x.at(2) = x0.at(2);

  if (getPtInv() < o2::constants::math::Almost0 || TMath::Abs(bz) < o2::constants::math::Almost0) { // straight-line tracks

    TVector3 trackDirection(1., 1., 1.);
    trackDirection.SetMag(1);
    trackDirection.SetTheta(getTheta());
    trackDirection.SetPhi(getPhi());

    // Double_t unit[3]; GetDirection(unit);
    x.at(0) += trackDirection.X() * len;
    x.at(1) += trackDirection.Y() * len;
    x.at(2) += trackDirection.Z() * len;

    p.at(0) = trackDirection.X() / o2::constants::math::Almost0;
    p.at(1) = trackDirection.Y() / o2::constants::math::Almost0;
    p.at(2) = trackDirection.Z() / o2::constants::math::Almost0;
  } else {

    getPxPyPzGlo(p);
    Double_t pp = getP();

    Double_t a = -kB2C * bz * getSign(); ////////// what is kB2C
    Double_t rho = a / pp;
    x.at(0) += p.at(0) * TMath::Sin(rho * len) / a - p.at(1) * (1 - TMath::Cos(rho * len)) / a;
    x.at(1) += p.at(1) * TMath::Sin(rho * len) / a + p.at(0) * (1 - TMath::Cos(rho * len)) / a;
    x.at(2) += p.at(2) * len / pp;
    Double_t p0 = p.at(0);
    p.at(0) = p0 * TMath::Cos(rho * len) - p.at(1) * TMath::Sin(rho * len);
    p.at(1) = p.at(1) * TMath::Cos(rho * len) + p0 * TMath::Sin(rho * len);
  }

} // Propagate()
} // namespace dataformats
} // namespace o2
