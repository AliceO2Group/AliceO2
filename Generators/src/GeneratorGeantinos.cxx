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
//
/// \author A+Morsch - March 2022

//------------------------------------------------------------------------
//    Generates geantino rays to check the material distributions and detector's
//    geometry
//------------------------------------------------------------------------

#include "Generators/GeneratorGeantinos.h"
#include <TParticle.h>
#include <FairPrimaryGenerator.h>

namespace o2
{
namespace eventgen
{
//_______________________________________________________________________
GeneratorGeantinos::GeneratorGeantinos() : FairGenerator(),
                                           mMode(-1),
                                           mRadMin(0),
                                           mRadMax(0),
                                           mZMax(0),
                                           mNCoor1(0),
                                           mNCoor2(0),
                                           mCoor1Min(0),
                                           mCoor1Max(0),
                                           mCoor2Min(0),
                                           mCoor2Max(0)
{
}

//_______________________________________________________________________
GeneratorGeantinos::GeneratorGeantinos(Int_t mode, Int_t nc1, Float_t c1min,
                                       Float_t c1max, Int_t nc2,
                                       Float_t c2min, Float_t c2max,
                                       Float_t rmin, Float_t rmax, Float_t zmax) :

                                                                                   FairGenerator(),
                                                                                   mMode(mode),
                                                                                   mRadMin(rmin),
                                                                                   mRadMax(rmax),
                                                                                   mZMax(zmax),
                                                                                   mNCoor1(nc1),
                                                                                   mNCoor2(nc2),
                                                                                   mCoor1Min(c1min),
                                                                                   mCoor1Max(c1max),
                                                                                   mCoor2Min(c2min),
                                                                                   mCoor2Max(c2max)
{
  //
  // Standard generator for geantinos
  //
}

//_______________________________________________________________________
Bool_t GeneratorGeantinos::ReadEvent(FairPrimaryGenerator* primGen)
{
  Float_t orig[3], pmom[3];
  Float_t t, cost, sint, cosp, sinp;
  Float_t s1 = (mCoor1Max - mCoor1Min) / mNCoor1;
  Float_t s2 = (mCoor2Max - mCoor2Min) / mNCoor2;
  const Float_t g2rad = TMath::Pi() / 180.;
  for (Int_t i = 0; i < mNCoor1; i++) {
    for (Int_t j = 0; j < mNCoor2; j++) {
      Float_t c1 = mCoor1Min + i * s1 + s1 / 2.;
      Float_t c2 = mCoor2Min + j * s2 + s2 / 2.;
      if (mMode == 2) {
        sint = 1.;
        cost = 0.;
      } else if (mMode == 1) {
        Float_t theta = 2. * TMath::ATan(TMath::Exp(-c1));
        sint = TMath::Sin(theta);
        cost = TMath::Cos(theta);
      } else {
        cost = TMath::Cos(c1 * g2rad);
        sint = TMath::Sin(c1 * g2rad);
      }

      cosp = TMath::Cos(c2 * g2rad);
      sinp = TMath::Sin(c2 * g2rad);

      pmom[0] = cosp * sint;
      pmom[1] = sinp * sint;
      pmom[2] = cost;

      // --- Where to start
      orig[0] = orig[1] = orig[2] = 0;
      if (mMode == 2) {
        orig[2] = c1;
      }
      Float_t dalicz = 3000;
      if (mRadMin > 0) {
        t = PropagateCylinder(orig, pmom, mRadMin, dalicz);
        orig[0] = pmom[0] * t;
        orig[1] = pmom[1] * t;
        orig[2] = pmom[2] * t;
        if (TMath::Abs(orig[2]) > mZMax) {
          return kFALSE;
        }
      }
      Float_t polar[3] = {0., 0., 0.};
      primGen->AddTrack(0, pmom[0], pmom[1], pmom[2], orig[0], orig[1], orig[2], -1, true, 0, 0, 1.);
    } // j
  }   // i
  return kTRUE;
}

//_______________________________________________________________________
Float_t GeneratorGeantinos::PropagateCylinder(Float_t* x, Float_t* v, Float_t r,
                                              Float_t z)
{
  //
  // Propagate to cylinder from inside
  //
  Double_t hnorm, sz, t, t1, t2, t3, sr;
  Double_t d[3];
  const Float_t kSmall = 1e-8;
  const Float_t kSmall2 = kSmall * kSmall;

  // ---> Find intesection with Z planes
  d[0] = v[0];
  d[1] = v[1];
  d[2] = v[2];
  hnorm = TMath::Sqrt(1 / (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]));
  d[0] *= hnorm;
  d[1] *= hnorm;
  d[2] *= hnorm;
  if (d[2] > kSmall) {
    sz = (z - x[2]) / d[2];
  } else if (d[2] < -kSmall) {
    sz = -(z + x[2]) / d[2];
  } else {
    sz = 1.e10; // ---> Direction parallel to X-Y, no intersection
  }

  t1 = d[0] * d[0] + d[1] * d[1];
  if (t1 <= kSmall2) {
    t = sz; // ---> Track parallel to the z-axis, take distance to planes
  } else {
    t2 = x[0] * d[0] + x[1] * d[1];
    t3 = x[0] * x[0] + x[1] * x[1];
    // ---> It should be positive, but there may be numerical problems
    sr = (-t2 + TMath::Sqrt(TMath::Max(t2 * t2 - (t3 - r * r) * t1, 0.))) / t1;
    // ---> Find minimum distance between planes and cylinder
    t = TMath::Min(sz, sr);
  }
  return t;
}
} // namespace eventgen
} // namespace o2
