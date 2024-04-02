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

/// \author Sandro Wenzel - April 2024

#include "Generators/BoxGenerator.h"
#include "TRandom.h"
#include "TDatabasePDG.h"

using namespace o2::eventgen;

double GetPDGMass(int pdg)
{
  static TDatabasePDG* pid = TDatabasePDG::Instance();
  TParticlePDG* p = pid->GetParticle(pdg);
  if (p != nullptr) {
    // LOG(info) << this->ClassName() << ": particle with PDG =" << GetPDGType() << " Found";
    return p->Mass(); // fPDGMass = p->Mass();
  }
  // LOG(warn) << "pdg not known";
  return 0.;
}

TParticle o2::eventgen::BoxGenerator::sampleParticle() const
{
  // Primary particles are distributed uniformly along
  // those kinematics variables which were limitted by setters.
  // if SetCosTheta() function is used, the distribution will be uniform in
  // cos(theta)

  static double mass = GetPDGMass(mPDG);

  double pabs = 0, phi, pt = 0, theta = 0, eta, y, mt, px, py, pz = 0;
  phi = gRandom->Uniform(mPhiMin, mPhiMax) * TMath::DegToRad();
  if (mPRangeIsSet) {
    pabs = gRandom->Uniform(mPMin, mPMax);
  } else if (mPtRangeIsSet) {
    pt = gRandom->Uniform(mPtMin, mPtMax);
  }
  if (mThetaRangeIsSet) {
    if (mCosThetaIsSet) {
      theta = acos(gRandom->Uniform(cos(mThetaMin * TMath::DegToRad()), cos(mThetaMax * TMath::DegToRad())));
    } else {
      theta = gRandom->Uniform(mThetaMin, mThetaMax) * TMath::DegToRad();
    }
  } else if (mEtaRangeIsSet) {
    eta = gRandom->Uniform(mEtaMin, mEtaMax);
    theta = 2 * TMath::ATan(TMath::Exp(-eta));
  } else if (mYRangeIsSet) {
    y = gRandom->Uniform(mYMin, mYMax);
    mt = TMath::Sqrt(mass * mass + pt * pt);
    pz = mt * TMath::SinH(y);
  }

  if (mThetaRangeIsSet || mEtaRangeIsSet) {
    if (mPRangeIsSet) {
      pz = pabs * TMath::Cos(theta);
      pt = pabs * TMath::Sin(theta);
    } else if (mPtRangeIsSet) {
      pz = pt / TMath::Tan(theta);
    }
  }
  px = pt * TMath::Cos(phi);
  py = pt * TMath::Sin(phi);

  double vx = 0., vy = 0., vz = 0.;
  double etot = TMath::Sqrt(px * px + py * py + pz * pz + mass * mass);
  return TParticle(mPDG, 1 /*status*/, -1 /* mother1 */, -1 /* mother2 */,
                   -1 /* daughter1 */, -1 /* daughter2 */, px, py, pz, etot, vx, vy, vz, 0. /*time*/);
}
