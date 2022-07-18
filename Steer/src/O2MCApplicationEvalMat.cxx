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

#include <Steer/O2MCApplicationEvalMat.h>
#include <SimConfig/MatMapParams.h>
#include <Generators/GeneratorGeantinos.h>
#include <TObjArray.h>
#include <TLorentzVector.h>
#include <FairPrimaryGenerator.h>

namespace o2
{
namespace steer
{
void O2MCApplicationEvalMat::FinishPrimary()
{
  mMaterialBudgetMap->FinishPrimary(mC1[mMode], mPhi);
  FairMCApplication::FinishPrimary();
}

void O2MCApplicationEvalMat::Stepping()
{
  // dispatch to MaterialBudget Map
  if (fMC->IsNewTrack()) {
    TLorentzVector p;
    fMC->TrackMomentum(p);
    Float_t x, y, z;
    fMC->TrackPosition(x, y, z);

    mPhi = p.Phi() * 180. / TMath::Pi();
    if (mPhi < 0.) {
      mPhi += 360.;
    }

    mC1[0] = p.Theta() * 180. / TMath::Pi();
    mC1[1] = p.Eta();
    mC1[2] = z;
  }

  mMaterialBudgetMap->Stepping();
}

void O2MCApplicationEvalMat::BeginEvent()
{
  if (!mMaterialBudgetMap) {
    auto& p = o2::conf::MatMapParams::Instance();
    Float_t c1min, c1max;
    Int_t n1;
    if (p.ntheta != 0) {
      // theta-phi binning
      mMode = 0;
      n1 = p.ntheta;
      c1min = p.thetamin;
      c1max = p.thetamax;
    } else if (p.neta != 0) {
      // eta-phi binning
      mMode = 1;
      n1 = p.neta;
      c1min = p.etamin;
      c1max = p.etamax;
    } else if (p.nzv != 0) {
      // z-phi binning
      mMode = 2;
      n1 = p.nzv;
      c1min = p.zvmin;
      c1max = p.zvmax;
    }
    printf("MaterialBudgetMap: %5d %13.3f %13.3f %5d %13.3f %13.3f %13.3f %13.3f\n", n1, c1min, c1max, p.nphi, p.phimin, p.phimax, p.rmax, p.zmax);
    mMaterialBudgetMap = new MaterialBudgetMap("Map", mMode,
                                               n1, c1min, c1max, p.nphi, p.phimin, p.phimax, p.rmin, p.rmax, p.zmax);

    auto gen = GetGenerator();
    gen->GetListOfGenerators()->Clear();
    gen->AddGenerator(new o2::eventgen::GeneratorGeantinos(mMode, n1, c1min, c1max, p.nphi, p.phimin, p.phimax, p.rmin, p.rmax, p.zmax));
  }
  mMaterialBudgetMap->BeginEvent();
}

void O2MCApplicationEvalMat::FinishEvent()
{
  mMaterialBudgetMap->FinishEvent();
  O2MCApplicationBase::FinishEvent();
}

} // namespace steer
} // namespace o2
