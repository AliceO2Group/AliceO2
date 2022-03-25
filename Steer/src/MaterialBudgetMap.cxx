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

/*
 *  Created on: March 17, 2022
 *      Author: amorsch
 */

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//
//  Utility class to evaluate the material budget from
//  a given radius to the surface of an arbitrary cylinder
//  along radial directions from the centre:
//
//   - radiation length
//   - Interaction length
//   - g/cm2
//
//  Geantinos are shot in the bins in the fNtheta bins in theta
//  and fNphi bins in phi with specified rectangular limits.
//  The statistics are accumulated per
//    fRadMin < r < fRadMax    and  <0 < z < fZMax
//
//////////////////////////////////////////////////////////////

#include "TH2.h"
#include "TMath.h"
#include "TString.h"
#include "TVirtualMC.h"
#include "TFile.h"

#include "Steer/MaterialBudgetMap.h"
using namespace o2::steer;
//_______________________________________________________________________
MaterialBudgetMap::MaterialBudgetMap() : mMode(-1),
                                         mTotRadl(0),
                                         mTotAbso(0),
                                         mTotGcm2(0),
                                         mHistRadl(0),
                                         mHistAbso(0),
                                         mHistGcm2(0),
                                         mHistReta(0),
                                         mRZR(0),
                                         mRZA(0),
                                         mRZG(0),
                                         mStopped(0)
{
  //
  // Default constructor
  //
}

//_______________________________________________________________________
MaterialBudgetMap::MaterialBudgetMap(const char* title, Int_t mode, Int_t nc1, Float_t c1min,
                                     Float_t c1max, Int_t nphi, Float_t phimin, Float_t phimax,
                                     Float_t rmin, Float_t rmax, Float_t zmax) : mMode(mode),
                                                                                 mTotRadl(0),
                                                                                 mTotAbso(0),
                                                                                 mTotGcm2(0),
                                                                                 mHistRadl(0),
                                                                                 mHistAbso(0),
                                                                                 mHistGcm2(0),
                                                                                 mHistReta(0),
                                                                                 mRZR(0),
                                                                                 mRZA(0),
                                                                                 mRZG(0),
                                                                                 mStopped(0),
                                                                                 mRmin(rmin),
                                                                                 mRmax(rmax),
                                                                                 mZmax(zmax)
{
  //
  // specify the angular limits and the size of the rectangular box
  //
  const char* xtitles[3] = {"#theta [degree]", "#eta", "z [cm]"};
  mHistRadl = new TH2F("hradl", "Radiation length map",
                       nc1, c1min, c1max, nphi, phimin, phimax);
  mHistRadl->SetYTitle("#phi [degree]");
  mHistRadl->SetXTitle(xtitles[mMode]);
  mHistAbso = new TH2F("habso", "Interaction length map",
                       nc1, c1min, c1max, nphi, phimin, phimax);
  mHistAbso->SetYTitle("#phi [degree]");
  mHistAbso->SetXTitle(xtitles[mMode]);
  mHistGcm2 = new TH2F("hgcm2", "g/cm2 length map",
                       nc1, c1min, c1max, nphi, phimin, phimax);
  mHistGcm2->SetYTitle("#phi [degree]");
  mHistGcm2->SetXTitle(xtitles[mMode]);
  mRZR = new TH2F("rzR", "Radiation length @ (r,z)",
                  zmax, -zmax, zmax, (rmax - rmin), rmin, rmax);
  mRZR->SetXTitle("#it{z} [cm]");
  mRZR->SetYTitle("#it{r} [cm]");
  mRZA = static_cast<TH2F*>(mRZR->Clone("rzA"));
  mRZA->SetTitle("Interaction length @ (r,z)");
  mRZG = static_cast<TH2F*>(mRZR->Clone("rzG"));
  mRZG->SetTitle("g/cm^{2} @ (r,z)");
}

//_______________________________________________________________________
MaterialBudgetMap::~MaterialBudgetMap()
{
  //
  // Destructor
  //
  delete mHistRadl;
  delete mHistAbso;
  delete mHistGcm2;
}

//_______________________________________________________________________
void MaterialBudgetMap::BeginEvent()
{
  //
  // --- Set to 0 radiation length, absorption length and g/cm2 ---
  //
  mTotRadl = 0;
  mTotAbso = 0;
  mTotGcm2 = 0;
  mStopped = 0;
}

//_______________________________________________________________________
void MaterialBudgetMap::FinishPrimary(Float_t c1, Float_t c2)
{
  //
  // Finish the event and update the histos
  //
  mHistRadl->Fill(c1, c2, mTotRadl);
  mHistAbso->Fill(c1, c2, mTotAbso);
  mHistGcm2->Fill(c1, c2, mTotGcm2);
  mTotRadl = 0;
  mTotAbso = 0;
  mTotGcm2 = 0;
  mStopped = 0;
}

//_______________________________________________________________________
void MaterialBudgetMap::FinishEvent()
{
  //
  // Store histograms in current Root file
  //
  printf("map::finish event \n");
  auto f = new TFile("o2sim_matbudget.root", "recreate");
  mHistRadl->Write();
  mHistAbso->Write();
  mHistGcm2->Write();
  mRZR->Write();
  mRZA->Write();
  mRZG->Write();
  f->Close();
  // Delete histograms from memory
  mHistRadl->Delete();
  mHistRadl = 0;
  mHistAbso->Delete();
  mHistAbso = 0;
  mHistGcm2->Delete();
  mHistGcm2 = 0;
}

//_______________________________________________________________________
void MaterialBudgetMap::Stepping()
{
  //
  // called from AliRun::Stepmanager from gustep.
  // Accumulate the 3 parameters step by step
  //
  static Float_t t;
  Float_t a, z, dens, radl, absl;
  Int_t i, id, copy;
  const char* vol;
  static Float_t vect[3], dir[3];

  TString tmp1, tmp2;
  copy = 1;
  id = TVirtualMC::GetMC()->CurrentVolID(copy);
  vol = TVirtualMC::GetMC()->VolName(id);
  Float_t step = TVirtualMC::GetMC()->TrackStep();

  TLorentzVector pos;
  TVirtualMC::GetMC()->TrackPosition(pos);

  Int_t status = 0;
  if (TVirtualMC::GetMC()->IsTrackEntering())
    status = 1;
  if (TVirtualMC::GetMC()->IsTrackExiting())
    status = 2;

  TVirtualMC::GetMC()->CurrentMaterial(a, z, dens, radl, absl);
  Double_t r = TMath::Sqrt(pos[0] * pos[0] + pos[1] * pos[1]);

  mRZR->Fill(pos[2], r, step / radl);
  mRZA->Fill(pos[2], r, step / absl);
  mRZG->Fill(pos[2], r, step * dens);
  if (z < 1)
    return;
  // --- See if we have to stop now
  if (TMath::Abs(pos[2]) > TMath::Abs(mZmax) || r > mRmax) {
    if (!TVirtualMC::GetMC()->IsNewTrack()) {
      // Not the first step, add past contribution
      if (!mStopped) {
        if (absl)
          mTotAbso += t / absl;
        if (radl)
          mTotRadl += t / radl;
        mTotGcm2 += t * dens;
      } // not stooped
    }   // not a new track !
    mStopped = kTRUE;
    TVirtualMC::GetMC()->StopTrack();
    return;
  } // outside scoring region ?
  if (step) {
    if (absl)
      mTotAbso += step / absl;
    if (radl)
      mTotRadl += step / radl;
    mTotGcm2 += step * dens;
  } // step
}
