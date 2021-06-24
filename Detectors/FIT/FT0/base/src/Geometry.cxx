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
#include "FT0Base/Geometry.h"
#include "TSystem.h"
#include "Framework/Logger.h"
#include <sstream>
#include <iomanip>
#include <string>
#include <iostream>

ClassImp(o2::ft0::Geometry);

using namespace TMath;
using namespace o2::detectors;
using namespace o2::ft0;

Geometry::Geometry() : mMCP{{0, 0, 0}}
{

  setAsideModules();
  setCsideModules();
}

void Geometry::setAsideModules()
{
  Float_t mPosModuleAx[Geometry::NCellsA] = {-12.2, -6.1, 0, 6.1, 12.2, -12.2, -6.1, 0,
                                             6.1, 12.2, -13.3743, -7.274299999999999,
                                             7.274299999999999, 13.3743, -12.2, -6.1, 0,
                                             6.1, 12.2, -12.2, -6.1, 0, 6.1, 12.2};

  Float_t mPosModuleAy[Geometry::NCellsA] = {12.2, 12.2, 13.53, 12.2, 12.2, 6.1, 6.1,
                                             7.43, 6.1, 6.1, 0, 0, 0, 0, -6.1, -6.1,
                                             -7.43, -6.1, -6.1, -12.2, -12.2, -13.53,
                                             -12.2, -12.2};

  // A side Translations
  for (Int_t ipmt = 0; ipmt < NCellsA; ipmt++) {
    mMCP[ipmt].SetXYZ(mPosModuleAx[ipmt], mPosModuleAy[ipmt], ZdetA);
  }
}
void Geometry::setCsideModules()
{
  // C side Concave Geometry
  Float_t mInStart[3] = {2.9491, 2.9491, 2.5};
  Float_t mStartC[3] = {20., 20, 5.5};

  Double_t crad = ZdetC; // define concave c-side radius here

  Double_t dP = mInStart[0]; // side length of mcp divided by 2

  // uniform angle between detector faces==
  Double_t btta = 2 * TMath::ATan(dP / crad);

  // get noncompensated translation data
  Double_t grdin[6] = {-3, -2, -1, 1, 2, 3};
  Double_t gridpoints[6];
  for (Int_t i = 0; i < 6; i++) {
    gridpoints[i] = crad * TMath::Sin((1 - 1 / (2 * TMath::Abs(grdin[i]))) * grdin[i] * btta);
  }

  Double_t xi[NCellsC] = {gridpoints[1], gridpoints[2], gridpoints[3], gridpoints[4], gridpoints[0],
                          gridpoints[1], gridpoints[2], gridpoints[3], gridpoints[4], gridpoints[5],
                          gridpoints[0], gridpoints[1], gridpoints[4], gridpoints[5], gridpoints[0],
                          gridpoints[1], gridpoints[4], gridpoints[5], gridpoints[0], gridpoints[1],
                          gridpoints[2], gridpoints[3], gridpoints[4], gridpoints[5], gridpoints[1],
                          gridpoints[2], gridpoints[3], gridpoints[4]};
  Double_t yi[NCellsC] = {gridpoints[5], gridpoints[5], gridpoints[5], gridpoints[5], gridpoints[4],
                          gridpoints[4], gridpoints[4], gridpoints[4], gridpoints[4], gridpoints[4],
                          gridpoints[3], gridpoints[3], gridpoints[3], gridpoints[3], gridpoints[2],
                          gridpoints[2], gridpoints[2], gridpoints[2], gridpoints[1], gridpoints[1],
                          gridpoints[1], gridpoints[1], gridpoints[1], gridpoints[1], gridpoints[0],
                          gridpoints[0], gridpoints[0], gridpoints[0]};
  Double_t zi[NCellsC];
  for (Int_t i = 0; i < NCellsC; i++) {
    zi[i] = TMath::Sqrt(TMath::Power(crad, 2) - TMath::Power(xi[i], 2) - TMath::Power(yi[i], 2));
  }

  // get rotation data
  Double_t ac[NCellsC], bc[NCellsC], gc[NCellsC];
  for (Int_t i = 0; i < NCellsC; i++) {
    ac[i] = TMath::ATan(yi[i] / xi[i]) - TMath::Pi() / 2 + 2 * TMath::Pi();
    if (xi[i] < 0) {
      bc[i] = TMath::ACos(zi[i] / crad);
    } else {
      bc[i] = -1 * TMath::ACos(zi[i] / crad);
    }
  }
  Double_t xc2[NCellsC], yc2[NCellsC], zc2[NCellsC];

  // compensation based on node position within individual detector geometries
  // determine compensated radius
  Double_t rcomp = crad + mStartC[2] / 2.0; //
  for (Int_t i = 0; i < NCellsC; i++) {
    // Get compensated translation data
    xc2[i] = rcomp * TMath::Cos(ac[i] + TMath::Pi() / 2) * TMath::Sin(-1 * bc[i]);
    yc2[i] = rcomp * TMath::Sin(ac[i] + TMath::Pi() / 2) * TMath::Sin(-1 * bc[i]);
    zc2[i] = rcomp * TMath::Cos(bc[i]);

    // Convert angles to degrees
    ac[i] *= 180 / TMath::Pi();
    bc[i] *= 180 / TMath::Pi();
    gc[i] = -1 * ac[i];
    mAngles[i].SetXYZ(ac[i], bc[i], gc[i]);
    mMCP[i + NCellsA].SetXYZ(xc2[i], yc2[i], zc2[i]);
  }
}
