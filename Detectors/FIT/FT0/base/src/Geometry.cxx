// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <iomanip>
//#include <TVector3.h>
#include "FT0Base/Geometry.h"

#include <FairLogger.h>
#include <sstream>

ClassImp(o2::ft0::Geometry);

using namespace o2::ft0;

Geometry::Geometry() : mMCP{{0, 0, 0}}
{

  Float_t zDetA = 333;
  Float_t xa[24] = {-11.8, -5.9, 0, 5.9, 11.8, -11.8, -5.9, 0, 5.9, 11.8, -12.8, -6.9,
                    6.9, 12.8, -11.8, -5.9, 0, 5.9, 11.8, -11.8, -5.9, 0, 5.9, 11.8};

  Float_t ya[24] = {11.9, 11.9, 12.9, 11.9, 11.9, 6.0, 6.0, 7.0, 6.0, 6.0, -0., -0.,
                    0., 0., -6.0, -6.0, -7.0, -6.0, -6.0, -11.9, -11.9, -12.9, -11.9, -11.9};

  Float_t pmcp[3] = {2.949, 2.949, 2.8}; // MCP

  // Matrix(idrotm[901], 90., 0., 90., 90., 180., 0.);

  // C side Concave Geometry

  Double_t crad = 82.; // define concave c-side radius here

  Double_t dP = pmcp[0]; // side length of mcp divided by 2

  // uniform angle between detector faces==
  Double_t btta = 2 * TMath::ATan(dP / crad);

  // get noncompensated translation data
  Double_t grdin[6] = {-3, -2, -1, 1, 2, 3};
  Double_t gridpoints[6];
  for (Int_t i = 0; i < 6; i++) {
    gridpoints[i] = crad * TMath::Sin((1 - 1 / (2 * TMath::Abs(grdin[i]))) * grdin[i] * btta);
  }

  Double_t xi[28] = {gridpoints[1], gridpoints[2], gridpoints[3], gridpoints[4], gridpoints[0], gridpoints[1],
                     gridpoints[2], gridpoints[3], gridpoints[4], gridpoints[5], gridpoints[0], gridpoints[1],
                     gridpoints[4], gridpoints[5], gridpoints[0], gridpoints[1], gridpoints[4], gridpoints[5],
                     gridpoints[0], gridpoints[1], gridpoints[2], gridpoints[3], gridpoints[4], gridpoints[5],
                     gridpoints[1], gridpoints[2], gridpoints[3], gridpoints[4]};
  Double_t yi[28] = {gridpoints[5], gridpoints[5], gridpoints[5], gridpoints[5], gridpoints[4], gridpoints[4],
                     gridpoints[4], gridpoints[4], gridpoints[4], gridpoints[4], gridpoints[3], gridpoints[3],
                     gridpoints[3], gridpoints[3], gridpoints[2], gridpoints[2], gridpoints[2], gridpoints[2],
                     gridpoints[1], gridpoints[1], gridpoints[1], gridpoints[1], gridpoints[1], gridpoints[1],
                     gridpoints[0], gridpoints[0], gridpoints[0], gridpoints[0]};
  Double_t zi[28];
  for (Int_t i = 0; i < 28; i++) {
    zi[i] = TMath::Sqrt(TMath::Power(crad, 2) - TMath::Power(xi[i], 2) - TMath::Power(yi[i], 2));
  }

  // get rotation data
  Double_t ac[28], bc[28], gc[28];
  for (Int_t i = 0; i < 28; i++) {
    ac[i] = TMath::ATan(yi[i] / xi[i]) - TMath::Pi() / 2 + 2 * TMath::Pi();
    if (xi[i] < 0) {
      bc[i] = TMath::ACos(zi[i] / crad);
    } else {
      bc[i] = -1 * TMath::ACos(zi[i] / crad);
    }
  }
  Double_t xc2[28], yc2[28], zc2[28];

  // compensation based on node position within individual detector geometries
  // determine compensated radius
  Double_t rcomp = crad /*+ pstartC[2] / 2.0*/; //
  for (Int_t i = 0; i < 28; i++) {
    // Get compensated translation data
    xc2[i] = rcomp * TMath::Cos(ac[i] + TMath::Pi() / 2) * TMath::Sin(-1 * bc[i]);
    yc2[i] = rcomp * TMath::Sin(ac[i] + TMath::Pi() / 2) * TMath::Sin(-1 * bc[i]);
    zc2[i] = rcomp * TMath::Cos(bc[i]);

    // Convert angles to degrees
    ac[i] *= 180 / TMath::Pi();
    bc[i] *= 180 / TMath::Pi();
    gc[i] = -1 * ac[i];
  }
  // Set coordinate
  for (int ipmt = 0; ipmt < 24; ipmt++)
    mMCP[ipmt].SetXYZ(xa[ipmt], xa[ipmt], zDetA);
  for (int ipmt = 24; ipmt < 52; ipmt++)
    mMCP[ipmt].SetXYZ(xc2[ipmt - 24], yc2[ipmt - 24], zc2[ipmt - 24]);
}
