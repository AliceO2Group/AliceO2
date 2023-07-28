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
  //These are coordinate positions for the sensitive elements within the FIT mother volume
  //measured from the CAD drawings.  The positive/negative X values are referenced
  //from the back side of the frame lookinmg toward the interaction point
  Float_t mPosModuleAx[Geometry::NCellsA] = {-12.25, -6.15, -0.05, 6.15, 12.25, -12.25, -6.15, -0.05, 6.15, 12.25, -13.58, -7.48, 7.48, 13.58, -12.25, -6.15, 0.05, 6.15, 12.25, -12.25, -6.15, 0.05, 6.15, 12.25};

  Float_t mPosModuleAy[Geometry::NCellsA] = {12.2, 12.2, 13.53, 12.2, 12.2, 6.1, 6.1, 7.43, 6.1, 6.1, 0.0, 0.0, 0.0, 0.0, -6.1, -6.1, -7.43, -6.1, -6.1, -12.2, -12.2, -13.53, -12.2, -12.2};

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
  Double_t xi[NCellsC] = {-15.038271418735729, 15.038271418735729,
                          -15.003757581112167, 15.003757581112167, -9.02690018974363,
                          9.02690018974363, -9.026897413747076, 9.026897413747076,
                          -9.026896531935773, 9.026896531935773, -3.0004568618531313,
                          3.0004568618531313, -3.0270795197907225, 3.0270795197907225,
                          3.0003978432927543, -3.0003978432927543, 3.0270569670429572,
                          -3.0270569670429572, 9.026750365564254, -9.026750365564254,
                          9.026837450695885, -9.026837450695885, 9.026849243816981,
                          -9.026849243816981, 15.038129472387304, -15.038129472387304,
                          15.003621961057961, -15.003621961057961};
  Double_t yi[NCellsC] = {3.1599494336464455, -3.1599494336464455,
                          9.165191680982874, -9.165191680982874, 3.1383331772537426,
                          -3.1383331772537426, 9.165226363918643, -9.165226363918643,
                          15.141616002932361, -15.141616002932361, 9.16517861649866,
                          -9.16517861649866, 15.188854859073416, -15.188854859073416,
                          9.165053319552113, -9.165053319552113, 15.188703787345304,
                          -15.188703787345304, 3.138263189805292, -3.138263189805292,
                          9.165104089644917, -9.165104089644917, 15.141494417823818,
                          -15.141494417823818, 3.1599158563428644, -3.1599158563428644,
                          9.165116302773846, -9.165116302773846};

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
void Geometry::calculateChannelCenter()
{
  // This method calculates the position of each channel center composing both FT0-A
  // and FT0-C, based on the position of their corresponding modules given by
  // "Geometry::setAsideModules()" and "Geometry::setCsideModules()".

  // Ensure the positions of the modules are well defined.
  setAsideModules();
  setCsideModules();

  // Calculate first the positions for the channels for FT0-A. These correspond to the
  // channels 0-95 in the modules mMCP[0-23].
  Double_t delta = 5.3 / 4.;          // Half-channel width (TODO: ask if it actually corresponds to ChannelWidth)
  Double_t xLocalChannels[Nchannels]; // x-positions of all channels ordered according to xi and internal numbering.
  Double_t yLocalChannels[Nchannels]; // y-positions of all channels ordered according to yi and internal numbering.
  Double_t zLocalChannels[Nchannels]; // z-positions of all channels ordered according to zi and internal numbering.
                                      // INFO: We assume here the modules are perpendicular to z, so z(channel) = z(module).

  for (int iModA = 0; iModA < NCellsA; iModA++) {
    xLocalChannels[4 * iModA + 0] = mMCP[iModA].X() - delta;
    xLocalChannels[4 * iModA + 1] = mMCP[iModA].X() + delta;
    xLocalChannels[4 * iModA + 2] = mMCP[iModA].X() - delta;
    xLocalChannels[4 * iModA + 3] = mMCP[iModA].X() + delta;

    yLocalChannels[4 * iModA + 0] = mMCP[iModA].Y() + delta;
    yLocalChannels[4 * iModA + 1] = mMCP[iModA].Y() + delta;
    yLocalChannels[4 * iModA + 2] = mMCP[iModA].Y() - delta;
    yLocalChannels[4 * iModA + 3] = mMCP[iModA].Y() - delta;

    zLocalChannels[4 * iModA + 0] = mMCP[iModA].Z();
    zLocalChannels[4 * iModA + 1] = mMCP[iModA].Z();
    zLocalChannels[4 * iModA + 2] = mMCP[iModA].Z();
    zLocalChannels[4 * iModA + 3] = mMCP[iModA].Z();
  }

  // Calculate then the positions for the channels for FT0-C, corresponding to the
  // channels 96-207 in the modules mMCP[24-51].
  for (int iModC = 0; iModC < NCellsC; iModC++) {
    xLocalChannels[4 * (iModC + NCellsA) + 0] = mMCP[iModC + NCellsA].X() - delta;
    xLocalChannels[4 * (iModC + NCellsA) + 1] = mMCP[iModC + NCellsA].X() + delta;
    xLocalChannels[4 * (iModC + NCellsA) + 2] = mMCP[iModC + NCellsA].X() - delta;
    xLocalChannels[4 * (iModC + NCellsA) + 3] = mMCP[iModC + NCellsA].X() + delta;

    yLocalChannels[4 * (iModC + NCellsA) + 0] = mMCP[iModC + NCellsA].Y() + delta;
    yLocalChannels[4 * (iModC + NCellsA) + 1] = mMCP[iModC + NCellsA].Y() + delta;
    yLocalChannels[4 * (iModC + NCellsA) + 2] = mMCP[iModC + NCellsA].Y() - delta;
    yLocalChannels[4 * (iModC + NCellsA) + 3] = mMCP[iModC + NCellsA].Y() - delta;

    zLocalChannels[4 * (iModC + NCellsA) + 0] = mMCP[iModC + NCellsA].Z();
    zLocalChannels[4 * (iModC + NCellsA) + 1] = mMCP[iModC + NCellsA].Z();
    zLocalChannels[4 * (iModC + NCellsA) + 2] = mMCP[iModC + NCellsA].Z();
    zLocalChannels[4 * (iModC + NCellsA) + 3] = mMCP[iModC + NCellsA].Z();
  }

  for (int iChannel = 0; iChannel < Nchannels; iChannel++) {
    mChannelCenter[localChannelOrder[iChannel]].SetXYZ(xLocalChannels[iChannel], yLocalChannels[iChannel], zLocalChannels[iChannel]);
  }
}
