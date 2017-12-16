// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <TGeoCompositeShape.h>
#include <TGeoCone.h>
#include <TGeoManager.h>
#include <TGeoMatrix.h>
#include <TGeoMedium.h>
#include <TGeoPcon.h>
#include <TGeoTube.h>
#include <TGeoVolume.h>
#include <TVirtualMC.h>

#include "EMCALSimulation/SpaceFrame.h"

using namespace o2::EMCAL;

SpaceFrame::SpaceFrame()
  : mNumCross(0),
    mNumSubSets(0),
    mTotalHalfWidth(0.),
    mBeginPhi(0.),
    mEndPhi(0.),
    mTotalPhi(0.),
    mBeginRadius(0.),
    mHalfFrameTrans(0.),
    mFlangeHeight(0.),
    mFlangeWidth(0.),
    mRibHeight(0.),
    mRibWidth(0.),
    mCrossBottomWidth(0.),
    mCrossTopWidth(0.),
    mCrossBottomHeight(0.),
    mCrossBottomRadThick(0.),
    mCrossBeamArcLength(0.),
    mCrossBottomStartRadius(0.),
    mCrossTopHeight(0.),
    mCrossTopRadThick(0.),
    mCrossTopStart(0.),
    mEndRadius(0.),
    mEndBeamRadThick(0),
    mEndBeamBeginRadius(0)
{
  mNumCross = 12;
  mNumSubSets = 3;
  mTotalHalfWidth = 152.3; // Half Width of a Half Frame
                           // (CalFrame comes in 2 sections)
  mBeginPhi = 76.8;
  mEndPhi = 193.03;
  mBeginRadius = 490.;

  mHalfFrameTrans = mTotalHalfWidth + 57.2 / 2.; // Half Frame Connector is 57.2cm wide,
                                                 // Supermodule is 340cm wide
                                                 // Sources: HALF-FRAME-CONNECTOR-27E226A.pdf
                                                 // provided by LBL

  mFlangeWidth = 15.2;
  mRibWidth = 1.5;
  mCrossBottomHeight = 15.2;
  mCrossBottomRadThick = 1.5;
  mCrossTopHeight = 1.5;
  mCrossTopRadThick = 35. / 2.;

  mTotalPhi = mEndPhi - mBeginPhi;
  mFlangeHeight = mBeginRadius + 3.;
  mRibHeight = mFlangeHeight + 35;
  mCrossBottomWidth = 0.5 / (Double_t)mNumSubSets * (2. * mTotalHalfWidth - 8. * mFlangeWidth);
  mCrossTopWidth = mCrossBottomWidth; // mCrossBottomWidth + mFlangeWidth - mRibWidth;
                                      // for future release pending
                                      // overlap correction - new TGeoVolume creation

  mCrossBeamArcLength = (112.62597) / (mNumCross - 1) - .001; // To account for shape of TGeoBBox
  mCrossBottomStartRadius = mBeginRadius + mCrossBottomRadThick;
  mCrossTopStart =
    mBeginRadius + 2. * mCrossBottomRadThick + mCrossTopRadThick + 0.015; // 0.015 is a
                                                                          // bubblegum and duct tape
                                                                          // fix for an overlap problem
                                                                          // will be worked out in future releases
  mEndRadius = mRibHeight + 1.15;
  mEndBeamRadThick = mCrossBottomRadThick + mCrossTopRadThick;
  mEndBeamBeginRadius = mBeginRadius + mEndBeamRadThick;
}

SpaceFrame::SpaceFrame(const SpaceFrame& frame)
  : mNumCross(frame.mNumCross),
    mNumSubSets(frame.mNumSubSets),
    mTotalHalfWidth(frame.mTotalHalfWidth),
    mBeginPhi(frame.mBeginPhi),
    mEndPhi(frame.mEndPhi),
    mTotalPhi(frame.mTotalPhi),
    mBeginRadius(frame.mBeginRadius),
    mHalfFrameTrans(frame.mHalfFrameTrans),
    mFlangeHeight(frame.mFlangeHeight),
    mFlangeWidth(frame.mFlangeWidth),
    mRibHeight(frame.mRibHeight),
    mRibWidth(frame.mRibWidth),
    mCrossBottomWidth(frame.mCrossBottomWidth),
    mCrossTopWidth(frame.mCrossTopWidth),
    mCrossBottomHeight(frame.mCrossBottomHeight),
    mCrossBottomRadThick(frame.mCrossBottomRadThick),
    mCrossBeamArcLength(frame.mCrossBeamArcLength),
    mCrossBottomStartRadius(frame.mCrossBottomStartRadius),
    mCrossTopHeight(frame.mCrossTopHeight),
    mCrossTopRadThick(frame.mCrossTopRadThick),
    mCrossTopStart(frame.mCrossTopStart),
    mEndRadius(frame.mEndRadius),
    mEndBeamRadThick(frame.mEndBeamRadThick),
    mEndBeamBeginRadius(frame.mEndBeamBeginRadius)
{
}

void SpaceFrame::CreateGeometry()
{
  LOG(DEBUG) << "Create CalFrame Geometry" << std::endl;

  //////////////////////////////////////Setup/////////////////////////////////////////
  TGeoVolume* top = gGeoManager->GetVolume("cave");
  TGeoMedium* steel = gGeoManager->GetMedium("EMC_S steel$");
  TGeoMedium* air = gGeoManager->GetMedium("EMC_Air$");

  //////////////////////////////////// Volumes ///////////////////////////////////////
  TGeoVolume* calFrameMO = gGeoManager->MakeTubs("CalFrame", air, mBeginRadius - 2.1, mEndRadius, mTotalHalfWidth * 3,
                                                 mBeginPhi - 3, mEndPhi + 3); // Mother Volume

  calFrameMO->SetVisibility(kFALSE);

  // Half Frame Mother Volume
  TGeoVolume* calHalfFrameMO = gGeoManager->MakeTubs("HalfFrame", air, mBeginRadius - 2, mEndRadius, mTotalHalfWidth,
                                                     mBeginPhi - 2.9, mEndPhi + 2.9);

  calHalfFrameMO->SetVisibility(kFALSE);

  TGeoVolume* endBeams =
    gGeoManager->MakeBox("End Beams", steel, mEndBeamRadThick, mCrossTopHeight, mTotalHalfWidth); // End Beams

  TGeoVolume* skin = gGeoManager->MakeTubs("skin", steel, mRibHeight + 0.15, mEndRadius, mTotalHalfWidth, mBeginPhi,
                                           mEndPhi); // back frame

  TGeoVolume* flangeVolume = gGeoManager->MakeTubs("supportBottom", steel, mBeginRadius, mFlangeHeight, mFlangeWidth,
                                                   mBeginPhi, mEndPhi); // FlangeVolume Beams

  TGeoVolume* ribVolume =
    gGeoManager->MakeTubs("RibVolume", steel, mFlangeHeight, mRibHeight, mRibWidth, mBeginPhi, mEndPhi);

  TGeoVolume* subSetCross = gGeoManager->MakeTubs(
    "subSetCross", air, mBeginRadius - 1, mBeginRadius + 2 * mCrossBottomRadThick + 2 * mCrossTopRadThick + 0.15,
    mCrossBottomWidth, mBeginPhi, mEndPhi); // Cross Beam Containers
  subSetCross->SetVisibility(kFALSE);
  /*                                            // Obsolete for now
   TGeoVolume *subSetCrossTop =
   gGeoManager->MakeTubs("SubSetCrossTop", air, mBeginRadius+2*mCrossBottomRadThick-1,
   mBeginRadius+2*mCrossBottomRadThick+ 2*mCrossTopRadThick+1, mCrossTopWidth, mBeginPhi, mEndPhi);     // Cross
   subSetCrossTop->SetVisibility(kFALSE);
   */
  TGeoVolume* crossBottomBeams = gGeoManager->MakeBox("crossBottom", steel, mCrossBottomRadThick, mCrossBottomHeight,
                                                      mCrossBottomWidth); // Cross Beams

  TGeoVolume* crossTopBeams =
    gGeoManager->MakeBox("crossTop", steel, mCrossTopRadThick, mCrossTopHeight, mCrossTopWidth); // Cross Beams

  TGeoTranslation* trTEST = new TGeoTranslation();
  TGeoRotation* rotTEST = new TGeoRotation();

  Double_t conv = TMath::Pi() / 180.;
  Double_t radAngle = 0;
  Double_t endBeamParam = .4;
  // cout<<"\nmCrossBottomStartRadius: "<<mCrossBottomStartRadius<<"\n";

  for (Int_t i = 0; i < mNumCross; i++) {
    Double_t loopPhi = mBeginPhi + 1.8;

    // Cross Bottom Beams

    radAngle = (loopPhi + i * mCrossBeamArcLength) * conv;

    rotTEST->SetAngles(mBeginPhi + i * mCrossBeamArcLength, 0,
                       0); //  SetTranslation(Double_t dx, Double_t dy, Double_t dz);
    trTEST->SetTranslation(cos(radAngle) * mCrossBottomStartRadius, sin(radAngle) * mCrossBottomStartRadius, 0);

    TGeoCombiTrans* combo = new TGeoCombiTrans(*trTEST, *rotTEST); // TGeoTranslation &tr, const TGeoRotation &rot);
    combo->RegisterYourself();
    crossBottomBeams->SetVisibility(1);
    subSetCross->AddNode(crossBottomBeams, i + 1, combo);
    if (i != 0 && i != mNumCross - 1) {
      // Cross Bottom Beams
      rotTEST->SetAngles(mBeginPhi + i * mCrossBeamArcLength, 0,
                         0); //  SetTranslation(Double_t dx, Double_t dy, Double_t dz);
      trTEST->SetTranslation(cos(radAngle) * mCrossTopStart, sin(radAngle) * mCrossTopStart, 0);
      crossTopBeams->SetVisibility(1);
      subSetCross->AddNode(crossTopBeams, i + 1, new TGeoCombiTrans(*trTEST, *rotTEST));
    }

    else if (i == 0) {
      rotTEST->SetAngles(mBeginPhi + i * mCrossBeamArcLength, 0,
                         0); //  SetTranslation(Double_t dx, Double_t dy, Double_t dz);
      trTEST->SetTranslation(cos((77 - endBeamParam) * conv) * (mEndBeamBeginRadius),
                             sin((77 - endBeamParam) * conv) * (mEndBeamBeginRadius), 0);
      endBeams->SetVisibility(1);
      calHalfFrameMO->AddNode(endBeams, 1, new TGeoCombiTrans(*trTEST, *rotTEST));
    } else {
      rotTEST->SetAngles(193.03, 0, 0); //  SetTranslation(Double_t dx, Double_t dy, Double_t dz);
      trTEST->SetTranslation(cos((193.03 + endBeamParam) * conv) * (mEndBeamBeginRadius) /*more duct tape*/,
                             sin((193.03 + endBeamParam) * conv) * (mEndBeamBeginRadius), 0);
      endBeams->SetVisibility(1);
      calHalfFrameMO->AddNode(endBeams, 2, new TGeoCombiTrans(*trTEST, *rotTEST));
    }
  }

  // Beam Containers

  // Translations

  TGeoTranslation* origin1 = new TGeoTranslation(0, 0, 0); // Equivalent to gGeoIdentity
  TGeoTranslation* origin2 = new TGeoTranslation(0, 0, 2 * (mCrossBottomWidth + mFlangeWidth));
  TGeoTranslation* origin3 = new TGeoTranslation(0, 0, -2 * (mCrossBottomWidth + mFlangeWidth));

  // FlangeVolume translations
  TGeoTranslation* str1 = new TGeoTranslation(0, 0, -3 * (mCrossBottomWidth + mFlangeWidth));
  TGeoTranslation* str2 = new TGeoTranslation(0, 0, -(mCrossBottomWidth + mFlangeWidth));
  TGeoTranslation* str3 = new TGeoTranslation(0, 0, (mCrossBottomWidth + mFlangeWidth));
  TGeoTranslation* str4 = new TGeoTranslation(0, 0, 3 * (mCrossBottomWidth + mFlangeWidth));

  // Half Frame Translations
  TGeoTranslation* halfTrans1 = new TGeoTranslation(0, 0, mHalfFrameTrans);
  TGeoTranslation* halfTrans2 = new TGeoTranslation(0, 0, -mHalfFrameTrans);

  // Beams Volume
  calHalfFrameMO->AddNode(flangeVolume, 1, str1);
  calHalfFrameMO->AddNode(flangeVolume, 2, str2);
  calHalfFrameMO->AddNode(flangeVolume, 3, str3);
  calHalfFrameMO->AddNode(flangeVolume, 4, str4);

  calHalfFrameMO->AddNode(ribVolume, 1, str1);
  calHalfFrameMO->AddNode(ribVolume, 2, str2);
  calHalfFrameMO->AddNode(ribVolume, 3, str3);
  calHalfFrameMO->AddNode(ribVolume, 4, str4);

  // Cross Beams
  calHalfFrameMO->AddNode(subSetCross, 1, origin1);
  calHalfFrameMO->AddNode(subSetCross, 2, origin2);
  calHalfFrameMO->AddNode(subSetCross, 3, origin3);
  /*                                    // Obsolete for now
   calHalfFrameMO->AddNode(subSetCrossTop, 1, origin1);
   calHalfFrameMO->AddNode(subSetCrossTop, 2, origin2);
   calHalfFrameMO->AddNode(subSetCrossTop, 3, origin3);
   */

  calHalfFrameMO->AddNode(skin, 1, gGeoIdentity);

  calFrameMO->AddNode(calHalfFrameMO, 1, halfTrans1);
  calFrameMO->AddNode(calHalfFrameMO, 2, halfTrans2);

  top->AddNode(calFrameMO, 1, gGeoIdentity);
  LOG(DEBUG) << "**********************************\nmEndRadius:\t" << mEndRadius << std::endl;
}
