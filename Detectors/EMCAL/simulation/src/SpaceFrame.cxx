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

#include <fairlogger/Logger.h>
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

using namespace o2::emcal;

void SpaceFrame::CreateGeometry()
{
  LOG(debug) << "Create CalFrame Geometry" << std::endl;

  /************ Definition of constants *****************************/

  // space frame parameters from "SINGLE FRAME ASSEMBLY 27D624H.pdf"
  // provided by Lawrence Berkeley Labs, USA
  const int NUMCROSS = 12;
  const int NUMSUBSETS = 3;
  const double TOTALHALFWIDTH = 152.3; // Half Width of a Half Frame
                                       // (CalFrame comes in 2 sections)
  const double BEGINPHI = 76.8;
  const double ENDPHI = 193.03;
  const double BEGINRADIUS = 490.;

  const double HALFFRAMETRANS = TOTALHALFWIDTH + 57.2 / 2.; // Half Frame Connector is 57.2cm wide,
                                                            // Supermodule is 340cm wide
                                                            // Sources: HALF-FRAME-CONNECTOR-27E226A.pdf
                                                            // provided by LBL

  const double FLAGEWIDTH = 15.2;
  const double RIBWIDTH = 1.5;
  const double CROSSBOTTOMHEIGHT = 15.2;
  const double CROSSBOTTOMRADTHICK = 1.5;
  const double CROSSTOPHEIGHT = 1.5;
  const double CROSSTOPRADTHICK = 35. / 2.;

  //const double TOTALPHI = ENDPHI - BEGINPHI;
  const double FLAGEHEIGHT = BEGINRADIUS + 3.;
  const double RIBHEIGHT = FLAGEHEIGHT + 35;
  const double CROSSBOTTOMWIDTH = 0.5 / (Double_t)NUMSUBSETS * (2. * TOTALHALFWIDTH - 8. * FLAGEWIDTH);
  const double CROSSTOPWIDTH = CROSSBOTTOMWIDTH; // mCrossBottomWidth + mFlangeWidth - mRibWidth;
                                                 // for future release pending
                                                 // overlap correction - new TGeoVolume creation

  const double CROSSBEAMARCLENGTH = (112.62597) / (NUMCROSS - 1) - .001; // To account for shape of TGeoBBox
  const double CROSSBOTTOMSTARTRADIUS = BEGINRADIUS + CROSSBOTTOMRADTHICK;
  const double CROSSTOPSTART =
    BEGINRADIUS + 2. * CROSSBOTTOMRADTHICK + CROSSTOPRADTHICK + 0.015; // 0.015 is a
                                                                       // bubblegum and duct tape
                                                                       // fix for an overlap problem
                                                                       // will be worked out in future releases
  const double ENDRADIUS = RIBHEIGHT + 1.15;
  const double ENDBEAMRADTHICK = CROSSBOTTOMRADTHICK + CROSSTOPRADTHICK;
  const double ENDBEAMBEGINRADIUS = BEGINRADIUS + ENDBEAMRADTHICK;

  /************ End efinition of constants **************************/

  //////////////////////////////////////Setup/////////////////////////////////////////
  TGeoVolume* top = gGeoManager->GetVolume("barrel");
  TGeoMedium* steel = gGeoManager->GetMedium("EMC_S steel$");
  TGeoMedium* air = gGeoManager->GetMedium("EMC_Air$");

  //////////////////////////////////// Volumes ///////////////////////////////////////
  TGeoVolume* calFrameMO = gGeoManager->MakeTubs("CalFrame", air, BEGINRADIUS - 2.1, ENDRADIUS, TOTALHALFWIDTH * 3,
                                                 BEGINPHI - 3, ENDPHI + 3); // Mother Volume

  calFrameMO->SetVisibility(kFALSE);

  // Half Frame Mother Volume
  TGeoVolume* calHalfFrameMO = gGeoManager->MakeTubs("HalfFrame", air, BEGINRADIUS - 2, ENDRADIUS, TOTALHALFWIDTH,
                                                     BEGINPHI - 2.9, ENDPHI + 2.9);

  calHalfFrameMO->SetVisibility(kFALSE);

  TGeoVolume* endBeams =
    gGeoManager->MakeBox("End Beams", steel, ENDBEAMRADTHICK, CROSSTOPHEIGHT, TOTALHALFWIDTH); // End Beams

  TGeoVolume* skin = gGeoManager->MakeTubs("skin", steel, RIBHEIGHT + 0.15, ENDRADIUS, TOTALHALFWIDTH, BEGINPHI,
                                           ENDPHI); // back frame

  TGeoVolume* flangeVolume = gGeoManager->MakeTubs("supportBottom", steel, BEGINRADIUS, FLAGEHEIGHT, FLAGEWIDTH,
                                                   BEGINPHI, ENDPHI); // FlangeVolume Beams

  TGeoVolume* ribVolume =
    gGeoManager->MakeTubs("RibVolume", steel, FLAGEHEIGHT, RIBHEIGHT, RIBWIDTH, BEGINPHI, ENDPHI);

  TGeoVolume* subSetCross = gGeoManager->MakeTubs(
    "subSetCross", air, BEGINRADIUS - 1, BEGINRADIUS + 2 * CROSSBOTTOMRADTHICK + 2 * CROSSTOPRADTHICK + 0.15,
    CROSSBOTTOMWIDTH, BEGINPHI, ENDPHI); // Cross Beam Containers
  subSetCross->SetVisibility(kFALSE);
  /*                                            // Obsolete for now
   TGeoVolume *subSetCrossTop =
   gGeoManager->MakeTubs("SubSetCrossTop", air, mBeginRadius+2*mCrossBottomRadThick-1,
   mBeginRadius+2*mCrossBottomRadThick+ 2*mCrossTopRadThick+1, mCrossTopWidth, mBeginPhi, mEndPhi);     // Cross
   subSetCrossTop->SetVisibility(kFALSE);
   */
  TGeoVolume* crossBottomBeams = gGeoManager->MakeBox("crossBottom", steel, CROSSBOTTOMRADTHICK, CROSSBOTTOMHEIGHT,
                                                      CROSSBOTTOMWIDTH); // Cross Beams

  TGeoVolume* crossTopBeams =
    gGeoManager->MakeBox("crossTop", steel, CROSSTOPRADTHICK, CROSSTOPHEIGHT, CROSSTOPWIDTH); // Cross Beams

  TGeoTranslation* trTEST = new TGeoTranslation();
  TGeoRotation* rotTEST = new TGeoRotation();

  Double_t conv = TMath::Pi() / 180.;
  Double_t radAngle = 0;
  Double_t endBeamParam = .4;
  // cout<<"\nmCrossBottomStartRadius: "<<mCrossBottomStartRadius<<"\n";

  for (Int_t i = 0; i < NUMCROSS; i++) {
    Double_t loopPhi = BEGINPHI + 1.8;

    // Cross Bottom Beams

    radAngle = (loopPhi + i * CROSSBEAMARCLENGTH) * conv;

    rotTEST->SetAngles(BEGINPHI + i * CROSSBEAMARCLENGTH, 0,
                       0); //  SetTranslation(Double_t dx, Double_t dy, Double_t dz);
    trTEST->SetTranslation(cos(radAngle) * CROSSBOTTOMSTARTRADIUS, sin(radAngle) * CROSSBOTTOMSTARTRADIUS, 0);

    TGeoCombiTrans* combo = new TGeoCombiTrans(*trTEST, *rotTEST); // TGeoTranslation &tr, const TGeoRotation &rot);
    combo->RegisterYourself();
    crossBottomBeams->SetVisibility(1);
    subSetCross->AddNode(crossBottomBeams, i + 1, combo);
    if (i != 0 && i != NUMCROSS - 1) {
      // Cross Bottom Beams
      rotTEST->SetAngles(BEGINPHI + i * CROSSBEAMARCLENGTH, 0,
                         0); //  SetTranslation(Double_t dx, Double_t dy, Double_t dz);
      trTEST->SetTranslation(cos(radAngle) * CROSSTOPSTART, sin(radAngle) * CROSSTOPSTART, 0);
      crossTopBeams->SetVisibility(1);
      subSetCross->AddNode(crossTopBeams, i + 1, new TGeoCombiTrans(*trTEST, *rotTEST));
    }

    else if (i == 0) {
      rotTEST->SetAngles(BEGINPHI + i * CROSSBEAMARCLENGTH, 0,
                         0); //  SetTranslation(Double_t dx, Double_t dy, Double_t dz);
      trTEST->SetTranslation(cos((77 - endBeamParam) * conv) * (ENDBEAMBEGINRADIUS),
                             sin((77 - endBeamParam) * conv) * (ENDBEAMBEGINRADIUS), 0);
      endBeams->SetVisibility(1);
      calHalfFrameMO->AddNode(endBeams, 1, new TGeoCombiTrans(*trTEST, *rotTEST));
    } else {
      rotTEST->SetAngles(193.03, 0, 0); //  SetTranslation(Double_t dx, Double_t dy, Double_t dz);
      trTEST->SetTranslation(cos((193.03 + endBeamParam) * conv) * (ENDBEAMBEGINRADIUS) /*more duct tape*/,
                             sin((193.03 + endBeamParam) * conv) * (ENDBEAMBEGINRADIUS), 0);
      endBeams->SetVisibility(1);
      calHalfFrameMO->AddNode(endBeams, 2, new TGeoCombiTrans(*trTEST, *rotTEST));
    }
  }

  // Beam Containers

  // Translations

  TGeoTranslation* origin1 = new TGeoTranslation(0, 0, 0); // Equivalent to gGeoIdentity
  TGeoTranslation* origin2 = new TGeoTranslation(0, 0, 2 * (CROSSBOTTOMWIDTH + FLAGEWIDTH));
  TGeoTranslation* origin3 = new TGeoTranslation(0, 0, -2 * (CROSSBOTTOMWIDTH + FLAGEWIDTH));

  // FlangeVolume translations
  TGeoTranslation* str1 = new TGeoTranslation(0, 0, -3 * (CROSSBOTTOMWIDTH + FLAGEWIDTH));
  TGeoTranslation* str2 = new TGeoTranslation(0, 0, -(CROSSBOTTOMWIDTH + FLAGEWIDTH));
  TGeoTranslation* str3 = new TGeoTranslation(0, 0, (CROSSBOTTOMWIDTH + FLAGEWIDTH));
  TGeoTranslation* str4 = new TGeoTranslation(0, 0, 3 * (CROSSBOTTOMWIDTH + FLAGEWIDTH));

  // Half Frame Translations
  TGeoTranslation* halfTrans1 = new TGeoTranslation(0, 0, HALFFRAMETRANS);
  TGeoTranslation* halfTrans2 = new TGeoTranslation(0, 0, -HALFFRAMETRANS);

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

  top->AddNode(calFrameMO, 1, new TGeoTranslation(0., 30., 0.));
  LOG(debug) << "**********************************\nmEndRadius:\t" << ENDRADIUS << std::endl;
}
