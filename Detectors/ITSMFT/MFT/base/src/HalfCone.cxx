// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HalfCone.cxx
/// \brief Class building geometry of one half of one MFT half-cone
/// \author sbest@pucp.pe, eric.endress@gmx.de, franck.manso@clermont.in2p3.fr
/// \Carlos csoncco@pucp.edu.pe
/// \date 01/07/2020

#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TGeoManager.h"
#include "TGeoCompositeShape.h"
#include "TGeoShape.h"
#include "TGeoCone.h"
#include "TGeoVolume.h"
#include "TGeoMaterial.h"
#include "TGeoMedium.h"
#include "TGeoTube.h"
#include "TGeoTrd1.h"
#include "TMath.h"
#include "TGeoXtru.h"

#include "MFTBase/HalfCone.h"
#include "MFTBase/Constants.h"

using namespace o2::mft;

ClassImp(o2::mft::HalfCone);

//_____________________________________________________________________________
HalfCone::HalfCone() : mHalfCone(nullptr)
{

  // default constructor
}

//_____________________________________________________________________________
HalfCone::~HalfCone() = default;

//_____________________________________________________________________________
TGeoVolumeAssembly* HalfCone::createHalfCone(Int_t half)
{

  auto* HalfConeVolume = new TGeoVolumeAssembly("HalfConeVolume");

  TGeoElement* Silicon = new TGeoElement("Silicon", "Silicon", 14, 28.0855);
  TGeoElement* Iron = new TGeoElement("Iron", "Iron", 26, 55.845);
  TGeoElement* Copper = new TGeoElement("Copper", "Copper", 29, 63.546);
  TGeoElement* Manganese =
    new TGeoElement("Manganese", "Manganese", 25, 54.938049);
  TGeoElement* Magnesium =
    new TGeoElement("Magnesium", "Magnesium", 12, 24.3050);
  TGeoElement* Zinc = new TGeoElement("Zinc", "Zinc", 30, 65.39);
  TGeoElement* Titanium = new TGeoElement("Titanium", "Titanium", 22, 47.867);
  TGeoElement* Chromium = new TGeoElement("Chromium", "Chromium", 24, 51.9961);
  TGeoElement* Aluminum = new TGeoElement("Aluminum", "Aluminum", 13, 26981538);

  TGeoMixture* alu5083 = new TGeoMixture("alu5083", 9, 2.650); // g/cm3
  alu5083->AddElement(Iron, 0.004);
  alu5083->AddElement(Silicon, 0.004);
  alu5083->AddElement(Copper, 0.001);
  alu5083->AddElement(Manganese, 0.002);
  alu5083->AddElement(Magnesium, 0.0025);
  alu5083->AddElement(Zinc, 0.0025);
  alu5083->AddElement(Titanium, 0.0015);
  alu5083->AddElement(Chromium, 0.0010);
  alu5083->AddElement(Aluminum, 0.9815);
  // alu5083->SetTemperature(); // 25.
  alu5083->SetDensity(2.65); // g/cm3
  alu5083->SetState(TGeoMaterial::kMatStateSolid);

  TGeoMedium* kMedAlu = gGeoManager->GetMedium("MFT_Alu$");
  TGeoMedium* malu5083 =
    new TGeoMedium("malu5083", 2, alu5083); // name, numer medio, material

  // Rotation
  TGeoRotation* rot1 = new TGeoRotation("rot1", 180, -180, 0);

  rot1->RegisterYourself();
  TGeoRotation* rot2 = new TGeoRotation("rot2", 90, -90, 0);
  rot2->RegisterYourself();

  TGeoRotation* rot3 = new TGeoRotation("rot3", 0, 90, 0);
  rot3->RegisterYourself();

  TGeoRotation* rot_90x = new TGeoRotation("rot_90x", 0, -90, 0); // half0
  rot_90x->RegisterYourself();

  TGeoRotation* rot_base = new TGeoRotation("rot_base", 180, 180, 0); // rail_r
  rot_base->RegisterYourself();

  TGeoCombiTrans* combi1 =
    new TGeoCombiTrans(0, -10.3, 1.29, rot1); // y=-10.80 belt
  combi1->RegisterYourself();
  TGeoCombiTrans* combi2 = new TGeoCombiTrans(-16.8, 0., 0., rot2);
  combi2->RegisterYourself();

  TGeoRotation* r0 = new TGeoRotation("r0", 10., 0., 0.);
  r0->RegisterYourself();

  // 1st piece Cross_beam_MB0

  auto* Cross_mb0 = new TGeoVolumeAssembly("Cross_mb0");

  // rectangular box
  Double_t x_boxmb0 = 14.4; // dx= 7.2 cm
  Double_t y_boxmb0 = 0.6;
  Double_t z_boxmb0 = 0.6;

  ///// holes tub  1hole tranversal
  Double_t radin_1hmb0 = 0.;
  Double_t radout_1hmb0 = 0.175; // diameter 3.5 H9  (0.35cm)
  Double_t high_1hmb0 = 0.7;

  TGeoRotation* rot_1hole_mb0 = new TGeoRotation("rot_1hole_mb0", 0, 90, 0);
  rot_1hole_mb0->RegisterYourself();
  // h= hole
  TGeoCombiTrans* acombi_1h_mb0 = new TGeoCombiTrans(5.2, 0, 0, rot_1hole_mb0);
  acombi_1h_mb0->SetName("acombi_1h_mb0");
  acombi_1h_mb0->RegisterYourself();
  TGeoCombiTrans* bcombi_1h_mb0 =
    new TGeoCombiTrans(-5.2, 0, 0, rot_1hole_mb0); // y=
  bcombi_1h_mb0->SetName("bcombi_1h_mb0");
  bcombi_1h_mb0->RegisterYourself();

  // 2hole coaxial
  Double_t radin_2hmb0 = 0.;
  Double_t radout_2hmb0 = 0.15; // diameter M3
  Double_t high_2hmb0 = 1.2;    // 12

  TGeoRotation* rot_2hole_mb0 = new TGeoRotation("rot_2hole_mb0", 90, 90, 0);
  rot_2hole_mb0->SetName("rot_2hole_mb0");
  rot_2hole_mb0->RegisterYourself();

  TGeoCombiTrans* combi_2hole_mb0 =
    new TGeoCombiTrans(6.7, 0, 0, rot_2hole_mb0);
  combi_2hole_mb0->SetName("combi_2hole_mb0");
  combi_2hole_mb0->RegisterYourself();

  TGeoCombiTrans* combi_2hole_mb0_b =
    new TGeoCombiTrans(-6.7, 0, 0, rot_2hole_mb0); // y=
  combi_2hole_mb0_b->SetName("combi_2hole_mb0_b");
  combi_2hole_mb0_b->RegisterYourself();

  // shape for cross_mb0..
  // TGeoShape* box_mb0 = new TGeoBBox("s_box_mb0", x_boxmb0 / 2, y_boxmb0 / 2,
  // z_boxmb0 / 2);
  new TGeoBBox("box_mb0", x_boxmb0 / 2, y_boxmb0 / 2, z_boxmb0 / 2);
  new TGeoTube("hole1_mb0", radin_1hmb0, radout_1hmb0, high_1hmb0 / 2);
  new TGeoTube("hole2_mb0", radin_2hmb0, radout_2hmb0, high_2hmb0 / 2);
  /// composite shape for mb0

  auto* c_mb0_Shape_0 = new TGeoCompositeShape(
    "c_mb0_Shape_0",
    "box_mb0  - hole2_mb0:combi_2hole_mb0 - "
    "hole2_mb0:combi_2hole_mb0_b"); //- hole1_mb0:acombi_1h_mb0 -
                                    // hole1_mb0:bcombi_1h_mb0

  /// auto* cross_mb0_Volume = new TGeoVolume("cross_mb0_Volume", c_mb0_Shape_0,
  /// kMedAlu);
  auto* cross_mb0_Volume =
    new TGeoVolume("cross_mb0_Volume", c_mb0_Shape_0, malu5083);

  Cross_mb0->AddNode(cross_mb0_Volume, 1);

  // 2nd piece  cross beam MFT (cbeam)

  auto* Cross_mft = new TGeoVolumeAssembly("Cross_mft");
  auto* Cross_mft_2 = new TGeoVolumeAssembly("Cross_mft_2");
  auto* Cross_mft_3 = new TGeoVolumeAssembly("Cross_mft_3");
  auto* Cross_mft_4 = new TGeoVolumeAssembly("Cross_mft_4");

  // 2020_cross_beam
  //  tub-main
  Double_t radin_cb = 0.;
  Double_t radout_cb = 0.3;
  Double_t high_cb = 14.2;

  TGeoRotation* rot_cb = new TGeoRotation("rot_cb", 90, 90, 0);
  rot_cb->SetName("rot_cb");
  rot_cb->RegisterYourself();

  // using the same "box" of the 1 piece
  // 2hole coaxial
  Double_t radin_hole_cbeam = 0.;
  Double_t radout_hole_cbeam = 0.15; // diameter M3
  Double_t high_hole_cbeam = 0.91;   //

  // same rotation in tub aximatAl "rot_2hole_mb0"

  TGeoCombiTrans* combi_hole_1cbeam =
    new TGeoCombiTrans(6.8, 0, 0, rot_2hole_mb0);
  combi_hole_1cbeam->SetName("combi_hole_1cbeam");
  combi_hole_1cbeam->RegisterYourself();

  TGeoCombiTrans* combi_hole_2cbeam =
    new TGeoCombiTrans(-6.8, 0, 0, rot_2hole_mb0);
  combi_hole_2cbeam->SetName("combi_hole_2cbeam");
  combi_hole_2cbeam->RegisterYourself();

  // shape for shape cross beam

  new TGeoTube("hole_cbeam", radin_hole_cbeam, radout_hole_cbeam,
               high_hole_cbeam / 2);
  new TGeoTube("s_cb", radin_cb, radout_cb, high_cb / 2); // new

  // composite shape for cross beam (using the same box of mb0)
  /// auto* c_cbeam_Shape = new TGeoCompositeShape("c_cbeam_Shape", "box_mb0 -
  /// hole_cbeam:combi_hole_1cbeam - hole_cbeam:combi_hole_2cbeam");
  auto* c_cbeam_Shape = new TGeoCompositeShape(
    "c_cbeam_Shape",
    " s_cb:rot_cb - hole_cbeam:combi_hole_1cbeam - "
    "hole_cbeam:combi_hole_2cbeam");

  /// auto* Cross_mft_Volume = new TGeoVolume("Cross_mft_Volume", c_cbeam_Shape,
  /// kMedAlu);
  auto* Cross_mft_Volume =
    new TGeoVolume("Cross_mft_Volume", c_cbeam_Shape, malu5083);
  auto* Cross_mft_Volume_2 =
    new TGeoVolume("Cross_mft_Volume_2", c_cbeam_Shape, malu5083);
  auto* Cross_mft_Volume_3 =
    new TGeoVolume("Cross_mft_Volume_3", c_cbeam_Shape, malu5083);
  auto* Cross_mft_Volume_4 =
    new TGeoVolume("Cross_mft_Volume_4", c_cbeam_Shape, malu5083);

  Cross_mft->AddNode(Cross_mft_Volume, 1);
  Cross_mft_2->AddNode(Cross_mft_Volume_2, 1);
  Cross_mft_3->AddNode(Cross_mft_Volume_3, 1);
  Cross_mft_4->AddNode(Cross_mft_Volume_4, 1);

  // 3th piece Framework front

  auto* Fra_front = new TGeoVolumeAssembly("Fra_front");
  auto* Fra_front_L = new TGeoVolumeAssembly("Fra_front_L");
  auto* Fra_front_R = new TGeoVolumeAssembly("Fra_front_R");

  // new fra_front seg tub
  Double_t radin_fwf = 15.2;
  Double_t radout_fwf = 15.9;
  Double_t high_fwf = 0.6;
  Double_t ang_in_fwf = 180. + 21.0;  // 21.44
  Double_t ang_fin_fwf = 180 + 60.21; //

  // box to quit -inferior part
  Double_t x_box_qdown = 1.6;
  Double_t y_box_qdown = 1.6;
  Double_t z_box_qdown = 0.65;

  TGeoTranslation* tr_qdown =
    new TGeoTranslation("tr_qdown", -7.1, -13.7985, 0.);
  tr_qdown->RegisterYourself();

  // box to add -inferior part
  Double_t x_box_addown = 1.83;
  Double_t y_box_addown = 0.6;
  Double_t z_box_addown = 0.6;

  TGeoTranslation* tr_addown =
    new TGeoTranslation("tr_addown", -8.015, -12.6985, 0.);
  tr_addown->RegisterYourself();
  ////
  TGeoXtru* tria_fwf = new TGeoXtru(2);
  tria_fwf->SetName("tria_fwf");

  Double_t x_tria_fwf[3] = {-13.2, -10.3, -10.3};
  Double_t y_tria_fwf[3] = {-8.0, -11.4, -8.0};
  tria_fwf->DefinePolygon(3, x_tria_fwf, y_tria_fwf);
  tria_fwf->DefineSection(0, -0.3, 0., 0., 1);
  tria_fwf->DefineSection(1, 0.3, 0., 0., 1);
  //////////
  TGeoXtru* top_adfwf = new TGeoXtru(2);
  top_adfwf->SetName("top_adfwf");

  Double_t x_top_adfwf[4] = {-14.8, -14.2, -14.2, -14.8};
  Double_t y_top_adfwf[4] = {-3.6 - 0.12, -3.6 - 0.12, -5.56,
                             -5.83}; // 5.52 , 5.9 --5.83
  top_adfwf->DefinePolygon(4, x_top_adfwf, y_top_adfwf);
  top_adfwf->DefineSection(0, -0.3, 0., 0.,
                           1); //(plane,-zplane/ +zplane, x0, y0,(x/y))
  top_adfwf->DefineSection(1, 0.3, 0., 0., 1);

  // box to quit up part
  Double_t x_q_upbox = 0.4;
  Double_t y_q_upbox = 4.;
  Double_t z_q_upbox = 1.;

  TGeoTranslation* tr_q_upbox =
    new TGeoTranslation("tr_q_upbox", -14.8 - 0.2, -3.6 - 0.6, 0.);
  tr_q_upbox->RegisterYourself();

  TGeoRotation* rot_180yR =
    new TGeoRotation("rot_180yR", 180, -180, 0); // half0
  rot_180yR->RegisterYourself();
  TGeoCombiTrans* combi_fwb_R = new TGeoCombiTrans(0, 0, 0, rot_180yR);
  combi_fwb_R->SetName("combi_fwb_R");
  combi_fwb_R->RegisterYourself();

  ///////////////////////////////////////////

  Double_t x_box_up = 0.6; // cm
  Double_t y_box_up = 0.605;
  Double_t z_box_up = 2.84;
  // hole up
  Double_t dia_tub_up = 0.35;
  Double_t high_tub_up = 0.65;
  // hole down
  Double_t dia_tubdown = 0.35;
  Double_t high_tubdown = 0.68;
  //
  Double_t x_boxA_down = 0.8;
  Double_t y_boxA_down = 0.6;
  Double_t z_boxA_down = 0.6;

  Double_t x_boxB_down = 0.6;
  Double_t y_boxB_down = 0.605;
  Double_t z_boxB_down = 1.26; // 12.6
  // seg tub
  Double_t radin_segtub = 16.9;
  Double_t radout_segtub = 17.5;
  Double_t high_segtub = 0.6;
  Double_t ang_in_segtub = 212.1;
  Double_t ang_fin_segtub = 241.92; //

  // trans. rot.
  TGeoCombiTrans* combi_3a = new TGeoCombiTrans(-7.4, 0, 8.975, rot2);
  combi_3a->SetName("combi_3a");
  combi_3a->RegisterYourself();

  TGeoTranslation* tr1_up = new TGeoTranslation("tr1_up", -7.4, 0, 8.28); //

  tr1_up->RegisterYourself();

  TGeoTranslation* tr1_tub1 = new TGeoTranslation("tr1_tub1", 0, 0., 3.075);
  tr1_tub1->RegisterYourself();

  TGeoCombiTrans* combi_3b = new TGeoCombiTrans(7.118, 0, 16.16, rot3);
  combi_3b->SetName("combi_3b");
  combi_3b->RegisterYourself();

  TGeoTranslation* tr_2_box = new TGeoTranslation("tr_2_box", -0.4, 0, 0.7);
  tr_2_box->RegisterYourself();

  TGeoTranslation* tr3_box = new TGeoTranslation("tr3_box", -1.1, 0, 0.63);
  tr3_box->RegisterYourself();

  TGeoTranslation* tr_tubdown = new TGeoTranslation("tr_tubdown", -0.4, 0, 0.7);
  tr_tubdown->RegisterYourself();

  // shape for framewor front
  auto* fwf_tub = new TGeoTubeSeg("fwf_tub", radin_fwf, radout_fwf,
                                  high_fwf / 2, ang_in_fwf, ang_fin_fwf);
  auto* box_qdown = new TGeoBBox("box_qdown", x_box_qdown / 2, y_box_qdown / 2,
                                 z_box_qdown / 2);
  auto* box_addown = new TGeoBBox("box_addown", x_box_addown / 2,
                                  y_box_addown / 2, z_box_addown / 2);
  auto* q_upbox =
    new TGeoBBox("q_upbox", x_q_upbox / 2, y_q_upbox / 2, z_q_upbox / 2);

  new TGeoBBox("box_up", x_box_up / 2, y_box_up / 2, z_box_up / 2);

  new TGeoTube("tub_up", 0., dia_tub_up / 2, high_tub_up / 2); //
  new TGeoTubeSeg("seg_tub", radin_segtub, radout_segtub, high_segtub / 2,
                  ang_in_segtub, ang_fin_segtub);

  new TGeoBBox("boxB_down", x_boxB_down / 2, y_boxB_down / 2, z_boxB_down / 2);

  new TGeoBBox("boxA_down", x_boxA_down / 2, y_boxA_down / 2, z_boxA_down / 2);

  new TGeoTube("tubdown", 0., dia_tubdown / 2, high_tubdown / 2);

  // Composite shapes for Fra_front

  new TGeoCompositeShape("fra_front_Shape_0",
                         "box_up:tr1_up + seg_tub:combi_3b + boxB_down:tr3_box "
                         "+ boxA_down:tr_2_box");

  auto* fra_front_Shape_1 = new TGeoCompositeShape(
    "fra_front_Shape_1",
    "fra_front_Shape_0 - tubdown:tr_tubdown - tub_up:combi_3a");

  TGeoRotation* rot_z180x90 =
    new TGeoRotation("rot_z180x90", 180, 90, 0); // half0_R
  rot_z180x90->RegisterYourself();

  TGeoRotation* rot_halfR =
    new TGeoRotation("rot_halfR", 180, 180, 0); // half0_R
  rot_halfR->RegisterYourself();
  /// TGeoCombiTrans* combi_front_L = new TGeoCombiTrans(-7.1, -16.2, 32.5 +
  /// 0.675, rot_90x); // x=7.35, y=0, z=15.79
  TGeoCombiTrans* combi_front_L =
    new TGeoCombiTrans(-7.1, -16.2, 32.5 + 0.675, rot_90x);

  combi_front_L->SetName("combi_front_L");
  combi_front_L->RegisterYourself();

  TGeoTranslation* tr_ff = new TGeoTranslation(
    "tr_ff", 0, -2.5 - 0.31, 32.5 + 0.675); // 7.1 , -16.2 z32.5
  tr_ff->RegisterYourself();
  // TGeoRotation  *rot_180yR = new TGeoRotation("rot_180yR", 180,-180,0); //
  // half0 rot_180yR->RegisterYourself();

  // TGeoCombiTrans* combi_front_R = new TGeoCombiTrans(7.1, -16.2, 32.5 +
  // 0.675, rot_z180x90); //x=7.35, y=0, z=15.79
  TGeoCombiTrans* combi_front_R =
    new TGeoCombiTrans(0, -2.5 - 0.31, 32.5 + 0.675, rot_180yR);
  combi_front_R->SetName("combi_front_R");
  combi_front_R->RegisterYourself();

  auto* fra_front_Shape_2 = new TGeoCompositeShape(
    "Fra_front_Shape_2",
    "fwf_tub  - box_qdown:tr_qdown + box_addown:tr_addown  + tria_fwf + "
    "top_adfwf - q_upbox:tr_q_upbox");

  // auto * fra_front_Shape_3 = new
  // TGeoCompositeShape("Fra_front_Shape_3","Fra_front_Shape_2 +
  // Fra_front_Shape_2:combi_front_R ");

  auto* fra_front_Shape_3 = new TGeoCompositeShape(
    "Fra_front_Shape_3", "Fra_front_Shape_2 + Fra_front_Shape_2:rot_180yR");

  // auto * fra_front_Shape_3 = new
  // TGeoCompositeShape("fra_front_Shape_3","fra_front_Shape_2:rot_halfR  ");

  /// auto* Fra_front_Volume = new TGeoVolume("Fra_front_Volume",
  /// fra_front_Shape_1, kMedAlu);
  auto* Fra_front_Volume_R =
    new TGeoVolume("Fra_front_Volume_R", fra_front_Shape_2, malu5083);
  auto* Fra_front_Volume_L =
    new TGeoVolume("Fra_front_Volume_L", fra_front_Shape_2, malu5083);

  auto* Fra_front_Volume_RL =
    new TGeoVolume("Fra_front_Volume_RL", fra_front_Shape_3, malu5083);

  // Fra_front_L->AddNode(Fra_front_Volume, 1, tr_ff);
  // Fra_front_R->AddNode(Fra_front_Volume, 1, combi_front_R);

  // Fra_front->AddNode(Fra_front_L, 1);
  // Fra_front->AddNode(Fra_front_R, 2);
  Fra_front->AddNode(Fra_front_Volume_RL, 1, tr_ff);

  // 4th piece "BASE" framework half support

  auto* base = new TGeoVolumeAssembly("base");

  // seg tub  disc
  Double_t radin_disc = 23.6;
  Double_t radout_disc = 30.3;
  Double_t high_disc = 1.35;
  Double_t ang_in_disc = 180;
  Double_t ang_fin_disc = 360;

  // holes tub  1hole tranversal o3.5
  Double_t radin_holeB = 0.;
  Double_t radout_holeB = 0.175; // diameter 3.5 H11
  Double_t high_holeB = 1.5;
  TGeoTranslation* tr1_holeB = new TGeoTranslation("tr1_holeB", -7.5, -28.8, 0);
  tr1_holeB->RegisterYourself();

  TGeoTranslation* tr2_holeB = new TGeoTranslation("tr2_holeB", 7.5, -28.8, 0);
  tr2_holeB->RegisterYourself();

  // box 1
  Double_t x_1box = 61.0;
  Double_t y_1box = 13.0;
  Double_t z_1box = 1.4;
  // box 2
  Double_t x_2box = 51.2;
  Double_t y_2box = 14.6;
  Double_t z_2box = 1.4;
  // box 3
  Double_t x_3box = 45.1;
  Double_t y_3box = 23.812;
  Double_t z_3box = 1.4;
  // seg tub hole
  Double_t radin_1hole = 29.3;
  Double_t radout_1hole = 30.3;
  Double_t high_1hole = 1.4;
  Double_t ang_in_1hole = 205;
  Double_t ang_fin_1hole = 225;
  // seg tub 2 hole
  Double_t radin_2hole = 23.0;
  Double_t radout_2hole = 25.5;
  Double_t high_2hole = 1.4;
  Double_t ang_in_2hole = 207.83;
  Double_t ang_fin_2hole = 249.998;
  // seg tub 3 ARC central xy  SEG_3ARC U
  Double_t radin_3hole = 25.5;
  Double_t radout_3hole = 27.5;
  Double_t high_3hole = 1.35;
  Double_t ang_in_3hole = 255.253;
  Double_t ang_fin_3hole = 284.746;
  // hole central down |_|   since x=-70 to 0
  Double_t xc_box = 7.0;
  Double_t yc_box = 5.772;
  Double_t zc_box = 1.4;

  TGeoTranslation* tr_cbox =
    new TGeoTranslation("tr_cbox", -xc_box / 2, -radout_disc + 0.888, 0);
  tr_cbox->RegisterYourself();
  // box 4 lamine 1
  Double_t x_labox = 60.0;
  Double_t y_labox = 30.3;
  Double_t z_labox = 0.305;
  TGeoTranslation* tr_la =
    new TGeoTranslation("tr_la", 0, -y_labox / 2 - 9.3, high_disc / 2);
  tr_la->RegisterYourself();

  // box 5 lamin 2
  Double_t x_2labox = 51.2;
  Double_t y_2labox = 2.8; // C-B
  Double_t z_2labox = 0.303;
  TGeoTranslation* tr_2la =
    new TGeoTranslation("tr_2la", 0, -8.1, high_disc / 2); //
  tr_2la->RegisterYourself();

  // circular border C SEG_BORD
  // seg tub 3 xy
  Double_t radin_bord = 0.5;
  Double_t radout_bord = 0.9; //
  Double_t high_bord = 1.355; // 13.5
  Double_t ang_in_bord = 0;
  Double_t ang_fin_bord = 90;
  // TGeoRotation *rot_bord1 = new TGeoRotation("rot_bord1", ang_in_1hole
  // +0.0167,0,0);
  TGeoRotation* rot1_bord1 = new TGeoRotation("rot1_bord1", 14.8, 0, 0);
  rot1_bord1->RegisterYourself();
  TGeoCombiTrans* combi_bord1 =
    new TGeoCombiTrans(-26.7995, -13.0215, 0, rot1_bord1); // y=
  combi_bord1->SetName("combi_bord1");
  combi_bord1->RegisterYourself();

  TGeoRotation* rot2_bord1 = new TGeoRotation("rot2_bord1", -50, 0, 0);
  rot2_bord1->RegisterYourself();
  TGeoCombiTrans* combi2_bord1 =
    new TGeoCombiTrans(-21.3795, -20.7636, 0, rot2_bord1); // y=
  combi2_bord1->SetName("combi2_bord1");
  combi2_bord1->RegisterYourself();
  //
  TGeoRotation* rot1_bord2 = new TGeoRotation("rot1_bord2", 250, 0, 0);
  rot1_bord2->RegisterYourself();
  TGeoCombiTrans* combi1_bord2 =
    new TGeoCombiTrans(-9.0527, -23.3006, 0, rot1_bord2); // y=
  combi1_bord2->SetName("combi1_bord2");
  combi1_bord2->RegisterYourself();
  // e|°____°|
  TGeoRotation* rot_cent_bord = new TGeoRotation("rot_cent_bord", 90, 0, 0);
  rot_cent_bord->RegisterYourself();
  TGeoCombiTrans* combi_cent_bord =
    new TGeoCombiTrans(-6.5, -27.094, 0, rot_cent_bord); // y=
  combi_cent_bord->SetName("combi_cent_bord");
  combi_cent_bord->RegisterYourself();
  // box tonge
  Double_t x_tong = 2.0;
  Double_t y_tong = 1.5;
  Double_t z_tong = 1.35;
  TGeoTranslation* tr_tong = new TGeoTranslation("tr_tong", 0, -26.75, 0); //
  tr_tong->RegisterYourself();
  // circular central hole1 to conexion with other parts
  Double_t radin_hole1 = 0;
  Double_t radout_hole1 = 0.4;
  Double_t high_hole1 = 1.36;
  TGeoTranslation* tr_hole1 =
    new TGeoTranslation("tr_hole1", 0, -28.0, 0); // tonge
  tr_hole1->RegisterYourself();

  TGeoTranslation* tr2_hole1 =
    new TGeoTranslation("tr2_hole1", -26.5, -8.5, 0); // left
  tr2_hole1->RegisterYourself();

  TGeoTranslation* tr3_hole1 =
    new TGeoTranslation("tr3_hole1", 26.5, -8.5, 0); // right
  tr3_hole1->RegisterYourself();

  // circular hole2 ; hole2 r=6.7
  Double_t radin_hole2 = 0;
  Double_t radout_hole2 = 0.335; // diameter 6.7
  Double_t high_hole2 = 1.36;    // 13.5
  TGeoTranslation* tr1_hole2 =
    new TGeoTranslation("tr1_hole2", -28.0, -8.5, 0); //
  tr1_hole2->RegisterYourself();

  TGeoTranslation* tr2_hole2 =
    new TGeoTranslation("tr2_hole2", 28.0, -8.5, 0); //
  tr2_hole2->RegisterYourself();

  //////////// hole "0" two tubs together
  Double_t radin_T1 = 0.325; // diam 0.65cm
  Double_t radout_T1 = 0.55; // dia 1.1
  Double_t high_T1 = 1.2;    //  dz 6

  Double_t radin_T2 = 0;
  Double_t radout_T2 = 1.1;
  Double_t high_T2 = 1.2; // dz 6
                          // seg tong xy
  Double_t radin_ccut = 27.5;
  Double_t radout_ccut = 29.; // 304
  Double_t high_ccut = 1.4;   /// 13.5
  Double_t ang_in_ccut = 260;
  Double_t ang_fin_ccut = 280;

  // shape for base

  new TGeoTubeSeg("disc", radin_disc, radout_disc, high_disc / 2, ang_in_disc,
                  ang_fin_disc);
  new TGeoTubeSeg("c_cut", radin_ccut, radout_ccut, high_ccut / 2, ang_in_ccut,
                  ang_fin_ccut);

  new TGeoBBox("box1", x_1box / 2, y_1box / 2, z_1box / 2);
  new TGeoBBox("box2", x_2box / 2, y_2box / 2, z_2box / 2);
  new TGeoBBox("box3", x_3box / 2, y_3box / 2, z_3box / 2);
  new TGeoBBox("labox1", x_labox / 2, y_labox / 2, z_labox / 2);
  new TGeoBBox("labox2", x_2labox / 2, y_2labox / 2, z_2labox / 2);
  new TGeoBBox("cbox", xc_box / 2, yc_box / 2, zc_box / 2);
  new TGeoBBox("tongbox", x_tong / 2, y_tong / 2, z_tong / 2);

  new TGeoTubeSeg("seg_1hole", radin_1hole, radout_1hole, high_1hole / 2,
                  ang_in_1hole, ang_fin_1hole); // r_in,r_out,dZ,ang,ang
  new TGeoTubeSeg("seg_2hole", radin_2hole, radout_2hole, high_2hole / 2,
                  ang_in_2hole, ang_fin_2hole);
  new TGeoTubeSeg("seg_3hole", radin_3hole, radout_3hole, high_3hole / 2,
                  ang_in_3hole, ang_fin_3hole); // y|u|
  new TGeoTubeSeg("seg_bord", radin_bord, radout_bord, high_bord / 2,
                  ang_in_bord, ang_fin_bord);

  new TGeoTube("circ_hole1", radin_hole1, radout_hole1, high_hole1 / 2);

  new TGeoTube("circ_hole2", radin_hole2, radout_hole2, high_hole2 / 2);

  new TGeoTube("circ_holeB", radin_holeB, radout_holeB, high_holeB / 2);

  // composite shape for base

  // new TGeoCompositeShape("base_Shape_0", " disc - box1 - box2 - box3 -
  // circ_holeB:tr1_holeB - circ_holeB:tr2_holeB");
  new TGeoCompositeShape("base_Shape_0", " disc - box1 - box2 - box3");
  new TGeoCompositeShape(
    "base_Shape_1",
    "(seg_1hole - seg_bord:combi_bord1 - seg_bord:combi2_bord1) + seg_2hole "
    "-seg_bord:combi1_bord2 + cbox:tr_cbox");

  new TGeoCompositeShape(
    "base_Shape_2",
    " seg_3hole + seg_bord:combi_cent_bord"); // seg_bord:combi_cent_bord

  new TGeoCompositeShape("base_Shape_3", " labox1:tr_la + labox2:tr_2la ");

  /// auto* base_Shape_4 = new TGeoCompositeShape("base_Shape_4", "base_Shape_0
  /// - base_Shape_1 - base_Shape_1:rot1 + base_Shape_2  + tongbox:tr_tong -
  /// circ_hole1:tr_hole1 - circ_hole1:tr2_hole1 - circ_hole1:tr3_hole1 -
  /// circ_hole2:tr1_hole2 - circ_hole2:tr2_hole2 - base_Shape_3 ");
  auto* base_Shape_4 = new TGeoCompositeShape(
    "base_Shape_4",
    "base_Shape_0 - base_Shape_1 - base_Shape_1:rot1 + base_Shape_2 - "
    "base_Shape_3 + tongbox:tr_tong - c_cut"); //- circ_hole1:tr_hole1 -
                                               // circ_hole1:tr2_hole1 -
                                               // circ_hole1:tr3_hole1  -
                                               // circ_hole2:tr1_hole2 -
                                               // circ_hole2:tr2_hole2

  // auto * base_Shape_5 = new TGeoCompositeShape("base_Shape_5","disc-box1
  // -box2 -box3 -seg_1hole -seg_2hole +seg_3hole -seg_1hole:rot1-seg_2hole:rot1
  // - cbox:tr_cbox - labox:tr_la - labox2:tr_2la  + seg_bord  ");

  auto* base4_Volume = new TGeoVolume("base4_Volume", base_Shape_4, malu5083);

  base->AddNode(base4_Volume, 2, rot_base);
  // base->AddNode(base4_Volume,2);

  // 5th piece middle  Framework middle

  auto* middle = new TGeoVolumeAssembly("middle");
  auto* middle_L = new TGeoVolumeAssembly("middle_L");
  auto* middle_R = new TGeoVolumeAssembly("middle_R");

  ////new2020 framework middle
  Double_t radin_fwm = 14.406;
  Double_t radout_fwm = 15.185; //
  Double_t high_fwm = 0.6;      ///
  Double_t ang_in_fwm = 180. + 12.93;
  Double_t ang_fin_fwm = 180. + 58.65;

  ////box add up
  Double_t x_fwm_1box = 0.8; // dx=4
  Double_t y_fwm_1box = 1.45;
  Double_t z_fwm_1box = 0.6; // 6.5 -> 6.6 to quit
  TGeoTranslation* tr_fwm_1box =
    new TGeoTranslation("tr_fwm_1box", -14.4, -3.398 + 1.45 / 2, 0); // 81
  tr_fwm_1box->RegisterYourself();

  ////box quit down
  Double_t x_fwm_2box = 0.8; // dx=4
  Double_t y_fwm_2box = 1.2;
  Double_t z_fwm_2box = 0.7; // 6.5 -> 6.6 to quit
  TGeoTranslation* tr_fwm_2box =
    new TGeoTranslation("tr_fwm_2box", -14.4 + 6.9, -3.398 - 9.1, 0); // 81
  tr_fwm_2box->RegisterYourself();

  ////
  TGeoXtru* tria_fwm = new TGeoXtru(2);
  tria_fwm->SetName("tria_fwm");

  Double_t x_tria_fwm[3] = {-13.5, -10., -10.};
  Double_t y_tria_fwm[3] = {-5.94, -5.94, -10.8};
  tria_fwm->DefinePolygon(3, x_tria_fwm, y_tria_fwm);
  tria_fwm->DefineSection(0, -0.3, 0., 0., 1);
  tria_fwm->DefineSection(1, 0.3, 0., 0., 1);
  //////////

  // box up to quit and to join
  Double_t x_middle = 0.8;   // dx=4
  Double_t y_middle = 3.495; // y=34.9
  Double_t z_middle = 0.62;  // z=6
  // tr1 to join with arc
  TGeoTranslation* tr1_middle_box =
    new TGeoTranslation("tr1_middle_box", -14.4, -0.745, 0); // -152,-17.45,0
  tr1_middle_box->RegisterYourself();
  // tr2 to quiet
  TGeoTranslation* tr2_middle_box =
    new TGeoTranslation("tr2_middle_box", -15.2, -0.745, 0); // -152,-17.45,0
  tr2_middle_box->RegisterYourself();

  // box down_1
  Double_t x_middle_d1box = 0.4; // dx=4
  Double_t y_middle_d1box = 0.28;
  Double_t z_middle_d1box = 0.66;
  TGeoTranslation* tr_middle_d1box =
    new TGeoTranslation("tr_middle_d1box", -7.3, -11.96, 0.); // 81
  tr_middle_d1box->RegisterYourself();

  // box down_2
  Double_t x_middle_d2box = 0.8; // dx=4
  Double_t y_middle_d2box = 1.0;
  Double_t z_middle_d2box = 0.66; //
  TGeoTranslation* tr_middle_d2box =
    new TGeoTranslation("tr_middle_d2box", -7.5, -12.6249, 0); // 81
  tr_middle_d2box->RegisterYourself();

  // arc circ part
  Double_t radin_middle = 14.0;
  Double_t radout_middle = 15.0; //
  Double_t high_middle = 0.6;    //
  Double_t ang_in_middle = 180;
  Double_t ang_fin_middle = 238.21; // alfa=57.60

  // circular hole1 ; hole_middle d=3.5
  Double_t radin_mid_1hole = 0.;
  Double_t radout_mid_1hole = 0.175; // diameter 3.5
  Double_t high_mid_1hole = 1.5;     // 2.4

  TGeoRotation* rot_mid_1hole = new TGeoRotation("rot_mid_1hole", 90, 90, 0);
  rot_mid_1hole->RegisterYourself();
  TGeoCombiTrans* combi_mid_1tubhole =
    new TGeoCombiTrans(-14.2, 0.325, 0, rot_mid_1hole); //
  combi_mid_1tubhole->SetName("combi_mid_1tubhole");
  combi_mid_1tubhole->RegisterYourself();

  // circular hole2 ; hole_middle d=3
  Double_t radin_mid_2hole = 0.;
  Double_t radout_mid_2hole = 0.15; // diameter 3
  Double_t high_mid_2hole = 1.8;    //

  TGeoCombiTrans* combi_mid_2tubhole =
    new TGeoCombiTrans(-7.7, -12.355, 0, rot_mid_1hole); // x=81
  combi_mid_2tubhole->SetName("combi_mid_2tubhole");
  combi_mid_2tubhole->RegisterYourself();

  // shape for middle
  new TGeoBBox("middle_box", x_middle / 2, y_middle / 2, z_middle / 2);

  new TGeoBBox("middle_d1box", x_middle_d1box / 2, y_middle_d1box / 2,
               z_middle_d1box / 2);

  new TGeoBBox("middle_d2box", x_middle_d2box / 2, y_middle_d2box / 2,
               z_middle_d2box / 2);

  new TGeoTubeSeg("arc_middle", radin_middle, radout_middle, high_middle / 2,
                  ang_in_middle, ang_fin_middle);

  new TGeoTube("mid_1tubhole", radin_mid_1hole, radout_mid_1hole,
               high_mid_1hole / 2);

  new TGeoTube("mid_2tubhole", radin_mid_2hole, radout_mid_2hole,
               high_mid_2hole / 2);

  auto* tube_fwm = new TGeoTubeSeg("tube_fwm", radin_fwm, radout_fwm,
                                   high_fwm / 2, ang_in_fwm, ang_fin_fwm);
  auto* fwm_1box =
    new TGeoBBox("fwm_1box", x_fwm_1box / 2, y_fwm_1box / 2, z_fwm_1box / 2);
  auto* fwm_2box =
    new TGeoBBox("fwm_2box", x_fwm_2box / 2, y_fwm_2box / 2, z_fwm_2box / 2);

  // composite shape for middle

  new TGeoCompositeShape(
    "middle_Shape_0",
    " arc_middle + middle_box:tr1_middle_box - middle_box:tr2_middle_box - "
    "middle_d1box:tr_middle_d1box - middle_d2box:tr_middle_d2box");

  auto* middle_Shape_1 = new TGeoCompositeShape(
    "middle_Shape_1",
    " middle_Shape_0 "
    "-mid_1tubhole:combi_mid_1tubhole-mid_2tubhole:combi_mid_2tubhole");

  TGeoRotation* rot_middlez = new TGeoRotation("rot_middley", 180, 180, 0);
  rot_middlez->RegisterYourself();
  TGeoCombiTrans* combi_middle_L = new TGeoCombiTrans(
    0, -7.625, 24.15 + 0.675,
    rot_90x); // x=7.35, y=0, z=15.79- 0,-7.625,24.15+0.675-80)
  combi_middle_L->SetName("combi_middle_L");
  combi_middle_L->RegisterYourself();

  TGeoTranslation* tr_middle_L = new TGeoTranslation(
    "tr_middle_L", 0, -4.45 - 0.1, 24.85 + 0.675); //+2.5,, -152,-17.45,0
  tr_middle_L->RegisterYourself();

  TGeoCombiTrans* combi_middle_R =
    new TGeoCombiTrans(0, -4.45 - 0.1, 24.85 + 0.675,
                       rot_middlez); // x=7.35, y=0, z=15.79 y7.625 ++2.5
  combi_middle_R->SetName("combi_middle_R");
  combi_middle_R->RegisterYourself();

  auto* middle_Shape_3 = new TGeoCompositeShape(
    "middle_Shape_3",
    "  tube_fwm + fwm_1box:tr_fwm_1box - fwm_2box:tr_fwm_2box +tria_fwm");

  auto* middle_Shape_4 = new TGeoCompositeShape(
    "middle_Shape_4",
    " tube_fwm + fwm_1box:tr_fwm_1box - fwm_2box:tr_fwm_2box +tria_fwm");

  // auto* middle_Volume = new TGeoVolume("middle_Volume", middle_Shape_1,
  // kMedAlu);
  auto* middle_Volume_L =
    new TGeoVolume("middle_Volume_L", middle_Shape_3, malu5083);
  auto* middle_Volume_R =
    new TGeoVolume("middle_Volume_R", middle_Shape_4, malu5083);

  TGeoTranslation* tr_middle = new TGeoTranslation(
    "tr_middle", 0, -4.45 - 0.1, 24.85 + 0.675); //+2.5,, -152,-17.45,0
  tr_middle->RegisterYourself();

  middle_L->AddNode(middle_Volume_L, 1, tr_middle);
  middle_R->AddNode(middle_Volume_R, 1, combi_middle_R); // combi_midle_R

  middle->AddNode(middle_L, 1);
  middle->AddNode(middle_R, 2);

  // new piece _/   \_
  // Support_rail_L & Support_rail_R

  auto* rail_L_R = new TGeoVolumeAssembly("rail_L_R");

  // 6 piece RAIL LEFT RL0000
  auto* rail_L = new TGeoVolumeAssembly("rail_L");

  // box down_2
  Double_t x_RL_1box = 3.0;  // dx=15
  Double_t y_RL_1box = 1.21; // dy=6, -dy=6
  Double_t z_RL_1box = 0.8;  // dz=4     to quit
  TGeoTranslation* tr_RL_1box =
    new TGeoTranslation(0, y_RL_1box / 2, 1.825); // 81
  tr_RL_1box->SetName("tr_RL_1box");
  tr_RL_1box->RegisterYourself();

  TGeoXtru* xtru_RL1 = new TGeoXtru(2);
  xtru_RL1->SetName("S_XTRU_RL1");

  Double_t x_RL1[5] = {-1.5, 1.5, 0.5, 0.5,
                       -1.5};                    // 93,93,73,73,-15}; //vertices
  Double_t y_RL1[5] = {1.2, 1.2, 2.2, 8.2, 8.2}; // 357.5,357.5,250.78,145.91};
  xtru_RL1->DefinePolygon(5, x_RL1, y_RL1);
  xtru_RL1->DefineSection(0, -2.225, 0., 0.,
                          1); // (plane,-zplane/ +zplane, x0, y0,(x/y))
  xtru_RL1->DefineSection(1, 2.225, 0., 0., 1);

  /// TGeoXtru* xtru_RL2 = new TGeoXtru(2);
  /// xtru_RL2->SetName("S_XTRU_RL2");

  /// Double_t x_RL2[8] = {-1.5, 0.5, 0.5, 9.3, 9.3, 7.3, 7.3, -1.5}; //
  /// vertices Double_t y_RL2[8] =
  /// {8.2, 8.2, 13.863, 24.35, 35.75, 35.75, 25.078, 14.591};

  /// xtru_RL2->DefinePolygon(8, x_RL2, y_RL2);

  /// xtru_RL2->DefineSection(0, 0.776, 0, 0, 1); // (plane,-zplane/+zplane, x0,
  /// y0,(x/y)) xtru_RL2->DefineSection(1, 2.225, 0, 0, 1);

  // box knee
  Double_t x_RL_kneebox = 1.5; // dx=7.5
  Double_t y_RL_kneebox = 3.5; // dy=17.5
  Double_t z_RL_kneebox = 1.5; // dz=7.5     to quit
  TGeoTranslation* tr_RL_kneebox =
    new TGeoTranslation(0, 0, 0); // 81 x =-2.5, y=145.91
  tr_RL_kneebox->SetName("tr_RL_kneebox");
  tr_RL_kneebox->RegisterYourself();

  TGeoRotation* rot_knee = new TGeoRotation("rot_knee", -40, 0, 0);
  rot_knee->SetName("rot_knee");
  rot_knee->RegisterYourself();
  TGeoCombiTrans* combi_knee =
    new TGeoCombiTrans(0.96, 1.75 + 0.81864, 0, rot_knee); // y=
  combi_knee->SetName("combi_knee");
  combi_knee->RegisterYourself();
  // quit diagona-> qdi
  Double_t x_qdi_box = 3.1;   //
  Double_t y_qdi_box = 7.159; //
  Double_t z_qdi_box = 3.005; //

  TGeoRotation* rot_qdi = new TGeoRotation("rot_qdi", 0, 24.775, 0);
  rot_qdi->RegisterYourself();
  TGeoCombiTrans* combi_qdi =
    new TGeoCombiTrans(0, 5.579, -2.087, rot_qdi); // y=
  combi_qdi->SetName("combi_qdi");
  combi_qdi->RegisterYourself();
  // knee small

  TGeoXtru* xtru3_RL = new TGeoXtru(2);
  xtru3_RL->SetName("xtru3_RL");

  Double_t x_3RL[6] = {-0.75, 0.75, 0.75, 2.6487, 1.4997, -0.75}; // vertices
  Double_t y_3RL[6] = {-1.75, -1.75, 1.203, 3.465, 4.4311, 1.75};

  xtru3_RL->DefinePolygon(6, x_3RL, y_3RL);
  xtru3_RL->DefineSection(0, -0.75, 0, 0,
                          1); // (plane,-zplane/+zplane, x0, y0,(x/y))
  xtru3_RL->DefineSection(1, 0.76, 0, 0, 1);

  TGeoTranslation* tr_vol3_RL = new TGeoTranslation(-0.25, 12.66, 0); //
  tr_vol3_RL->SetName("tr_vol3_RL");
  tr_vol3_RL->RegisterYourself();

  // circular holes  could be for rail R and L ..
  // circular hole1_RL (a(6,22)); hole_midle d=6.5 H11
  Double_t radin_RL1hole = 0.;
  Double_t radout_RL1hole = 0.325; // diameter 3.5
  Double_t high_RL1hole = 1.0;     //

  TGeoRotation* rot_RL1hole = new TGeoRotation("rot_RL1hole", 0, 0, 0);
  rot_RL1hole->RegisterYourself();
  TGeoCombiTrans* combi_RL1hole =
    new TGeoCombiTrans(0.7, 0.6, 1.85, rot_RL1hole); // y=
  combi_RL1hole->SetName("combi_RL1hole");
  combi_RL1hole->RegisterYourself();
  // similar hole for R Join.
  // circular hole_ir. diameter=M3 (3 mm)) prof trou:8, tar:6mm
  Double_t radin_ir_railL = 0.;
  Double_t radout_ir_railL = 0.15; // diameter 0.3cm
  Double_t high_ir_railL = 3.9;    //
  TGeoRotation* rot_ir_RL = new TGeoRotation("rot_ir_RL", 90, 90, 0);
  rot_ir_RL->RegisterYourself();
  // in y = l_253.5 - 6. enter in (0,6,0)
  TGeoCombiTrans* combi_ir1_RL =
    new TGeoCombiTrans(8.62, 24.75, 1.5, rot_ir_RL);
  combi_ir1_RL->SetName("combi_ir1_RL");
  combi_ir1_RL->RegisterYourself();

  TGeoCombiTrans* combi_ir2_RL = new TGeoCombiTrans(8.6, 33.15, 1.5, rot_ir_RL);
  combi_ir2_RL->SetName("combi_ir2_RL");
  combi_ir2_RL->RegisterYourself();
  //
  // 1) modification
  TGeoXtru* xtru_RL2 = new TGeoXtru(2);
  xtru_RL2->SetName("S_XTRU_RL2");

  // modi  the arm L---> new L
  Double_t x_RL2[8] = {-1.5, 0.5, 0.5, 9.3, 9.3, 7.3, 7.3, -1.5};
  Double_t y_RL2[8] = {8.2, 8.2, 13.863, 24.35, 25.65, 25.65, 25.078, 14.591};

  xtru_RL2->DefinePolygon(8, x_RL2, y_RL2);
  xtru_RL2->DefineSection(0, 0.7752, 0, 0,
                          1); //(plane,-zplane/+zplane, x0, y0,(x/y))  0.775
  xtru_RL2->DefineSection(1, 2.225, 0, 0, 1);

  // 2) modi  adding box  new element 1 box
  TGeoXtru* adi1_RL = new TGeoXtru(2);
  adi1_RL->SetName("S_ADI1_RL");

  Double_t x_adi1RL[4] = {-1.5, -1.5, 0.5, 0.5};
  Double_t y_adi1RL[4] = {2.2, 13.863, 13.863, 2.2};

  /// Double_t x_adi1RL[4]={-0.5,-0.5,1.5,1.5};   //vertices
  ////Double_t y_adi1RL[4]={2.2,13.86,13.86,2.2};

  adi1_RL->DefinePolygon(4, x_adi1RL, y_adi1RL);
  adi1_RL->DefineSection(0, -0.75, 0, 0,
                         1);                 //(plane,-zplane/+zplane, x0, y0,(x/y))
  adi1_RL->DefineSection(1, 0.775, 0, 0, 1); // 0.76

  //// 3) modi adding new knee new element 2 new knee
  TGeoXtru* adi2_RL = new TGeoXtru(2); //
  adi2_RL->SetName("S_ADI2_RL");
  Double_t x_adi2RL[6] = {-1.5, 0.5, 9.3, 9.3, 7.8, 7.8};
  Double_t y_adi2RL[6] = {13.863, 13.863, 24.35, 25.65, 25.65, 25.078};

  adi2_RL->DefinePolygon(6, x_adi2RL, y_adi2RL);
  adi2_RL->DefineSection(0, -0.75, 0, 0,
                         1);                  //(plane,-zplane/+zplane, x0, y0,(x/y))
  adi2_RL->DefineSection(1, 0.7755, 0, 0, 1); // 0.76,  0.775

  // 4)modi  to quit ---> trap
  Double_t RL_dx1 = 2.66; // at -z
  Double_t RL_dx2 = 1;    // dat +z
  Double_t RL_dy = 2.2;   // dz=7.5     to quit
  Double_t RL_dz = 1.5;   // dz=1.5     to quit
  // auto *trap1 = new TGeoTrd1("TRAP1",RL_dx1,RL_dx2 ,RL_dy ,RL_dz);

  TGeoRotation* rot_RL_Z50 = new TGeoRotation("rot_RL_Z50", 50, 0, 0);
  rot_RL_Z50->RegisterYourself();
  TGeoCombiTrans* combi_RL_trap =
    new TGeoCombiTrans(5, 18.633, -1.5 - 0.025, rot_RL_Z50); // y=
  combi_RL_trap->SetName("combi_RL_trap");
  combi_RL_trap->RegisterYourself();

  /////  5) modi  to quit     inferior part  box
  Double_t x_qinf_box = 10.66; //
  Double_t y_qinf_box = 10.2;  //
  Double_t z_qinf_box = 3.;    // dz =1.5
  auto* s_RL_qinf_box = new TGeoBBox("S_RL_QINF_BOX", x_qinf_box / 2,
                                     y_qinf_box / 2, z_qinf_box / 2);
  TGeoCombiTrans* combi_RL_qbox =
    new TGeoCombiTrans(7, 23., -1.5 - 0.025, rot_RL_Z50); // y= , z=-0.75-0.75
  combi_RL_qbox->SetName("combi_RL_qbox");
  combi_RL_qbox->RegisterYourself();

  // 6) modi.  add penta face z
  TGeoXtru* pentfa_RL = new TGeoXtru(2); // not work
  pentfa_RL->SetName("S_PENTFA_RL");
  Double_t x_pentfaRL[5] = {-1., -1., 0.13, 1., 1.};
  Double_t y_pentfaRL[5] = {1.125, 0.045, -1.125, -1.125, 1.125}; // 1.125

  pentfa_RL->DefinePolygon(5, x_pentfaRL, y_pentfaRL);
  pentfa_RL->DefineSection(0, -5.05, 0, 0,
                           1);                 //(plane,-zplane/+zplane, x0, y0,(x/y))  // 0.75
  pentfa_RL->DefineSection(1, 5.055, 0, 0, 1); // 0.76.. 0.9036

  TGeoRotation* rot_X90 = new TGeoRotation("rot_X90", 0, 90, 0);
  rot_X90->RegisterYourself();
  TGeoCombiTrans* combi_RL_pent = new TGeoCombiTrans(
    8.3, 30.705, 1.125 - 0.025, rot_X90); // x =8.3 , z= 1.125
  combi_RL_pent->SetName("combi_RL_pent");
  combi_RL_pent->RegisterYourself();
  //

  // shape for Rail L geom
  new TGeoBBox("RL_1box", x_RL_1box / 2, y_RL_1box / 2, z_RL_1box / 2);
  new TGeoBBox("RL_kneebox", x_RL_kneebox / 2, y_RL_kneebox / 2,
               z_RL_kneebox / 2); // no_ used
  new TGeoBBox("qdi_box", x_qdi_box / 2, y_qdi_box / 2, z_qdi_box / 2);
  new TGeoTrd1("TRAP1", RL_dx1, RL_dx2, RL_dy, RL_dz);
  // auto *s_RL1hole=new
  // TGeoTube("S_RL1HOLE",radin_RL1hole,radout_RL1hole,high_RL1hole/2); auto
  // *s_irL_hole=new
  // TGeoTube("S_irL_HOLE",radin_ir_railL,radout_ir_railL,high_ir_railL/2);
  // composite shape for rail L

  auto* RL_Shape_0 = new TGeoCompositeShape(
    "RL_Shape_0",
    "  S_XTRU_RL1 + S_XTRU_RL2 + RL_1box:tr_RL_1box - "
    "qdi_box:combi_qdi + "
    "S_ADI1_RL + S_ADI2_RL - TRAP1:combi_RL_trap - "
    "S_RL_QINF_BOX:combi_RL_qbox + "
    "S_PENTFA_RL:combi_RL_pent"); // xtru3_RL:tr_vol3_RL
                                  // +

  TGeoVolume* rail_L_vol0 = new TGeoVolume("RAIL_L_VOL0", RL_Shape_0, malu5083);
  // TGeoVolume* rail_L_vol0 = new TGeoVolume("RAIL_L_VOL0", RL_Shape_0,
  // kMedAlu);

  rail_L->AddNode(rail_L_vol0, 1, new TGeoTranslation(0., 0., 1.5));

  // piece 7th RAIL RIGHT
  // auto *rail_R = new TGeoVolumeAssembly("rail_R");

  Double_t x_RR_1box = 3.0; // dx=15
  Double_t y_RR_1box = 1.2; // dy=6, -dy=6
  Double_t z_RR_1box = 0.8; // dz=4     to quit
  TGeoTranslation* tr_RR_1box =
    new TGeoTranslation("tr_RR_1box", 0, 0.6, 1.825); // 81
  tr_RR_1box->RegisterYourself();

  TGeoXtru* part_RR1 = new TGeoXtru(2);
  part_RR1->SetName("part_RR1");
  // TGeoVolume *vol_RR1 = gGeoManager->MakeXtru("S_part_RR1",kMedAlu,2);
  // TGeoXtru *part_RR1 = (TGeoXtru*)vol_RR1->GetShape();

  Double_t x_RR1[5] = {-1.5, -0.5, -0.5, 1.5, 1.5}; // C,D,K,L,C' //vertices
  Double_t y_RR1[5] = {1.2, 2.2, 8.2, 8.2, 1.2};    // 357.5,357.5,250.78,145.91};

  part_RR1->DefinePolygon(5, x_RR1, y_RR1);
  part_RR1->DefineSection(0, -2.225, 0, 0,
                          1); // (plane,-zplane/ +zplane, x0, y0,(x/y))
  part_RR1->DefineSection(1, 2.225, 0, 0, 1);

  // TGeoXtru* part_RR2 = new TGeoXtru(2);
  // part_RR2->SetName("part_RR2");
  // TGeoVolume *vol_RR2 = gGeoManager->MakeXtru("part_RR2",Al,2);
  // TGeoXtru *xtru_RR2 = (TGeoXtru*)vol_RR2->GetShape();

  // Double_t x_RR2[8] = {-0.5, -0.5, -9.3, -9.3, -7.3, -7.3, 1.5, 1.5}; //
  // K,E,F,G,H,I,J,L // vertices Double_t y_RR2[8] =
  // {8.2, 13.863, 24.35, 35.75, 35.75, 25.078, 14.591, 8.2};

  // part_RR2->DefinePolygon(8, x_RR2, y_RR2);
  // part_RR2->DefineSection(0, 0.776, 0, 0, 1); // (plane,-zplane/+zplane, x0,
  // y0,(x/y)) part_RR2->DefineSection(1, 2.225, 0, 0, 1);

  // knee (small)

  TGeoXtru* part_RR3 = new TGeoXtru(2);
  part_RR3->SetName("part_RR3");

  Double_t x_3RR[6] = {1.0, 1.0, -1.2497,
                       -2.2138, -0.5, -0.5}; // R,Q,P,O,N.M // vertices
  Double_t y_3RR[6] = {10.91, 14.41, 17.0911, 15.9421, 13.86, 10.91};

  part_RR3->DefinePolygon(6, x_3RR, y_3RR);
  part_RR3->DefineSection(0, -0.75, 0, 0,
                          1); // (plane,-zplane/+zplane, x0, y0,(x/y))
  part_RR3->DefineSection(1, 0.78, 0, 0, 1);

  TGeoTranslation* tr_vol3_RR =
    new TGeoTranslation("tr_vol3_RR", -0.25, 12.66, 0); //
  tr_vol3_RR->RegisterYourself();

  //  quit diagona-> qdi
  Double_t x_qdi_Rbox = 3.1;   // dx=1.5
  Double_t y_qdi_Rbox = 7.159; //
  Double_t z_qdi_Rbox = 3.005; //

  TGeoRotation* rot_Rqdi = new TGeoRotation("rot_Rqdi", 0, 24.775, 0);
  rot_Rqdi->RegisterYourself();
  TGeoCombiTrans* combi_Rqdi =
    new TGeoCombiTrans(0, 5.579, -2.087, rot_Rqdi); // y=
  combi_Rqdi->SetName("combi_Rqdi");
  combi_Rqdi->RegisterYourself();

  // holes   circular hole_a. diameter=6.5 (a(6,22)); hole_midle d=6.5 H11
  Double_t radin_a_rail = 0.;
  Double_t radout_a_rail = 0.325; // diameter 3.5
  Double_t high_a_rail = 0.82;    //

  TGeoTranslation* tr_a_RR =
    new TGeoTranslation("tr_a_RR", -0.7, 0.6, 1.825); // right
  tr_a_RR->RegisterYourself();
  // circular hole_ir. diameter=M3 (3 mm)) prof trou:8, tar:6mm
  Double_t radin_ir_rail = 0.;
  Double_t radout_ir_rail = 0.15; // diameter 3
  Double_t high_ir_rail = 3.2;    // 19
  TGeoRotation* rot_ir_RR = new TGeoRotation("rot_ir_RR", 90, 90, 0);
  rot_ir_RR->RegisterYourself();
  // in y = l_253.5 - 6. center in (0,6,0)
  TGeoCombiTrans* combi_ir_RR =
    new TGeoCombiTrans(-8.62, 24.75, 1.5, rot_ir_RR);
  combi_ir_RR->SetName("combi_ir_RR");
  combi_ir_RR->RegisterYourself();

  TGeoCombiTrans* combi_ir2_RR =
    new TGeoCombiTrans(-8.6, 33.15, 1.5, rot_ir_RR);
  combi_ir2_RR->SetName("combi_ir2_RR");
  combi_ir2_RR->RegisterYourself();

  TGeoCombiTrans* combi_rail_R =
    new TGeoCombiTrans(24.1, -1.825, 0, rot_90x); // y=
  combi_rail_R->SetName("combi_rail_R");
  combi_rail_R->RegisterYourself();
  TGeoCombiTrans* combi_rail_L =
    new TGeoCombiTrans(-24.1, -1.825, 0, rot_90x); // y=
  combi_rail_L->SetName("combi_rail_L");
  combi_rail_L->RegisterYourself();

  // trasl L and R
  TGeoTranslation* tr_sr_l = new TGeoTranslation("tr_sr_l", -15.01, 0, 0); //
  tr_sr_l->RegisterYourself();
  TGeoTranslation* tr_sr_r = new TGeoTranslation("tr_sr_r", 15.01, 0, 0); //
  tr_sr_r->RegisterYourself();
  //
  //////// new modfi b ///////  cut arm
  TGeoXtru* part_RR2 = new TGeoXtru(2);
  part_RR2->SetName("part_RR2");
  //-TGeoVolume *vol_RR2 = gGeoManager->MakeXtru("S_XTRU_RR2",Al,2);
  //-TGeoXtru *xtru_RR2 = (TGeoXtru*)vol_RR2->GetShape();
  // 1b) modi, reducing arm
  // 1b) modi, reducing arm
  Double_t x_RR2[8] = {-0.5, -0.5, -9.3, -9.3,
                       -7.3, -7.3, 1.5, 1.5}; // K,E,F,G,H,I,J,L//vertices
  Double_t y_RR2[8] = {8.2, 13.863, 24.35, 25.65,
                       25.65, 25.078, 14.591, 8.2}; // 35.75, 35.75 -->25.65

  part_RR2->DefinePolygon(8, x_RR2, y_RR2);
  part_RR2->DefineSection(0, 0.776, 0, 0,
                          1); //(plane,-zplane/+zplane, x0, y0,(x/y))
  part_RR2->DefineSection(1, 2.225, 0, 0, 1);

  // 2b) modi  adding box  new element 1 box
  TGeoXtru* adi1_RR = new TGeoXtru(2);
  adi1_RR->SetName("S_ADI1_RR");

  Double_t x_adi1RR[4] = {-0.5, -.5, 1.5, 1.5}; // vertices
  Double_t y_adi1RR[4] = {2.2, 13.863, 13.863, 2.2};

  adi1_RR->DefinePolygon(4, x_adi1RR, y_adi1RR);
  adi1_RR->DefineSection(0, -0.75, 0, 0,
                         1);                 //(plane,-zplane/+zplane, x0, y0,(x/y))
  adi1_RR->DefineSection(1, 0.775, 0, 0, 1); // 0.76

  // 3b) modi adding new knee new element 2 new knee
  TGeoXtru* adi2_RR = new TGeoXtru(2); //
  adi2_RR->SetName("S_ADI2_RR");
  Double_t x_adi2RR[6] = {1.5, -0.5, -9.3, -9.3, -7.8, -7.8};
  Double_t y_adi2RR[6] = {13.863, 13.863, 24.35, 25.65, 25.65, 25.078};

  adi2_RR->DefinePolygon(6, x_adi2RR, y_adi2RR);
  adi2_RR->DefineSection(0, -0.75, 0, 0,
                         1);                  //(plane,-zplane/+zplane, x0, y0,(x/y))
  adi2_RR->DefineSection(1, 0.7755, 0, 0, 1); // 0.775

  //  4)modi  to quit ---> trap
  // Double_t RL_dx1=2.66; //at -z  (defined in rail L)
  // Double_t RL_dx2=1;  // dat +z
  // Double_t RL_dy=2.2; // dz=7.5     to quit
  // Double_t RL_dz=1.5; // dz=1.5     to quit
  // auto *trap1 = new TGeoTrd1("TRAP1",RL_dx1,RL_dx2 ,RL_dy ,RL_dz);
  TGeoRotation* rot_RR_Z310 = new TGeoRotation("rot_RR_Z310", -50, 0, 0);
  rot_RR_Z310->RegisterYourself();
  TGeoCombiTrans* combi_RR_trap =
    new TGeoCombiTrans(-5, 18.633, -1.5 - 0.025, rot_RR_Z310); // y=
  combi_RR_trap->SetName("combi_RR_trap");
  combi_RR_trap->RegisterYourself();

  //  5) modi  to quit     inferior part  box
  // Double_t x_qinf_box=10.66; // defined in RL
  // Double_t y_qinf_box=10.2;  //
  /// Double_t z_qinf_box=3.; // dz =1.5
  /// auto *s_RL_qinf_box =new TGeoBBox("S_RL_QINF_BOX",
  /// x_qinf_box/2,y_qinf_box/2,z_qinf_box/2);
  TGeoCombiTrans* combi_RR_qbox =
    new TGeoCombiTrans(-7, 23., -1.5 - 0.025, rot_RR_Z310); // rot to RR
  combi_RR_qbox->SetName("combi_RR_qbox");
  combi_RR_qbox->RegisterYourself();

  // 6) modi.  add penta face z
  TGeoXtru* pentfa_RR = new TGeoXtru(2);
  pentfa_RR->SetName("S_PENTFA_RR");
  Double_t x_pentfaRR[5] = {1., 1., -0.13, -1., -1.};
  Double_t y_pentfaRR[5] = {1.125, 0.045, -1.125, -1.125, 1.125}; // 1.125

  pentfa_RR->DefinePolygon(5, x_pentfaRR, y_pentfaRR);
  pentfa_RR->DefineSection(0, -5.05, 0, 0,
                           1);                 //(plane,-zplane/+zplane, x0, y0,(x/y))  // 0.75
  pentfa_RR->DefineSection(1, 5.055, 0, 0, 1); // 0.76.. 0.9036

  // TGeoRotation *rot_X90 = new TGeoRotation("rot_X90", 0,90,0);
  // rot_X90->RegisterYourself();
  TGeoCombiTrans* combi_RR_pent = new TGeoCombiTrans(
    -8.3, 30.705, 1.125 - 0.025, rot_X90); // x =8.3 , z= 1.125
  combi_RR_pent->SetName("combi_RR_pent");
  combi_RR_pent->RegisterYourself();

  // shape for rail R
  new TGeoBBox("RR_1box", x_RR_1box / 2, y_RR_1box / 2, z_RR_1box / 2);

  // auto *s_qdi_Rbox =new TGeoBBox("S_QDI_RBOX",
  // x_qdi_Rbox/2,y_qdi_Rbox/2,z_qdi_Rbox/2);

  // auto *s_ir_hole=new
  // TGeoTube("S_ir_HOLE",radin_ir_rail,radout_ir_rail,high_ir_rail/2);

  // auto *s_cc_hole=new
  // TGeoTube("S_CC_HOLE",radin_cc_rail,radout_cc_rail,high_cc_rail/2);

  // composite shape for rail R
  new TGeoCompositeShape(
    "RR_Shape_0",
    "RR_1box:tr_RR_1box + part_RR1 + part_RR2  - qdi_box:combi_qdi + "
    "S_ADI1_RR + S_ADI2_RR  -  TRAP1:combi_RR_trap - "
    "S_RL_QINF_BOX:combi_RR_qbox +S_PENTFA_RR:combi_RR_pent "); // quit +
                                                                // part_RR3
                                                                // old knee

  // auto * RR_Shape_0 = new
  // TGeoCompositeShape("RR_Shape_0","RR_1box:tr_RR_1box+ S_part_RR1  + part_RR2
  // +part_RR3- qdi_box:combi_qdi + S_ir_HOLE:combi_ir_RR
  // +S_ir_HOLE:combi_ir2_RR     "); //-RR_1box:tr_RL_1box- S_b_HOLE:tr_b_RR
  // -S_CC_HOLE:combi_cc2_RR

  // JOIN only for show L and R parts
  auto* rail_L_R_Shape = new TGeoCompositeShape(
    "RAIL_L_R_Shape", "  RL_Shape_0:combi_rail_L + RR_Shape_0:combi_rail_R");

  TGeoVolume* rail_L_R_vol0 =
    new TGeoVolume("RAIL_L_R_VOL0", rail_L_R_Shape, malu5083);
  // TGeoVolume* rail_L_R_vol0 = new TGeoVolume("RAIL_L_R_VOL0", rail_L_R_Shape,
  // kMedAlu);

  TGeoRotation* rot_rLR = new TGeoRotation("rot_rLR", 180, 180, 0);
  rot_rLR->RegisterYourself();
  TGeoCombiTrans* combi_rLR =
    new TGeoCombiTrans(0, -6.9, -0.5, rot_rLR); // 0,-6.9,-0.5-80
  combi_rLR->SetName("combi_rLR");
  combi_rLR->RegisterYourself();

  rail_L_R->AddNode(rail_L_R_vol0, 2, combi_rLR);

  // piece 8th support rail MB \_

  auto* sup_rail_MBL = new TGeoVolumeAssembly("sup_rail_MBL");

  /// new sup_rail_MB_L 2020  --------------------------------

  TGeoXtru* sup_MB_L = new TGeoXtru(2);
  sup_MB_L->SetName("sup_MB_L");

  // vertices a,b,c,d,e,f,g,h new...
  Double_t x_sMB_L[11] = {0., 0., 8.12, 24.55, 24.55, 28.25,
                          28.25, 34.55, 34.55, 31.737, 6.287};
  Double_t y_sMB_L[11] = {0., 1.8, 1.8, 9.934, 12.6, 12.6,
                          13.4, 13.4, 12.6, 12.6, 0.};

  sup_MB_L->DefinePolygon(11, x_sMB_L, y_sMB_L);
  sup_MB_L->DefineSection(0, -0.4, 0, 0,
                          1); //(plane, -zplane/ +zplane,x0,y0,(x/y))
  sup_MB_L->DefineSection(1, 0.4, 0, 0, 1);

  TGeoXtru* part_MBL_0 = new TGeoXtru(2);
  part_MBL_0->SetName("part_MBL_0"); // V-MBL_0

  // vertices a,b,c,d,e,f,g,h
  Double_t x[8] = {0., 0, 6.1, 31.55, 34.55, 34.55, 31.946, 6.496};
  Double_t y[8] = {-0.4, 0.4, 0.4, 13.0, 13.0, 12.2, 12.2, -0.4};

  part_MBL_0->DefinePolygon(8, x, y);
  part_MBL_0->DefineSection(0, -0.4, 0, 0,
                            1); // (plane, -zplane/ +zplane,x0,y0,(x/y))
  part_MBL_0->DefineSection(1, 0.4, 0, 0, 1);

  TGeoRotation* rot1_MBL_0 = new TGeoRotation("rot1_MBL_0", -90, -90, 90);
  rot1_MBL_0->RegisterYourself();

  // quit box in diag
  Double_t x_mb_box = 0.8;  // dx=4
  Double_t y_mb_box = 0.8;  // dy=4
  Double_t z_mb_box = 0.81; // dz=4 to quit
  TGeoTranslation* tr_mb_box =
    new TGeoTranslation("tr_mb_box", 24.05, 9.55, 0); // 240.5
  tr_mb_box->RegisterYourself();

  // lateral hole-box
  Double_t x_lat_box = 0.7; // dx=0.35
  Double_t y_lat_box = 1.8; // dy=0.9
  Double_t z_lat_box = 0.2; // dz=0.1
  TGeoTranslation* tr_lat1L_box =
    new TGeoTranslation("tr_lat1L_box", 4.6, 0, 0.4); //
  tr_lat1L_box->RegisterYourself();
  TGeoTranslation* tr_lat2L_box =
    new TGeoTranslation("tr_lat2L_box", 9.6, 1.65, 0.4); //
  tr_lat2L_box->RegisterYourself();
  /// TGeoTranslation* tr_lat3L_box = new
  /// TGeoTranslation("tr_lat3L_box", 18.53, 6.1, 0.4); //
  TGeoTranslation* tr_lat3L_box =
    new TGeoTranslation("tr_lat3L_box", 17.35, 5.923, 0.4); // 18.53
  tr_lat3L_box->RegisterYourself();
  TGeoTranslation* tr_lat4L_box =
    new TGeoTranslation("tr_lat4L_box", 26.45, 10, 0.4); //
  tr_lat4L_box->RegisterYourself();
  TGeoTranslation* tr_lat5L_box =
    new TGeoTranslation("tr_lat5L_box", 29.9, 11.6, 0.4); //
  tr_lat5L_box->RegisterYourself();

  TGeoTranslation* tr_lat1R_box =
    new TGeoTranslation("tr_lat1R_box", 4.6, 0, -0.4); //
  tr_lat1R_box->RegisterYourself();
  TGeoTranslation* tr_lat2R_box =
    new TGeoTranslation("tr_lat2R_box", 9.6, 1.65, -0.4); //
  tr_lat2R_box->RegisterYourself();
  /// TGeoTranslation* tr_lat3R_box = new
  /// TGeoTranslation("tr_lat3R_box", 18.53, 6.1, -0.4); //
  TGeoTranslation* tr_lat3R_box =
    new TGeoTranslation("tr_lat3R_box", 17.35, 5.923, -0.4); //
  tr_lat3R_box->RegisterYourself();
  TGeoTranslation* tr_lat4R_box =
    new TGeoTranslation("tr_lat4R_box", 26.45, 10, -0.4); //
  tr_lat4R_box->RegisterYourself();
  TGeoTranslation* tr_lat5R_box =
    new TGeoTranslation("tr_lat5R_box", 29.9, 11.6, -0.4); //
  tr_lat5R_box->RegisterYourself();

  // circular hole_1mbl. diameter=3.5 H9
  Double_t radin_1mb = 0.;
  Double_t radout_1mb = 0.175; // diameter 3.5mm _0.35 cm
  Double_t high_1mb = 2.825;   //  dh=+/- 4
  TGeoTranslation* tr1_mb =
    new TGeoTranslation("tr1_mb", 18.48, 6.1, 0.); // right
  tr1_mb->RegisterYourself();

  TGeoTranslation* tr2_mb =
    new TGeoTranslation("tr2_mb", 24.15, 8.9, 0.); // right
  tr2_mb->RegisterYourself();

  // circular hole_2mbl inclined and hole-up.diameter=M3 (3 mm)) prof , tar:8mm
  Double_t radin_2mb = 0.;
  Double_t radout_2mb = 0.15; // diameter 0.3
  Double_t high_2mb = 0.82;   ///  dh=+/- 4

  TGeoRotation* rot_hole2_MBL = new TGeoRotation("rot_hole2_MBL", 0, 90, 0);
  rot_hole2_MBL->RegisterYourself();

  TGeoTranslation* tr_mbl = new TGeoTranslation("tr_mbl", -7.5, 0., 0.); //
  tr_mbl->RegisterYourself();

  TGeoTranslation* tr_mbr = new TGeoTranslation("tr_mbr", 7.5, 0, 0); //
  tr_mbr->RegisterYourself();

  // hole up || hup
  TGeoCombiTrans* combi_hup_mb =
    new TGeoCombiTrans(32.5, 12.6, 0, rot_90x); // y=
  combi_hup_mb->SetName("combi_hup_mb");
  combi_hup_mb->RegisterYourself();

  // shape for rail MB
  new TGeoBBox("mb_box", x_mb_box / 2, y_mb_box / 2, z_mb_box / 2);
  new TGeoTube("hole_1mbl", radin_1mb, radout_1mb, high_1mb / 2); // d3.5
  new TGeoTube("hole_2mbl", radin_2mb, radout_2mb, high_2mb / 2); // d3
  new TGeoBBox("lat_box", x_lat_box / 2, y_lat_box / 2, z_lat_box / 2);

  // composite shape for rail_MB R + L
  auto* MB_Shape_0 = new TGeoCompositeShape(
    "MB_Shape_0", " sup_MB_L  - hole_2mbl:combi_hup_mb "); // new2020
  auto* MB_Shape_0L = new TGeoCompositeShape(
    "MB_Shape_0L", "MB_Shape_0  - lat_box:tr_lat3L_box ");
  auto* MB_Shape_0R = new TGeoCompositeShape(
    "MB_Shape_0R", "MB_Shape_0 - lat_box:tr_lat3R_box ");

  // auto * MB_Shape_0 = new TGeoCompositeShape("MB_Shape_0","  V_MBL_0 -
  // mb_box:tr_mb_box - hole_1mbl:tr1_mb + hole_1mbl:tr2_mb
  // -hole_2mbl:combi_hup_mb  ");
  ////new TGeoCompositeShape("MB_Shape_0", "part_MBL_0 - mb_box:tr_mb_box -
  /// hole_1mbl:tr1_mb - hole_2mbl:combi_hup_mb");

  /// new TGeoCompositeShape("MB_Shape_0L", "MB_Shape_0 - lat_box:tr_lat1L_box -
  /// lat_box:tr_lat2L_box - lat_box:tr_lat3L_box - lat_box:tr_lat4L_box -
  /// lat_box:tr_lat5L_box");

  /// new TGeoCompositeShape("MB_Shape_0R", "MB_Shape_0 - lat_box:tr_lat1R_box -
  /// lat_box:tr_lat2R_box - lat_box:tr_lat3R_box - lat_box:tr_lat4R_box -
  /// lat_box:tr_lat5R_box");

  new TGeoCompositeShape(
    "MB_Shape_1L",
    "MB_Shape_0L:rot1_MBL_0 - hole_2mbl"); // one piece "completed"
  // left and right
  new TGeoCompositeShape("MB_Shape_1R", "MB_Shape_0R:rot1_MBL_0 - hole_2mbl");

  auto* MB_Shape_2 = new TGeoCompositeShape(
    "MB_Shape_2", " MB_Shape_1L:tr_mbl +  MB_Shape_1R:tr_mbr ");

  // TGeoVolume *sup_rail_MBL_vol0 = new
  // TGeoVolume("SUPPORT_MBL_VOL0",MB_Shape_0,Al);
  TGeoVolume* sup_rail_MBL_vol =
    new TGeoVolume("SUPPORT_MBL_VOL", MB_Shape_2, malu5083);

  sup_rail_MBL->AddNode(sup_rail_MBL_vol, 1, rot_halfR);

  auto* stair = new TGeoVolumeAssembly("stair");

  stair->AddNode(sup_rail_MBL, 1,
                 new TGeoTranslation(0, 0 - 28.8 - 0.4, 0 + 0.675));
  stair->AddNode(Cross_mft, 1, new TGeoTranslation(0, -28.8, 4.55 + 0.675));
  stair->AddNode(Cross_mft_2, 2,
                 new TGeoTranslation(0, 1.65 - 28.8, 9.55 + 0.675));
  stair->AddNode(Cross_mb0, 4,
                 new TGeoTranslation(0, 5.423 - 28.8, 17.35 + 0.675)); // 6.1
  stair->AddNode(Cross_mft_3, 5,
                 new TGeoTranslation(0, 11.7 - 28.8, 25.55 + 0.675));
  stair->AddNode(Cross_mft_4, 6,
                 new TGeoTranslation(0, 12.5 - 28.8, 29.05 + 0.675));

  Double_t t_final_x;
  Double_t t_final_y;
  Double_t t_final_z;

  Double_t r_final_x;
  Double_t r_final_y;
  Double_t r_final_z;

  Double_t tyMB0;
  Double_t tzMB0;

  if (half == 0) {
    t_final_x = 0;
    t_final_y = 0.0;
    t_final_z = -80 - 0.675 - 0.15;

    r_final_x = 0;
    r_final_y = 0;
    r_final_z = 0;

    tyMB0 = -16.72;
    tzMB0 = -(45.3 + 46.7) / 2;
  }

  if (half == 1) {
    t_final_x = 0;
    t_final_y = 0.0;
    t_final_z = -80 - 0.675 - 0.15;

    r_final_x = 0;
    r_final_y = 0;
    r_final_z = 180;

    tyMB0 = 16.72;
    tzMB0 = -(45.3 + 46.7) / 2;
  }

  auto* t_final =
    new TGeoTranslation("t_final", t_final_x, t_final_y, t_final_z);
  auto* r_final = new TGeoRotation("r_final", r_final_x, r_final_y, r_final_z);
  auto* c_final = new TGeoCombiTrans(*t_final, *r_final);

  // 9th new 2020 ELEMENT middle framework back -----------------------------
  auto* frame_back = new TGeoVolumeAssembly("frame_back");
  /// variables
  // rectangular box1 to quit
  Double_t x_box_fwb = 15.8; // dx= 7.2 cm
  Double_t y_box_fwb = 5;
  Double_t z_box_fwb = 1;

  // rectangular box2 to add
  Double_t x_box2_fwb = 1.9; // dx= 7.2 cm
  Double_t y_box2_fwb = 0.5;
  Double_t z_box2_fwb = 0.6;

  ///// holes tub  1hole tranversal
  Double_t radin_fwb = 25.75;
  Double_t radout_fwb = 26.75; // diameter 3.5 H9  (0.35cm)
  Double_t high_fwb = 0.6;     ///

  // seg tub_back
  Double_t radin_stub = 23.6;  // 25.75
  Double_t radout_stub = 24.4; // 26.05
  Double_t high_stub = 0.6;
  Double_t ang_in_stub = 288.9; // 270 + 19.56
  Double_t ang_fin_stub = 342.; // 360-17.56

  TGeoRotation* rot_1hole_fwb = new TGeoRotation("rot_1hole_fwb", 0, 90, 0);
  rot_1hole_fwb->RegisterYourself();
  /// h= hole
  TGeoCombiTrans* acombi_fwb = new TGeoCombiTrans(5.2, 0, 0, rot_1hole_fwb);
  acombi_fwb->SetName("acombi_1h_fwb");
  acombi_fwb->RegisterYourself();

  TGeoTranslation* tr_box_y24 =
    new TGeoTranslation("tr_box_y24", 0, -24., 0.); //
  tr_box_y24->RegisterYourself();

  TGeoTranslation* tr_box2_fwb = new TGeoTranslation(
    "tr_box2_fwb", 24.4 - 1.9 / 2, -7.121 - 0.5 / 2, 0.); //
  tr_box2_fwb->RegisterYourself();

  TGeoRotation* rot_Z180_X180 = new TGeoRotation("rot_Z180_X180", 180, 180, 0);
  rot_Z180_X180->RegisterYourself();

  TGeoTranslation* tr_fb =
    new TGeoTranslation("tr_fb", 0, -2.3 - 0.06, 13.85 + 0.675); //
  tr_fb->RegisterYourself();

  //
  auto* q_box_fwb =
    new TGeoBBox("q_box_fwb", x_box_fwb / 2, y_box_fwb / 2, z_box_fwb / 2);
  auto* box2_fwb =
    new TGeoBBox("box2_fwb", x_box2_fwb / 2, y_box2_fwb / 2, z_box2_fwb / 2);
  auto* s_tub_fwb =
    new TGeoTube("s_tub_fwb", radin_fwb, radout_fwb, high_fwb / 2);
  // auto *s_ctub_fwb=new
  // TGeoTube("S_ctub_fwb",radin_fwb,radout_fwb,high_2hole_mb0/2);
  auto* s_stub_fwb =
    new TGeoTubeSeg("s_stub_fwb", radin_stub, radout_stub, high_stub / 2,
                    ang_in_stub, ang_fin_stub); // r_in,r_out,dZ,ang,ang

  /// composite shape for mb0

  auto* fwb_Shape_0 = new TGeoCompositeShape(
    "fwb_Shape_0",
    "  s_stub_fwb - q_box_fwb:tr_box_y24 + box2_fwb:tr_box2_fwb ");
  auto* fwb_Shape_1 = new TGeoCompositeShape(
    "fwb_Shape_1", "fwb_Shape_0 + fwb_Shape_0:rot_Z180_X180");

  auto* fwb_Volume =
    new TGeoVolume("fwb_Volume", fwb_Shape_1, malu5083); // malu5083
  frame_back->AddNode(fwb_Volume, 1, tr_fb);

  ////////////////////////////////////////////////
  /// 10 th colonne_support_MB012   new 2020
  auto* colonne_mb = new TGeoVolumeAssembly("colonne_mb");
  /// variables
  // rectangular box
  Double_t x_box_cmb = 1.9; // dx= 7.2 cm
  Double_t y_box_cmb = 0.6;
  Double_t z_box_cmb = 2.2033;

  ///// holes tub  1hole tranversal
  Double_t radin_c_mb = 0.;
  Double_t radout_c_mb = 0.3;  // diameter 3.5 H9  (0.35cm)
  Double_t high_c_mb = 2.2033; ///

  TGeoRotation* rot_1c_mb0 = new TGeoRotation("rot_1c_mb0", 0, 90, 0);
  rot_1c_mb0->RegisterYourself();
  /// h= hole
  TGeoCombiTrans* acombi_1c_mb0 = new TGeoCombiTrans(0.95, 0, 0, rot_1c_mb0);
  acombi_1c_mb0->SetName("acombi_1c_mb0");
  acombi_1c_mb0->RegisterYourself();
  TGeoCombiTrans* bcombi_1c_mb0 =
    new TGeoCombiTrans(-0.95, 0, 0, rot_1c_mb0); // y=
  bcombi_1c_mb0->SetName("bcombi_1c_mb0");
  bcombi_1c_mb0->RegisterYourself();

  // box to cut
  Double_t x_boxq_cmb = 3.; // dx= 7.2 cm
  Double_t y_boxq_cmb = 1.05;
  Double_t z_boxq_cmb = 4.;

  TGeoRotation* rot_X19 = new TGeoRotation("rot_X19", 0, -19, 0);
  rot_X19->RegisterYourself();
  TGeoCombiTrans* combi_qbox =
    new TGeoCombiTrans(0, +2.1 / 2 + 0.5, 0, rot_X19); // x =8.3 , z= 1.125
  combi_qbox->SetName("combi_qbox");
  combi_qbox->RegisterYourself();

  // shapes

  auto* box_cmb =
    new TGeoBBox("box_cmb", x_box_cmb / 2, y_box_cmb / 2, z_box_cmb / 2);
  auto* tub_cmb =
    new TGeoTube("tub_cmb", radin_c_mb, radout_c_mb, high_c_mb / 2);
  auto* boxq_cmb =
    new TGeoBBox("boxq_cmb", x_boxq_cmb / 2, y_boxq_cmb / 2, z_boxq_cmb / 2);

  // auto *s_2hole_mb0=new
  // TGeoTube("S_2HOLE_MB0",radin_2hole_mb0,radout_2hole_mb0,high_2hole_mb0/2);

  /// composite shape for mb0

  auto* c_mb_Shape_0 =
    new TGeoCompositeShape(
      "box_cmb:rot_1c_mb0 + tub_cmb:acombi_1c_mb0 + "
      "tub_cmb:bcombi_1c_mb0 - boxq_cmb:combi_qbox");

  TGeoTranslation* tr_cmb = new TGeoTranslation(
    "tr_cmb", 0, 5.923 - 28.8 + 2.2033 / 2 - 0.2, 17.35 + 0.675); //
  tr_cmb->RegisterYourself();
  ///////////////////
  // auto * cross_mb0_Volume = new
  // TGeoVolume("cross_mb0_Volume",c_mb0_Shape_0,Al); //
  auto* colonne_mb_Volume =
    new TGeoVolume("colonne_mb_Volume", c_mb_Shape_0, malu5083); // malu5083
  colonne_mb->AddNode(colonne_mb_Volume, 1, tr_cmb);

  //
  auto* Half_3 = new TGeoVolumeAssembly("Half_3");

  // Shell radii
  Float_t Shell_rmax = 60.6 + .7;
  Float_t Shell_rmin = 37.5 + .7;

  // Rotations and translations
  auto* tShell_0 =
    new TGeoTranslation("tShell_0", 0., 0., 3.1 + (25.15 + 1.) / 2.);
  auto* tShell_1 =
    new TGeoTranslation("tShell_1", 0., 0., -1.6 - (25.15 + 1.) / 2.);
  auto* tShellHole =
    new TGeoTranslation("tShellHole", 0., 0., 2. / 2. + (25.15 + 1.) / 2.);
  auto* tShellHole_0 =
    new TGeoTranslation("tShellHole_0", 0., -6.9, -26.1 / 2. - 6.2 / 2. - .1);
  auto* tShellHole_1 =
    new TGeoTranslation("tShellHole_1", 0., 0., -26.1 / 2. - 6.2 / 2. - .1);
  auto* tShell_Cut = new TGeoTranslation("tShell_Cut", 0., 25. / 2., 0.);
  auto* tShell_Cut_1 = new TGeoTranslation("tShell_Cut_1", -23., 0., -8.);
  auto* tShell_Cut_1_inv =
    new TGeoTranslation("tShell_Cut_1_inv", 23., 0., -8.);
  auto* Rz = new TGeoRotation("Rz", 50., 0., 0.);
  auto* Rz_inv = new TGeoRotation("Rz_inv", -50., 0., 0.);
  auto* RShell_Cut = new TGeoRotation("RShell_Cut", 90., 90. - 24., -7.5);
  auto* RShell_Cut_inv =
    new TGeoRotation("RShell_Cut_inv", 90., 90. + 24., -7.5);

  auto* cShell_Cut = new TGeoCombiTrans(*tShell_Cut_1, *RShell_Cut);
  auto* cShell_Cut_inv = new TGeoCombiTrans(*tShell_Cut_1_inv, *RShell_Cut_inv);

  tShell_0->RegisterYourself();
  tShell_1->RegisterYourself();
  tShellHole->RegisterYourself();
  tShellHole_0->RegisterYourself();
  tShellHole_1->RegisterYourself();
  tShell_Cut->RegisterYourself();
  Rz->RegisterYourself();
  Rz_inv->RegisterYourself();
  RShell_Cut->RegisterYourself();
  cShell_Cut->SetName("cShell_Cut");
  cShell_Cut->RegisterYourself();
  cShell_Cut_inv->SetName("cShell_Cut_inv");
  cShell_Cut_inv->RegisterYourself();

  // Basic shapes for Half_3
  TGeoShape* Shell_0 = new TGeoTubeSeg("Shell_0", Shell_rmax / 2. - .1,
                                       Shell_rmax / 2., 6.2 / 2., 12., 168.);
  TGeoShape* Shell_1 = new TGeoTubeSeg("Shell_1", Shell_rmin / 2. - .1,
                                       Shell_rmin / 2., 3.2 / 2., 0., 180.);
  new TGeoConeSeg("Shell_2", (25.15 + 1.0) / 2., Shell_rmin / 2. - .1,
                  Shell_rmin / 2., Shell_rmax / 2. - .1, Shell_rmax / 2., 0.,
                  180.);
  TGeoShape* Shell_3 =
    new TGeoTube("Shell_3", 0., Shell_rmin / 2. + .1, .1 / 2.);
  TGeoShape* ShellHole_0 = new TGeoTrd1("ShellHole_0", 17.5 / 4., 42.5 / 4.,
                                        80. / 2., (25.15 + 1.) / 2.);
  TGeoShape* ShellHole_1 =
    new TGeoBBox("ShellHole_1", 42.5 / 4., 80. / 2., 2. / 2. + 0.00001);
  TGeoShape* ShellHole_2 = new TGeoBBox(
    "ShellHole_2", 58.9 / 4., (Shell_rmin - 2.25) / 2., .4 / 2. + 0.00001);
  TGeoShape* ShellHole_3 = new TGeoBBox(
    "ShellHole_3", 80. / 4., (Shell_rmin - 11.6) / 2., .4 / 2. + 0.00001);

  // For the extra cut, not sure if this is the shape (apprx. distances)
  TGeoShape* Shell_Cut_0 = new TGeoTube("Shell_Cut_0", 0., 3.5, 5. / 2.);
  TGeoShape* Shell_Cut_1 =
    new TGeoBBox("Shell_Cut_1", 7. / 2., 25. / 2., 5. / 2.);

  // Composite shapes for Half_3
  auto* Half_3_Shape_0 = new TGeoCompositeShape(
    "Half_3_Shape_0", "Shell_Cut_0+Shell_Cut_1:tShell_Cut");
  new TGeoCompositeShape(
    "Half_3_Shape_1",
    "Shell_2 - Half_3_Shape_0:cShell_Cut - Half_3_Shape_0:cShell_Cut_inv");
  auto* Half_3_Shape_2 = new TGeoCompositeShape(
    "Half_3_Shape_2", "ShellHole_0+ShellHole_1:tShellHole");
  new TGeoCompositeShape("Half_3_Shape_3",
                         "Shell_3:tShellHole_1 -(ShellHole_2:tShellHole_1 + "
                         "ShellHole_3:tShellHole_0)");
  auto* Half_3_Shape_4 = new TGeoCompositeShape(
    "Half_3_Shape_4",
    "(Shell_0:tShell_0 + Half_3_Shape_1+ Shell_1:tShell_1) - (Half_3_Shape_2 "
    "+ "
    "Half_3_Shape_2:Rz + Half_3_Shape_2:Rz_inv)+Half_3_Shape_3");

  auto* Half_3_Volume =
    new TGeoVolume("Half_3_Volume", Half_3_Shape_4, kMedAlu);
  // Position of the piece relative to the origin which for this code is the
  // center of the the Framework piece (See Half_2)
  // Half_3->AddNode(Half_3_Volume, 1, new TGeoTranslation(0., 0., -19.));

  TGeoRotation* rot_z180 =
    new TGeoRotation("rot_z180", 0, 180, 0); // orig: (180,0,0)
  rot_z180->RegisterYourself();
  // in y = l_253.5 - 6. center in (0,6,0)
  TGeoCombiTrans* combi_coat = new TGeoCombiTrans(0, 0, 19.5 - 0.45, rot_z180);
  combi_coat->SetName("combi_coat");
  combi_coat->RegisterYourself();

  Half_3->AddNode(Half_3_Volume, 1, combi_coat);

  HalfConeVolume->AddNode(stair, 1, c_final); //
  HalfConeVolume->AddNode(base, 2, c_final);
  HalfConeVolume->AddNode(rail_L_R, 3, c_final); // R&L
  HalfConeVolume->AddNode(Fra_front, 4, c_final);
  HalfConeVolume->AddNode(middle, 5, c_final);     //
  HalfConeVolume->AddNode(frame_back, 6, c_final); //
  HalfConeVolume->AddNode(colonne_mb, 7, c_final); //

  //========================== Mother Boards =========================================

  // =============  MotherBoard 0 and 1
  Double_t mMB0cu[3];
  Double_t mMB0fr4;
  Double_t mMB0pol;
  Double_t mMB0epo;
  // Sizes
  mMB0cu[0] = {13.65};
  mMB0cu[1] = {0.00615}; // 122.5 microns * taux d'occupation 50% = 61.5 microns
  mMB0cu[2] = {2.39};
  mMB0fr4 = 0.1;    // 1 mm
  mMB0pol = 0.0150; // 150 microns
  mMB0epo = 0.0225; // 225 microns
  // Materials
  auto* mCu = gGeoManager->GetMedium("MFT_Cu$");
  auto* mFR4 = gGeoManager->GetMedium("MFT_FR4$");
  auto* mPol = gGeoManager->GetMedium("MFT_Polyimide$");
  auto* mEpo = gGeoManager->GetMedium("MFT_Epoxy$");
  auto* mInox = gGeoManager->GetMedium("MFT_Inox$");

  auto* MotherBoard0 = new TGeoVolumeAssembly(Form("MotherBoard0_H%d", half));
  // 4 layers
  TGeoVolume* vMB0cu = gGeoManager->MakeBox("vMB0cu", mCu, mMB0cu[0] / 2, mMB0cu[1] / 2, mMB0cu[2] / 2);
  TGeoVolume* vMB0fr4 = gGeoManager->MakeBox("vMB0fr4", mFR4, mMB0cu[0] / 2, mMB0fr4 / 2, mMB0cu[2] / 2);
  TGeoVolume* vMB0pol = gGeoManager->MakeBox("vMB0pol", mPol, mMB0cu[0] / 2, mMB0pol / 2, mMB0cu[2] / 2);
  TGeoVolume* vMB0epo = gGeoManager->MakeBox("vMB0epo", mEpo, mMB0cu[0] / 2, mMB0epo / 2, mMB0cu[2] / 2);
  // Screws = Head + Thread
  TGeoVolume* vMB0screwH = gGeoManager->MakeTube("vMB0screwH", mInox, 0.0, 0.7 / 2, 0.35 / 2); // tete
  TGeoVolume* vMB0screwT = gGeoManager->MakeTube("vMB0screwT", mInox, 0.0, 0.4 / 2, 1.2 / 2);  // filetage
  // Insert Sertitec
  TGeoVolume* vMB0serti = gGeoManager->MakeTube("vMB0serti", mInox, 0.16 / 2, 0.556 / 2, 0.15 / 2); // tete

  vMB0cu->SetLineColor(kRed);
  vMB0fr4->SetLineColor(kBlack);
  vMB0pol->SetLineColor(kGreen);
  vMB0epo->SetLineColor(kBlue);
  vMB0screwH->SetLineColor(kOrange);
  vMB0screwT->SetLineColor(kOrange);
  vMB0serti->SetLineColor(kOrange);
  // Positioning the layers
  MotherBoard0->AddNode(vMB0cu, 1);
  Int_t signe;
  if (half == 0) {
    signe = -1;
  }
  if (half == 1) {
    signe = +1;
  }
  auto* t_MB0fr4 = new TGeoTranslation("translation_fr4", 0.0, signe * (mMB0fr4 + mMB0cu[1]) / 2, 0.0);
  t_MB0fr4->RegisterYourself();
  MotherBoard0->AddNode(vMB0fr4, 1, t_MB0fr4);
  auto* t_MB0pol = new TGeoTranslation("translation_pol", 0.0, signe * (mMB0fr4 + (mMB0cu[1] + mMB0pol) / 2), 0.0);
  t_MB0pol->RegisterYourself();
  MotherBoard0->AddNode(vMB0pol, 1, t_MB0pol);
  auto* t_MB0epo = new TGeoTranslation("translation_epo", 0.0, signe * (mMB0fr4 + mMB0pol + (mMB0cu[1] + mMB0epo) / 2), 0.0);
  t_MB0epo->RegisterYourself();
  MotherBoard0->AddNode(vMB0epo, 1, t_MB0epo);
  auto* r_MB0screw = new TGeoRotation("rotation_vMB0screw", 0, 90, 0);
  auto* t_MB0screwH1 = new TGeoCombiTrans(mMB0cu[0] / 2 - 1.65,
                                          signe * (mMB0fr4 + mMB0pol + mMB0epo + (mMB0cu[1] + 0.35) / 2), 0.0, r_MB0screw);
  t_MB0screwH1->RegisterYourself();
  auto* t_MB0screwT1 = new TGeoCombiTrans(mMB0cu[0] / 2 - 1.65, -signe * (mMB0cu[1] + 1.2) / 2, 0.0, r_MB0screw);
  t_MB0screwT1->RegisterYourself();
  auto* t_MB0screwH2 = new TGeoCombiTrans(-(mMB0cu[0] / 2 - 1.65),
                                          signe * (mMB0fr4 + mMB0pol + mMB0epo + (mMB0cu[1] + 0.35) / 2), 0.0, r_MB0screw);
  t_MB0screwH2->RegisterYourself();
  auto* t_MB0screwT2 = new TGeoCombiTrans(-(mMB0cu[0] / 2 - 1.65), -signe * (mMB0cu[1] + 1.2) / 2, 0.0, r_MB0screw);
  t_MB0screwT2->RegisterYourself();
  auto* t_MB0serti1 = new TGeoCombiTrans(mMB0cu[0] / 2 - 2.65,
                                         signe * (mMB0fr4 + mMB0pol + mMB0epo + (mMB0cu[1] + 0.153) / 2), 0.0, r_MB0screw);
  t_MB0serti1->RegisterYourself();
  auto* t_MB0serti2 = new TGeoCombiTrans(-(mMB0cu[0] / 2 - 2.65),
                                         signe * (mMB0fr4 + mMB0pol + mMB0epo + (mMB0cu[1] + 0.153) / 2), 0.0, r_MB0screw);
  t_MB0serti2->RegisterYourself();
  MotherBoard0->AddNode(vMB0screwH, 1, t_MB0screwH1);
  MotherBoard0->AddNode(vMB0screwT, 1, t_MB0screwT1);
  MotherBoard0->AddNode(vMB0screwH, 1, t_MB0screwH2);
  MotherBoard0->AddNode(vMB0screwT, 1, t_MB0screwT2);
  MotherBoard0->AddNode(vMB0serti, 1, t_MB0serti1);
  MotherBoard0->AddNode(vMB0serti, 1, t_MB0serti2);

  // Positioning the board
  auto* t_MB0 = new TGeoTranslation("translation_MB0", 0.0, tyMB0, tzMB0);
  t_MB0->RegisterYourself();
  auto* t_MB1 = new TGeoTranslation("translation_MB1", 0.0, tyMB0, tzMB0 - 3.3); // 3.3 cm is the interdistance between disk 0 and 1
  t_MB1->RegisterYourself();
  auto* r_MB0 = new TGeoRotation("rotation_MB0", 0.0, 0.0, 0.0);
  r_MB0->RegisterYourself();
  auto* p_MB0 = new TGeoCombiTrans(*t_MB0, *r_MB0);
  p_MB0->RegisterYourself();
  auto* p_MB1 = new TGeoCombiTrans(*t_MB1, *r_MB0);
  p_MB1->RegisterYourself();
  // Final addition of the board
  HalfConeVolume->AddNode(MotherBoard0, 1, p_MB0);
  HalfConeVolume->AddNode(MotherBoard0, 1, p_MB1);

  auto* MotherBoard0_1 = new TGeoVolumeAssembly(Form("MotherBoard0_1_H%d", half));
  // 4 layers
  TGeoVolume* vMB0cu_1 = gGeoManager->MakeBox("vMB0cu_1", mCu, 18.0 / 2, mMB0cu[1] / 2, 1.2 / 2);
  TGeoVolume* vMB0fr4_1 = gGeoManager->MakeBox("vMB0fr4_1", mFR4, 18.0 / 2, mMB0fr4 / 2, 1.2 / 2);
  TGeoVolume* vMB0pol_1 = gGeoManager->MakeBox("vMB0pol_1", mPol, 18.0 / 2, mMB0pol / 2, 1.2 / 2);
  TGeoVolume* vMB0epo_1 = gGeoManager->MakeBox("vMB0epo_1", mEpo, 18.0 / 2, mMB0epo / 2, 1.2 / 2);
  vMB0cu_1->SetLineColor(kRed);
  vMB0fr4_1->SetLineColor(kBlack);
  vMB0pol_1->SetLineColor(kGreen);
  vMB0epo_1->SetLineColor(kBlue);

  MotherBoard0_1->AddNode(vMB0cu_1, 1);
  MotherBoard0_1->AddNode(vMB0fr4_1, 1, t_MB0fr4);
  MotherBoard0_1->AddNode(vMB0pol_1, 1, t_MB0pol);
  MotherBoard0_1->AddNode(vMB0epo_1, 1, t_MB0epo);

  // ================ MotherBoard 2
  Double_t mMB2cu[4];
  Double_t mMB2fr4;
  Double_t mMB2pol;
  Double_t mMB2epo;
  // Sizes
  mMB2cu[0] = {24.0};
  mMB2cu[1] = {21.0};
  mMB2cu[2] = {0.0079}; // 315 microns * taux d'occupation 25% = 79 microns
  mMB2cu[3] = {8.5};
  mMB2fr4 = 0.2;    // 2 mm
  mMB2pol = 0.0175; // 175 microns
  mMB2epo = 0.0075; // 75 microns
  auto* MotherBoard2 = new TGeoVolumeAssembly(Form("MotherBoard2_H%d", half));
  // 4 layers
  TGeoVolume* vMB2cu = gGeoManager->MakeTrd1("vMB2cu", mCu, mMB2cu[0] / 2, mMB2cu[1] / 2, mMB2cu[2] / 2, mMB2cu[3] / 2);
  TGeoVolume* vMB2fr4 = gGeoManager->MakeTrd1("vMB2fr4", mFR4, mMB2cu[0] / 2, mMB2cu[1] / 2, mMB2fr4 / 2, mMB2cu[3] / 2);
  TGeoVolume* vMB2pol = gGeoManager->MakeTrd1("vMB2pol", mPol, mMB2cu[0] / 2, mMB2cu[1] / 2, mMB2pol / 2, mMB2cu[3] / 2);
  TGeoVolume* vMB2epo = gGeoManager->MakeTrd1("vMB2epo", mEpo, mMB2cu[0] / 2, mMB2cu[1] / 2, mMB2epo / 2, mMB2cu[3] / 2);

  vMB2cu->SetLineColor(kRed);
  vMB2fr4->SetLineColor(kBlack);
  vMB2pol->SetLineColor(kGreen);
  vMB2epo->SetLineColor(kBlue);

  auto* t_MB2fr4 = new TGeoTranslation("translation_fr4", 0.0, signe * (mMB2fr4 + mMB2cu[2]) / 2, 0.0);
  t_MB2fr4->RegisterYourself();
  auto* t_MB2pol = new TGeoTranslation("translation_pol", 0.0, signe * (mMB2fr4 + (mMB2cu[2] + mMB2pol) / 2), 0.0);
  t_MB2pol->RegisterYourself();
  auto* t_MB2epo = new TGeoTranslation("translation_epo", 0.0, signe * (mMB2fr4 + mMB2pol + (mMB2cu[2] + mMB2epo) / 2), 0.0);
  t_MB2epo->RegisterYourself();

  MotherBoard2->AddNode(vMB2cu, 1);
  MotherBoard2->AddNode(vMB2fr4, 1, t_MB2fr4);
  MotherBoard2->AddNode(vMB2pol, 1, t_MB2pol);
  MotherBoard2->AddNode(vMB2epo, 1, t_MB2epo);
  for (Float_t i = -1; i < 3; i++) {
    auto* t_MB2serti1 = new TGeoTranslation("translationMB2serti1", 8.5, -signe * (mMB2cu[2] + 0.153) / 2, 1.3 * i);
    t_MB2serti1->RegisterYourself();
    auto* t_MB2serti2 = new TGeoTranslation("translationMB2serti2", -8.5, -signe * (mMB2cu[2] + 0.153) / 2, 1.3 * i);
    t_MB2serti2->RegisterYourself();
    auto* p_MB2serti1 = new TGeoCombiTrans(*t_MB2serti1, *r_MB0screw);
    p_MB2serti1->RegisterYourself();
    auto* p_MB2serti2 = new TGeoCombiTrans(*t_MB2serti2, *r_MB0screw);
    p_MB2serti2->RegisterYourself();
    MotherBoard2->AddNode(vMB0serti, 1, p_MB2serti1);
    MotherBoard2->AddNode(vMB0serti, 1, p_MB2serti2);
  }

  for (Float_t i = -2; i < 1; i++) {
    auto* t_MB2serti3 = new TGeoTranslation("translationMB2serti3", 0.7, -signe * (mMB2cu[2] + 0.153) / 2, 1.3 * i);
    t_MB2serti3->RegisterYourself();
    auto* t_MB2serti4 = new TGeoTranslation("translationMB2serti4", -0.7, -signe * (mMB2cu[2] + 0.153) / 2, 1.3 * i);
    t_MB2serti4->RegisterYourself();
    auto* p_MB2serti3 = new TGeoCombiTrans(*t_MB2serti3, *r_MB0screw);
    p_MB2serti3->RegisterYourself();
    auto* p_MB2serti4 = new TGeoCombiTrans(*t_MB2serti4, *r_MB0screw);
    p_MB2serti4->RegisterYourself();
    MotherBoard2->AddNode(vMB0serti, 1, p_MB2serti3);
    MotherBoard2->AddNode(vMB0serti, 1, p_MB2serti4);
  }

  for (Float_t i = -2; i < 2; i++) {
    auto* t_MB2serti5 = new TGeoTranslation("translationMB2serti5", 7.0 * i + 3.5, -signe * (mMB2cu[2] + 0.153) / 2, -2.5);
    t_MB2serti5->RegisterYourself();
    auto* p_MB2serti5 = new TGeoCombiTrans(*t_MB2serti5, *r_MB0screw);
    p_MB2serti5->RegisterYourself();
    auto* t_MB2serti6 = new TGeoTranslation("translationMB2serti6", 7.0 * i + 3.5, -signe * (mMB2cu[2] + 0.153) / 2, -3.5);
    t_MB2serti6->RegisterYourself();
    auto* p_MB2serti6 = new TGeoCombiTrans(*t_MB2serti6, *r_MB0screw);
    p_MB2serti6->RegisterYourself();
    MotherBoard2->AddNode(vMB0serti, 1, p_MB2serti5);
    MotherBoard2->AddNode(vMB0serti, 1, p_MB2serti6);
  }
  // Connector board of MB0 on MB2
  auto* t_MotherBoard0_1 = new TGeoTranslation("translation_MB0_1", 0.0, -signe * (-0.5), 3.5);
  t_MotherBoard0_1->RegisterYourself();
  auto* t_MotherBoard0_2 = new TGeoTranslation("translation_MB0_2", 0.0, -signe * (-0.5), 1.5);
  t_MotherBoard0_2->RegisterYourself();
  MotherBoard2->AddNode(MotherBoard0_1, 1, t_MotherBoard0_1);
  MotherBoard2->AddNode(MotherBoard0_1, 1, t_MotherBoard0_2);
  // Positioning the board
  auto* t_MotherBoard2 = new TGeoTranslation("translation_MB2", 0.0,
                                             -signe * (-20.52 + mMB2fr4 + mMB2pol + mMB2epo + 2.2 * TMath::Sin(19.0)),
                                             -62.8 + 2.2 * TMath::Cos(19.0));
  t_MotherBoard2->RegisterYourself();
  auto* r_MotherBoard2 = new TGeoRotation("rotation_MB2", 0.0, -signe * (-19.0), 0.0);
  r_MotherBoard2->RegisterYourself();
  auto* p_MB2 = new TGeoCombiTrans(*t_MotherBoard2, *r_MotherBoard2);
  p_MB2->RegisterYourself();
  HalfConeVolume->AddNode(MotherBoard2, 1, p_MB2);
  //===================================================================
  return HalfConeVolume;
}
