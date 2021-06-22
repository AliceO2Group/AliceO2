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

#include "TGeoBBox.h"
#include "TGeoBoolNode.h"
#include "TGeoCompositeShape.h"
#include "TGeoCone.h"
#include "TGeoManager.h"
#include "TGeoMaterial.h"
#include "TGeoMatrix.h"
#include "TGeoMedium.h"
#include "TGeoShape.h"
#include "TGeoTrd1.h"
#include "TGeoTube.h"
#include "TGeoVolume.h"
#include "TGeoXtru.h"
#include "TMath.h"
#include <TGeoArb8.h> // for TGeoTrap
#include "MFTBase/MFTBaseParam.h"
#include "MFTBase/GeometryBuilder.h"

#include "MFTBase/Constants.h"
#include "MFTBase/HalfCone.h"

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

  TGeoMedium* malu5083 = gGeoManager->GetMedium("MFT_Alu5083$");

  // Rotation
  TGeoRotation* rot1 = new TGeoRotation("rot1", 180, -180, 0);

  rot1->RegisterYourself();
  TGeoRotation* rot2 = new TGeoRotation("rot2", 90, -90, 0);
  rot2->RegisterYourself();

  TGeoRotation* rot3 = new TGeoRotation("rot3", 0, 90, 0);
  rot3->RegisterYourself();

  TGeoRotation* rot_90x = new TGeoRotation("rot_90x", 0, -90, 0);
  rot_90x->RegisterYourself();

  TGeoRotation* rot_base = new TGeoRotation("rot_base", 180, 180, 0);
  rot_base->RegisterYourself();

  TGeoCombiTrans* combi1 = new TGeoCombiTrans(0, -10.3, 1.29, rot1);
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
  Double_t radout_1hmb0 = 0.175;
  Double_t high_1hmb0 = 0.7;

  TGeoRotation* rot_1hole_mb0 = new TGeoRotation("rot_1hole_mb0", 0, 90, 0);
  rot_1hole_mb0->RegisterYourself();
  // h= hole
  TGeoCombiTrans* acombi_1h_mb0 = new TGeoCombiTrans(5.2, 0, 0, rot_1hole_mb0);
  acombi_1h_mb0->SetName("acombi_1h_mb0");
  acombi_1h_mb0->RegisterYourself();
  TGeoCombiTrans* bcombi_1h_mb0 = new TGeoCombiTrans(-5.2, 0, 0, rot_1hole_mb0);
  bcombi_1h_mb0->SetName("bcombi_1h_mb0");
  bcombi_1h_mb0->RegisterYourself();

  // 2hole coaxial
  Double_t radin_2hmb0 = 0.;
  Double_t radout_2hmb0 = 0.15;
  Double_t high_2hmb0 = 1.2;
  TGeoRotation* rot_2hole_mb0 = new TGeoRotation("rot_2hole_mb0", 90, 90, 0);
  rot_2hole_mb0->SetName("rot_2hole_mb0");
  rot_2hole_mb0->RegisterYourself();

  TGeoCombiTrans* combi_2hole_mb0 =
    new TGeoCombiTrans(6.7, 0, 0, rot_2hole_mb0);
  combi_2hole_mb0->SetName("combi_2hole_mb0");
  combi_2hole_mb0->RegisterYourself();

  TGeoCombiTrans* combi_2hole_mb0_b =
    new TGeoCombiTrans(-6.7, 0, 0, rot_2hole_mb0);
  combi_2hole_mb0_b->SetName("combi_2hole_mb0_b");
  combi_2hole_mb0_b->RegisterYourself();

  // shape for cross_mb0..
  new TGeoBBox("box_mb0", x_boxmb0 / 2, y_boxmb0 / 2, z_boxmb0 / 2);
  new TGeoTube("hole1_mb0", radin_1hmb0, radout_1hmb0, high_1hmb0 / 2);
  new TGeoTube("hole2_mb0", radin_2hmb0, radout_2hmb0, high_2hmb0 / 2);
  /// composite shape for mb0

  auto* c_mb0_Shape_0 = new TGeoCompositeShape(
    "c_mb0_Shape_0",
    "box_mb0  - hole2_mb0:combi_2hole_mb0 - "
    "hole2_mb0:combi_2hole_mb0_b");

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
  Double_t radout_hole_cbeam = 0.15;
  Double_t high_hole_cbeam = 0.91;

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

  auto* c_cbeam_Shape = new TGeoCompositeShape(
    "c_cbeam_Shape",
    " s_cb:rot_cb - hole_cbeam:combi_hole_1cbeam - "
    "hole_cbeam:combi_hole_2cbeam");

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
  Double_t y_top_adfwf[4] = {-3.6 - 0.12, -3.6 - 0.12, -5.56, -5.83};
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

  TGeoRotation* rot_180yR = new TGeoRotation("rot_180yR", 180, -180, 0);
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
  Double_t z_boxB_down = 1.26;
  // seg tub
  Double_t radin_segtub = 16.9;
  Double_t radout_segtub = 17.5;
  Double_t high_segtub = 0.6;
  Double_t ang_in_segtub = 212.1;
  Double_t ang_fin_segtub = 241.92;

  // trans. rot.
  TGeoCombiTrans* combi_3a = new TGeoCombiTrans(-7.4, 0, 8.975, rot2);
  combi_3a->SetName("combi_3a");
  combi_3a->RegisterYourself();

  TGeoTranslation* tr1_up = new TGeoTranslation("tr1_up", -7.4, 0, 8.28);

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
  TGeoCombiTrans* combi_front_L =
    new TGeoCombiTrans(-7.1, -16.2, 32.5 + 0.675, rot_90x);

  combi_front_L->SetName("combi_front_L");
  combi_front_L->RegisterYourself();

  TGeoTranslation* tr_ff = new TGeoTranslation(
    "tr_ff", 0, -2.5 - 0.31, 32.5 + 0.675); // 7.1 , -16.2 z32.5
  tr_ff->RegisterYourself();

  TGeoCombiTrans* combi_front_R =
    new TGeoCombiTrans(0, -2.5 - 0.31, 32.5 + 0.675, rot_180yR);
  combi_front_R->SetName("combi_front_R");
  combi_front_R->RegisterYourself();

  auto* fra_front_Shape_2 = new TGeoCompositeShape(
    "Fra_front_Shape_2",
    "fwf_tub  - box_qdown:tr_qdown + box_addown:tr_addown  + tria_fwf + "
    "top_adfwf - q_upbox:tr_q_upbox");

  auto* fra_front_Shape_3 = new TGeoCompositeShape(
    "Fra_front_Shape_3", "Fra_front_Shape_2 + Fra_front_Shape_2:rot_180yR");

  auto* Fra_front_Volume_R =
    new TGeoVolume("Fra_front_Volume_R", fra_front_Shape_2, malu5083);
  auto* Fra_front_Volume_L =
    new TGeoVolume("Fra_front_Volume_L", fra_front_Shape_2, malu5083);

  auto* Fra_front_Volume_RL =
    new TGeoVolume("Fra_front_Volume_RL", fra_front_Shape_3, malu5083);

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
    new TGeoTranslation("tr_2la", 0, -8.1, high_disc / 2);
  tr_2la->RegisterYourself();

  // circular border C SEG_BORD
  // seg tub 3 xy
  Double_t radin_bord = 0.5;
  Double_t radout_bord = 0.9; //
  Double_t high_bord = 1.355; // 13.5
  Double_t ang_in_bord = 0;
  Double_t ang_fin_bord = 90;
  TGeoRotation* rot1_bord1 = new TGeoRotation("rot1_bord1", 14.8, 0, 0);
  rot1_bord1->RegisterYourself();
  TGeoCombiTrans* combi_bord1 =
    new TGeoCombiTrans(-26.7995, -13.0215, 0, rot1_bord1); // y=
  combi_bord1->SetName("combi_bord1");
  combi_bord1->RegisterYourself();

  TGeoRotation* rot2_bord1 = new TGeoRotation("rot2_bord1", -50, 0, 0);
  rot2_bord1->RegisterYourself();
  TGeoCombiTrans* combi2_bord1 =
    new TGeoCombiTrans(-21.3795, -20.7636, 0, rot2_bord1);
  combi2_bord1->SetName("combi2_bord1");
  combi2_bord1->RegisterYourself();
  //
  TGeoRotation* rot1_bord2 = new TGeoRotation("rot1_bord2", 250, 0, 0);
  rot1_bord2->RegisterYourself();
  TGeoCombiTrans* combi1_bord2 =
    new TGeoCombiTrans(-9.0527, -23.3006, 0, rot1_bord2);
  combi1_bord2->SetName("combi1_bord2");
  combi1_bord2->RegisterYourself();
  // e|°____°|
  TGeoRotation* rot_cent_bord = new TGeoRotation("rot_cent_bord", 90, 0, 0);
  rot_cent_bord->RegisterYourself();
  TGeoCombiTrans* combi_cent_bord =
    new TGeoCombiTrans(-6.5, -27.094, 0, rot_cent_bord);
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
  TGeoTranslation* tr_hole1 = new TGeoTranslation("tr_hole1", 0, -28.0, 0);
  tr_hole1->RegisterYourself();

  TGeoTranslation* tr2_hole1 =
    new TGeoTranslation("tr2_hole1", -26.5, -8.5, 0); // left
  tr2_hole1->RegisterYourself();

  TGeoTranslation* tr3_hole1 =
    new TGeoTranslation("tr3_hole1", 26.5, -8.5, 0); // right
  tr3_hole1->RegisterYourself();

  // circular hole2 ; hole2 r=6.7
  Double_t radin_hole2 = 0;
  Double_t radout_hole2 = 0.335;
  Double_t high_hole2 = 1.36;
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

  Double_t radin_ccut = 27.5;
  Double_t radout_ccut = 29.;
  Double_t high_ccut = 1.4;
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
                  ang_in_1hole, ang_fin_1hole);
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

  new TGeoCompositeShape("base_Shape_0", " disc - box1 - box2 - box3");
  new TGeoCompositeShape(
    "base_Shape_1",
    "(seg_1hole - seg_bord:combi_bord1 - seg_bord:combi2_bord1) + seg_2hole "
    "-seg_bord:combi1_bord2 + cbox:tr_cbox");

  new TGeoCompositeShape("base_Shape_2",
                         " seg_3hole + seg_bord:combi_cent_bord");

  new TGeoCompositeShape("base_Shape_3", " labox1:tr_la + labox2:tr_2la ");

  auto* base_Shape_4 = new TGeoCompositeShape(
    "base_Shape_4",
    "base_Shape_0 - base_Shape_1 - base_Shape_1:rot1 + base_Shape_2 - "
    "base_Shape_3 + tongbox:tr_tong - c_cut");

  auto* base4_Volume = new TGeoVolume("base4_Volume", base_Shape_4, malu5083);

  base->AddNode(base4_Volume, 2, rot_base);

  // 5th piece middle  Framework middle
  auto* middle = new TGeoVolumeAssembly("middle");
  auto* middle_L = new TGeoVolumeAssembly("middle_L");
  auto* middle_R = new TGeoVolumeAssembly("middle_R");

  ////new2020 framework middle
  Double_t radin_fwm = 14.406;
  Double_t radout_fwm = 15.185;
  Double_t high_fwm = 0.6;
  Double_t ang_in_fwm = 180. + 12.93;
  Double_t ang_fin_fwm = 180. + 58.65;

  ////box add up
  Double_t x_fwm_1box = 0.8; // dx=4
  Double_t y_fwm_1box = 1.45;
  Double_t z_fwm_1box = 0.6; // 6.5 -> 6.6 to quit
  TGeoTranslation* tr_fwm_1box =
    new TGeoTranslation("tr_fwm_1box", -14.4, -3.398 + 1.45 / 2, 0);
  tr_fwm_1box->RegisterYourself();

  ////box quit down
  Double_t x_fwm_2box = 0.8; // dx=4
  Double_t y_fwm_2box = 1.2;
  Double_t z_fwm_2box = 0.7; // 6.5 -> 6.6 to quit
  TGeoTranslation* tr_fwm_2box =
    new TGeoTranslation("tr_fwm_2box", -14.4 + 6.9, -3.398 - 9.1, 0);
  tr_fwm_2box->RegisterYourself();

  TGeoXtru* tria_fwm = new TGeoXtru(2);
  tria_fwm->SetName("tria_fwm");

  Double_t x_tria_fwm[3] = {-13.5, -10., -10.};
  Double_t y_tria_fwm[3] = {-5.94, -5.94, -10.8};
  tria_fwm->DefinePolygon(3, x_tria_fwm, y_tria_fwm);
  tria_fwm->DefineSection(0, -0.3, 0., 0., 1);
  tria_fwm->DefineSection(1, 0.3, 0., 0., 1);
  //////////

  // box up to quit and to join
  Double_t x_middle = 0.8;
  Double_t y_middle = 3.495;
  Double_t z_middle = 0.62;
  // tr1 to join with arc
  TGeoTranslation* tr1_middle_box =
    new TGeoTranslation("tr1_middle_box", -14.4, -0.745, 0);
  tr1_middle_box->RegisterYourself();
  // tr2 to quiet
  TGeoTranslation* tr2_middle_box =
    new TGeoTranslation("tr2_middle_box", -15.2, -0.745, 0);
  tr2_middle_box->RegisterYourself();

  // box down_1
  Double_t x_middle_d1box = 0.4; // dx=4
  Double_t y_middle_d1box = 0.28;
  Double_t z_middle_d1box = 0.66;
  TGeoTranslation* tr_middle_d1box =
    new TGeoTranslation("tr_middle_d1box", -7.3, -11.96, 0.);
  tr_middle_d1box->RegisterYourself();

  // box down_2
  Double_t x_middle_d2box = 0.8; // dx=4
  Double_t y_middle_d2box = 1.0;
  Double_t z_middle_d2box = 0.66; //
  TGeoTranslation* tr_middle_d2box =
    new TGeoTranslation("tr_middle_d2box", -7.5, -12.6249, 0);
  tr_middle_d2box->RegisterYourself();

  // arc circ part
  Double_t radin_middle = 14.0;
  Double_t radout_middle = 15.0;
  Double_t high_middle = 0.6;
  Double_t ang_in_middle = 180;
  Double_t ang_fin_middle = 238.21; // alfa=57.60

  // circular hole1 ; hole_middle d=3.5
  Double_t radin_mid_1hole = 0.;
  Double_t radout_mid_1hole = 0.175;
  Double_t high_mid_1hole = 1.5;

  TGeoRotation* rot_mid_1hole = new TGeoRotation("rot_mid_1hole", 90, 90, 0);
  rot_mid_1hole->RegisterYourself();
  TGeoCombiTrans* combi_mid_1tubhole =
    new TGeoCombiTrans(-14.2, 0.325, 0, rot_mid_1hole);
  combi_mid_1tubhole->SetName("combi_mid_1tubhole");
  combi_mid_1tubhole->RegisterYourself();

  // circular hole2 ; hole_middle d=3
  Double_t radin_mid_2hole = 0.;
  Double_t radout_mid_2hole = 0.15;
  Double_t high_mid_2hole = 1.8;

  TGeoCombiTrans* combi_mid_2tubhole =
    new TGeoCombiTrans(-7.7, -12.355, 0, rot_mid_1hole);
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
  TGeoCombiTrans* combi_middle_L =
    new TGeoCombiTrans(0, -7.625, 24.15 + 0.675, rot_90x);
  combi_middle_L->SetName("combi_middle_L");
  combi_middle_L->RegisterYourself();

  TGeoTranslation* tr_middle_L =
    new TGeoTranslation("tr_middle_L", 0, -4.45 - 0.1, 24.85 + 0.675);
  tr_middle_L->RegisterYourself();

  TGeoCombiTrans* combi_middle_R =
    new TGeoCombiTrans(0, -4.45 - 0.1, 24.85 + 0.675, rot_middlez);
  combi_middle_R->SetName("combi_middle_R");
  combi_middle_R->RegisterYourself();

  auto* middle_Shape_3 = new TGeoCompositeShape(
    "middle_Shape_3",
    "  tube_fwm + fwm_1box:tr_fwm_1box - fwm_2box:tr_fwm_2box +tria_fwm");

  auto* middle_Shape_4 = new TGeoCompositeShape(
    "middle_Shape_4",
    " tube_fwm + fwm_1box:tr_fwm_1box - fwm_2box:tr_fwm_2box +tria_fwm");

  auto* middle_Volume_L =
    new TGeoVolume("middle_Volume_L", middle_Shape_3, malu5083);
  auto* middle_Volume_R =
    new TGeoVolume("middle_Volume_R", middle_Shape_4, malu5083);

  TGeoTranslation* tr_middle =
    new TGeoTranslation("tr_middle", 0, -4.45 - 0.1, 24.85 + 0.675);
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
  TGeoTranslation* tr_RL_1box = new TGeoTranslation(0, y_RL_1box / 2, 1.825);
  tr_RL_1box->SetName("tr_RL_1box");
  tr_RL_1box->RegisterYourself();

  TGeoXtru* xtru_RL1 = new TGeoXtru(2);
  xtru_RL1->SetName("S_XTRU_RL1");

  Double_t x_RL1[5] = {-1.5, 1.5, 0.5, 0.5, -1.5};
  Double_t y_RL1[5] = {1.2, 1.2, 2.2, 8.2, 8.2};
  xtru_RL1->DefinePolygon(5, x_RL1, y_RL1);
  xtru_RL1->DefineSection(0, -2.225, 0., 0., 1);
  xtru_RL1->DefineSection(1, 2.225, 0., 0., 1);

  // box knee
  Double_t x_RL_kneebox = 1.5;
  Double_t y_RL_kneebox = 3.5;
  Double_t z_RL_kneebox = 1.5;
  TGeoTranslation* tr_RL_kneebox = new TGeoTranslation(0, 0, 0);
  tr_RL_kneebox->SetName("tr_RL_kneebox");
  tr_RL_kneebox->RegisterYourself();

  TGeoRotation* rot_knee = new TGeoRotation("rot_knee", -40, 0, 0);
  rot_knee->SetName("rot_knee");
  rot_knee->RegisterYourself();
  TGeoCombiTrans* combi_knee =
    new TGeoCombiTrans(0.96, 1.75 + 0.81864, 0, rot_knee);
  combi_knee->SetName("combi_knee");
  combi_knee->RegisterYourself();
  // quit diagona-> qdi
  Double_t x_qdi_box = 3.1;   //
  Double_t y_qdi_box = 7.159; //
  Double_t z_qdi_box = 3.005; //

  TGeoRotation* rot_qdi = new TGeoRotation("rot_qdi", 0, 24.775, 0);
  rot_qdi->RegisterYourself();
  TGeoCombiTrans* combi_qdi = new TGeoCombiTrans(0, 5.579, -2.087, rot_qdi);
  combi_qdi->SetName("combi_qdi");
  combi_qdi->RegisterYourself();
  // knee small

  TGeoXtru* xtru3_RL = new TGeoXtru(2);
  xtru3_RL->SetName("xtru3_RL");

  Double_t x_3RL[6] = {-0.75, 0.75, 0.75, 2.6487, 1.4997, -0.75};
  Double_t y_3RL[6] = {-1.75, -1.75, 1.203, 3.465, 4.4311, 1.75};

  xtru3_RL->DefinePolygon(6, x_3RL, y_3RL);
  xtru3_RL->DefineSection(0, -0.75, 0, 0, 1);
  xtru3_RL->DefineSection(1, 0.76, 0, 0, 1);

  TGeoTranslation* tr_vol3_RL = new TGeoTranslation(-0.25, 12.66, 0);
  tr_vol3_RL->SetName("tr_vol3_RL");
  tr_vol3_RL->RegisterYourself();

  // circular holes for rail R and L
  // circular hole1_RL (a(6,22)); hole_midle d=6.5 H11
  Double_t radin_RL1hole = 0.;
  Double_t radout_RL1hole = 0.325;
  Double_t high_RL1hole = 1.0;

  TGeoRotation* rot_RL1hole = new TGeoRotation("rot_RL1hole", 0, 0, 0);
  rot_RL1hole->RegisterYourself();
  TGeoCombiTrans* combi_RL1hole =
    new TGeoCombiTrans(0.7, 0.6, 1.85, rot_RL1hole);
  combi_RL1hole->SetName("combi_RL1hole");
  combi_RL1hole->RegisterYourself();
  // similar hole for R Join.
  // circular hole_ir. diameter=M3 (3 mm)) prof trou:8, tar:6mm
  Double_t radin_ir_railL = 0.;
  Double_t radout_ir_railL = 0.15;
  Double_t high_ir_railL = 3.9;
  TGeoRotation* rot_ir_RL = new TGeoRotation("rot_ir_RL", 90, 90, 0);
  rot_ir_RL->RegisterYourself();

  TGeoCombiTrans* combi_ir1_RL =
    new TGeoCombiTrans(8.62, 24.75, 1.5, rot_ir_RL);
  combi_ir1_RL->SetName("combi_ir1_RL");
  combi_ir1_RL->RegisterYourself();

  TGeoCombiTrans* combi_ir2_RL = new TGeoCombiTrans(8.6, 33.15, 1.5, rot_ir_RL);
  combi_ir2_RL->SetName("combi_ir2_RL");
  combi_ir2_RL->RegisterYourself();
  //
  TGeoXtru* xtru_RL2 = new TGeoXtru(2);
  xtru_RL2->SetName("S_XTRU_RL2");

  Double_t x_RL2[8] = {-1.5, 0.5, 0.5, 9.3, 9.3, 7.3, 7.3, -1.5};
  Double_t y_RL2[8] = {8.2, 8.2, 13.863, 24.35, 25.65, 25.65, 25.078, 14.591};

  xtru_RL2->DefinePolygon(8, x_RL2, y_RL2);
  xtru_RL2->DefineSection(0, 0.7752, 0, 0, 1);
  xtru_RL2->DefineSection(1, 2.225, 0, 0, 1);

  // 2) modi  adding box  new element 1 box
  TGeoXtru* adi1_RL = new TGeoXtru(2);
  adi1_RL->SetName("S_ADI1_RL");

  Double_t x_adi1RL[4] = {-1.5, -1.5, 0.5, 0.5};
  Double_t y_adi1RL[4] = {2.2, 13.863, 13.863, 2.2};

  adi1_RL->DefinePolygon(4, x_adi1RL, y_adi1RL);
  adi1_RL->DefineSection(0, -0.75, 0, 0, 1);
  adi1_RL->DefineSection(1, 0.775, 0, 0, 1);

  //// 3) modi adding new knee new element 2 new knee
  TGeoXtru* adi2_RL = new TGeoXtru(2); //
  adi2_RL->SetName("S_ADI2_RL");
  Double_t x_adi2RL[6] = {-1.5, 0.5, 9.3, 9.3, 7.8, 7.8};
  Double_t y_adi2RL[6] = {13.863, 13.863, 24.35, 25.65, 25.65, 25.078};

  adi2_RL->DefinePolygon(6, x_adi2RL, y_adi2RL);
  adi2_RL->DefineSection(0, -0.75, 0, 0, 1);
  adi2_RL->DefineSection(1, 0.7755, 0, 0, 1);

  // 4)modi  to quit ---> trap
  Double_t RL_dx1 = 2.66;
  Double_t RL_dx2 = 1;
  Double_t RL_dy = 2.2;
  Double_t RL_dz = 1.5;

  TGeoRotation* rot_RL_Z50 = new TGeoRotation("rot_RL_Z50", 50, 0, 0);
  rot_RL_Z50->RegisterYourself();
  TGeoCombiTrans* combi_RL_trap =
    new TGeoCombiTrans(5, 18.633, -1.5 - 0.025, rot_RL_Z50);
  combi_RL_trap->SetName("combi_RL_trap");
  combi_RL_trap->RegisterYourself();

  /////  5) modi  to quit inferior part  box
  Double_t x_qinf_box = 10.66;
  Double_t y_qinf_box = 10.2;
  Double_t z_qinf_box = 3.;
  auto* s_RL_qinf_box = new TGeoBBox("S_RL_QINF_BOX", x_qinf_box / 2,
                                     y_qinf_box / 2, z_qinf_box / 2);
  TGeoCombiTrans* combi_RL_qbox =
    new TGeoCombiTrans(7, 23., -1.5 - 0.025, rot_RL_Z50);
  combi_RL_qbox->SetName("combi_RL_qbox");
  combi_RL_qbox->RegisterYourself();

  // 6) modi.  add penta face z
  TGeoXtru* pentfa_RL = new TGeoXtru(2);
  pentfa_RL->SetName("S_PENTFA_RL");
  Double_t x_pentfaRL[5] = {-1., -1., 0.13, 1., 1.};
  Double_t y_pentfaRL[5] = {1.125, 0.045, -1.125, -1.125, 1.125};

  pentfa_RL->DefinePolygon(5, x_pentfaRL, y_pentfaRL);
  pentfa_RL->DefineSection(0, -5.05, 0, 0, 1);
  pentfa_RL->DefineSection(1, 5.055, 0, 0, 1);

  TGeoRotation* rot_X90 = new TGeoRotation("rot_X90", 0, 90, 0);
  rot_X90->RegisterYourself();
  TGeoCombiTrans* combi_RL_pent =
    new TGeoCombiTrans(8.3, 30.705, 1.125 - 0.025, rot_X90);
  combi_RL_pent->SetName("combi_RL_pent");
  combi_RL_pent->RegisterYourself();

  // shape for Rail L geom
  new TGeoBBox("RL_1box", x_RL_1box / 2, y_RL_1box / 2, z_RL_1box / 2);
  new TGeoBBox("RL_kneebox", x_RL_kneebox / 2, y_RL_kneebox / 2,
               z_RL_kneebox / 2);
  new TGeoBBox("qdi_box", x_qdi_box / 2, y_qdi_box / 2, z_qdi_box / 2);
  new TGeoTrd1("TRAP1", RL_dx1, RL_dx2, RL_dy, RL_dz);

  // composite shape for rail L

  auto* RL_Shape_0 = new TGeoCompositeShape(
    "RL_Shape_0",
    "  S_XTRU_RL1 + S_XTRU_RL2 + RL_1box:tr_RL_1box - "
    "qdi_box:combi_qdi + "
    "S_ADI1_RL + S_ADI2_RL - TRAP1:combi_RL_trap - "
    "S_RL_QINF_BOX:combi_RL_qbox + "
    "S_PENTFA_RL:combi_RL_pent");

  TGeoVolume* rail_L_vol0 = new TGeoVolume("RAIL_L_VOL0", RL_Shape_0, malu5083);

  rail_L->AddNode(rail_L_vol0, 1, new TGeoTranslation(0., 0., 1.5));

  // piece 7th RAIL RIGHT
  Double_t x_RR_1box = 3.0; // dx=15
  Double_t y_RR_1box = 1.2; // dy=6, -dy=6
  Double_t z_RR_1box = 0.8; // dz=4     to quit
  TGeoTranslation* tr_RR_1box =
    new TGeoTranslation("tr_RR_1box", 0, 0.6, 1.825);
  tr_RR_1box->RegisterYourself();

  TGeoXtru* part_RR1 = new TGeoXtru(2);
  part_RR1->SetName("part_RR1");

  Double_t x_RR1[5] = {-1.5, -0.5, -0.5, 1.5, 1.5};
  Double_t y_RR1[5] = {1.2, 2.2, 8.2, 8.2, 1.2};

  part_RR1->DefinePolygon(5, x_RR1, y_RR1);
  part_RR1->DefineSection(0, -2.225, 0, 0, 1);
  part_RR1->DefineSection(1, 2.225, 0, 0, 1);

  // knee (small)
  TGeoXtru* part_RR3 = new TGeoXtru(2);
  part_RR3->SetName("part_RR3");

  Double_t x_3RR[6] = {1.0, 1.0, -1.2497, -2.2138, -0.5, -0.5};
  Double_t y_3RR[6] = {10.91, 14.41, 17.0911, 15.9421, 13.86, 10.91};

  part_RR3->DefinePolygon(6, x_3RR, y_3RR);
  part_RR3->DefineSection(0, -0.75, 0, 0, 1);
  part_RR3->DefineSection(1, 0.78, 0, 0, 1);

  TGeoTranslation* tr_vol3_RR =
    new TGeoTranslation("tr_vol3_RR", -0.25, 12.66, 0);
  tr_vol3_RR->RegisterYourself();

  //  quit diagona-> qdi
  Double_t x_qdi_Rbox = 3.1;
  Double_t y_qdi_Rbox = 7.159;
  Double_t z_qdi_Rbox = 3.005;

  TGeoRotation* rot_Rqdi = new TGeoRotation("rot_Rqdi", 0, 24.775, 0);
  rot_Rqdi->RegisterYourself();
  TGeoCombiTrans* combi_Rqdi = new TGeoCombiTrans(0, 5.579, -2.087, rot_Rqdi);
  combi_Rqdi->SetName("combi_Rqdi");
  combi_Rqdi->RegisterYourself();

  // holes   circular hole_a. diameter=6.5 (a(6,22)); hole_midle d=6.5 H11
  Double_t radin_a_rail = 0.;
  Double_t radout_a_rail = 0.325;
  Double_t high_a_rail = 0.82;

  TGeoTranslation* tr_a_RR = new TGeoTranslation("tr_a_RR", -0.7, 0.6, 1.825);
  tr_a_RR->RegisterYourself();

  Double_t radin_ir_rail = 0.;
  Double_t radout_ir_rail = 0.15;
  Double_t high_ir_rail = 3.2;
  TGeoRotation* rot_ir_RR = new TGeoRotation("rot_ir_RR", 90, 90, 0);
  rot_ir_RR->RegisterYourself();

  TGeoCombiTrans* combi_ir_RR =
    new TGeoCombiTrans(-8.62, 24.75, 1.5, rot_ir_RR);
  combi_ir_RR->SetName("combi_ir_RR");
  combi_ir_RR->RegisterYourself();

  TGeoCombiTrans* combi_ir2_RR =
    new TGeoCombiTrans(-8.6, 33.15, 1.5, rot_ir_RR);
  combi_ir2_RR->SetName("combi_ir2_RR");
  combi_ir2_RR->RegisterYourself();

  TGeoCombiTrans* combi_rail_R = new TGeoCombiTrans(24.1, -1.825, 0, rot_90x);
  combi_rail_R->SetName("combi_rail_R");
  combi_rail_R->RegisterYourself();
  TGeoCombiTrans* combi_rail_L = new TGeoCombiTrans(-24.1, -1.825, 0, rot_90x);
  combi_rail_L->SetName("combi_rail_L");
  combi_rail_L->RegisterYourself();

  // trasl L and R
  TGeoTranslation* tr_sr_l = new TGeoTranslation("tr_sr_l", -15.01, 0, 0);
  tr_sr_l->RegisterYourself();
  TGeoTranslation* tr_sr_r = new TGeoTranslation("tr_sr_r", 15.01, 0, 0);
  tr_sr_r->RegisterYourself();

  //////// new modfi b ///////  cut arm
  TGeoXtru* part_RR2 = new TGeoXtru(2);
  part_RR2->SetName("part_RR2");

  Double_t x_RR2[8] = {-0.5, -0.5, -9.3, -9.3, -7.3, -7.3, 1.5, 1.5};
  Double_t y_RR2[8] = {8.2, 13.863, 24.35, 25.65, 25.65, 25.078, 14.591, 8.2};

  part_RR2->DefinePolygon(8, x_RR2, y_RR2);
  part_RR2->DefineSection(0, 0.776, 0, 0, 1);
  part_RR2->DefineSection(1, 2.225, 0, 0, 1);

  // 2b) modi  adding box  new element 1 box
  TGeoXtru* adi1_RR = new TGeoXtru(2);
  adi1_RR->SetName("S_ADI1_RR");

  Double_t x_adi1RR[4] = {-0.5, -.5, 1.5, 1.5};
  Double_t y_adi1RR[4] = {2.2, 13.863, 13.863, 2.2};

  adi1_RR->DefinePolygon(4, x_adi1RR, y_adi1RR);
  adi1_RR->DefineSection(0, -0.75, 0, 0, 1);
  adi1_RR->DefineSection(1, 0.775, 0, 0, 1);

  // 3b) modi adding new knee new element
  TGeoXtru* adi2_RR = new TGeoXtru(2); //
  adi2_RR->SetName("S_ADI2_RR");
  Double_t x_adi2RR[6] = {1.5, -0.5, -9.3, -9.3, -7.8, -7.8};
  Double_t y_adi2RR[6] = {13.863, 13.863, 24.35, 25.65, 25.65, 25.078};

  adi2_RR->DefinePolygon(6, x_adi2RR, y_adi2RR);
  adi2_RR->DefineSection(0, -0.75, 0, 0, 1);
  adi2_RR->DefineSection(1, 0.7755, 0, 0, 1);

  //  4)modi  to quit ---> trap
  TGeoRotation* rot_RR_Z310 = new TGeoRotation("rot_RR_Z310", -50, 0, 0);
  rot_RR_Z310->RegisterYourself();
  TGeoCombiTrans* combi_RR_trap =
    new TGeoCombiTrans(-5, 18.633, -1.5 - 0.025, rot_RR_Z310);
  combi_RR_trap->SetName("combi_RR_trap");
  combi_RR_trap->RegisterYourself();

  //  5) modi  to quit     inferior part  box
  TGeoCombiTrans* combi_RR_qbox =
    new TGeoCombiTrans(-7, 23., -1.5 - 0.025, rot_RR_Z310);
  combi_RR_qbox->SetName("combi_RR_qbox");
  combi_RR_qbox->RegisterYourself();

  // 6) modi.  add penta face z
  TGeoXtru* pentfa_RR = new TGeoXtru(2);
  pentfa_RR->SetName("S_PENTFA_RR");
  Double_t x_pentfaRR[5] = {1., 1., -0.13, -1., -1.};
  Double_t y_pentfaRR[5] = {1.125, 0.045, -1.125, -1.125, 1.125};

  pentfa_RR->DefinePolygon(5, x_pentfaRR, y_pentfaRR);
  pentfa_RR->DefineSection(0, -5.05, 0, 0, 1);
  pentfa_RR->DefineSection(1, 5.055, 0, 0, 1);

  TGeoCombiTrans* combi_RR_pent =
    new TGeoCombiTrans(-8.3, 30.705, 1.125 - 0.025, rot_X90);
  combi_RR_pent->SetName("combi_RR_pent");
  combi_RR_pent->RegisterYourself();

  // shape for rail R
  new TGeoBBox("RR_1box", x_RR_1box / 2, y_RR_1box / 2, z_RR_1box / 2);

  // composite shape for rail R
  new TGeoCompositeShape(
    "RR_Shape_0",
    "RR_1box:tr_RR_1box + part_RR1 + part_RR2  - qdi_box:combi_qdi + "
    "S_ADI1_RR + S_ADI2_RR  -  TRAP1:combi_RR_trap - "
    "S_RL_QINF_BOX:combi_RR_qbox +S_PENTFA_RR:combi_RR_pent ");

  // JOIN only for show L and R parts
  auto* rail_L_R_Shape = new TGeoCompositeShape(
    "RAIL_L_R_Shape", "  RL_Shape_0:combi_rail_L + RR_Shape_0:combi_rail_R");

  TGeoVolume* rail_L_R_vol0 =
    new TGeoVolume("RAIL_L_R_VOL0", rail_L_R_Shape, malu5083);

  TGeoRotation* rot_rLR = new TGeoRotation("rot_rLR", 180, 180, 0);
  rot_rLR->RegisterYourself();
  TGeoCombiTrans* combi_rLR = new TGeoCombiTrans(0, -6.9, -0.5, rot_rLR);
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
  sup_MB_L->DefineSection(0, -0.4, 0, 0, 1);
  sup_MB_L->DefineSection(1, 0.4, 0, 0, 1);

  TGeoXtru* part_MBL_0 = new TGeoXtru(2);
  part_MBL_0->SetName("part_MBL_0");

  Double_t x[8] = {0., 0, 6.1, 31.55, 34.55, 34.55, 31.946, 6.496};
  Double_t y[8] = {-0.4, 0.4, 0.4, 13.0, 13.0, 12.2, 12.2, -0.4};

  part_MBL_0->DefinePolygon(8, x, y);
  part_MBL_0->DefineSection(0, -0.4, 0, 0, 1);
  part_MBL_0->DefineSection(1, 0.4, 0, 0, 1);

  TGeoRotation* rot1_MBL_0 = new TGeoRotation("rot1_MBL_0", -90, -90, 90);
  rot1_MBL_0->RegisterYourself();

  // quit box in diag
  Double_t x_mb_box = 0.8;
  Double_t y_mb_box = 0.8;
  Double_t z_mb_box = 0.81;
  TGeoTranslation* tr_mb_box = new TGeoTranslation("tr_mb_box", 24.05, 9.55, 0);
  tr_mb_box->RegisterYourself();

  // lateral hole-box
  Double_t x_lat_box = 0.7;
  Double_t y_lat_box = 1.8;
  Double_t z_lat_box = 0.2;
  TGeoTranslation* tr_lat1L_box =
    new TGeoTranslation("tr_lat1L_box", 4.6, 0, 0.4);
  tr_lat1L_box->RegisterYourself();
  TGeoTranslation* tr_lat2L_box =
    new TGeoTranslation("tr_lat2L_box", 9.6, 1.65, 0.4);
  tr_lat2L_box->RegisterYourself();

  TGeoTranslation* tr_lat3L_box =
    new TGeoTranslation("tr_lat3L_box", 17.35, 5.923, 0.4);
  tr_lat3L_box->RegisterYourself();
  TGeoTranslation* tr_lat4L_box =
    new TGeoTranslation("tr_lat4L_box", 26.45, 10, 0.4);
  tr_lat4L_box->RegisterYourself();
  TGeoTranslation* tr_lat5L_box =
    new TGeoTranslation("tr_lat5L_box", 29.9, 11.6, 0.4);
  tr_lat5L_box->RegisterYourself();

  TGeoTranslation* tr_lat1R_box =
    new TGeoTranslation("tr_lat1R_box", 4.6, 0, -0.4);
  tr_lat1R_box->RegisterYourself();
  TGeoTranslation* tr_lat2R_box =
    new TGeoTranslation("tr_lat2R_box", 9.6, 1.65, -0.4);
  tr_lat2R_box->RegisterYourself();

  TGeoTranslation* tr_lat3R_box =
    new TGeoTranslation("tr_lat3R_box", 17.35, 5.923, -0.4);
  tr_lat3R_box->RegisterYourself();
  TGeoTranslation* tr_lat4R_box =
    new TGeoTranslation("tr_lat4R_box", 26.45, 10, -0.4);
  tr_lat4R_box->RegisterYourself();
  TGeoTranslation* tr_lat5R_box =
    new TGeoTranslation("tr_lat5R_box", 29.9, 11.6, -0.4);
  tr_lat5R_box->RegisterYourself();

  // circular hole_1mbl. diameter=3.5 H9
  Double_t radin_1mb = 0.;
  Double_t radout_1mb = 0.175;
  Double_t high_1mb = 2.825;
  TGeoTranslation* tr1_mb = new TGeoTranslation("tr1_mb", 18.48, 6.1, 0.);
  tr1_mb->RegisterYourself();

  TGeoTranslation* tr2_mb = new TGeoTranslation("tr2_mb", 24.15, 8.9, 0.);
  tr2_mb->RegisterYourself();

  // circular hole_2mbl inclined and hole-up.diameter=M3 (3 mm)) prof , tar:8mm
  Double_t radin_2mb = 0.;
  Double_t radout_2mb = 0.15;
  Double_t high_2mb = 0.82;

  TGeoRotation* rot_hole2_MBL = new TGeoRotation("rot_hole2_MBL", 0, 90, 0);
  rot_hole2_MBL->RegisterYourself();

  TGeoTranslation* tr_mbl = new TGeoTranslation("tr_mbl", -7.5, 0., 0.);
  tr_mbl->RegisterYourself();

  TGeoTranslation* tr_mbr = new TGeoTranslation("tr_mbr", 7.5, 0, 0);
  tr_mbr->RegisterYourself();

  // hole up || hup
  TGeoCombiTrans* combi_hup_mb = new TGeoCombiTrans(32.5, 12.6, 0, rot_90x);
  combi_hup_mb->SetName("combi_hup_mb");
  combi_hup_mb->RegisterYourself();

  // shape for rail MB
  new TGeoBBox("mb_box", x_mb_box / 2, y_mb_box / 2, z_mb_box / 2);
  new TGeoTube("hole_1mbl", radin_1mb, radout_1mb, high_1mb / 2);
  new TGeoTube("hole_2mbl", radin_2mb, radout_2mb, high_2mb / 2);
  new TGeoBBox("lat_box", x_lat_box / 2, y_lat_box / 2, z_lat_box / 2);

  // composite shape for rail_MB R + L
  auto* MB_Shape_0 = new TGeoCompositeShape(
    "MB_Shape_0", " sup_MB_L  - hole_2mbl:combi_hup_mb ");
  auto* MB_Shape_0L = new TGeoCompositeShape(
    "MB_Shape_0L", "MB_Shape_0  - lat_box:tr_lat3L_box ");
  auto* MB_Shape_0R = new TGeoCompositeShape(
    "MB_Shape_0R", "MB_Shape_0 - lat_box:tr_lat3R_box ");

  new TGeoCompositeShape("MB_Shape_1L", "MB_Shape_0L:rot1_MBL_0 - hole_2mbl");
  // left and right
  new TGeoCompositeShape("MB_Shape_1R", "MB_Shape_0R:rot1_MBL_0 - hole_2mbl");

  auto* MB_Shape_2 = new TGeoCompositeShape(
    "MB_Shape_2", " MB_Shape_1L:tr_mbl +  MB_Shape_1R:tr_mbr ");

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
                 new TGeoTranslation(0, 5.423 - 28.8, 17.35 + 0.675));
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
  Double_t tyMB0_3;
  Double_t tzMB0;

  if (half == 0) {
    t_final_x = 0;
    t_final_y = 0.0;
    t_final_z = -80 - 0.675 - 0.15;

    r_final_x = 0;
    r_final_y = 0;
    r_final_z = 0;

    tyMB0 = -16.72;
    tyMB0_3 = -18.0;
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
    tyMB0_3 = 18.0;
    tzMB0 = -(45.3 + 46.7) / 2;
  }

  auto* t_final =
    new TGeoTranslation("t_final", t_final_x, t_final_y, t_final_z);
  auto* r_final = new TGeoRotation("r_final", r_final_x, r_final_y, r_final_z);
  auto* c_final = new TGeoCombiTrans(*t_final, *r_final);

  // 9th new 2020 ELEMENT middle framework back -----------------------------
  auto* frame_back = new TGeoVolumeAssembly("frame_back");

  // rectangular box1 to quit
  Double_t x_box_fwb = 15.8;
  Double_t y_box_fwb = 5;
  Double_t z_box_fwb = 1;

  // rectangular box2 to add
  Double_t x_box2_fwb = 1.9;
  Double_t y_box2_fwb = 0.5;
  Double_t z_box2_fwb = 0.6;

  ///// holes tub  1hole tranversal
  Double_t radin_fwb = 25.75;
  Double_t radout_fwb = 26.75;
  Double_t high_fwb = 0.6;

  // seg tub_back
  Double_t radin_stub = 23.6;
  Double_t radout_stub = 24.4;
  Double_t high_stub = 0.6;
  Double_t ang_in_stub = 288.9;
  Double_t ang_fin_stub = 342.;

  TGeoRotation* rot_1hole_fwb = new TGeoRotation("rot_1hole_fwb", 0, 90, 0);
  rot_1hole_fwb->RegisterYourself();

  TGeoCombiTrans* acombi_fwb = new TGeoCombiTrans(5.2, 0, 0, rot_1hole_fwb);
  acombi_fwb->SetName("acombi_1h_fwb");
  acombi_fwb->RegisterYourself();

  TGeoTranslation* tr_box_y24 = new TGeoTranslation("tr_box_y24", 0, -24., 0.);
  tr_box_y24->RegisterYourself();

  TGeoTranslation* tr_box2_fwb =
    new TGeoTranslation("tr_box2_fwb", 24.4 - 1.9 / 2, -7.121 - 0.5 / 2, 0.);
  tr_box2_fwb->RegisterYourself();

  TGeoRotation* rot_Z180_X180 = new TGeoRotation("rot_Z180_X180", 180, 180, 0);
  rot_Z180_X180->RegisterYourself();

  TGeoTranslation* tr_fb =
    new TGeoTranslation("tr_fb", 0, -2.3 - 0.06, 13.85 + 0.675);
  tr_fb->RegisterYourself();

  auto* q_box_fwb =
    new TGeoBBox("q_box_fwb", x_box_fwb / 2, y_box_fwb / 2, z_box_fwb / 2);
  auto* box2_fwb =
    new TGeoBBox("box2_fwb", x_box2_fwb / 2, y_box2_fwb / 2, z_box2_fwb / 2);
  auto* s_tub_fwb =
    new TGeoTube("s_tub_fwb", radin_fwb, radout_fwb, high_fwb / 2);

  auto* s_stub_fwb = new TGeoTubeSeg("s_stub_fwb", radin_stub, radout_stub,
                                     high_stub / 2, ang_in_stub, ang_fin_stub);

  /// composite shape for mb0

  auto* fwb_Shape_0 = new TGeoCompositeShape(
    "fwb_Shape_0",
    "  s_stub_fwb - q_box_fwb:tr_box_y24 + box2_fwb:tr_box2_fwb ");
  auto* fwb_Shape_1 = new TGeoCompositeShape(
    "fwb_Shape_1", "fwb_Shape_0 + fwb_Shape_0:rot_Z180_X180");

  auto* fwb_Volume = new TGeoVolume("fwb_Volume", fwb_Shape_1, malu5083);
  frame_back->AddNode(fwb_Volume, 1, tr_fb);

  /// 10 th colonne_support_MB012   new 2020
  auto* colonne_mb = new TGeoVolumeAssembly("colonne_mb");

  // rectangular box
  Double_t x_box_cmb = 1.9;
  Double_t y_box_cmb = 0.6;
  Double_t z_box_cmb = 2.2033;

  ///// holes tub  1hole tranversal
  Double_t radin_c_mb = 0.;
  Double_t radout_c_mb = 0.3;
  Double_t high_c_mb = 2.2033;

  TGeoRotation* rot_1c_mb0 = new TGeoRotation("rot_1c_mb0", 0, 90, 0);
  rot_1c_mb0->RegisterYourself();

  TGeoCombiTrans* acombi_1c_mb0 = new TGeoCombiTrans(0.95, 0, 0, rot_1c_mb0);
  acombi_1c_mb0->SetName("acombi_1c_mb0");
  acombi_1c_mb0->RegisterYourself();
  TGeoCombiTrans* bcombi_1c_mb0 = new TGeoCombiTrans(-0.95, 0, 0, rot_1c_mb0);
  bcombi_1c_mb0->SetName("bcombi_1c_mb0");
  bcombi_1c_mb0->RegisterYourself();

  // box to cut
  Double_t x_boxq_cmb = 3.;
  Double_t y_boxq_cmb = 1.05;
  Double_t z_boxq_cmb = 4.;

  TGeoRotation* rot_X19 = new TGeoRotation("rot_X19", 0, -19, 0);
  rot_X19->RegisterYourself();
  TGeoCombiTrans* combi_qbox =
    new TGeoCombiTrans(0, +2.1 / 2 + 0.5, 0, rot_X19);
  combi_qbox->SetName("combi_qbox");
  combi_qbox->RegisterYourself();

  // shapes
  auto* box_cmb =
    new TGeoBBox("box_cmb", x_box_cmb / 2, y_box_cmb / 2, z_box_cmb / 2);
  auto* tub_cmb =
    new TGeoTube("tub_cmb", radin_c_mb, radout_c_mb, high_c_mb / 2);
  auto* boxq_cmb =
    new TGeoBBox("boxq_cmb", x_boxq_cmb / 2, y_boxq_cmb / 2, z_boxq_cmb / 2);

  /// composite shape for mb0
  auto* c_mb_Shape_0 =
    new TGeoCompositeShape(
      "box_cmb:rot_1c_mb0 + tub_cmb:acombi_1c_mb0 + "
      "tub_cmb:bcombi_1c_mb0 - boxq_cmb:combi_qbox");

  TGeoTranslation* tr_cmb = new TGeoTranslation(
    "tr_cmb", 0, 5.923 - 28.8 + 2.2033 / 2 - 0.2, 17.35 + 0.675);
  tr_cmb->RegisterYourself();
  ///////////////////

  auto* colonne_mb_Volume =
    new TGeoVolume("colonne_mb_Volume", c_mb_Shape_0, malu5083);
  colonne_mb->AddNode(colonne_mb_Volume, 1, tr_cmb);

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
    new TGeoVolume("Half_3_Volume", Half_3_Shape_4, malu5083);

  TGeoRotation* rot_z180 = new TGeoRotation("rot_z180", 0, 180, 0);
  rot_z180->RegisterYourself();

  TGeoCombiTrans* combi_coat = new TGeoCombiTrans(0, 0, 19.5 - 0.45, rot_z180);
  combi_coat->SetName("combi_coat");
  combi_coat->RegisterYourself();

  Half_3->AddNode(Half_3_Volume, 1, combi_coat);

  HalfConeVolume->AddNode(stair, 1, c_final);
  HalfConeVolume->AddNode(base, 2, c_final);
  HalfConeVolume->AddNode(rail_L_R, 3, c_final); // R&L
  HalfConeVolume->AddNode(Fra_front, 4, c_final);
  HalfConeVolume->AddNode(middle, 5, c_final);
  HalfConeVolume->AddNode(frame_back, 6, c_final);
  HalfConeVolume->AddNode(colonne_mb, 7, c_final);

  // ======================== Mother Boards and Services =======================
  Int_t signe;
  if (half == 0) {
    signe = -1;
  }
  if (half == 1) {
    signe = +1;
  }
  auto& mftBaseParam = MFTBaseParam::Instance();
  if (mftBaseParam.buildServices) {
    makeMotherBoards(HalfConeVolume, half, signe, tyMB0, tyMB0_3, tzMB0);
    makeAirVentilation(HalfConeVolume, half, signe);
    makeFlexCables(HalfConeVolume, half, signe);
    makePowerCables(HalfConeVolume, half, signe);
    if (!mftBaseParam.minimal && mftBaseParam.buildCone && mftBaseParam.buildReadoutCables) {
      makeReadoutCables(HalfConeVolume, half, signe);
    }
  }
  // ===========================================================================

  return HalfConeVolume;
}

void HalfCone::makeAirVentilation(TGeoVolumeAssembly* HalfConeVolume, Int_t half, Int_t signe)
{
  TGeoMedium* mVentilation = gGeoManager->GetMedium("MFT_Polypropylene$");
  // Bottom
  TGeoSubtraction* vent_subB1;
  Float_t lB1 = 5.5; // half length
  Float_t xB1 = 0.3;
  Float_t yB1 = 0.4;
  auto* ventB1 = new TGeoBBox(Form("ventB1_H%d", half), xB1, yB1, lB1);
  auto* ventB1_int =
    new TGeoBBox(Form("ventB1_int_H%d", half), 0.2, 0.3, lB1 + 0.0001);
  vent_subB1 = new TGeoSubtraction(ventB1, ventB1_int, nullptr, nullptr);
  TGeoCompositeShape* vent_finalB1 =
    new TGeoCompositeShape(Form("vent_finalB1_H%d", half), vent_subB1);
  auto* vent_B1 =
    new TGeoVolume(Form("ventB1_H%d", half), vent_finalB1, mVentilation);
  vent_B1->SetLineColor(kGray);
  auto* t_airB1 = new TGeoTranslation("t_airB1", signe * (15.3 - xB1),
                                      -8.75 - yB1 - 0.1, -45.570 - lB1);
  t_airB1->RegisterYourself();
  auto* r_airB1 = new TGeoRotation("r_airB1", 0.0, 0.0, 0.0);
  r_airB1->RegisterYourself();
  auto* p_airB1 = new TGeoCombiTrans(*t_airB1, *r_airB1);
  p_airB1->RegisterYourself();
  HalfConeVolume->AddNode(vent_B1, 1, p_airB1);

  TGeoSubtraction* vent_subB2;
  Float_t lB2 = 10.6; // half length
  auto* ventB2 = new TGeoBBox(Form("ventB2_H%d", half), yB1, xB1, lB2);
  auto* ventB2_int =
    new TGeoBBox(Form("ventB2_int_H%d", half), 0.3, 0.2, lB2 + 0.0001);
  vent_subB2 = new TGeoSubtraction(ventB2, ventB2_int, nullptr, nullptr);
  TGeoCompositeShape* vent_finalB2 =
    new TGeoCompositeShape(Form("vent_finalB2_H%d", half), vent_subB2);
  auto* vent_B2 =
    new TGeoVolume(Form("ventB2_H%d", half), vent_finalB2, mVentilation);
  vent_B2->SetLineColor(kGray);
  Float_t theta = -signe * 32.;
  Float_t phi = signe * 23.;
  Float_t thetaRad = theta * TMath::Pi() / 180.;
  Float_t phiRad = phi * TMath::Pi() / 180.;
  auto* r_airB2 = new TGeoRotation("r_airB2", 90.0 - phi, theta, 0.);
  r_airB2->RegisterYourself();
  Float_t XairB2 =
    signe *
    (15.3 + lB2 * TMath::Sin(TMath::Abs(thetaRad) * TMath::Cos(phiRad)));
  Float_t YairB2 =
    -8.75 - 2 * yB1 + TMath::Sin(phiRad) * TMath::Sin(thetaRad) * lB2 + 0.2;
  Float_t ZairB2 = -45.570 - 2 * lB1 - TMath::Cos(thetaRad) * lB2 - 0.2;
  auto* t_airB2 = new TGeoTranslation("t_airB2", XairB2, YairB2, ZairB2);
  t_airB2->RegisterYourself();
  auto* p_airB2 = new TGeoCombiTrans(*t_airB2, *r_airB2);
  p_airB2->RegisterYourself();
  HalfConeVolume->AddNode(vent_B2, 1, p_airB2);

  TGeoSubtraction* vent_subB3;
  Float_t lB3 = 4.8; // half length
  auto* ventB3 = new TGeoBBox(Form("ventB3_H%d", half), yB1, xB1, lB3);
  auto* ventB3_int =
    new TGeoBBox(Form("ventB3_int_H%d", half), 0.3, 0.2, lB3 + 0.0001);
  vent_subB3 = new TGeoSubtraction(ventB3, ventB3_int, nullptr, nullptr);
  TGeoCompositeShape* vent_finalB3 =
    new TGeoCompositeShape(Form("vent_finalB3_H%d", half), vent_subB3);
  auto* vent_B3 =
    new TGeoVolume(Form("ventB3_H%d", half), vent_finalB3, mVentilation);
  vent_B3->SetLineColor(kGray);
  auto* r_airB3 = new TGeoRotation("r_airB3", 90.0 - phi, theta, 0.);
  r_airB3->RegisterYourself();
  Float_t XairB3 =
    signe *
    (15.3 +
     (2 * lB2 - lB3) * TMath::Sin(TMath::Abs(thetaRad) * TMath::Cos(phiRad)) -
     xB1);
  Float_t YairB3 = -8.75 - 2 * yB1 +
                   TMath::Sin(phiRad) * TMath::Sin(thetaRad) * (2 * lB2 - lB3) +
                   0.2 - 1.9 * yB1;
  Float_t ZairB3 =
    -45.570 - 2 * lB1 - TMath::Cos(thetaRad) * (2 * lB2 - lB3) - 0.2;

  auto* t_airB3 = new TGeoTranslation("t_airB3", XairB3, YairB3, ZairB3);

  t_airB3->RegisterYourself();
  auto* p_airB3 = new TGeoCombiTrans(*t_airB3, *r_airB3);
  p_airB3->RegisterYourself();
  HalfConeVolume->AddNode(vent_B3, 1, p_airB3);

  TGeoSubtraction* vent_subB4;
  Float_t lB4 = 4.5; // half length
  Float_t xB4 = 0.3;
  Float_t yB4 = 0.8;
  auto* ventB4 = new TGeoBBox(Form("ventB4_H%d", half), yB4, xB4, lB4);
  auto* ventB4_int =
    new TGeoBBox(Form("ventB4_int_H%d", half), 0.7, 0.2, lB4 + 0.0001);
  vent_subB4 = new TGeoSubtraction(ventB4, ventB4_int, nullptr, nullptr);
  TGeoCompositeShape* vent_finalB4 =
    new TGeoCompositeShape(Form("vent_finalB4_H%d", half), vent_subB4);
  auto* vent_B4 =
    new TGeoVolume(Form("ventB3_H%d", half), vent_finalB4, mVentilation);
  vent_B4->SetLineColor(kGray);
  auto* r_airB4 =
    new TGeoRotation("r_airB4", 90.0 - signe * 25., -signe * 5, 0.);
  r_airB4->RegisterYourself();
  auto* t_airB4 = new TGeoTranslation(
    "t_airB4",
    XairB2 +
      signe * (lB2 * TMath::Sin(TMath::Abs(thetaRad) * TMath::Cos(phiRad)) +
               0.4),
    YairB2 + TMath::Sin(phiRad) * TMath::Sin(thetaRad) * lB2 - 0.6,
    ZairB3 - TMath::Cos(thetaRad) * lB2 * 0.965);
  t_airB4->RegisterYourself();
  auto* p_airB4 = new TGeoCombiTrans(*t_airB4, *r_airB4);
  p_airB4->RegisterYourself();
  HalfConeVolume->AddNode(vent_B4, 1, p_airB4);

  // Top
  TGeoSubtraction* vent_subT1;
  auto* ventT1 = new TGeoBBox(Form("ventT1_H%d", half), xB1, yB1, lB1);
  auto* ventT1_int =
    new TGeoBBox(Form("ventT1_int_H%d", half), 0.2, 0.3, lB1 + 0.0001);
  vent_subT1 = new TGeoSubtraction(ventB1, ventB1_int, nullptr, nullptr);
  TGeoCompositeShape* vent_finalT1 =
    new TGeoCompositeShape(Form("vent_finalT1_H%d", half), vent_subT1);
  auto* vent_T1 =
    new TGeoVolume(Form("ventT1_H%d", half), vent_finalT1, mVentilation);
  vent_T1->SetLineColor(kGray);
  auto* t_airT1 = new TGeoTranslation("t_airT1", signe * (15.3 - xB1),
                                      -(-8.75 - yB1 - 0.1), -45.570 - lB1);
  t_airB1->RegisterYourself();
  auto* r_airT1 = new TGeoRotation("r_airT1", 0.0, 0.0, 0.0);
  r_airT1->RegisterYourself();
  auto* p_airT1 = new TGeoCombiTrans(*t_airT1, *r_airT1);
  p_airT1->RegisterYourself();
  HalfConeVolume->AddNode(vent_T1, 1, p_airT1);

  TGeoSubtraction* vent_subT2;
  auto* ventT2 = new TGeoBBox(Form("ventT2_H%d", half), yB1, xB1, lB2);
  auto* ventT2_int =
    new TGeoBBox(Form("ventT2_int_H%d", half), 0.3, 0.2, lB2 + 0.0001);
  vent_subT2 = new TGeoSubtraction(ventT2, ventT2_int, nullptr, nullptr);
  TGeoCompositeShape* vent_finalT2 =
    new TGeoCompositeShape(Form("vent_finalT2_H%d", half), vent_subT2);
  auto* vent_T2 =
    new TGeoVolume(Form("ventT2_H%d", half), vent_finalT2, mVentilation);
  vent_T2->SetLineColor(kGray);
  theta = -signe * 32.;
  phi = signe * 23.;
  thetaRad = theta * TMath::Pi() / 180.;
  phiRad = phi * TMath::Pi() / 180.;
  auto* r_airT2 = new TGeoRotation("r_airT2", 90.0 - phi, -theta, 0.);
  r_airT2->RegisterYourself();
  auto* t_airT2 = new TGeoTranslation("t_airT2", -XairB2, -YairB2, ZairB2);
  t_airT2->RegisterYourself();
  auto* p_airT2 = new TGeoCombiTrans(*t_airT2, *r_airT2);
  p_airT2->RegisterYourself();
  HalfConeVolume->AddNode(vent_T2, 1, p_airT2);

  TGeoSubtraction* vent_subT3;
  auto* ventT3 = new TGeoBBox(Form("ventT3_H%d", half), yB1, xB1, lB3);
  auto* ventT3_int =
    new TGeoBBox(Form("ventT3_int_H%d", half), 0.3, 0.2, lB3 + 0.0001);
  vent_subT3 = new TGeoSubtraction(ventT3, ventT3_int, nullptr, nullptr);
  TGeoCompositeShape* vent_finalT3 =
    new TGeoCompositeShape(Form("vent_finalT3_H%d", half), vent_subT3);
  auto* vent_T3 =
    new TGeoVolume(Form("ventT3_H%d", half), vent_finalT3, mVentilation);
  vent_T3->SetLineColor(kGray);
  auto* r_airT3 = new TGeoRotation("r_airT3", 90.0 - phi, -theta, 0.);
  r_airT3->RegisterYourself();
  auto* t_airT3 = new TGeoTranslation("t_airT3", -XairB3, -YairB3, ZairB3);
  t_airT3->RegisterYourself();
  auto* p_airT3 = new TGeoCombiTrans(*t_airT3, *r_airT3);
  p_airT3->RegisterYourself();
  HalfConeVolume->AddNode(vent_T3, 1, p_airT3);

  TGeoSubtraction* vent_subT4;
  auto* ventT4 = new TGeoBBox(Form("ventT4_H%d", half), yB4, xB4, lB4);
  auto* ventT4_int =
    new TGeoBBox(Form("ventT4_int_H%d", half), 0.7, 0.2, lB4 + 0.0001);
  vent_subT4 = new TGeoSubtraction(ventT4, ventT4_int, nullptr, nullptr);
  TGeoCompositeShape* vent_finalT4 =
    new TGeoCompositeShape(Form("vent_finalT4_H%d", half), vent_subT4);
  auto* vent_T4 =
    new TGeoVolume(Form("ventT4_H%d", half), vent_finalT4, mVentilation);
  vent_T4->SetLineColor(kGray);
  auto* r_airT4 =
    new TGeoRotation("r_airT4", 90.0 - signe * 25., signe * 5, 0.);
  r_airT4->RegisterYourself();
  auto* t_airT4 = new TGeoTranslation(
    "t_airT4",
    -(XairB2 +
      signe * (lB2 * TMath::Sin(TMath::Abs(thetaRad) * TMath::Cos(phiRad)) +
               0.4)),
    -(YairB2 + TMath::Sin(phiRad) * TMath::Sin(thetaRad) * lB2 - 0.6),
    ZairB3 - TMath::Cos(thetaRad) * lB2 * 0.965);
  t_airT4->RegisterYourself();
  auto* p_airT4 = new TGeoCombiTrans(*t_airT4, *r_airT4);
  p_airT4->RegisterYourself();
  HalfConeVolume->AddNode(vent_T4, 1, p_airT4);
}

void HalfCone::makeMotherBoards(TGeoVolumeAssembly* HalfConeVolume, Int_t half, Int_t signe, Double_t tyMB0, Double_t tyMB0_3, Double_t tzMB0)
{
  // =============  MotherBoard 0 and 1
  Double_t mMB0cu[3];
  Double_t mMB0fr4;
  Double_t mMB0pol;
  Double_t mMB0epo;
  // Sizes
  mMB0cu[0] = {14.00};   // 13.65 old
  mMB0cu[1] = {0.00615}; // 122.5 microns * taux d'occupation 50% = 61.5 microns
  mMB0cu[2] = {2.45};    // 2.39 old
  mMB0fr4 = 0.1;         // 1 m
  mMB0pol = 0.0150;      // 150 microns
  mMB0epo = 0.0225;      // 225 microns

  // Materials
  auto* mCu = gGeoManager->GetMedium("MFT_Cu$");
  auto* mFR4 = gGeoManager->GetMedium("MFT_FR4$");
  auto* mPol = gGeoManager->GetMedium("MFT_Polyimide$");
  auto* mEpo = gGeoManager->GetMedium("MFT_Epoxy$");
  auto* mInox = gGeoManager->GetMedium("MFT_Inox$");
  auto* mPolyu = gGeoManager->GetMedium("MFT_Polyurethane$");

  // Mother boards connected to the first three disk
  auto* MotherBoard0 = new TGeoVolumeAssembly(Form("MotherBoard0_H%d", half));
  // 4 layers
  TGeoVolume* vMB0cu = gGeoManager->MakeBox("vMB0cu", mCu, mMB0cu[0] / 2, mMB0cu[1] / 2, mMB0cu[2] / 2);
  TGeoVolume* vMB0fr4 = gGeoManager->MakeBox("vMB0fr4", mFR4, mMB0cu[0] / 2, mMB0fr4 / 2, mMB0cu[2] / 2);
  TGeoVolume* vMB0pol = gGeoManager->MakeBox("vMB0pol", mPol, mMB0cu[0] / 2, mMB0pol / 2, mMB0cu[2] / 2);
  TGeoVolume* vMB0epo = gGeoManager->MakeBox("vMB0epo", mEpo, mMB0cu[0] / 2, mMB0epo / 2, mMB0cu[2] / 2);

  // Screws = Head + Thread
  TGeoVolume* vMB0screwH = gGeoManager->MakeTube("vMB0screwH", mInox, 0.0, 0.7 / 2, 0.35 / 2); // tete
  TGeoVolume* vMB0screwT = gGeoManager->MakeTube("vMB0screwT", mInox, 0.0, 0.4 / 2, 0.95 / 2); // filetage
  // Insert Sertitec
  TGeoVolume* vMB0serti = gGeoManager->MakeTube("vMB0serti", mInox, 0.16 / 2, 0.556 / 2, 0.15 / 2); // tete

  Float_t heigthConnector = 0.4;                                                                              // male + female
  TGeoVolume* vConnector = gGeoManager->MakeBox("vConnector", mPolyu, 6.2 / 2, heigthConnector / 2, 1.4 / 2); // in liquid-crystal polymer --> polyurethane?
  vMB0cu->SetLineColor(kGreen);
  vMB0fr4->SetLineColor(kBlack);
  vMB0pol->SetLineColor(kBlue);
  vMB0epo->SetLineColor(kGreen);
  vMB0screwH->SetLineColor(kOrange);
  vMB0screwT->SetLineColor(kOrange);
  vMB0serti->SetLineColor(kOrange);
  vConnector->SetLineColor(kGray + 3);

  // Positioning the layers
  MotherBoard0->AddNode(vMB0cu, 1);

  auto* t_MB0fr4 = new TGeoTranslation("translation_fr4", 0.0, signe * (mMB0fr4 + mMB0cu[1]) / 2, 0.0);
  t_MB0fr4->RegisterYourself();
  MotherBoard0->AddNode(vMB0fr4, 1, t_MB0fr4);
  auto* t_MB0pol =
    new TGeoTranslation("translation_pol", 0.0, signe * (mMB0fr4 + (mMB0cu[1] + mMB0pol) / 2), 0.0);
  t_MB0pol->RegisterYourself();
  MotherBoard0->AddNode(vMB0pol, 1, t_MB0pol);
  auto* t_MB0epo = new TGeoTranslation("translation_epo", 0.0, signe * (mMB0fr4 + mMB0pol + (mMB0cu[1] + mMB0epo) / 2), 0.0);
  t_MB0epo->RegisterYourself();
  MotherBoard0->AddNode(vMB0epo, 1, t_MB0epo);
  auto* r_MB0screw = new TGeoRotation("rotation_vMB0screw", 0, 90, 0);
  auto* t_MB0screwH1 = new TGeoCombiTrans(mMB0cu[0] / 2 - 1.65, signe * (mMB0fr4 + mMB0pol + mMB0epo + (mMB0cu[1] + 0.35) / 2), 0.0, r_MB0screw);
  t_MB0screwH1->RegisterYourself();
  auto* t_MB0screwT1 = new TGeoCombiTrans(mMB0cu[0] / 2 - 1.65, -signe * (mMB0cu[1] + 0.95) / 2, 0.0, r_MB0screw);
  t_MB0screwT1->RegisterYourself();
  auto* t_MB0screwH2 = new TGeoCombiTrans(-(mMB0cu[0] / 2 - 1.65), signe * (mMB0fr4 + mMB0pol + mMB0epo + (mMB0cu[1] + 0.35) / 2), 0.0, r_MB0screw);
  t_MB0screwH2->RegisterYourself();
  auto* t_MB0screwT2 = new TGeoCombiTrans(-(mMB0cu[0] / 2 - 1.65), -signe * (mMB0cu[1] + 0.95) / 2, 0.0, r_MB0screw);
  t_MB0screwT2->RegisterYourself();
  auto* t_MB0serti1 = new TGeoCombiTrans(mMB0cu[0] / 2 - 2.65, signe * (mMB0fr4 + mMB0pol + mMB0epo + (mMB0cu[1] + 0.153) / 2), 0.0, r_MB0screw);
  t_MB0serti1->RegisterYourself();
  auto* t_MB0serti2 = new TGeoCombiTrans(-(mMB0cu[0] / 2 - 2.65), signe * (mMB0fr4 + mMB0pol + mMB0epo + (mMB0cu[1] + 0.153) / 2), 0.0, r_MB0screw);
  t_MB0serti2->RegisterYourself();
  auto* t_MB0Connector = new TGeoTranslation("translation_connector", 0.0, signe * (-mMB0cu[1] - 0.2), 0.0);
  t_MB0Connector->RegisterYourself();

  MotherBoard0->AddNode(vMB0screwH, 1, t_MB0screwH1);
  MotherBoard0->AddNode(vMB0screwT, 1, t_MB0screwT1);
  MotherBoard0->AddNode(vMB0screwH, 1, t_MB0screwH2);
  MotherBoard0->AddNode(vMB0screwT, 1, t_MB0screwT2);
  MotherBoard0->AddNode(vMB0serti, 1, t_MB0serti1);
  MotherBoard0->AddNode(vMB0serti, 1, t_MB0serti2);
  MotherBoard0->AddNode(vConnector, 1, t_MB0Connector);

  // Positioning the boards on the disks
  Float_t shift = 0.23;
  Float_t shift3 = 0.09;
  auto* t_disk0 = new TGeoTranslation("translation_disk0", 0.0, tyMB0 - signe * shift, tzMB0);
  t_disk0->RegisterYourself();
  auto* t_disk1 = new TGeoTranslation("translation_disk1", 0.0, tyMB0 - signe * shift, tzMB0 - 3.3);
  t_disk1->RegisterYourself();
  auto* t_disk2 = new TGeoTranslation("translation_disk2", 0.0, tyMB0_3 - signe * shift3, tzMB0 - 7.1);
  t_disk2->RegisterYourself();
  auto* r_MB0 = new TGeoRotation("rotation_MB0", 0.0, 0.0, 0.0);
  r_MB0->RegisterYourself();
  auto* p_disk0 = new TGeoCombiTrans(*t_disk0, *r_MB0);
  p_disk0->RegisterYourself();
  auto* p_disk1 = new TGeoCombiTrans(*t_disk1, *r_MB0);
  p_disk1->RegisterYourself();
  auto* p_disk2 = new TGeoCombiTrans(*t_disk2, *r_MB0);
  p_disk2->RegisterYourself();
  // Final addition of the boards
  HalfConeVolume->AddNode(MotherBoard0, 1, p_disk0);
  HalfConeVolume->AddNode(MotherBoard0, 1, p_disk1);
  HalfConeVolume->AddNode(MotherBoard0, 1, p_disk2);

  // Small boards on to the main mother board
  auto* MotherBoard0_1 = new TGeoVolumeAssembly(Form("MotherBoard0_1_H%d", half));
  // 4 layers
  TGeoVolume* vMB0cu_1 =
    gGeoManager->MakeBox("vMB0cu_1", mCu, 18.0 / 2, mMB0cu[1] / 2, 1.2 / 2);
  TGeoVolume* vMB0fr4_1 =
    gGeoManager->MakeBox("vMB0fr4_1", mFR4, 18.0 / 2, mMB0fr4 / 2, 1.2 / 2);
  TGeoVolume* vMB0pol_1 =
    gGeoManager->MakeBox("vMB0pol_1", mPol, 18.0 / 2, mMB0pol / 2, 1.2 / 2);
  TGeoVolume* vMB0epo_1 =
    gGeoManager->MakeBox("vMB0epo_1", mEpo, 18.0 / 2, mMB0epo / 2, 1.2 / 2);
  vMB0cu_1->SetLineColor(kGreen);
  vMB0fr4_1->SetLineColor(kBlack);
  vMB0pol_1->SetLineColor(kBlue);
  vMB0epo_1->SetLineColor(kGreen);
  MotherBoard0_1->AddNode(vMB0cu_1, 1);
  MotherBoard0_1->AddNode(vMB0fr4_1, 1, t_MB0fr4);
  MotherBoard0_1->AddNode(vMB0pol_1, 1, t_MB0pol);
  MotherBoard0_1->AddNode(vMB0epo_1, 1, t_MB0epo);

  // ================ Main Mother Board
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
  TGeoVolume* vMB2cu =
    gGeoManager->MakeTrd1("vMB2cu", mCu, mMB2cu[0] / 2, mMB2cu[1] / 2,
                          mMB2cu[2] / 2, mMB2cu[3] / 2);
  TGeoVolume* vMB2fr4 =
    gGeoManager->MakeTrd1("vMB2fr4", mFR4, mMB2cu[0] / 2, mMB2cu[1] / 2,
                          mMB2fr4 / 2, mMB2cu[3] / 2);
  TGeoVolume* vMB2pol =
    gGeoManager->MakeTrd1("vMB2pol", mPol, mMB2cu[0] / 2, mMB2cu[1] / 2,
                          mMB2pol / 2, mMB2cu[3] / 2);
  TGeoVolume* vMB2epo =
    gGeoManager->MakeTrd1("vMB2epo", mEpo, mMB2cu[0] / 2, mMB2cu[1] / 2,
                          mMB2epo / 2, mMB2cu[3] / 2);
  vMB2cu->SetLineColor(kGreen);
  vMB2fr4->SetLineColor(kBlack);
  vMB2pol->SetLineColor(kBlue);
  vMB2epo->SetLineColor(kGreen + 2);
  auto* t_MB2fr4 = new TGeoTranslation("translation_fr4", 0.0,
                                       signe * (mMB2fr4 + mMB2cu[2]) / 2, 0.0);
  t_MB2fr4->RegisterYourself();
  auto* t_MB2pol =
    new TGeoTranslation("translation_pol", 0.0,
                        signe * (mMB2fr4 + (mMB2cu[2] + mMB2pol) / 2), 0.0);
  t_MB2pol->RegisterYourself();
  auto* t_MB2epo = new TGeoTranslation(
    "translation_epo", 0.0,
    signe * (mMB2fr4 + mMB2pol + (mMB2cu[2] + mMB2epo) / 2), 0.0);
  t_MB2epo->RegisterYourself();
  MotherBoard2->AddNode(vMB2cu, 1);
  MotherBoard2->AddNode(vMB2fr4, 1, t_MB2fr4);
  MotherBoard2->AddNode(vMB2pol, 1, t_MB2pol);
  MotherBoard2->AddNode(vMB2epo, 1, t_MB2epo);

  for (Float_t i = -1; i < 3; i++) {
    auto* t_MB2serti1 = new TGeoTranslation(
      "translationMB2serti1", 8.5, -signe * (mMB2cu[2] + 0.153) / 2, 1.3 * i);
    t_MB2serti1->RegisterYourself();
    auto* t_MB2serti2 =
      new TGeoTranslation("translationMB2serti2", -8.5,
                          -signe * (mMB2cu[2] + 0.153) / 2, 1.3 * i);
    t_MB2serti2->RegisterYourself();
    auto* p_MB2serti1 = new TGeoCombiTrans(*t_MB2serti1, *r_MB0screw);
    p_MB2serti1->RegisterYourself();
    auto* p_MB2serti2 = new TGeoCombiTrans(*t_MB2serti2, *r_MB0screw);
    p_MB2serti2->RegisterYourself();
    MotherBoard2->AddNode(vMB0serti, 1, p_MB2serti1);
    MotherBoard2->AddNode(vMB0serti, 1, p_MB2serti2);
  }

  for (Float_t i = -2; i < 1; i++) {
    auto* t_MB2serti3 = new TGeoTranslation(
      "translationMB2serti3", 0.7, -signe * (mMB2cu[2] + 0.153) / 2, 1.3 * i);
    t_MB2serti3->RegisterYourself();
    auto* t_MB2serti4 =
      new TGeoTranslation("translationMB2serti4", -0.7,
                          -signe * (mMB2cu[2] + 0.153) / 2, 1.3 * i);
    t_MB2serti4->RegisterYourself();
    auto* p_MB2serti3 = new TGeoCombiTrans(*t_MB2serti3, *r_MB0screw);
    p_MB2serti3->RegisterYourself();
    auto* p_MB2serti4 = new TGeoCombiTrans(*t_MB2serti4, *r_MB0screw);
    p_MB2serti4->RegisterYourself();
    MotherBoard2->AddNode(vMB0serti, 1, p_MB2serti3);
    MotherBoard2->AddNode(vMB0serti, 1, p_MB2serti4);
  }

  for (Float_t i = -2; i < 2; i++) {
    auto* t_MB2serti5 =
      new TGeoTranslation("translationMB2serti5", 7.0 * i + 3.5,
                          -signe * (mMB2cu[2] + 0.153) / 2, -2.5);
    t_MB2serti5->RegisterYourself();
    auto* p_MB2serti5 = new TGeoCombiTrans(*t_MB2serti5, *r_MB0screw);
    p_MB2serti5->RegisterYourself();
    auto* t_MB2serti6 =
      new TGeoTranslation("translationMB2serti6", 7.0 * i + 3.5,
                          -signe * (mMB2cu[2] + 0.153) / 2, -3.5);
    t_MB2serti6->RegisterYourself();
    auto* p_MB2serti6 = new TGeoCombiTrans(*t_MB2serti6, *r_MB0screw);
    p_MB2serti6->RegisterYourself();
    MotherBoard2->AddNode(vMB0serti, 1, p_MB2serti5);
    MotherBoard2->AddNode(vMB0serti, 1, p_MB2serti6);
  }
  // Two boards (from the two first disks) located on the main mother board
  auto* t_MotherBoard0_1 =
    new TGeoTranslation("translation_MB0_1", 0.0, -signe * (-heigthConnector - mMB2cu[2] - mMB2fr4 - mMB2pol - mMB2epo), 3.5);
  t_MotherBoard0_1->RegisterYourself();
  auto* t_MotherBoard0_2 =
    new TGeoTranslation("translation_MB0_2", 0.0, -signe * (-heigthConnector - mMB2cu[2] - mMB2fr4 - mMB2pol - mMB2epo), 1.5);

  t_MotherBoard0_2->RegisterYourself();
  MotherBoard2->AddNode(MotherBoard0_1, 1, t_MotherBoard0_1);
  MotherBoard2->AddNode(MotherBoard0_1, 2, t_MotherBoard0_2);

  TGeoVolume* vConnector2 = gGeoManager->MakeBox("vConnector2", mPolyu, 12.00 / 2, heigthConnector / 2, 0.7 / 2);
  TGeoVolume* vConnector2p = gGeoManager->MakeBox("vConnector2p", mPolyu, 6.00 / 2, heigthConnector / 2, 0.7 / 2);
  // === Readout cables connectors  ===

  TGeoVolume* vConnectorRC = gGeoManager->MakeBox("vConnectorRC-MB2", mPolyu, 6.00 / 2, heigthConnector / 2, 1.4 / 2);
  // ==================================
  Double_t yMB2Connector = signe * (mMB2cu[2] / 2 + mMB2fr4 + mMB2pol + mMB2epo + heigthConnector / 2);
  auto* t_MB2Connector1 = new TGeoTranslation("translation_connector1", 0.0, yMB2Connector, 1.5);
  t_MB2Connector1->RegisterYourself();
  auto* t_MB2Connector2 = new TGeoTranslation("translation_connector2", 0.0, yMB2Connector, 3.5);
  t_MB2Connector2->RegisterYourself();
  auto* t_MB2ConnectorRC_1 = new TGeoTranslation("translation_connectorRC_1", 3.5, yMB2Connector, -0.5);
  t_MB2ConnectorRC_1->RegisterYourself();
  auto* t_MB2ConnectorRC_2 = new TGeoTranslation("translation_connectorRC_2", -3.5, yMB2Connector, -0.5);
  t_MB2ConnectorRC_1->RegisterYourself();
  auto* t_MB2ConnectorRC_3 = new TGeoTranslation("translation_connectorRC_3", 7.0, yMB2Connector, -3.5);
  t_MB2ConnectorRC_2->RegisterYourself();
  auto* t_MB2ConnectorRC_4 = new TGeoTranslation("translation_connectorRC_4", -7.0, yMB2Connector, -3.5);
  t_MB2ConnectorRC_2->RegisterYourself();

  yMB2Connector = signe * (-mMB2cu[2] / 2 - heigthConnector / 2);
  auto* t_MB2ConnectorRCp_1 = new TGeoTranslation("translation_connectorRCp_1", 4.0, yMB2Connector, 0.5);
  t_MB2ConnectorRCp_1->RegisterYourself();
  auto* t_MB2ConnectorRCp_2 = new TGeoTranslation("translation_connectorRCp_2", -4.0, yMB2Connector, 0.5);
  t_MB2ConnectorRCp_2->RegisterYourself();
  auto* t_MB2ConnectorRCp_3 = new TGeoTranslation("translation_connectorRCp_3", 7.0, yMB2Connector, -3.5);
  t_MB2ConnectorRCp_3->RegisterYourself();
  auto* t_MB2ConnectorRCp_4 = new TGeoTranslation("translation_connectorRCp_4", -7.0, yMB2Connector, -3.5);
  t_MB2ConnectorRCp_4->RegisterYourself();

  vConnector2->SetLineColor(kGray + 3);
  vConnectorRC->SetLineColor(kGray + 3);
  MotherBoard2->AddNode(vConnector2, 1, t_MB2Connector1);
  MotherBoard2->AddNode(vConnector2, 1, t_MB2Connector2);
  MotherBoard2->AddNode(vConnectorRC, 1, t_MB2ConnectorRC_1);
  MotherBoard2->AddNode(vConnectorRC, 1, t_MB2ConnectorRC_2);
  MotherBoard2->AddNode(vConnectorRC, 1, t_MB2ConnectorRC_3);
  MotherBoard2->AddNode(vConnectorRC, 1, t_MB2ConnectorRC_4);
  MotherBoard2->AddNode(vConnectorRC, 1, t_MB2ConnectorRCp_1);
  MotherBoard2->AddNode(vConnectorRC, 1, t_MB2ConnectorRCp_2);
  MotherBoard2->AddNode(vConnectorRC, 1, t_MB2ConnectorRCp_3);
  MotherBoard2->AddNode(vConnectorRC, 1, t_MB2ConnectorRCp_4);

  // Positioning the main mother board
  auto* t_MotherBoard2 = new TGeoTranslation(
    "translation_MB2", 0.0,
    -signe * (-20.52 + mMB2fr4 + mMB2pol + mMB2epo + 2.2 * TMath::Sin(19.0)),
    -62.8 + 2.2 * TMath::Cos(19.0));
  t_MotherBoard2->RegisterYourself();
  auto* r_MotherBoard2 =
    new TGeoRotation("rotation_MB2", 0.0, -signe * (-19.0), 0.0);
  r_MotherBoard2->RegisterYourself();
  auto* p_MB2 = new TGeoCombiTrans(*t_MotherBoard2, *r_MotherBoard2);
  p_MB2->RegisterYourself();
  HalfConeVolume->AddNode(MotherBoard2, 1, p_MB2);
}

void HalfCone::makeFlexCables(TGeoVolumeAssembly* HalfConeVolume, Int_t half, Int_t signe)
{
  auto* mCu = gGeoManager->GetMedium("MFT_Cu$");
  // Flat cables between disk 0 and main board
  Float_t width_flat = 1.0;        // 4 flexs, fully arbitrary!
  Float_t thickness_flat = 0.0041; // 10 microns lines (228 lines, 0.0175x0.1 mm2) + 31 microns ground plane (~width 70 mm, thickness 0.0175mm)
  Double_t theta1 = -22.70 + signe * 16.5;
  Double_t theta2 = 22.60 + signe * 16.5;
  TGeoVolume* vFlat0_1 = gGeoManager->MakeTubs("vFlat0_1", mCu, 15.0, 15.0 + thickness_flat, width_flat / 2., theta1, theta2);
  TGeoVolume* vFlat0_2 = gGeoManager->MakeTubs("vFlat0_2", mCu, 15.0, 15.0 + thickness_flat, width_flat / 2., theta1, theta2);
  TGeoVolume* vFlat0_3 = gGeoManager->MakeTubs("vFlat0_3", mCu, 15.0, 15.0 + thickness_flat, width_flat / 2., theta1, theta2);
  TGeoVolume* vFlat0_4 = gGeoManager->MakeTubs("vFlat0_4", mCu, 15.0, 15.0 + thickness_flat, width_flat / 2., theta1, theta2);
  auto* t_flat0_1 = new TGeoTranslation("translation_flat0_1", 4.5, signe * 5.0, -56.73);
  t_flat0_1->RegisterYourself();
  auto* t_flat0_2 = new TGeoTranslation("translation_flat0_2", 1.5, signe * 5.0, -56.73);
  t_flat0_2->RegisterYourself();
  auto* t_flat0_3 = new TGeoTranslation("translation_flat0_3", -1.5, signe * 5.0, -56.73);
  t_flat0_3->RegisterYourself();
  auto* t_flat0_4 = new TGeoTranslation("translation_flat0_4", -4.5, signe * 5.0, -56.73);
  t_flat0_4->RegisterYourself();
  auto* r_flat0 = new TGeoRotation("rotation_flat0", signe * 90.0, signe * 90.0, 0.0);
  r_flat0->RegisterYourself();
  auto* p_flat0_1 = new TGeoCombiTrans(*t_flat0_1, *r_flat0);
  p_flat0_1->RegisterYourself();
  auto* p_flat0_2 = new TGeoCombiTrans(*t_flat0_2, *r_flat0);
  p_flat0_2->RegisterYourself();
  auto* p_flat0_3 = new TGeoCombiTrans(*t_flat0_3, *r_flat0);
  p_flat0_3->RegisterYourself();
  auto* p_flat0_4 = new TGeoCombiTrans(*t_flat0_4, *r_flat0);
  p_flat0_4->RegisterYourself();
  HalfConeVolume->AddNode(vFlat0_1, 1, p_flat0_1);
  HalfConeVolume->AddNode(vFlat0_2, 1, p_flat0_2);
  HalfConeVolume->AddNode(vFlat0_3, 1, p_flat0_3);
  HalfConeVolume->AddNode(vFlat0_4, 1, p_flat0_4);

  // Flat lines between disk 1 and main board
  theta1 = -32.0 + signe * 24.69;
  theta2 = 32.7 + signe * 24.69;
  TGeoVolume* vFlat1_1 = gGeoManager->MakeTubs("vFlat1_1", mCu, 6.0, 6.00 + thickness_flat, width_flat / 2., theta1, theta2);
  TGeoVolume* vFlat1_2 = gGeoManager->MakeTubs("vFlat1_2", mCu, 6.0, 6.00 + thickness_flat, width_flat / 2., theta1, theta2);
  TGeoVolume* vFlat1_3 = gGeoManager->MakeTubs("vFlat1_3", mCu, 6.0, 6.00 + thickness_flat, width_flat / 2., theta1, theta2);
  TGeoVolume* vFlat1_4 = gGeoManager->MakeTubs("vFlat1_4", mCu, 6.0, 6.00 + thickness_flat, width_flat / 2., theta1, theta2);
  auto* t_flat1_1 = new TGeoTranslation("translation_flat1_1", 3., signe * 13.4, -55.6);
  t_flat1_1->RegisterYourself();
  auto* t_flat1_2 = new TGeoTranslation("translation_flat1_2", 1., signe * 13.4, -55.6);
  t_flat1_2->RegisterYourself();
  auto* t_flat1_3 = new TGeoTranslation("translation_flat1_3", -1., signe * 13.4, -55.6);
  t_flat1_3->RegisterYourself();
  auto* t_flat1_4 = new TGeoTranslation("translation_flat1_4", -3., signe * 13.4, -55.6);
  t_flat1_4->RegisterYourself();
  auto* r_flat1 = new TGeoRotation("rotation_flat1", signe * 90.0, signe * 90.0, 0.0);
  r_flat1->RegisterYourself();
  auto* p_flat1_1 = new TGeoCombiTrans(*t_flat1_1, *r_flat1);
  p_flat1_1->RegisterYourself();
  auto* p_flat1_2 = new TGeoCombiTrans(*t_flat1_2, *r_flat1);
  p_flat1_2->RegisterYourself();
  auto* p_flat1_3 = new TGeoCombiTrans(*t_flat1_3, *r_flat1);
  p_flat1_3->RegisterYourself();
  auto* p_flat1_4 = new TGeoCombiTrans(*t_flat1_4, *r_flat1);
  p_flat1_4->RegisterYourself();
  HalfConeVolume->AddNode(vFlat1_1, 1, p_flat1_1);
  HalfConeVolume->AddNode(vFlat1_2, 1, p_flat1_2);
  HalfConeVolume->AddNode(vFlat1_3, 1, p_flat1_3);
  HalfConeVolume->AddNode(vFlat1_4, 1, p_flat1_4);

  // Flat lines between disk 2 and main board
  width_flat = 14.0;
  thickness_flat = 0.017; // 3 microns lines (268 lines, 0.0175x0.09 mm2), 14 microns ground plane (140 mm x 0.0175mm) including flex ~20% longer in o2!
  TGeoVolume* vFlat2 = gGeoManager->MakeBox("vFlat2", mCu, width_flat / 2, thickness_flat / 2, 2.3 / 2);
  auto* t_flat2 = new TGeoTranslation("translation_flat2", 0.0, -signe * 18.35, -55.45);
  t_flat2->RegisterYourself();
  auto* r_flat2 = new TGeoRotation("rotation_flat2", 0.0, -signe * (18.), 0.0);
  r_flat2->RegisterYourself();
  auto* p_flat2 = new TGeoCombiTrans(*t_flat2, *r_flat2);
  p_flat2->RegisterYourself();
  HalfConeVolume->AddNode(vFlat2, 1, p_flat2);
}

void HalfCone::makeReadoutCables(TGeoVolumeAssembly* HalfConeVolume, Int_t half, Int_t signe)
{
  auto* mCu = gGeoManager->GetMedium("MFT_Cu$");

  // Connector
  Float_t heigthConnector = 0.4; // male + female
  auto* mPolyu = gGeoManager->GetMedium("MFT_Polyurethane$");
  TGeoVolume* vConnectorRC = gGeoManager->MakeBox("vConnectorRC", mPolyu, 6.00 / 2, heigthConnector / 2, 1.4 / 2);
  vConnectorRC->SetLineColor(kGray + 3);

  // Fondamental numbers for the sections of Copper in the 2 types of readout cables
  Double_t section_ROcable_48pairs = 0.26544; // in cm2
  Double_t section_ROcable_16pairs = 0.08848; // in cm2

  // Starting from the MFT backside: from the patch panel to MB2
  Double_t mRO1[3];
  mRO1[0] = {14.0};                                                                 //width
  mRO1[1] = (16 * section_ROcable_48pairs + 6 * section_ROcable_16pairs) / mRO1[0]; // thickness
  mRO1[2] = {6.5};                                                                  // length
  TGeoVolume* vRO1 = gGeoManager->MakeBox("vRO1", mCu, mRO1[0] / 2, mRO1[1] / 2, mRO1[2] / 2);
  Double_t zRO1 = -80.20;
  auto* t_RO1 = new TGeoTranslation("translation_RO1", 0.0, -signe * 28.0, zRO1);
  t_RO1->RegisterYourself();

  Double_t mRO2[3];
  mRO2[0] = mRO1[0];
  mRO2[1] = (14 * section_ROcable_48pairs + 5 * section_ROcable_16pairs) / mRO2[0];
  mRO2[2] = {3.0};
  TGeoVolume* vRO2 = gGeoManager->MakeBox("vRO2", mCu, mRO2[0] / 2, mRO2[1] / 2, mRO2[2] / 2);
  auto* t_RO2 = new TGeoTranslation("translation_RO2", 0.0, -signe * 28.0, zRO1 + mRO1[2] / 2 + mRO2[2] / 2);
  t_RO2->RegisterYourself();

  Double_t mRO3[3];
  mRO3[0] = mRO1[0];
  mRO3[1] = (12 * section_ROcable_48pairs + 4 * section_ROcable_16pairs) / mRO3[0];
  mRO3[2] = {3.6};
  TGeoVolume* vRO3 = gGeoManager->MakeBox("vRO3", mCu, mRO3[0] / 2, mRO3[1] / 2, mRO3[2] / 2);
  auto* t_RO3 = new TGeoTranslation("translation_RO3", 0.0, -signe * 28.0, zRO1 + mRO1[2] / 2 + mRO2[2] + mRO3[2] / 2);
  t_RO3->RegisterYourself();

  Double_t eRO4 = 12 * section_ROcable_48pairs / mRO1[0];
  TGeoVolume* vRO4 = gGeoManager->MakeTubs("vRO4", mCu, 6.5, 6.5 + eRO4, mRO1[0] / 2, -21.0 + signe * 26.11, 21.0 + signe * 26.11);
  auto* t_RO4 = new TGeoTranslation("translation_RO4", 0.0, signe * 21.4, -70.9);
  t_RO4->RegisterYourself();
  auto* r_RO4 = new TGeoRotation("rotation_RO4", signe * 90.0, signe * 90.0, 0.0);
  r_RO4->RegisterYourself();
  auto* p_RO4 = new TGeoCombiTrans(*t_RO4, *r_RO4);
  p_RO4->RegisterYourself();
  HalfConeVolume->AddNode(vRO4, 1, p_RO4);

  // RO5 = cable connected to the upper part of MB2
  Double_t mRO5[3];
  mRO5[0] = {4.5};
  mRO5[1] = section_ROcable_48pairs / mRO5[0];
  mRO5[2] = {7.071};
  TGeoVolume* vRO5 = gGeoManager->MakeBox("vRO5", mCu, mRO5[0] / 2, mRO5[1] / 2, mRO5[2] / 2);
  auto* t_RO5_1 = new TGeoTranslation("translation_RO5_1", 3.5, -signe * (20.83 + mRO5[2] / 2 * TMath::Sin(45.0 * TMath::Pi() / 180)), -61 - mRO5[2] / 2 * TMath::Cos(45 * TMath::Pi() / 180));
  t_RO5_1->RegisterYourself();
  auto* t_RO5_2 = new TGeoTranslation("translation_RO5_1", -3.5, -signe * (20.83 + mRO5[2] / 2 * TMath::Sin(45.0 * TMath::Pi() / 180)), -61 - mRO5[2] / 2 * TMath::Cos(45 * TMath::Pi() / 180));
  t_RO5_2->RegisterYourself();
  auto* r_RO5 = new TGeoRotation("rotation_RO5", 0.0, -signe * (45.0), 0.0);
  r_RO5->RegisterYourself();
  auto* p_RO5_1 = new TGeoCombiTrans(*t_RO5_1, *r_RO5);
  p_RO5_1->RegisterYourself();
  auto* p_RO5_2 = new TGeoCombiTrans(*t_RO5_2, *r_RO5);
  p_RO5_2->RegisterYourself();

  // ===========   TGeoTrap ---> trapezoid shapes ===============
  Float_t length = 2.20 / 2;
  Float_t angleZ = 69.;
  Float_t angleXY = -signe * 50.;
  Float_t width = 3.5;
  Float_t thickness = (section_ROcable_48pairs / width) / TMath::Cos(angleXY * TMath::Pi() / 180); // special thickness!!
  TGeoVolume* vRO6 = new TGeoVolume("vRO6", new TGeoTrap(length, angleZ, angleXY, width / 2, thickness / 2, thickness / 2, 0., width / 2, thickness / 2, thickness / 2, 0.), mCu);
  Double_t yRO6 = -signe * (22.45 + length * 1.26 * TMath::Sin(angleZ * TMath::Pi() / 180));
  Double_t zRO6 = -64.1 - length * 2. * TMath::Cos(angleZ * TMath::Pi() / 180);
  auto* t_RO6 = new TGeoTranslation("translation_RO6", 5.3, yRO6, zRO6);
  t_RO6->RegisterYourself();
  auto* r_RO6 = new TGeoRotation("rotation_RO6", signe * 90.0, -signe * (0), 0.0);
  r_RO6->RegisterYourself();
  auto* p_RO6 = new TGeoCombiTrans(*t_RO6, *r_RO6);
  p_RO6->RegisterYourself();

  TGeoVolume* vRO7 = new TGeoVolume("vRO7", new TGeoTrap(length, angleZ, -angleXY, width / 2, thickness / 2, thickness / 2, 0., width / 2, thickness / 2, thickness / 2, 0.), mCu);
  auto* t_RO7 = new TGeoTranslation("translation_RO6", -5.3, yRO6, zRO6);
  t_RO7->RegisterYourself();
  auto* r_RO7 = new TGeoRotation("rotation_RO7", signe * 90.0, -signe * (0), 0.0);
  r_RO7->RegisterYourself();
  auto* p_RO7 = new TGeoCombiTrans(*t_RO7, *r_RO7);
  p_RO7->RegisterYourself();

  // ============= under the main mother board =============
  Double_t mRO8[3];
  mRO8[0] = mRO5[0];                               // width
  mRO8[1] = 2 * section_ROcable_48pairs / mRO8[0]; // thickness, 2 cables coded in one cable
  mRO8[2] = {6.40};                                // length
  TGeoVolume* vRO8 = gGeoManager->MakeBox("vRO8", mCu, mRO8[0] / 2, mRO8[1] / 2, mRO8[2] / 2);
  Double_t yRO8 = -signe * (19.1 + mRO8[2] / 2 * TMath::Sin(19.0 * TMath::Pi() / 180));
  Double_t zRO8 = -59.95 - mRO8[2] / 2 * TMath::Cos(19.0 * TMath::Pi() / 180);
  auto* t_RO8_1 = new TGeoTranslation("translation_RO8_1", 3.5, yRO8, zRO8);
  t_RO8_1->RegisterYourself();
  auto* t_RO8_2 = new TGeoTranslation("translation_RO8_1", -3.5, yRO8, zRO8);
  t_RO8_2->RegisterYourself();
  auto* r_RO8 = new TGeoRotation("rotation_RO8", 0.0, -signe * (19.0), 0.0);
  r_RO8->RegisterYourself();
  auto* p_RO8_1 = new TGeoCombiTrans(*t_RO8_1, *r_RO8);
  p_RO8_1->RegisterYourself();
  auto* p_RO8_2 = new TGeoCombiTrans(*t_RO8_2, *r_RO8);
  p_RO8_2->RegisterYourself();

  Double_t mRO9[3];
  mRO9[0] = mRO5[0];                               // width
  mRO9[1] = 2 * section_ROcable_48pairs / mRO9[0]; // thickness, 2 cables coded in one cable
  mRO9[2] = {4.5};                                 // length
  TGeoVolume* vRO9 = gGeoManager->MakeBox("vRO9", mCu, mRO9[0] / 2, mRO9[1] / 2, mRO9[2] / 2);
  auto* t_RO9_1 = new TGeoTranslation("translation_RO9_1", 3.5, -signe * (23.5), -66.15);
  t_RO9_1->RegisterYourself();
  auto* t_RO9_2 = new TGeoTranslation("translation_RO9_2", -3.5, -signe * (23.5), -66.15);
  t_RO9_2->RegisterYourself();
  auto* r_RO9 = new TGeoRotation("rotation_RO9", 0.0, -signe * (90.0), 0.0);
  r_RO9->RegisterYourself();
  auto* p_RO9_1 = new TGeoCombiTrans(*t_RO9_1, *r_RO9);
  p_RO9_1->RegisterYourself();
  auto* p_RO9_2 = new TGeoCombiTrans(*t_RO9_2, *r_RO9);
  p_RO9_2->RegisterYourself();

  // ===========  For the 2 latest disks  ================
  // =============  Disk 3  ============
  length = 6.9 / 2;
  angleZ = -45.;
  angleXY = 0.;
  thickness = 3.5;                             // inversion width/thickness! this is the width
  width = section_ROcable_48pairs / thickness; // inversion width/thickness! this is the thickness
  // ==== front
  TGeoVolume* vRO_D3_1 = new TGeoVolume("vRO_D3_1", new TGeoTrap(length, angleZ + 10., angleXY, width / 2, thickness / 2, thickness / 2, 0., width / 2, thickness / 2, thickness / 2, 0.), mCu);
  auto* t_ROD3_1 = new TGeoTranslation("translation_ROD3_1", 4.5, -signe * (20.9 + length * TMath::Cos(angleZ * TMath::Pi() / 180)), -67.4);
  t_ROD3_1->RegisterYourself();
  auto* r_ROD3_1 = new TGeoRotation("rotation_ROD3_1", signe * 180.0, signe * (90), 0.0);
  r_ROD3_1->RegisterYourself();
  auto* p_ROD3_1 = new TGeoCombiTrans(*t_ROD3_1, *r_ROD3_1);
  p_ROD3_1->RegisterYourself();

  TGeoVolume* vRO_D3_2 = new TGeoVolume("vRO_D3_2", new TGeoTrap(length, -angleZ - 10., angleXY, width / 2, thickness / 2, thickness / 2, 0., width / 2, thickness / 2, thickness / 2, 0.), mCu);
  auto* t_ROD3_2 = new TGeoTranslation("translation_ROD3_2", -4.5, -signe * (20.9 + length * TMath::Cos(angleZ * TMath::Pi() / 180)), -67.4);
  t_ROD3_2->RegisterYourself();
  auto* r_ROD3_2 = new TGeoRotation("rotation_ROD3_2", signe * 180.0, signe * (90), 0.0);
  r_ROD3_2->RegisterYourself();
  auto* p_ROD3_2 = new TGeoCombiTrans(*t_ROD3_2, *r_ROD3_2);
  p_ROD3_2->RegisterYourself();

  // ===== rear
  TGeoVolume* vRO_D3_3 = new TGeoVolume("vRO_D3_3", new TGeoTrap(length + 0.4, angleZ + 10., angleXY, width / 2, thickness / 2, thickness / 2, 0., width / 2, thickness / 2, thickness / 2, 0.), mCu);
  auto* t_ROD3_3 = new TGeoTranslation("translation_ROD3_3", 4.5, -signe * (21.3 + length * TMath::Cos(angleZ * TMath::Pi() / 180)), -69.4);
  t_ROD3_3->RegisterYourself();
  auto* r_ROD3_3 = new TGeoRotation("rotation_ROD3_3", signe * 180.0, signe * (90), 0.0);
  r_ROD3_3->RegisterYourself();
  auto* p_ROD3_3 = new TGeoCombiTrans(*t_ROD3_3, *r_ROD3_3);
  p_ROD3_3->RegisterYourself();

  TGeoVolume* vRO_D3_4 = new TGeoVolume("vRO_D3_4", new TGeoTrap(length + 0.4, -angleZ - 10., angleXY, width / 2, thickness / 2, thickness / 2, 0., width / 2, thickness / 2, thickness / 2, 0.), mCu);
  auto* t_ROD3_4 = new TGeoTranslation("translation_ROD3_4", -4.5, -signe * (21.3 + length * TMath::Cos(angleZ * TMath::Pi() / 180)), -69.4);
  t_ROD3_4->RegisterYourself();
  auto* r_ROD3_4 = new TGeoRotation("rotation_ROD3_4", signe * 180.0, signe * (90), 0.0);
  r_ROD3_4->RegisterYourself();
  auto* p_ROD3_4 = new TGeoCombiTrans(*t_ROD3_4, *r_ROD3_4);
  p_ROD3_4->RegisterYourself();

  auto* t_ConnectorRC3_1 = new TGeoTranslation("translation_RC3_1", 5.9, -signe * (20.2), -67.87 + heigthConnector / 2);
  t_ConnectorRC3_1->RegisterYourself();
  auto* r_ConnectorRC3_1 = new TGeoRotation("rotation_RC3_1", 0.0, 90, 0.0);
  r_ConnectorRC3_1->RegisterYourself();
  auto* p_ConnectorRC3_1 = new TGeoCombiTrans(*t_ConnectorRC3_1, *r_ConnectorRC3_1);
  p_ConnectorRC3_1->RegisterYourself();
  auto* t_ConnectorRC3_2 = new TGeoTranslation("translation_RC3_2", -5.9, -signe * (20.2), -67.87 + heigthConnector / 2);
  t_ConnectorRC3_2->RegisterYourself();
  auto* p_ConnectorRC3_2 = new TGeoCombiTrans(*t_ConnectorRC3_2, *r_ConnectorRC3_1);
  p_ConnectorRC3_2->RegisterYourself();
  auto* t_ConnectorRC3_3 = new TGeoTranslation("translation_RC3_3", 5.9, -signe * (20.2), -68.93 - heigthConnector / 2);
  t_ConnectorRC3_3->RegisterYourself();
  auto* p_ConnectorRC3_3 = new TGeoCombiTrans(*t_ConnectorRC3_3, *r_ConnectorRC3_1);
  p_ConnectorRC3_3->RegisterYourself();
  auto* t_ConnectorRC3_4 = new TGeoTranslation("translation_RC3_4", -5.9, -signe * (20.2), -68.93 - heigthConnector / 2);
  t_ConnectorRC3_4->RegisterYourself();
  auto* p_ConnectorRC3_4 = new TGeoCombiTrans(*t_ConnectorRC3_4, *r_ConnectorRC3_1);
  p_ConnectorRC3_4->RegisterYourself();

  // ==============  Disk 4  ================
  length = 6.3 / 2;
  angleZ = -20.;
  angleXY = 0.;
  thickness = 3.5;                             // inversion width and thickness! this is the width
  width = section_ROcable_48pairs / thickness; // inversion width and thickness! this is the thickness
  Float_t xD4 = 6.1;
  Float_t yD4 = 21.6;
  // ===== front
  TGeoVolume* vRO_D4_1 = new TGeoVolume("vRO_D4_1", new TGeoTrap(length, angleZ, angleXY, width / 2, thickness / 2, thickness / 2, 0., width / 2, thickness / 2, thickness / 2, 0.), mCu);
  auto* t_ROD4_1 = new TGeoTranslation("translation_ROD4_1", xD4, -signe * (yD4 + length * TMath::Cos(angleZ * TMath::Pi() / 180)), -75.8);
  t_ROD4_1->RegisterYourself();
  auto* r_ROD4_1 = new TGeoRotation("rotation_ROD4_1", signe * 180.0, signe * (90), 0.0);
  r_ROD4_1->RegisterYourself();
  auto* p_ROD4_1 = new TGeoCombiTrans(*t_ROD4_1, *r_ROD3_1);
  p_ROD4_1->RegisterYourself();

  TGeoVolume* vRO_D4_2 = new TGeoVolume("vRO_D4_2", new TGeoTrap(length, -angleZ, angleXY, width / 2, thickness / 2, thickness / 2, 0., width / 2, thickness / 2, thickness / 2, 0.), mCu);
  auto* t_ROD4_2 = new TGeoTranslation("translation_ROD4_2", -xD4, -signe * (yD4 + length * TMath::Cos(angleZ * TMath::Pi() / 180)), -75.8);
  t_ROD4_2->RegisterYourself();
  auto* r_ROD4_2 = new TGeoRotation("rotation_ROD4_2", signe * 180.0, signe * (90), 0.0);
  r_ROD4_2->RegisterYourself();
  auto* p_ROD4_2 = new TGeoCombiTrans(*t_ROD4_2, *r_ROD4_2);
  p_ROD4_2->RegisterYourself();

  // ===== rear
  TGeoVolume* vRO_D4_3 = new TGeoVolume("vRO_D4_3", new TGeoTrap(length, angleZ, angleXY, width / 2, thickness / 2, thickness / 2, 0., width / 2, thickness / 2, thickness / 2, 0.), mCu);
  auto* t_ROD4_3 = new TGeoTranslation("translation_ROD4_3", xD4, -signe * (yD4 + length * TMath::Cos(angleZ * TMath::Pi() / 180)), -77.8);
  t_ROD4_3->RegisterYourself();
  auto* r_ROD4_3 = new TGeoRotation("rotation_ROD4_3", signe * 180.0, signe * (90), 0.0);
  r_ROD4_3->RegisterYourself();
  auto* p_ROD4_3 = new TGeoCombiTrans(*t_ROD4_3, *r_ROD3_3);
  p_ROD4_3->RegisterYourself();

  TGeoVolume* vRO_D4_4 = new TGeoVolume("vRO_D4_4", new TGeoTrap(length, -angleZ, angleXY, width / 2, thickness / 2, thickness / 2, 0., width / 2, thickness / 2, thickness / 2, 0.), mCu);
  auto* t_ROD4_4 = new TGeoTranslation("translation_ROD4_4", -xD4, -signe * (yD4 + length * TMath::Cos(angleZ * TMath::Pi() / 180)), -77.8);
  t_ROD4_4->RegisterYourself();
  auto* r_ROD4_4 = new TGeoRotation("rotation_ROD4_4", signe * 180.0, signe * (90), 0.0);
  r_ROD4_4->RegisterYourself();
  auto* p_ROD4_4 = new TGeoCombiTrans(*t_ROD4_4, *r_ROD4_4);
  p_ROD4_4->RegisterYourself();

  // Middle
  // 16 pairs cable
  width = section_ROcable_16pairs / thickness; // inversion width and thickness! this is thickness...
  TGeoVolume* vRO_D4_5 = new TGeoVolume("vRO_D4_5", new TGeoTrap(length, 0., 0., width / 2, thickness / 2, thickness / 2, 0., width / 2, thickness / 2, thickness / 2, 0.), mCu);
  auto* t_ROD4_5 = new TGeoTranslation("translation_ROD4_5", 0, -signe * (yD4 + length - 0.2), -75.8);
  t_ROD4_5->RegisterYourself();
  auto* r_ROD4_5 = new TGeoRotation("rotation_ROD4_5", signe * 180.0, signe * (90), 0.0);
  r_ROD4_5->RegisterYourself();
  auto* p_ROD4_5 = new TGeoCombiTrans(*t_ROD4_5, *r_ROD4_5);
  p_ROD4_5->RegisterYourself();
  TGeoVolume* vRO_D4_6 = new TGeoVolume("vRO_D4_6", new TGeoTrap(length, 0., 0., width / 2, thickness / 2, thickness / 2, 0., width / 2, thickness / 2, thickness / 2, 0.), mCu);
  auto* t_ROD4_6 = new TGeoTranslation("translation_ROD4_6", 0, -signe * (yD4 + length - 0.2), -77.8);
  t_ROD4_6->RegisterYourself();
  auto* r_ROD4_6 = new TGeoRotation("rotation_ROD4_6", signe * 180.0, signe * (90), 0.0);
  r_ROD4_6->RegisterYourself();
  auto* p_ROD4_6 = new TGeoCombiTrans(*t_ROD4_6, *r_ROD4_6);
  p_ROD4_6->RegisterYourself();

  auto* t_ConnectorRC4_1 = new TGeoTranslation("translation_RC4_1", 7.5, -signe * (21.5), -76.26 + heigthConnector / 2);
  t_ConnectorRC4_1->RegisterYourself();
  auto* p_ConnectorRC4_1 = new TGeoCombiTrans(*t_ConnectorRC4_1, *r_ConnectorRC3_1);
  p_ConnectorRC4_1->RegisterYourself();
  auto* t_ConnectorRC4_2 = new TGeoTranslation("translation_RC4_2", -7.5, -signe * (21.5), -76.26 + heigthConnector / 2);
  t_ConnectorRC4_2->RegisterYourself();
  auto* p_ConnectorRC4_2 = new TGeoCombiTrans(*t_ConnectorRC4_2, *r_ConnectorRC3_1);
  p_ConnectorRC4_2->RegisterYourself();
  auto* t_ConnectorRC4_3 = new TGeoTranslation("translation_RC4_3", 7.5, -signe * (21.5), -77.34 - heigthConnector / 2);
  t_ConnectorRC4_3->RegisterYourself();
  auto* p_ConnectorRC4_3 = new TGeoCombiTrans(*t_ConnectorRC4_3, *r_ConnectorRC3_1);
  p_ConnectorRC4_3->RegisterYourself();
  auto* t_ConnectorRC4_4 = new TGeoTranslation("translation_RC4_4", -7.5, -signe * (21.5), -77.34 - heigthConnector / 2);
  t_ConnectorRC4_4->RegisterYourself();
  auto* p_ConnectorRC4_4 = new TGeoCombiTrans(*t_ConnectorRC4_4, *r_ConnectorRC3_1);
  p_ConnectorRC4_4->RegisterYourself();
  auto* t_ConnectorRC4_5 = new TGeoTranslation("translation_RC4_5", 0, -signe * (21.5), -76.26 + heigthConnector / 2);
  t_ConnectorRC4_5->RegisterYourself();
  auto* p_ConnectorRC4_5 = new TGeoCombiTrans(*t_ConnectorRC4_5, *r_ConnectorRC3_1);
  p_ConnectorRC4_5->RegisterYourself();
  auto* t_ConnectorRC4_6 = new TGeoTranslation("translation_RC4_6", 0, -signe * (21.5), -77.34 - heigthConnector / 2);
  t_ConnectorRC4_6->RegisterYourself();
  auto* p_ConnectorRC4_6 = new TGeoCombiTrans(*t_ConnectorRC4_6, *r_ConnectorRC3_1);
  p_ConnectorRC4_6->RegisterYourself();

  // ========== For the RO Supply Unit (PSU) ==========
  length = 1.6 / 2;
  angleZ = 70.;
  angleXY = -signe * 25.;
  width = 3.5;
  thickness = section_ROcable_16pairs / width / TMath::Cos(angleZ * TMath::Pi() / 180);
  TGeoVolume* vRO_PSU_1 = new TGeoVolume("vRO_PSU_1", new TGeoTrap(length, angleZ, angleXY, width / 2, thickness / 2, thickness / 2, 0., width / 2, thickness / 2, thickness / 2, 0.), mCu);
  Double_t yPSU = -signe * (24.8 + length * 1.26 * TMath::Sin(angleZ * TMath::Pi() / 180));
  Double_t zPSU = -71.1 - length * 2. * TMath::Cos(angleZ * TMath::Pi() / 180);
  auto* t_RO_PSU_1 = new TGeoTranslation("translation_RO_PSU_1", 5.0, yPSU, zPSU);
  t_RO_PSU_1->RegisterYourself();
  auto* r_RO_PSU_1 = new TGeoRotation("rotation_RO_PSU_1", signe * 90.0, -signe * (0), 0.0);
  r_RO_PSU_1->RegisterYourself();
  auto* p_RO_PSU_1 = new TGeoCombiTrans(*t_RO_PSU_1, *r_RO_PSU_1);
  p_RO_PSU_1->RegisterYourself();

  TGeoVolume* vRO_PSU_2 = new TGeoVolume("vRO_PSU_2", new TGeoTrap(length, angleZ, -angleXY, width / 2, thickness / 2, thickness / 2, 0., width / 2, thickness / 2, thickness / 2, 0.), mCu);
  auto* t_RO_PSU_2 = new TGeoTranslation("translation_RO_PSU_2", -5.0, yPSU, zPSU);
  t_RO_PSU_2->RegisterYourself();
  auto* r_RO_PSU_2 = new TGeoRotation("rotation_RO_PSU_2", signe * 90.0, -signe * (0), 0.0);
  r_RO_PSU_2->RegisterYourself();
  auto* p_RO_PSU_2 = new TGeoCombiTrans(*t_RO_PSU_2, *r_RO_PSU_2);
  p_RO_PSU_2->RegisterYourself();

  zPSU = -72.98 - length * 2. * TMath::Cos(angleZ * TMath::Pi() / 180);
  TGeoVolume* vRO_PSU_3 = new TGeoVolume("vRO_PSU_3", new TGeoTrap(length, -angleZ, angleXY, width / 2, thickness / 2, thickness / 2, 0., width / 2, thickness / 2, thickness / 2, 0.), mCu);
  auto* t_RO_PSU_3 = new TGeoTranslation("translation_RO_PSU_3", 5.0, yPSU, zPSU);
  t_RO_PSU_3->RegisterYourself();
  auto* r_RO_PSU_3 = new TGeoRotation("rotation_RO_PSU_3", signe * 90.0, -signe * (0), 0.0);
  r_RO_PSU_3->RegisterYourself();
  auto* p_RO_PSU_3 = new TGeoCombiTrans(*t_RO_PSU_3, *r_RO_PSU_3);
  p_RO_PSU_3->RegisterYourself();

  TGeoVolume* vRO_PSU_4 = new TGeoVolume("vRO_PSU_4", new TGeoTrap(length, -angleZ, -angleXY, width / 2, thickness / 2, thickness / 2, 0., width / 2, thickness / 2, thickness / 2, 0.), mCu);
  auto* t_RO_PSU_4 = new TGeoTranslation("translation_RO_PSU_4", -5.0, yPSU, zPSU);
  t_RO_PSU_4->RegisterYourself();
  auto* r_RO_PSU_4 = new TGeoRotation("rotation_RO_PSU_4", signe * 90.0, -signe * (0), 0.0);
  r_RO_PSU_4->RegisterYourself();
  auto* p_RO_PSU_4 = new TGeoCombiTrans(*t_RO_PSU_4, *r_RO_PSU_4);
  p_RO_PSU_4->RegisterYourself();

  auto* t_ConnectorPSU_1 = new TGeoTranslation("translation_PSU_1", 6.9, -signe * (23.5), -70.44 - heigthConnector / 2);
  t_ConnectorPSU_1->RegisterYourself();
  auto* p_ConnectorPSU_1 = new TGeoCombiTrans(*t_ConnectorPSU_1, *r_ConnectorRC3_1);
  p_ConnectorPSU_1->RegisterYourself();
  auto* t_ConnectorPSU_2 = new TGeoTranslation("translation_PSU_2", -6.9, -signe * (23.5), -70.44 - heigthConnector / 2);
  t_ConnectorPSU_2->RegisterYourself();
  auto* p_ConnectorPSU_2 = new TGeoCombiTrans(*t_ConnectorPSU_2, *r_ConnectorRC3_1);
  p_ConnectorPSU_2->RegisterYourself();
  auto* t_ConnectorPSU_3 = new TGeoTranslation("translation_PSU_3", 6.9, -signe * (23.5), -74.77 + heigthConnector / 2);
  t_ConnectorPSU_3->RegisterYourself();
  auto* p_ConnectorPSU_3 = new TGeoCombiTrans(*t_ConnectorPSU_3, *r_ConnectorRC3_1);
  p_ConnectorPSU_3->RegisterYourself();
  auto* t_ConnectorPSU_4 = new TGeoTranslation("translation_PSU_4", -6.9, -signe * (23.5), -74.77 + heigthConnector / 2);
  t_ConnectorPSU_4->RegisterYourself();
  auto* p_ConnectorPSU_4 = new TGeoCombiTrans(*t_ConnectorPSU_4, *r_ConnectorRC3_1);
  p_ConnectorPSU_4->RegisterYourself();

  vRO1->SetLineColor(kGray + 2);
  vRO2->SetLineColor(kGray + 2);
  vRO3->SetLineColor(kGray + 2);
  vRO4->SetLineColor(kGray + 2);
  vRO5->SetLineColor(kGray + 2);
  vRO6->SetLineColor(kGray + 2);
  vRO7->SetLineColor(kGray + 2);
  vRO8->SetLineColor(kGray + 2);
  vRO9->SetLineColor(kGray + 2);
  vRO_D3_1->SetLineColor(kGray + 2);
  vRO_D3_2->SetLineColor(kGray + 2);
  vRO_D3_3->SetLineColor(kGray + 2);
  vRO_D3_4->SetLineColor(kGray + 2);
  vRO_D4_1->SetLineColor(kGray + 2);
  vRO_D4_2->SetLineColor(kGray + 2);
  vRO_D4_3->SetLineColor(kGray + 2);
  vRO_D4_4->SetLineColor(kGray + 2);
  vRO_D4_5->SetLineColor(kGray + 2);
  vRO_D4_6->SetLineColor(kGray + 2);
  vRO_PSU_1->SetLineColor(kGray + 2);
  vRO_PSU_2->SetLineColor(kGray + 2);
  vRO_PSU_3->SetLineColor(kGray + 2);
  vRO_PSU_4->SetLineColor(kGray + 2);

  HalfConeVolume->AddNode(vRO1, 1, t_RO1);
  HalfConeVolume->AddNode(vRO2, 1, t_RO2);
  HalfConeVolume->AddNode(vRO3, 1, t_RO3);
  HalfConeVolume->AddNode(vRO4, 1, p_RO4);
  HalfConeVolume->AddNode(vRO5, 1, p_RO5_1);
  HalfConeVolume->AddNode(vRO5, 1, p_RO5_2);
  HalfConeVolume->AddNode(vRO6, 1, p_RO6);
  HalfConeVolume->AddNode(vRO7, 1, p_RO7);
  HalfConeVolume->AddNode(vRO8, 1, p_RO8_1);
  HalfConeVolume->AddNode(vRO8, 1, p_RO8_2);
  HalfConeVolume->AddNode(vRO9, 1, p_RO9_1);
  HalfConeVolume->AddNode(vRO9, 1, p_RO9_2);
  HalfConeVolume->AddNode(vRO_D3_1, 1, p_ROD3_1);
  HalfConeVolume->AddNode(vRO_D3_2, 1, p_ROD3_2);
  HalfConeVolume->AddNode(vRO_D3_3, 1, p_ROD3_3);
  HalfConeVolume->AddNode(vRO_D3_4, 1, p_ROD3_4);
  HalfConeVolume->AddNode(vConnectorRC, 1, p_ConnectorRC3_1);
  HalfConeVolume->AddNode(vConnectorRC, 1, p_ConnectorRC3_2);
  HalfConeVolume->AddNode(vConnectorRC, 1, p_ConnectorRC3_3);
  HalfConeVolume->AddNode(vConnectorRC, 1, p_ConnectorRC3_4);

  HalfConeVolume->AddNode(vRO_D4_1, 1, p_ROD4_1);
  HalfConeVolume->AddNode(vRO_D4_2, 1, p_ROD4_2);
  HalfConeVolume->AddNode(vRO_D4_3, 1, p_ROD4_3);
  HalfConeVolume->AddNode(vRO_D4_4, 1, p_ROD4_4);
  HalfConeVolume->AddNode(vRO_D4_5, 1, p_ROD4_5);
  HalfConeVolume->AddNode(vRO_D4_6, 1, p_ROD4_6);
  HalfConeVolume->AddNode(vConnectorRC, 1, p_ConnectorRC4_1);
  HalfConeVolume->AddNode(vConnectorRC, 1, p_ConnectorRC4_2);
  HalfConeVolume->AddNode(vConnectorRC, 1, p_ConnectorRC4_3);
  HalfConeVolume->AddNode(vConnectorRC, 1, p_ConnectorRC4_4);
  HalfConeVolume->AddNode(vConnectorRC, 1, p_ConnectorRC4_5);
  HalfConeVolume->AddNode(vConnectorRC, 1, p_ConnectorRC4_6);

  HalfConeVolume->AddNode(vRO_PSU_1, 1, p_RO_PSU_1);
  HalfConeVolume->AddNode(vRO_PSU_2, 1, p_RO_PSU_2);
  HalfConeVolume->AddNode(vRO_PSU_3, 1, p_RO_PSU_3);
  HalfConeVolume->AddNode(vRO_PSU_4, 1, p_RO_PSU_4);
  HalfConeVolume->AddNode(vConnectorRC, 1, p_ConnectorPSU_1);
  HalfConeVolume->AddNode(vConnectorRC, 1, p_ConnectorPSU_2);
  HalfConeVolume->AddNode(vConnectorRC, 1, p_ConnectorPSU_3);
  HalfConeVolume->AddNode(vConnectorRC, 1, p_ConnectorPSU_4);
}

void HalfCone::makePowerCables(TGeoVolumeAssembly* HalfConeVolume, Int_t half, Int_t signe)
{
  auto* mCu = gGeoManager->GetMedium("MFT_Cu$");

  // ========= Cables for the first 3 disks ===========
  // ==================== Bottom side ===================
  Double_t xPC0 = 8.0;
  Double_t yPC0 = 0.0313;
  Double_t zPC0 = 2.0; // length
  Int_t side;
  if (signe == -1) {
    side = 1; // left
  }
  if (signe == 1) {
    side = 2; // right
  }

  TGeoVolume* vPCb0 = gGeoManager->MakeBox(Form("vPCb0_S%d", side), mCu, xPC0 / 2, yPC0 / 2, zPC0 / 2);
  auto* r_PC0 = new TGeoRotation("rotation_PC0", 0., 0., 0.);
  r_PC0->RegisterYourself();
  Double_t XPC0 = signe * 18.5;
  Double_t YPC0 = -4. + yPC0 / 2;
  Double_t ZPC0 = -72.5;
  auto* p_PC0 = new TGeoCombiTrans(XPC0, YPC0, ZPC0, r_PC0);

  //===================== first cable ===============================
  Double_t rmaxPC1 = 0.163;
  Double_t zPC1 = 5.5;
  TGeoVolume* vPCb1_1 = gGeoManager->MakeTube(Form("vPCb1_1_S%d", side), mCu, 0., rmaxPC1, zPC1 / 2);
  Double_t XPC1 = XPC0 + signe * (2 * rmaxPC1 - 0.3);
  Double_t YPC1 = YPC0;
  Double_t ZPC1 = ZPC0 + (zPC0 + zPC1) / 2;
  auto* p_PC1_1 = new TGeoCombiTrans(XPC1, YPC1, ZPC1, r_PC0);
  p_PC1_1->RegisterYourself();
  //===========
  Double_t rPC2 = 0.95;
  TGeoVolume* vPCb2_1 = gGeoManager->MakeTorus(Form("vPCb2_1_S%d", side), mCu, rPC2, 0., rmaxPC1, 0., 90.);
  auto* r_PC2 = new TGeoRotation("rotation_PC2", 90., -90., -90.);
  r_PC2->RegisterYourself();
  Double_t XPC2 = XPC1;
  Double_t YPC2 = YPC1 - rPC2;
  Double_t ZPC2 = ZPC1 + zPC1 / 2;
  auto* p_PC2_1 = new TGeoCombiTrans(XPC2, YPC2, ZPC2, r_PC2);
  p_PC2_1->RegisterYourself();
  //===========
  Double_t zPC3 = 2.6;
  TGeoVolume* vPCb3_1 = gGeoManager->MakeTube(Form("vPCb3_S%d", side), mCu, 0., rmaxPC1, zPC3 / 2);
  auto* r_PC3 = new TGeoRotation("rotation_PC3", 0., 90., 0.);
  r_PC3->RegisterYourself();
  Double_t XPC3 = XPC2;
  Double_t YPC3 = YPC2 - zPC3 / 2;
  Double_t ZPC3 = ZPC2 + rPC2;
  auto* p_PC3_1 = new TGeoCombiTrans(XPC3, YPC3, ZPC3, r_PC3);
  p_PC3_1->RegisterYourself();
  //===========
  Double_t rPC4 = 0.95;
  TGeoVolume* vPCb4_1 = gGeoManager->MakeTorus(Form("vPCb4_1_H%d", side), mCu, rPC2, 0., rmaxPC1, 0., 90.);
  auto* r_PC4 = new TGeoRotation("rotation_PC4", 90., -90. - signe * 45., 90.);
  r_PC4->RegisterYourself();
  Double_t XPC4 = XPC3 - signe * rPC4 * TMath::Sin(45. * TMath::Pi() / 180.);
  Double_t YPC4 = YPC3 - zPC3 / 2;
  Double_t ZPC4 = ZPC3 + rPC4 * TMath::Cos(45. * TMath::Pi() / 180.);
  auto* p_PC4_1 = new TGeoCombiTrans(XPC4, YPC4, ZPC4, r_PC4);
  p_PC4_1->RegisterYourself();
  //===========
  Double_t zPC5 = 8.1;
  TGeoVolume* vPCb5_1 = gGeoManager->MakeTube(Form("vPCb5_S%d", side), mCu, 0., rmaxPC1, zPC5 / 2);
  auto* r_PC5 = new TGeoRotation("rotation_PC5", 90., -signe * 45., 0);
  r_PC5->RegisterYourself();
  Double_t XPC5 = XPC4 - signe * zPC5 / 2 * TMath::Sin(45. * TMath::Pi() / 180.);
  Double_t YPC5 = YPC4 - rPC4;
  Double_t ZPC5 = ZPC4 + zPC5 / 2 * TMath::Cos(45. * TMath::Pi() / 180.);
  auto* p_PC5_1 = new TGeoCombiTrans(XPC5, YPC5, ZPC5, r_PC5);
  p_PC5_1->RegisterYourself();
  //===========
  Double_t rPC6 = 0.95;
  TGeoVolume* vPCb6_1 = gGeoManager->MakeTorus(Form("vPCb6_1_H%d", side), mCu, rPC2, 0.0, rmaxPC1, 45., 45.);
  auto* r_PC6 = new TGeoRotation("rotation_PC6", 0., -signe * 90., signe * 90.);
  r_PC6->RegisterYourself();
  Double_t XPC6 = XPC5 - signe * (zPC5 / 2 * TMath::Sin(45 * TMath::Pi() / 180.) - rPC6 * TMath::Cos(45. * TMath::Pi() / 180.));
  Double_t YPC6 = YPC5;
  Double_t ZPC6 = ZPC5 + zPC5 / 2 * TMath::Cos(45. * TMath::Pi() / 180.) + rPC6 * TMath::Sin(45. * TMath::Pi() / 180.);
  auto* p_PC6_1 = new TGeoCombiTrans(XPC6, YPC6, ZPC6, r_PC6);
  p_PC6_1->RegisterYourself();
  //===========
  Double_t zPC7 = 13.0;
  TGeoVolume* vPCb7_1 = gGeoManager->MakeTube(Form("vPCb7_1_H%d", side), mCu, 0., rmaxPC1, zPC7 / 2);
  auto* r_PC7 = new TGeoRotation("rotation_PC7", 0., 0., 0.);
  r_PC7->RegisterYourself();
  Double_t XPC7 = XPC6 - signe * rPC6;
  Double_t YPC7 = YPC6;
  Double_t ZPC7 = ZPC6 + zPC7 / 2;
  auto* p_PC7_1 = new TGeoCombiTrans(XPC7, YPC7, ZPC7, r_PC7);
  p_PC7_1->RegisterYourself();

  //===================== second cable ==============================
  zPC1 = 5.5;
  TGeoVolume* vPCb1_2 = gGeoManager->MakeTube(Form("vPCb1_2_S%d", side), mCu, 0., rmaxPC1, zPC1 / 2);
  XPC1 = XPC0 + signe * (8 * rmaxPC1 - 0.3);
  auto* p_PC1_2 = new TGeoCombiTrans(XPC1, YPC1, ZPC1, r_PC0);
  p_PC1_2->RegisterYourself();
  //===========
  TGeoVolume* vPCb2_2 = gGeoManager->MakeTorus(Form("vPCb2_2_S%d", side), mCu, rPC2, 0.0, rmaxPC1, 0., 90.);
  XPC2 = XPC1;
  auto* p_PC2_2 = new TGeoCombiTrans(XPC2, YPC2, ZPC2, r_PC2);
  p_PC2_2->RegisterYourself();
  //===========
  TGeoVolume* vPCb3_2 = gGeoManager->MakeTube(Form("vPCb3_2_S%d", side), mCu, 0., rmaxPC1, zPC3 / 2);
  XPC3 = XPC2;
  YPC3 = YPC2 - zPC3 / 2;
  ZPC3 = ZPC2 + rPC2;
  auto* p_PC3_2 = new TGeoCombiTrans(XPC3, YPC3, ZPC3, r_PC3);
  p_PC3_2->RegisterYourself();
  //===========
  TGeoVolume* vPCb4_2 = gGeoManager->MakeTorus(Form("vPCb4_2_S%d", side), mCu, rPC2, 0., rmaxPC1, 0., 90.);
  XPC4 = XPC3 - signe * rPC4 * TMath::Sin(45. * TMath::Pi() / 180.);
  YPC4 = YPC3 - zPC3 / 2;
  ZPC4 = ZPC3 + rPC4 * TMath::Cos(45. * TMath::Pi() / 180.);
  auto* p_PC4_2 = new TGeoCombiTrans(XPC4, YPC4, ZPC4, r_PC4);
  p_PC4_2->RegisterYourself();
  //===========
  zPC5 = 8.1;
  TGeoVolume* vPCb5_2 = gGeoManager->MakeTube(Form("vPCb5_2_S%d", side), mCu, 0., rmaxPC1, zPC5 / 2);
  XPC5 = XPC4 - signe * zPC5 / 2 * TMath::Sin(45. * TMath::Pi() / 180.);
  YPC5 = YPC4 - rPC4;
  ZPC5 = ZPC4 + zPC5 / 2 * TMath::Cos(45. * TMath::Pi() / 180.);
  auto* p_PC5_2 = new TGeoCombiTrans(XPC5, YPC5, ZPC5, r_PC5);
  p_PC5_2->RegisterYourself();
  //===========
  TGeoVolume* vPCb6_2 = gGeoManager->MakeTorus(Form("vPCb6_2_S%d", side), mCu, rPC2, 0.0, rmaxPC1, 45., 45.);
  XPC6 = XPC5 - signe * (zPC5 / 2 * TMath::Sin(45. * TMath::Pi() / 180.) - rPC6 * TMath::Cos(45. * TMath::Pi() / 180.));
  YPC6 = YPC5;
  ZPC6 = ZPC5 + zPC5 / 2 * TMath::Cos(45. * TMath::Pi() / 180.) + rPC6 * TMath::Sin(45. * TMath::Pi() / 180.);
  auto* p_PC6_2 = new TGeoCombiTrans(XPC6, YPC6, ZPC6, r_PC6);
  p_PC6_2->RegisterYourself();
  //===========
  zPC7 = 9.8;
  TGeoVolume* vPCb7_2 = gGeoManager->MakeTube(Form("vPCb7_2_S%d", side), mCu, 0., rmaxPC1, zPC7 / 2);
  XPC7 = XPC6 - signe * rPC6;
  YPC7 = YPC6;
  ZPC7 = ZPC6 + zPC7 / 2;
  auto* p_PC7_2 = new TGeoCombiTrans(XPC7, YPC7, ZPC7, r_PC7);
  p_PC7_2->RegisterYourself();

  //===================== third cable ==============================
  zPC1 = 5.5;
  TGeoVolume* vPCb1_3 = gGeoManager->MakeTube(Form("vPCb1_3_S%d", side), mCu, 0., rmaxPC1, zPC1 / 2);
  XPC1 = XPC0 + signe * (14 * rmaxPC1 - 0.3);
  auto* p_PC1_3 = new TGeoCombiTrans(XPC1, YPC1, ZPC1, r_PC0);
  p_PC1_3->RegisterYourself();
  //===========
  TGeoVolume* vPCb2_3 = gGeoManager->MakeTorus(Form("vPCb2_3_S%d", side), mCu, rPC2, 0., rmaxPC1, 0., 90.);
  XPC2 = XPC1;
  auto* p_PC2_3 = new TGeoCombiTrans(XPC2, YPC2, ZPC2, r_PC2);
  p_PC2_3->RegisterYourself();
  //===========
  zPC3 = 2.6;
  TGeoVolume* vPCb3_3 = gGeoManager->MakeTube(Form("vPCb3_3_S%d", side), mCu, 0., rmaxPC1, zPC3 / 2);
  XPC3 = XPC2;
  YPC3 = YPC2 - zPC3 / 2;
  ZPC3 = ZPC2 + rPC2;
  auto* p_PC3_3 = new TGeoCombiTrans(XPC3, YPC3, ZPC3, r_PC3);
  p_PC3_3->RegisterYourself();
  //===========
  TGeoVolume* vPCb4_3 = gGeoManager->MakeTorus(Form("vPCb4_3_S%d", side), mCu, rPC2, 0., rmaxPC1, 0., 90.);
  XPC4 = XPC3 - signe * rPC4 * TMath::Sin(45. * TMath::Pi() / 180.);
  YPC4 = YPC3 - zPC3 / 2;
  ZPC4 = ZPC3 + rPC4 * TMath::Cos(45. * TMath::Pi() / 180.);
  auto* p_PC4_3 = new TGeoCombiTrans(XPC4, YPC4, ZPC4, r_PC4);
  p_PC4_3->RegisterYourself();
  //===========
  zPC5 = 8.1 + 0.3;
  TGeoVolume* vPCb5_3 = gGeoManager->MakeTube(Form("vPCb5_3_S%d", side), mCu, 0., rmaxPC1, zPC5 / 2);
  XPC5 = XPC4 - signe * zPC5 / 2 * TMath::Sin(45. * TMath::Pi() / 180.);
  YPC5 = YPC4 - rPC4;
  ZPC5 = ZPC4 + zPC5 / 2 * TMath::Cos(45. * TMath::Pi() / 180.);
  auto* p_PC5_3 = new TGeoCombiTrans(XPC5, YPC5, ZPC5, r_PC5);
  p_PC5_3->RegisterYourself();
  //===========
  TGeoVolume* vPCb6_3 = gGeoManager->MakeTorus(Form("vPCb6_3_S%d", side), mCu, rPC2, 0.0, rmaxPC1, 45., 45.);
  XPC6 = XPC5 - signe * (zPC5 / 2 * TMath::Sin(45. * TMath::Pi() / 180.) - rPC6 * TMath::Cos(45. * TMath::Pi() / 180.));
  YPC6 = YPC5;
  ZPC6 = ZPC5 + zPC5 / 2 * TMath::Cos(45. * TMath::Pi() / 180.) + rPC6 * TMath::Sin(45. * TMath::Pi() / 180.);
  auto* p_PC6_3 = new TGeoCombiTrans(XPC6, YPC6, ZPC6, r_PC6);
  p_PC6_3->RegisterYourself();
  //===========
  zPC7 = 5.8;
  TGeoVolume* vPCb7_3 = gGeoManager->MakeTube(Form("vPCb7_3_S%d", side), mCu, 0., rmaxPC1, zPC7 / 2);
  XPC7 = XPC6 - signe * rPC6;
  YPC7 = YPC6;
  ZPC7 = ZPC6 + zPC7 / 2;
  auto* p_PC7_3 = new TGeoCombiTrans(XPC7, YPC7, ZPC7, r_PC7);
  p_PC7_3->RegisterYourself();
  //==========================================================

  vPCb0->SetLineColor(kBlue);
  vPCb1_1->SetLineColor(kBlue);
  vPCb1_2->SetLineColor(kBlue);
  vPCb1_3->SetLineColor(kBlue);
  vPCb2_1->SetLineColor(kBlue);
  vPCb2_2->SetLineColor(kBlue);
  vPCb2_3->SetLineColor(kBlue);
  vPCb3_1->SetLineColor(kBlue);
  vPCb3_2->SetLineColor(kBlue);
  vPCb3_3->SetLineColor(kBlue);
  vPCb4_1->SetLineColor(kBlue);
  vPCb4_2->SetLineColor(kBlue);
  vPCb4_3->SetLineColor(kBlue);
  vPCb5_1->SetLineColor(kBlue);
  vPCb5_2->SetLineColor(kBlue);
  vPCb5_3->SetLineColor(kBlue);
  vPCb6_1->SetLineColor(kBlue);
  vPCb6_2->SetLineColor(kBlue);
  vPCb6_3->SetLineColor(kBlue);
  vPCb7_1->SetLineColor(kBlue);
  vPCb7_2->SetLineColor(kBlue);
  vPCb7_3->SetLineColor(kBlue);
  HalfConeVolume->AddNode(vPCb0, 1, p_PC0);
  HalfConeVolume->AddNode(vPCb1_1, 1, p_PC1_1);
  HalfConeVolume->AddNode(vPCb1_2, 1, p_PC1_2);
  HalfConeVolume->AddNode(vPCb1_3, 1, p_PC1_3);
  HalfConeVolume->AddNode(vPCb2_1, 1, p_PC2_1);
  HalfConeVolume->AddNode(vPCb2_2, 1, p_PC2_2);
  HalfConeVolume->AddNode(vPCb2_3, 1, p_PC2_3);
  HalfConeVolume->AddNode(vPCb3_1, 1, p_PC3_1);
  HalfConeVolume->AddNode(vPCb3_2, 1, p_PC3_2);
  HalfConeVolume->AddNode(vPCb3_3, 1, p_PC3_3);
  HalfConeVolume->AddNode(vPCb4_1, 1, p_PC4_1);
  HalfConeVolume->AddNode(vPCb4_2, 1, p_PC4_2);
  HalfConeVolume->AddNode(vPCb4_3, 1, p_PC4_3);
  HalfConeVolume->AddNode(vPCb5_1, 1, p_PC5_1);
  HalfConeVolume->AddNode(vPCb5_2, 1, p_PC5_2);
  HalfConeVolume->AddNode(vPCb5_3, 1, p_PC5_3);
  HalfConeVolume->AddNode(vPCb6_1, 1, p_PC6_1);
  HalfConeVolume->AddNode(vPCb6_2, 1, p_PC6_2);
  HalfConeVolume->AddNode(vPCb6_3, 1, p_PC6_3);
  HalfConeVolume->AddNode(vPCb7_1, 1, p_PC7_1);
  HalfConeVolume->AddNode(vPCb7_2, 1, p_PC7_2);
  HalfConeVolume->AddNode(vPCb7_3, 1, p_PC7_3);

  // ==================== Top side ===================
  TGeoVolume* vPCt0 = gGeoManager->MakeBox(Form("vPCt0_S%d", side), mCu, xPC0 / 2, yPC0 / 2, zPC0 / 2);
  p_PC0 = new TGeoCombiTrans(XPC0, -YPC0, ZPC0, r_PC0);

  //===================== first cable ===============================
  zPC1 = 5.5;
  TGeoVolume* vPCt1_1 = gGeoManager->MakeTube(Form("vPCt1_1_S%d", side), mCu, 0.0, rmaxPC1, zPC1 / 2);
  XPC1 = XPC0 + signe * (2 * rmaxPC1 - 0.3);
  p_PC1_1 = new TGeoCombiTrans(XPC1, -YPC1, ZPC1, r_PC0);
  p_PC1_1->RegisterYourself();
  //==========
  TGeoVolume* vPCt2_1 = gGeoManager->MakeTorus(Form("vPCt2_1_S%d", side), mCu, rPC2, 0., rmaxPC1, 0., 90.);
  r_PC2 = new TGeoRotation("rotation_PC2", 90., 90., 90.);
  r_PC2->RegisterYourself();
  XPC2 = XPC1;
  YPC2 = YPC1 - rPC2;
  ZPC2 = ZPC1 + zPC1 / 2;
  p_PC2_1 = new TGeoCombiTrans(XPC2, -YPC2, ZPC2, r_PC2);
  p_PC2_1->RegisterYourself();
  //===========
  zPC3 = 2.6;
  TGeoVolume* vPCt3_1 = gGeoManager->MakeTube(Form("vPCt3_S%d", side), mCu, 0., rmaxPC1, zPC3 / 2);
  r_PC3->RegisterYourself();
  XPC3 = XPC2;
  YPC3 = YPC2 - zPC3 / 2;
  ZPC3 = ZPC2 + rPC2;
  p_PC3_1 = new TGeoCombiTrans(XPC3, -YPC3, ZPC3, r_PC3);
  p_PC3_1->RegisterYourself();
  //===========
  rPC4 = 0.95;
  TGeoVolume* vPCt4_1 = gGeoManager->MakeTorus(Form("vPCt4_1_H%d", side), mCu, rPC2, 0., rmaxPC1, 0., 90.);
  r_PC4 = new TGeoRotation("rotation_PC4", 90., 90. - signe * 45., 180. + 90.);
  r_PC4->RegisterYourself();
  XPC4 = XPC3 - signe * rPC4 * TMath::Sin(45. * TMath::Pi() / 180.);
  YPC4 = YPC3 - zPC3 / 2;
  ZPC4 = ZPC3 + rPC4 * TMath::Cos(45. * TMath::Pi() / 180.);
  p_PC4_1 = new TGeoCombiTrans(XPC4, -YPC4, ZPC4, r_PC4);
  p_PC4_1->RegisterYourself();
  //===========
  zPC5 = 8.1;
  TGeoVolume* vPCt5_1 = gGeoManager->MakeTube(Form("vPCt5_S%d", side), mCu, 0.0, rmaxPC1, zPC5 / 2);
  r_PC5 = new TGeoRotation("rotation_PC5", 90., 180.0 - signe * 45., 180);
  r_PC5->RegisterYourself();
  XPC5 = XPC4 - signe * zPC5 / 2 * TMath::Sin(45. * TMath::Pi() / 180.);
  YPC5 = YPC4 - rPC4;
  ZPC5 = ZPC4 + zPC5 / 2 * TMath::Cos(45. * TMath::Pi() / 180.);
  p_PC5_1 = new TGeoCombiTrans(XPC5, -YPC5, ZPC5, r_PC5);
  p_PC5_1->RegisterYourself();
  //===========
  rPC6 = 0.95;
  TGeoVolume* vPCt6_1 = gGeoManager->MakeTorus(Form("vPCt6_1_H%d", side), mCu, rPC2, 0.0, rmaxPC1, 45., 45.);
  XPC6 = XPC5 - signe * (zPC5 / 2 * TMath::Sin(45.0 * TMath::Pi() / 180.) - rPC6 * TMath::Cos(45 * TMath::Pi() / 180.));
  YPC6 = YPC5;
  ZPC6 = ZPC5 + zPC5 / 2 * TMath::Cos(45. * TMath::Pi() / 180.) + rPC6 * TMath::Sin(45 * TMath::Pi() / 180.);
  p_PC6_1 = new TGeoCombiTrans(XPC6, -YPC6, ZPC6, r_PC6);
  p_PC6_1->RegisterYourself();
  //===========
  zPC7 = 13.0;
  TGeoVolume* vPCt7_1 = gGeoManager->MakeTube(Form("vPCt7_1_H%d", side), mCu, 0., rmaxPC1, zPC7 / 2);
  XPC7 = XPC6 - signe * rPC6;
  YPC7 = YPC6;
  ZPC7 = ZPC6 + zPC7 / 2;
  p_PC7_1 = new TGeoCombiTrans(XPC7, -YPC7, ZPC7, r_PC7);
  p_PC7_1->RegisterYourself();

  //===================== second cable ==============================
  zPC1 = 5.5;
  TGeoVolume* vPCt1_2 = gGeoManager->MakeTube(Form("vPCt1_2_S%d", side), mCu, 0., rmaxPC1, zPC1 / 2);
  XPC1 = XPC0 + signe * (8 * rmaxPC1 - 0.3);
  p_PC1_2 = new TGeoCombiTrans(XPC1, -YPC1, ZPC1, r_PC0);
  p_PC1_2->RegisterYourself();
  //===========
  TGeoVolume* vPCt2_2 = gGeoManager->MakeTorus(Form("TorusPCt2_2_S%d", side), mCu, rPC2, 0.0, rmaxPC1, 0., 90.);
  XPC2 = XPC1;
  p_PC2_2 = new TGeoCombiTrans(XPC2, -YPC2, ZPC2, r_PC2);
  p_PC2_2->RegisterYourself();
  //===========
  TGeoVolume* vPCt3_2 = gGeoManager->MakeTube(Form("vPCt3_2_S%d", side), mCu, 0., rmaxPC1, zPC3 / 2);
  XPC3 = XPC2;
  p_PC3_2 = new TGeoCombiTrans(XPC3, -YPC3, ZPC3, r_PC3);
  p_PC3_2->RegisterYourself();
  //===========
  TGeoVolume* vPCt4_2 = gGeoManager->MakeTorus(Form("TorusPCt4_2_S%d", side), mCu, rPC2, 0.0, rmaxPC1, 0., 90.);
  XPC4 = XPC3 - signe * rPC4 * TMath::Sin(45. * TMath::Pi() / 180.);
  YPC4 = YPC3 - zPC3 / 2;
  ZPC4 = ZPC3 + rPC4 * TMath::Cos(45 * TMath::Pi() / 180.);
  p_PC4_2 = new TGeoCombiTrans(XPC4, -YPC4, ZPC4, r_PC4);
  p_PC4_2->RegisterYourself();
  //===========
  zPC5 = 8.1;
  TGeoVolume* vPCt5_2 = gGeoManager->MakeTube(Form("vPCt5_2_S%d", side), mCu, 0., rmaxPC1, zPC5 / 2);
  XPC5 = XPC4 - signe * zPC5 / 2 * TMath::Sin(45.0 * TMath::Pi() / 180.);
  YPC5 = YPC4 - rPC4;
  ZPC5 = ZPC4 + zPC5 / 2 * TMath::Cos(45.0 * TMath::Pi() / 180.);
  p_PC5_2 = new TGeoCombiTrans(XPC5, -YPC5, ZPC5, r_PC5);
  p_PC5_2->RegisterYourself();
  //===========
  TGeoVolume* vPCt6_2 = gGeoManager->MakeTorus(Form("TorusPCt6_2_S%d", side), mCu, rPC2, 0.0, rmaxPC1, 45., 45.);
  XPC6 = XPC5 - signe * (zPC5 / 2 * TMath::Sin(45.0 * TMath::Pi() / 180.) - rPC6 * TMath::Cos(45. * TMath::Pi() / 180.));
  YPC6 = YPC5;
  ZPC6 = ZPC5 + zPC5 / 2 * TMath::Cos(45. * TMath::Pi() / 180.) + rPC6 * TMath::Sin(45. * TMath::Pi() / 180.);
  p_PC6_2 = new TGeoCombiTrans(XPC6, -YPC6, ZPC6, r_PC6);
  p_PC6_2->RegisterYourself();
  //===========
  zPC7 = 9.8;
  TGeoVolume* vPCt7_2 = gGeoManager->MakeTube(Form("vPCt7_2_S%d", side), mCu, 0.0, rmaxPC1, zPC7 / 2);
  XPC7 = XPC6 - signe * rPC6;
  YPC7 = YPC6;
  ZPC7 = ZPC6 + zPC7 / 2;
  p_PC7_2 = new TGeoCombiTrans(XPC7, -YPC7, ZPC7, r_PC7);
  p_PC7_2->RegisterYourself();

  //===================== third cable ==============================
  zPC1 = 5.5;
  TGeoVolume* vPCt1_3 = gGeoManager->MakeTube(Form("vPCt1_3_S%d", side), mCu, 0., rmaxPC1, zPC1 / 2);
  XPC1 = XPC0 + signe * (14 * rmaxPC1 - 0.3);
  p_PC1_3 = new TGeoCombiTrans(XPC1, -YPC1, ZPC1, r_PC0);
  p_PC1_3->RegisterYourself();
  //===========
  zPC3 = 2.6;
  TGeoVolume* vPCt2_3 = gGeoManager->MakeTorus(Form("TorusPCt2_3_S%d", side), mCu, rPC2, 0., rmaxPC1, 0., 90.);
  XPC2 = XPC1;
  p_PC2_3 = new TGeoCombiTrans(XPC2, -YPC2, ZPC2, r_PC2);
  p_PC2_3->RegisterYourself();
  //===========
  TGeoVolume* vPCt3_3 = gGeoManager->MakeTube(Form("vPCt3_3_S%d", side), mCu, 0., rmaxPC1, zPC3 / 2);
  XPC3 = XPC2;
  YPC3 = YPC2 - zPC3 / 2;
  ZPC3 = ZPC2 + rPC2;
  p_PC3_3 = new TGeoCombiTrans(XPC3, -YPC3, ZPC3, r_PC3);
  p_PC3_3->RegisterYourself();
  //===========
  TGeoVolume* vPCt4_3 = gGeoManager->MakeTorus(Form("TorusPCt4_3_S%d", side), mCu, rPC2, 0., rmaxPC1, 0., 90.);
  XPC4 = XPC3 - signe * rPC4 * TMath::Sin(45. * TMath::Pi() / 180.);
  YPC4 = YPC3 - zPC3 / 2;
  ZPC4 = ZPC3 + rPC4 * TMath::Cos(45.0 * TMath::Pi() / 180.);
  p_PC4_3 = new TGeoCombiTrans(XPC4, -YPC4, ZPC4, r_PC4);
  p_PC4_3->RegisterYourself();
  //===========
  zPC5 = 8.1 + 0.3;
  TGeoVolume* vPCt5_3 = gGeoManager->MakeTube(Form("vPCt5_3_S%d", side), mCu, 0., rmaxPC1, zPC5 / 2);
  XPC5 = XPC4 - signe * zPC5 / 2 * TMath::Sin(45. * TMath::Pi() / 180.);
  YPC5 = YPC4 - rPC4;
  ZPC5 = ZPC4 + zPC5 / 2 * TMath::Cos(45 * TMath::Pi() / 180.);
  p_PC5_3 = new TGeoCombiTrans(XPC5, -YPC5, ZPC5, r_PC5);
  p_PC5_3->RegisterYourself();
  //===========
  TGeoVolume* vPCt6_3 = gGeoManager->MakeTorus(Form("TorusPCt6_3_S%d", side), mCu, rPC2, 0., rmaxPC1, 45., 45.);
  XPC6 = XPC5 - signe * (zPC5 / 2 * TMath::Sin(45.0 * TMath::Pi() / 180.) - rPC6 * TMath::Cos(45. * TMath::Pi() / 180.));
  YPC6 = YPC5;
  ZPC6 = ZPC5 + zPC5 / 2 * TMath::Cos(45.0 * TMath::Pi() / 180.) + rPC6 * TMath::Sin(45. * TMath::Pi() / 180.);
  p_PC6_3 = new TGeoCombiTrans(XPC6, -YPC6, ZPC6, r_PC6);
  p_PC6_3->RegisterYourself();
  //===========
  zPC7 = 5.8;
  TGeoVolume* vPCt7_3 = gGeoManager->MakeTube(Form("vPCt7_3_S%d", side), mCu, 0., rmaxPC1, zPC7 / 2);
  XPC7 = XPC6 - signe * rPC6;
  YPC7 = YPC6;
  ZPC7 = ZPC6 + zPC7 / 2;
  p_PC7_3 = new TGeoCombiTrans(XPC7, -YPC7, ZPC7, r_PC7);
  p_PC7_3->RegisterYourself();
  //==========================================================

  vPCt0->SetLineColor(kBlue);
  vPCt1_1->SetLineColor(kBlue);
  vPCt2_1->SetLineColor(kBlue);
  vPCt3_1->SetLineColor(kBlue);
  vPCt4_1->SetLineColor(kBlue);
  vPCt5_1->SetLineColor(kBlue);
  vPCt6_1->SetLineColor(kBlue);
  vPCt7_1->SetLineColor(kBlue);
  vPCt1_2->SetLineColor(kBlue);
  vPCt2_2->SetLineColor(kBlue);
  vPCt3_2->SetLineColor(kBlue);
  vPCt4_2->SetLineColor(kBlue);
  vPCt5_2->SetLineColor(kBlue);
  vPCt6_2->SetLineColor(kBlue);
  vPCt7_2->SetLineColor(kBlue);
  vPCt1_3->SetLineColor(kBlue);
  vPCt2_3->SetLineColor(kBlue);
  vPCt3_3->SetLineColor(kBlue);
  vPCt4_3->SetLineColor(kBlue);
  vPCt5_3->SetLineColor(kBlue);
  vPCt6_3->SetLineColor(kBlue);
  vPCt7_3->SetLineColor(kBlue);
  HalfConeVolume->AddNode(vPCt0, 1, p_PC0);
  HalfConeVolume->AddNode(vPCt1_1, 1, p_PC1_1);
  HalfConeVolume->AddNode(vPCt2_1, 1, p_PC2_1);
  HalfConeVolume->AddNode(vPCt3_1, 1, p_PC3_1);
  HalfConeVolume->AddNode(vPCt4_1, 1, p_PC4_1);
  HalfConeVolume->AddNode(vPCt5_1, 1, p_PC5_1);
  HalfConeVolume->AddNode(vPCt6_1, 1, p_PC6_1);
  HalfConeVolume->AddNode(vPCt7_1, 1, p_PC7_1);
  HalfConeVolume->AddNode(vPCt1_2, 1, p_PC1_2);
  HalfConeVolume->AddNode(vPCt2_2, 1, p_PC2_2);
  HalfConeVolume->AddNode(vPCt3_2, 1, p_PC3_2);
  HalfConeVolume->AddNode(vPCt4_2, 1, p_PC4_2);
  HalfConeVolume->AddNode(vPCt5_2, 1, p_PC5_2);
  HalfConeVolume->AddNode(vPCt6_2, 1, p_PC6_2);
  HalfConeVolume->AddNode(vPCt7_2, 1, p_PC7_2);
  HalfConeVolume->AddNode(vPCt1_3, 1, p_PC1_3);
  HalfConeVolume->AddNode(vPCt2_3, 1, p_PC2_3);
  HalfConeVolume->AddNode(vPCt3_3, 1, p_PC3_3);
  HalfConeVolume->AddNode(vPCt4_3, 1, p_PC4_3);
  HalfConeVolume->AddNode(vPCt5_3, 1, p_PC5_3);
  HalfConeVolume->AddNode(vPCt6_3, 1, p_PC6_3);
  HalfConeVolume->AddNode(vPCt7_3, 1, p_PC7_3);

  // ==================================================
  // ========= Cables for the 2 latest disks ==========
  // ==================================================
  // PSU --> DISK3
  // one diagonal
  Double_t xPC_0 = 1.0;    // width
  Double_t yPC_0 = 0.0836; // thickness
  Double_t zPC_0 = 5.0;    // length
  TGeoVolume* vPC_0 = gGeoManager->MakeBox(Form("vPC_0_S%d", side), mCu, xPC_0 / 2, yPC_0 / 2, zPC_0 / 2);
  auto* r_PC_0 = new TGeoRotation("rotation_PC_0", -45., 0., 0.);
  r_PC_0->RegisterYourself();
  Double_t XPC_0 = signe * 19.7;
  Double_t YPC_0 = signe * 20.0;
  Double_t ZPC_0 = -71.0;
  auto* p_PC_0 = new TGeoCombiTrans(XPC_0, YPC_0, ZPC_0, r_PC_0);
  //===========
  Double_t xPC_1 = 4.8; // length
  Double_t yPC_1 = yPC_0;
  Double_t zPC_1 = xPC_0;
  TGeoVolume* vPC_1 = gGeoManager->MakeBox(Form("vPC_1_S%d", side), mCu, xPC_1 / 2, yPC_1 / 2, zPC_1 / 2);
  auto* r_PC_1 = new TGeoRotation("rotation_PC_1", 45., 90., 0.);
  r_PC_1->RegisterYourself();
  Double_t XPC_1 = XPC_0 - signe * xPC_1 / 2 * TMath::Cos(45. * TMath::Pi() / 180.);
  Double_t YPC_1 = YPC_0 - signe * xPC_1 / 2 * TMath::Sin(45. * TMath::Pi() / 180.);
  Double_t ZPC_1 = ZPC_0 + (zPC_0 + yPC_1) / 2;
  auto* p_PC_1 = new TGeoCombiTrans(XPC_1, YPC_1, ZPC_1, r_PC_1);
  //===========
  vPC_0->SetLineColor(kBlue);
  vPC_1->SetLineColor(kBlue);
  HalfConeVolume->AddNode(vPC_0, 1, p_PC_0);
  HalfConeVolume->AddNode(vPC_1, 1, p_PC_1);

  // other diagonal
  TGeoVolume* vPC_P0 = gGeoManager->MakeBox(Form("vPC_P0_S%d", side), mCu, xPC_0 / 2, yPC_0 / 2, zPC_0 / 2);
  auto* r_PC_P0 = new TGeoRotation("rotation_PC_P0", 45., 0., 0.);
  r_PC_P0->RegisterYourself();
  Double_t XPC_P0 = -signe * 19.7;
  Double_t YPC_P0 = signe * 20.0;
  Double_t ZPC_P0 = -71.0;
  auto* p_PC_P0 = new TGeoCombiTrans(XPC_P0, YPC_P0, ZPC_P0, r_PC_P0);
  //===========
  Double_t xPC_P1 = xPC_0;
  Double_t yPC_P1 = yPC_0;
  Double_t zPC_P1 = 4.8;
  TGeoVolume* vPC_P1 = gGeoManager->MakeBox(Form("vPC_P1_S%d", side), mCu, xPC_P1 / 2, yPC_P1 / 2, zPC_P1 / 2);
  auto* r_PC_P1 = new TGeoRotation("rotation_PC_P1", 45., 90., 0.);
  r_PC_P1->RegisterYourself();
  Double_t XPC_P1 = XPC_P0 + signe * zPC_P1 / 2 * TMath::Cos(45 * TMath::Pi() / 180);
  Double_t YPC_P1 = YPC_P0 - signe * zPC_P1 / 2 * TMath::Sin(45 * TMath::Pi() / 180);
  Double_t ZPC_P1 = ZPC_P0 + (zPC_0 + yPC_P1) / 2;
  auto* p_PC_P1 = new TGeoCombiTrans(XPC_P1, YPC_P1, ZPC_P1, r_PC_P1);
  //===========
  vPC_P0->SetLineColor(kBlue);
  vPC_P1->SetLineColor(kBlue);
  HalfConeVolume->AddNode(vPC_P0, 1, p_PC_P0);
  HalfConeVolume->AddNode(vPC_P1, 1, p_PC_P1);

  // PSU --> DISK4
  // one diagonal
  TGeoVolume* vPC_2 = gGeoManager->MakeBox(Form("vPC_2_S%d", side), mCu, xPC_0 / 2, yPC_0 / 2, zPC_0 / 2);
  auto* r_PC_2 = new TGeoRotation("rotation_PC_2", -39., 0., 0.);
  r_PC_2->RegisterYourself();
  Double_t XPC_2 = signe * 17.2;
  Double_t YPC_2 = signe * 22.0;
  Double_t ZPC_2 = -74.0;
  auto* p_PC_2 = new TGeoCombiTrans(XPC_2, YPC_2, ZPC_2, r_PC_2);
  //===========
  Double_t xPC_3 = 4.8; // length
  Double_t yPC_3 = yPC_0;
  Double_t zPC_3 = xPC_0;
  TGeoVolume* vPC_3 = gGeoManager->MakeBox(Form("vPC_3_S%d", side), mCu, xPC_3 / 2, yPC_3 / 2, zPC_3 / 2);
  auto* r_PC_3 = new TGeoRotation("rotation_PC_3", 90. - 39., 90., 0.);
  r_PC_3->RegisterYourself();
  Double_t XPC_3 = XPC_2 - signe * xPC_3 / 2 * TMath::Cos((90. - 39.) * TMath::Pi() / 180.);
  Double_t YPC_3 = YPC_2 - signe * xPC_3 / 2 * TMath::Sin((90. - 39.) * TMath::Pi() / 180.);
  Double_t ZPC_3 = ZPC_2 - (zPC_0 + yPC_3) / 2;
  auto* p_PC_3 = new TGeoCombiTrans(XPC_3, YPC_3, ZPC_3, r_PC_3);
  //===========
  vPC_2->SetLineColor(kBlue);
  vPC_3->SetLineColor(kBlue);
  HalfConeVolume->AddNode(vPC_2, 1, p_PC_2);
  HalfConeVolume->AddNode(vPC_3, 1, p_PC_3);

  // other diagonal
  TGeoVolume* vPC_P2 = gGeoManager->MakeBox(Form("vPC_P2_S%d", side), mCu, xPC_0 / 2, yPC_0 / 2, zPC_0 / 2);
  auto* r_PC_P2 = new TGeoRotation("rotation_PC_P2", 39., 0., 0.);
  r_PC_P2->RegisterYourself();
  Double_t XPC_P2 = -signe * 17.2;
  Double_t YPC_P2 = signe * 22.0;
  Double_t ZPC_P2 = -74.0;
  auto* p_PC_P2 = new TGeoCombiTrans(XPC_P2, YPC_P2, ZPC_P2, r_PC_P2);
  //===========
  Double_t xPC_P3 = xPC_0;
  Double_t yPC_P3 = yPC_0;
  Double_t zPC_P3 = 4.8;
  TGeoVolume* vPC_P3 = gGeoManager->MakeBox(Form("vPC_P3_S%d", side), mCu, xPC_P1 / 2, yPC_P1 / 2, zPC_P1 / 2);
  auto* r_PC_P3 = new TGeoRotation("rotation_PC_P3", 39., 90., 0.);
  r_PC_P3->RegisterYourself();
  Double_t XPC_P3 = XPC_P2 + signe * zPC_P3 / 2 * TMath::Cos((90. - 39.) * TMath::Pi() / 180.);
  Double_t YPC_P3 = YPC_P2 - signe * zPC_P3 / 2 * TMath::Sin((90. - 39.) * TMath::Pi() / 180.);
  Double_t ZPC_P3 = ZPC_P2 - (zPC_0 + yPC_P3) / 2;
  auto* p_PC_P3 = new TGeoCombiTrans(XPC_P3, YPC_P3, ZPC_P3, r_PC_P3);
  //===========
  vPC_P2->SetLineColor(kBlue);
  vPC_P3->SetLineColor(kBlue);
  HalfConeVolume->AddNode(vPC_P2, 1, p_PC_P2);
  HalfConeVolume->AddNode(vPC_P3, 1, p_PC_P3);

  // Cables for PSU from the patch panel
  // b :bottom
  Double_t xPC_PSUb0 = 2.5;    // width
  Double_t yPC_PSUb0 = 0.0418; // thickness
  Double_t zPC_PSUb0 = 0.8;    // length
  // on the PSU connectors
  TGeoVolume* vPC_PSUb0 = gGeoManager->MakeBox(Form("vPC_PSUb0_S%d", side), mCu, xPC_PSUb0 / 2, yPC_PSUb0 / 2, zPC_PSUb0 / 2);
  auto* r_PC_PSU = new TGeoRotation("rotation_PC_PSU", -signe * 52., 0., 0.);
  r_PC_PSU->RegisterYourself();
  Double_t XPC_PSUb0 = -signe * 22.4;
  Double_t YPC_PSUb0 = -16.4;
  Double_t ZPC_PSUb0 = -72.5 + signe * 1.;
  auto* p_PC_PSU = new TGeoCombiTrans(XPC_PSUb0, YPC_PSUb0, ZPC_PSUb0, r_PC_PSU);
  vPC_PSUb0->SetLineColor(kBlue);
  HalfConeVolume->AddNode(vPC_PSUb0, 1, p_PC_PSU);
  //===========
  Double_t xPC_PSUb1 = 2.1;   // length
  Double_t yPC_PSUb1 = 0.274; // square side
  Double_t zPC_PSUb1 = yPC_PSUb1;
  TGeoVolume* vPC_PSUb1 = gGeoManager->MakeBox(Form("vPC_PSUb1_S%d", side), mCu, xPC_PSUb1 / 2, yPC_PSUb1 / 2, zPC_PSUb1 / 2);
  r_PC_PSU = new TGeoRotation("rotation_PC_PSU", 90. - signe * 52., 90., 0.);
  r_PC_PSU->RegisterYourself();
  Double_t XPC_PSUb1 = XPC_PSUb0 - signe * xPC_PSUb1 / 2 * TMath::Cos((90. - 52.) * TMath::Pi() / 180.);
  Double_t YPC_PSUb1 = YPC_PSUb0 - xPC_PSUb1 / 2 * TMath::Sin((90. - 52.) * TMath::Pi() / 180.);
  Double_t ZPC_PSUb1 = ZPC_PSUb0 - zPC_PSUb0 / 2 - yPC_PSUb1 / 2;
  p_PC_PSU = new TGeoCombiTrans(XPC_PSUb1, YPC_PSUb1, ZPC_PSUb1, r_PC_PSU);
  vPC_PSUb1->SetLineColor(kBlue);
  HalfConeVolume->AddNode(vPC_PSUb1, 1, p_PC_PSU);
  //===========
  Double_t xPC_PSUb2 = yPC_PSUb1;
  Double_t yPC_PSUb2 = yPC_PSUb1;
  Double_t zPC_PSUb2 = 11.; // length
  TGeoVolume* vPC_PSUb2 = gGeoManager->MakeBox(Form("vPC_PSUb0_S%d", side), mCu, xPC_PSUb2 / 2, yPC_PSUb2 / 2, zPC_PSUb2 / 2);
  r_PC_PSU = new TGeoRotation("rotation_PC_PSU", -signe * 52., 0., 0.);
  r_PC_PSU->RegisterYourself();
  Double_t XPC_PSUb2 = -signe * 24.2;
  Double_t YPC_PSUb2 = -17.8;
  Double_t ZPC_PSUb2 = ZPC_PSUb0 - zPC_PSUb0 / 2 - zPC_PSUb2 / 2;
  p_PC_PSU = new TGeoCombiTrans(XPC_PSUb2, YPC_PSUb2, ZPC_PSUb2, r_PC_PSU);
  vPC_PSUb2->SetLineColor(kBlue);
  HalfConeVolume->AddNode(vPC_PSUb2, 1, p_PC_PSU);

  // t : top
  TGeoVolume* vPC_PSUt0 = gGeoManager->MakeBox(Form("vPC_PSUt0_S%d", side), mCu, xPC_PSUb0 / 2, yPC_PSUb0 / 2, zPC_PSUb0 / 2);
  r_PC_PSU = new TGeoRotation("rotation_PC_PSU", signe * 52., 0., 0.);
  r_PC_PSU->RegisterYourself();
  Double_t XPC_PSUt0 = -signe * 22.4;
  Double_t YPC_PSUt0 = 16.4;
  Double_t ZPC_PSUt0 = -72.5 - signe * 1.;
  p_PC_PSU = new TGeoCombiTrans(XPC_PSUt0, YPC_PSUt0, ZPC_PSUt0, r_PC_PSU);
  vPC_PSUt0->SetLineColor(kBlue);
  HalfConeVolume->AddNode(vPC_PSUt0, 1, p_PC_PSU);
  //===========
  TGeoVolume* vPC_PSUt1 = gGeoManager->MakeBox(Form("vPC_PSUt1_S%d", side), mCu, xPC_PSUb1 / 2, yPC_PSUb1 / 2, zPC_PSUb1 / 2);
  r_PC_PSU = new TGeoRotation("rotation_PC_PSU", -(90.0 - signe * 52.0), 90., 0.);
  r_PC_PSU->RegisterYourself();
  Double_t XPC_PSUt1 = XPC_PSUt0 - signe * xPC_PSUb1 / 2 * TMath::Cos((90. - 52.) * TMath::Pi() / 180);
  Double_t YPC_PSUt1 = YPC_PSUt0 + xPC_PSUb1 / 2 * TMath::Sin((90. - 52.) * TMath::Pi() / 180.);
  Double_t ZPC_PSUt1 = ZPC_PSUt0 - zPC_PSUb0 / 2 - yPC_PSUb1 / 2;
  p_PC_PSU = new TGeoCombiTrans(XPC_PSUt1, YPC_PSUt1, ZPC_PSUt1, r_PC_PSU);
  vPC_PSUt1->SetLineColor(kBlue);
  HalfConeVolume->AddNode(vPC_PSUt1, 1, p_PC_PSU);
  //===========
  TGeoVolume* vPC_PSUt2 = gGeoManager->MakeBox(Form("vPC_PSUt0_S%d", side), mCu, xPC_PSUb2 / 2, yPC_PSUb2 / 2, zPC_PSUb2 / 2);
  r_PC_PSU = new TGeoRotation("rotation_PC_PSU", signe * 52., 0., 0.);
  r_PC_PSU->RegisterYourself();
  Double_t XPC_PSUt2 = XPC_PSUb2;
  Double_t YPC_PSUt2 = -YPC_PSUb2;
  Double_t ZPC_PSUt2 = ZPC_PSUt0 - zPC_PSUb0 / 2 - zPC_PSUb2 / 2;
  p_PC_PSU = new TGeoCombiTrans(XPC_PSUt2, YPC_PSUt2, ZPC_PSUt2, r_PC_PSU);
  vPC_PSUt2->SetLineColor(kBlue);
  HalfConeVolume->AddNode(vPC_PSUt2, 1, p_PC_PSU);
}
