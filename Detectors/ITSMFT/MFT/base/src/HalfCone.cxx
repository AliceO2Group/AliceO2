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
/// \date 13/03/2019

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

  TGeoMedium* kMedAlu = gGeoManager->GetMedium("MFT_Alu$");

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

  TGeoCombiTrans* combi1 = new TGeoCombiTrans(0, -10.3, 1.29, rot1); // y=-10.80 belt
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
  TGeoCombiTrans* bcombi_1h_mb0 = new TGeoCombiTrans(-5.2, 0, 0, rot_1hole_mb0); //y=
  bcombi_1h_mb0->SetName("bcombi_1h_mb0");
  bcombi_1h_mb0->RegisterYourself();

  // 2hole coaxial
  Double_t radin_2hmb0 = 0.;
  Double_t radout_2hmb0 = 0.15; // diameter M3
  Double_t high_2hmb0 = 1.2;    // 12

  TGeoRotation* rot_2hole_mb0 = new TGeoRotation("rot_2hole_mb0", 90, 90, 0);
  rot_2hole_mb0->SetName("rot_2hole_mb0");
  rot_2hole_mb0->RegisterYourself();

  TGeoCombiTrans* combi_2hole_mb0 = new TGeoCombiTrans(6.7, 0, 0, rot_2hole_mb0);
  combi_2hole_mb0->SetName("combi_2hole_mb0");
  combi_2hole_mb0->RegisterYourself();

  TGeoCombiTrans* combi_2hole_mb0_b = new TGeoCombiTrans(-6.7, 0, 0, rot_2hole_mb0); //y=
  combi_2hole_mb0_b->SetName("combi_2hole_mb0_b");
  combi_2hole_mb0_b->RegisterYourself();

  // shape for cross_mb0..
  // TGeoShape* box_mb0 = new TGeoBBox("s_box_mb0", x_boxmb0 / 2, y_boxmb0 / 2, z_boxmb0 / 2);
  new TGeoBBox("box_mb0", x_boxmb0 / 2, y_boxmb0 / 2, z_boxmb0 / 2);
  new TGeoTube("hole1_mb0", radin_1hmb0, radout_1hmb0, high_1hmb0 / 2);
  new TGeoTube("hole2_mb0", radin_2hmb0, radout_2hmb0, high_2hmb0 / 2);

  ///composite shape for mb0

  auto* c_mb0_Shape_0 = new TGeoCompositeShape("c_mb0_Shape_0", "box_mb0 - hole1_mb0:acombi_1h_mb0 - hole1_mb0:bcombi_1h_mb0 - hole2_mb0:combi_2hole_mb0 - hole2_mb0:combi_2hole_mb0_b");

  auto* cross_mb0_Volume = new TGeoVolume("cross_mb0_Volume", c_mb0_Shape_0, kMedAlu);

  Cross_mb0->AddNode(cross_mb0_Volume, 1);

  // 2nd piece  cross beam MFT (cbeam)

  auto* Cross_mft = new TGeoVolumeAssembly("Cross_mft");

  // using the same "box" of the 1 piece
  // 2hole coaxial
  Double_t radin_hole_cbeam = 0.;
  Double_t radout_hole_cbeam = 0.15; // diameter M3
  Double_t high_hole_cbeam = 0.91;   //

  // same rotation in tub aximatAl "rot_2hole_mb0"

  TGeoCombiTrans* combi_hole_1cbeam = new TGeoCombiTrans(6.8, 0, 0, rot_2hole_mb0);
  combi_hole_1cbeam->SetName("combi_hole_1cbeam");
  combi_hole_1cbeam->RegisterYourself();

  TGeoCombiTrans* combi_hole_2cbeam = new TGeoCombiTrans(-6.8, 0, 0, rot_2hole_mb0);
  combi_hole_2cbeam->SetName("combi_hole_2cbeam");
  combi_hole_2cbeam->RegisterYourself();

  // shape for shape cross beam

  new TGeoTube("hole_cbeam", radin_hole_cbeam, radout_hole_cbeam, high_hole_cbeam / 2);

  // composite shape for cross beam (using the same box of mb0)
  auto* c_cbeam_Shape = new TGeoCompositeShape("c_cbeam_Shape", "box_mb0 - hole_cbeam:combi_hole_1cbeam - hole_cbeam:combi_hole_2cbeam");

  auto* Cross_mft_Volume = new TGeoVolume("Cross_mft_Volume", c_cbeam_Shape, kMedAlu);

  Cross_mft->AddNode(Cross_mft_Volume, 1);

  // 3th piece Framework front

  auto* Fra_front = new TGeoVolumeAssembly("Fra_front");
  auto* Fra_front_L = new TGeoVolumeAssembly("Fra_front_L");
  auto* Fra_front_R = new TGeoVolumeAssembly("Fra_front_R");

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

  new TGeoBBox("box_up", x_box_up / 2, y_box_up / 2, z_box_up / 2);

  new TGeoTube("tub_up", 0., dia_tub_up / 2, high_tub_up / 2); //
  new TGeoTubeSeg("seg_tub", radin_segtub, radout_segtub, high_segtub / 2, ang_in_segtub, ang_fin_segtub);

  new TGeoBBox("boxB_down", x_boxB_down / 2, y_boxB_down / 2, z_boxB_down / 2);

  new TGeoBBox("boxA_down", x_boxA_down / 2, y_boxA_down / 2, z_boxA_down / 2);

  new TGeoTube("tubdown", 0., dia_tubdown / 2, high_tubdown / 2);

  // Composite shapes for Fra_front
  new TGeoCompositeShape("fra_front_Shape_0", "box_up:tr1_up + seg_tub:combi_3b + boxB_down:tr3_box + boxA_down:tr_2_box");

  auto* fra_front_Shape_1 = new TGeoCompositeShape("fra_front_Shape_1", "fra_front_Shape_0 - tubdown:tr_tubdown - tub_up:combi_3a");

  TGeoRotation* rot_z180x90 = new TGeoRotation("rot_z180x90", 180, 90, 0); //half0_R
  rot_z180x90->RegisterYourself();

  TGeoRotation* rot_halfR = new TGeoRotation("rot_halfR", 180, 180, 0); // half0_R
  rot_halfR->RegisterYourself();
  TGeoCombiTrans* combi_front_L = new TGeoCombiTrans(-7.1, -16.2, 32.5 + 0.675, rot_90x); // x=7.35, y=0, z=15.79
  combi_front_L->SetName("combi_front_L");
  combi_front_L->RegisterYourself();

  TGeoCombiTrans* combi_front_R = new TGeoCombiTrans(7.1, -16.2, 32.5 + 0.675, rot_z180x90); //x=7.35, y=0, z=15.79
  combi_front_R->SetName("combi_front_R");
  combi_front_R->RegisterYourself();

  // auto * fra_front_Shape_3 = new TGeoCompositeShape("fra_front_Shape_3","fra_front_Shape_2:rot_halfR  ");

  auto* Fra_front_Volume = new TGeoVolume("Fra_front_Volume", fra_front_Shape_1, kMedAlu);

  Fra_front_L->AddNode(Fra_front_Volume, 1, combi_front_L);
  Fra_front_R->AddNode(Fra_front_Volume, 1, combi_front_R);

  Fra_front->AddNode(Fra_front_L, 1);
  Fra_front->AddNode(Fra_front_R, 2);

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

  TGeoTranslation* tr_cbox = new TGeoTranslation("tr_cbox", -xc_box / 2, -radout_disc + 0.888, 0);
  tr_cbox->RegisterYourself();
  // box 4 lamine 1
  Double_t x_labox = 60.0;
  Double_t y_labox = 30.3;
  Double_t z_labox = 0.305;
  TGeoTranslation* tr_la = new TGeoTranslation("tr_la", 0, -y_labox / 2 - 9.3, high_disc / 2);
  tr_la->RegisterYourself();

  // box 5 lamin 2
  Double_t x_2labox = 51.2;
  Double_t y_2labox = 2.8; // C-B
  Double_t z_2labox = 0.303;
  TGeoTranslation* tr_2la = new TGeoTranslation("tr_2la", 0, -8.1, high_disc / 2); //
  tr_2la->RegisterYourself();

  // circular border C SEG_BORD
  // seg tub 3 xy
  Double_t radin_bord = 0.5;
  Double_t radout_bord = 0.9; //
  Double_t high_bord = 1.355; // 13.5
  Double_t ang_in_bord = 0;
  Double_t ang_fin_bord = 90;
  // TGeoRotation *rot_bord1 = new TGeoRotation("rot_bord1", ang_in_1hole +0.0167,0,0);
  TGeoRotation* rot1_bord1 = new TGeoRotation("rot1_bord1", 14.8, 0, 0);
  rot1_bord1->RegisterYourself();
  TGeoCombiTrans* combi_bord1 = new TGeoCombiTrans(-26.7995, -13.0215, 0, rot1_bord1); // y=
  combi_bord1->SetName("combi_bord1");
  combi_bord1->RegisterYourself();

  TGeoRotation* rot2_bord1 = new TGeoRotation("rot2_bord1", -50, 0, 0);
  rot2_bord1->RegisterYourself();
  TGeoCombiTrans* combi2_bord1 = new TGeoCombiTrans(-21.3795, -20.7636, 0, rot2_bord1); // y=
  combi2_bord1->SetName("combi2_bord1");
  combi2_bord1->RegisterYourself();
  //
  TGeoRotation* rot1_bord2 = new TGeoRotation("rot1_bord2", 250, 0, 0);
  rot1_bord2->RegisterYourself();
  TGeoCombiTrans* combi1_bord2 = new TGeoCombiTrans(-9.0527, -23.3006, 0, rot1_bord2); // y=
  combi1_bord2->SetName("combi1_bord2");
  combi1_bord2->RegisterYourself();
  // e|°____°|
  TGeoRotation* rot_cent_bord = new TGeoRotation("rot_cent_bord", 90, 0, 0);
  rot_cent_bord->RegisterYourself();
  TGeoCombiTrans* combi_cent_bord = new TGeoCombiTrans(-6.5, -27.094, 0, rot_cent_bord); // y=
  combi_cent_bord->SetName("combi_cent_bord");
  combi_cent_bord->RegisterYourself();
  // box tonge
  Double_t x_tong = 2.0;
  Double_t y_tong = 2.81;
  Double_t z_tong = 1.35;
  TGeoTranslation* tr_tong = new TGeoTranslation("tr_tong", 0, -28.6, 0); //
  tr_tong->RegisterYourself();
  // circular central hole1 to conexion with other parts
  Double_t radin_hole1 = 0;
  Double_t radout_hole1 = 0.4;
  Double_t high_hole1 = 1.36;
  TGeoTranslation* tr_hole1 = new TGeoTranslation("tr_hole1", 0, -28.0, 0); // tonge
  tr_hole1->RegisterYourself();

  TGeoTranslation* tr2_hole1 = new TGeoTranslation("tr2_hole1", -26.5, -8.5, 0); // left
  tr2_hole1->RegisterYourself();

  TGeoTranslation* tr3_hole1 = new TGeoTranslation("tr3_hole1", 26.5, -8.5, 0); // right
  tr3_hole1->RegisterYourself();

  // circular hole2 ; hole2 r=6.7
  Double_t radin_hole2 = 0;
  Double_t radout_hole2 = 0.335;                                                 // diameter 6.7
  Double_t high_hole2 = 1.36;                                                    // 13.5
  TGeoTranslation* tr1_hole2 = new TGeoTranslation("tr1_hole2", -28.0, -8.5, 0); //
  tr1_hole2->RegisterYourself();

  TGeoTranslation* tr2_hole2 = new TGeoTranslation("tr2_hole2", 28.0, -8.5, 0); //
  tr2_hole2->RegisterYourself();

  //////////// hole "0" two tubs together
  Double_t radin_T1 = 0.325; // diam 0.65cm
  Double_t radout_T1 = 0.55; // dia 1.1
  Double_t high_T1 = 1.2;    //  dz 6

  Double_t radin_T2 = 0;
  Double_t radout_T2 = 1.1;
  Double_t high_T2 = 1.2; // dz 6

  // shape for base

  new TGeoTubeSeg("disc", radin_disc, radout_disc, high_disc / 2, ang_in_disc, ang_fin_disc);

  new TGeoBBox("box1", x_1box / 2, y_1box / 2, z_1box / 2);
  new TGeoBBox("box2", x_2box / 2, y_2box / 2, z_2box / 2);
  new TGeoBBox("box3", x_3box / 2, y_3box / 2, z_3box / 2);
  new TGeoBBox("labox1", x_labox / 2, y_labox / 2, z_labox / 2);
  new TGeoBBox("labox2", x_2labox / 2, y_2labox / 2, z_2labox / 2);
  new TGeoBBox("cbox", xc_box / 2, yc_box / 2, zc_box / 2);
  new TGeoBBox("tongbox", x_tong / 2, y_tong / 2, z_tong / 2);

  new TGeoTubeSeg("seg_1hole", radin_1hole, radout_1hole, high_1hole / 2, ang_in_1hole, ang_fin_1hole); // r_in,r_out,dZ,ang,ang
  new TGeoTubeSeg("seg_2hole", radin_2hole, radout_2hole, high_2hole / 2, ang_in_2hole, ang_fin_2hole);
  new TGeoTubeSeg("seg_3hole", radin_3hole, radout_3hole, high_3hole / 2, ang_in_3hole, ang_fin_3hole); // y|u|
  new TGeoTubeSeg("seg_bord", radin_bord, radout_bord, high_bord / 2, ang_in_bord, ang_fin_bord);

  new TGeoTube("circ_hole1", radin_hole1, radout_hole1, high_hole1 / 2);

  new TGeoTube("circ_hole2", radin_hole2, radout_hole2, high_hole2 / 2);

  new TGeoTube("circ_holeB", radin_holeB, radout_holeB, high_holeB / 2);

  // composite shape for base

  new TGeoCompositeShape("base_Shape_0", " disc - box1 - box2 - box3 - circ_holeB:tr1_holeB - circ_holeB:tr2_holeB");
  new TGeoCompositeShape("base_Shape_1", "(seg_1hole - seg_bord:combi_bord1 - seg_bord:combi2_bord1) + seg_2hole -seg_bord:combi1_bord2 + cbox:tr_cbox");

  new TGeoCompositeShape("base_Shape_2", " seg_3hole + seg_bord:combi_cent_bord"); // seg_bord:combi_cent_bord

  new TGeoCompositeShape("base_Shape_3", " labox1:tr_la + labox2:tr_2la ");

  auto* base_Shape_4 = new TGeoCompositeShape("base_Shape_4", "base_Shape_0 - base_Shape_1 - base_Shape_1:rot1 + base_Shape_2  + tongbox:tr_tong - circ_hole1:tr_hole1 - circ_hole1:tr2_hole1 - circ_hole1:tr3_hole1 - circ_hole2:tr1_hole2 - circ_hole2:tr2_hole2 - base_Shape_3 ");

  // auto * base_Shape_5 = new TGeoCompositeShape("base_Shape_5","disc-box1 -box2 -box3 -seg_1hole -seg_2hole +seg_3hole -seg_1hole:rot1-seg_2hole:rot1 - cbox:tr_cbox - labox:tr_la - labox2:tr_2la  + seg_bord  ");

  // auto * base0_Volume = new TGeoVolume("base0_Volume",base_Shape_0,kMedAlu);
  // auto * base1_Volume = new TGeoVolume("base1_Volume",base_Shape_1,kMedAlu);
  // auto * base2_Volume = new TGeoVolume("base2_Volume",base_Shape_2,kMedAlu);
  // auto * base3_Volume = new TGeoVolume("base3_Volume",base_Shape_3,kMedAlu);

  auto* base4_Volume = new TGeoVolume("base4_Volume", base_Shape_4, kMedAlu);

  base->AddNode(base4_Volume, 2, rot_base);
  // base->AddNode(base4_Volume,2);

  // 5th piece MIDLE  Framework midle

  auto* midle = new TGeoVolumeAssembly("Midle");
  auto* midle_L = new TGeoVolumeAssembly("Midle_L");
  auto* midle_R = new TGeoVolumeAssembly("Midle_R");

  // box up to quit and to join
  Double_t x_midle = 0.8;   //dx=4
  Double_t y_midle = 3.495; //y=34.9
  Double_t z_midle = 0.62;  //z=6
  // tr1 to join with arc
  TGeoTranslation* tr1_midle_box = new TGeoTranslation("tr1_midle_box", -14.4, -0.745, 0); // -152,-17.45,0
  tr1_midle_box->RegisterYourself();
  // tr2 to quiet
  TGeoTranslation* tr2_midle_box = new TGeoTranslation("tr2_midle_box", -15.2, -0.745, 0); // -152,-17.45,0
  tr2_midle_box->RegisterYourself();

  // box down_1
  Double_t x_midle_d1box = 0.4; // dx=4
  Double_t y_midle_d1box = 0.28;
  Double_t z_midle_d1box = 0.66;
  TGeoTranslation* tr_midle_d1box = new TGeoTranslation("tr_midle_d1box", -7.3, -11.96, 0.); // 81
  tr_midle_d1box->RegisterYourself();

  // box down_2
  Double_t x_midle_d2box = 0.8; // dx=4
  Double_t y_midle_d2box = 1.0;
  Double_t z_midle_d2box = 0.66;                                                              //
  TGeoTranslation* tr_midle_d2box = new TGeoTranslation("tr_midle_d2box", -7.5, -12.6249, 0); // 81
  tr_midle_d2box->RegisterYourself();

  // arc circ part
  Double_t radin_midle = 14.0;
  Double_t radout_midle = 15.0; //
  Double_t high_midle = 0.6;    //
  Double_t ang_in_midle = 180;
  Double_t ang_fin_midle = 238.21; // alfa=57.60

  // circular hole1 ; hole_midle d=3.5
  Double_t radin_mid_1hole = 0.;
  Double_t radout_mid_1hole = 0.175; // diameter 3.5
  Double_t high_mid_1hole = 1.5;     // 2.4

  TGeoRotation* rot_mid_1hole = new TGeoRotation("rot_mid_1hole", 90, 90, 0);
  rot_mid_1hole->RegisterYourself();
  TGeoCombiTrans* combi_mid_1tubhole = new TGeoCombiTrans(-14.2, 0.325, 0, rot_mid_1hole); //
  combi_mid_1tubhole->SetName("combi_mid_1tubhole");
  combi_mid_1tubhole->RegisterYourself();

  // circular hole2 ; hole_midle d=3
  Double_t radin_mid_2hole = 0.;
  Double_t radout_mid_2hole = 0.15; // diameter 3
  Double_t high_mid_2hole = 1.8;    //

  TGeoCombiTrans* combi_mid_2tubhole = new TGeoCombiTrans(-7.7, -12.355, 0, rot_mid_1hole); //x=81
  combi_mid_2tubhole->SetName("combi_mid_2tubhole");
  combi_mid_2tubhole->RegisterYourself();

  // shape for midle
  new TGeoBBox("midle_box", x_midle / 2, y_midle / 2, z_midle / 2);

  new TGeoBBox("midle_d1box", x_midle_d1box / 2, y_midle_d1box / 2, z_midle_d1box / 2);

  new TGeoBBox("midle_d2box", x_midle_d2box / 2, y_midle_d2box / 2, z_midle_d2box / 2);

  new TGeoTubeSeg("arc_midle", radin_midle, radout_midle, high_midle / 2, ang_in_midle, ang_fin_midle);

  new TGeoTube("mid_1tubhole", radin_mid_1hole, radout_mid_1hole, high_mid_1hole / 2);

  new TGeoTube("mid_2tubhole", radin_mid_2hole, radout_mid_2hole, high_mid_2hole / 2);

  // composite shape for midle

  new TGeoCompositeShape("midle_Shape_0", " arc_midle + midle_box:tr1_midle_box - midle_box:tr2_midle_box - midle_d1box:tr_midle_d1box - midle_d2box:tr_midle_d2box");

  auto* midle_Shape_1 = new TGeoCompositeShape("midle_Shape_1", " midle_Shape_0 -mid_1tubhole:combi_mid_1tubhole-mid_2tubhole:combi_mid_2tubhole");

  TGeoRotation* rot_midlez = new TGeoRotation("rot_midley", 180, 180, 0);
  rot_midlez->RegisterYourself();
  TGeoCombiTrans* combi_midle_L = new TGeoCombiTrans(0, -7.625, 24.15 + 0.675, rot_90x); // x=7.35, y=0, z=15.79- 0,-7.625,24.15+0.675-80)
  combi_midle_L->SetName("combi_midle_L");
  combi_midle_L->RegisterYourself();

  TGeoTranslation* tr_midle_L = new TGeoTranslation("tr_midle_L", 0, -7.625, 24.15 + 0.675); // -152,-17.45,0
  tr_midle_L->RegisterYourself();

  TGeoCombiTrans* combi_midle_R = new TGeoCombiTrans(0, -7.625, 24.15 + 0.675, rot_midlez); // x=7.35, y=0, z=15.79
  combi_midle_R->SetName("combi_midle_R");
  combi_midle_R->RegisterYourself();

  auto* midle_Volume = new TGeoVolume("midle_Volume", midle_Shape_1, kMedAlu);

  midle_L->AddNode(midle_Volume, 1, tr_midle_L);
  midle_R->AddNode(midle_Volume, 1, combi_midle_R);

  // midle->AddNode(midle_Volume,1);
  midle->AddNode(midle_L, 1);
  midle->AddNode(midle_R, 2);

  // new piece _/   \_
  // Support_rail_L & Support_rail_R

  auto* rail_L_R = new TGeoVolumeAssembly("rail_L_R");

  // 6 piece RAIL LEFT RL0000
  auto* rail_L = new TGeoVolumeAssembly("rail_L");

  // box down_2
  Double_t x_RL_1box = 3.0;                                                   // dx=15
  Double_t y_RL_1box = 1.21;                                                  // dy=6, -dy=6
  Double_t z_RL_1box = 0.8;                                                   // dz=4     to quit
  TGeoTranslation* tr_RL_1box = new TGeoTranslation(0, y_RL_1box / 2, 1.825); // 81
  tr_RL_1box->SetName("tr_RL_1box");
  tr_RL_1box->RegisterYourself();

  TGeoXtru* xtru_RL1 = new TGeoXtru(2);
  xtru_RL1->SetName("S_XTRU_RL1");

  Double_t x_RL1[5] = { -1.5, 1.5, 0.5, 0.5, -1.5 }; // 93,93,73,73,-15}; //vertices
  Double_t y_RL1[5] = { 1.2, 1.2, 2.2, 8.2, 8.2 };   // 357.5,357.5,250.78,145.91};
  xtru_RL1->DefinePolygon(5, x_RL1, y_RL1);
  xtru_RL1->DefineSection(0, -2.225, 0., 0., 1); // (plane,-zplane/ +zplane, x0, y0,(x/y))
  xtru_RL1->DefineSection(1, 2.225, 0., 0., 1);

  TGeoXtru* xtru_RL2 = new TGeoXtru(2);
  xtru_RL2->SetName("S_XTRU_RL2");

  Double_t x_RL2[8] = { -1.5, 0.5, 0.5, 9.3, 9.3, 7.3, 7.3, -1.5 }; // vertices
  Double_t y_RL2[8] = { 8.2, 8.2, 13.863, 24.35, 35.75, 35.75, 25.078, 14.591 };

  xtru_RL2->DefinePolygon(8, x_RL2, y_RL2);

  xtru_RL2->DefineSection(0, 0.776, 0, 0, 1); // (plane,-zplane/+zplane, x0, y0,(x/y))
  xtru_RL2->DefineSection(1, 2.225, 0, 0, 1);

  // box knee
  Double_t x_RL_kneebox = 1.5;                                   // dx=7.5
  Double_t y_RL_kneebox = 3.5;                                   // dy=17.5
  Double_t z_RL_kneebox = 1.5;                                   // dz=7.5     to quit
  TGeoTranslation* tr_RL_kneebox = new TGeoTranslation(0, 0, 0); // 81 x =-2.5, y=145.91
  tr_RL_kneebox->SetName("tr_RL_kneebox");
  tr_RL_kneebox->RegisterYourself();

  TGeoRotation* rot_knee = new TGeoRotation("rot_knee", -40, 0, 0);
  rot_knee->SetName("rot_knee");
  rot_knee->RegisterYourself();
  TGeoCombiTrans* combi_knee = new TGeoCombiTrans(0.96, 1.75 + 0.81864, 0, rot_knee); // y=
  combi_knee->SetName("combi_knee");
  combi_knee->RegisterYourself();
  // quit diagona-> qdi
  Double_t x_qdi_box = 3.1;   //
  Double_t y_qdi_box = 7.159; //
  Double_t z_qdi_box = 3.005; //

  TGeoRotation* rot_qdi = new TGeoRotation("rot_qdi", 0, 24.775, 0);
  rot_qdi->RegisterYourself();
  TGeoCombiTrans* combi_qdi = new TGeoCombiTrans(0, 5.579, -2.087, rot_qdi); // y=
  combi_qdi->SetName("combi_qdi");
  combi_qdi->RegisterYourself();
  // knee small

  TGeoXtru* xtru3_RL = new TGeoXtru(2);
  xtru3_RL->SetName("xtru3_RL");

  Double_t x_3RL[6] = { -0.75, 0.75, 0.75, 2.6487, 1.4997, -0.75 }; // vertices
  Double_t y_3RL[6] = { -1.75, -1.75, 1.203, 3.465, 4.4311, 1.75 };

  xtru3_RL->DefinePolygon(6, x_3RL, y_3RL);
  xtru3_RL->DefineSection(0, -0.75, 0, 0, 1); // (plane,-zplane/+zplane, x0, y0,(x/y))
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
  TGeoCombiTrans* combi_RL1hole = new TGeoCombiTrans(0.7, 0.6, 1.85, rot_RL1hole); //y=
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
  TGeoCombiTrans* combi_ir1_RL = new TGeoCombiTrans(8.62, 24.75, 1.5, rot_ir_RL);
  combi_ir1_RL->SetName("combi_ir1_RL");
  combi_ir1_RL->RegisterYourself();

  TGeoCombiTrans* combi_ir2_RL = new TGeoCombiTrans(8.6, 33.15, 1.5, rot_ir_RL);
  combi_ir2_RL->SetName("combi_ir2_RL");
  combi_ir2_RL->RegisterYourself();

  // shape for Rail L geom
  new TGeoBBox("RL_1box", x_RL_1box / 2, y_RL_1box / 2, z_RL_1box / 2);
  new TGeoBBox("RL_kneebox", x_RL_kneebox / 2, y_RL_kneebox / 2, z_RL_kneebox / 2); // no_ used
  new TGeoBBox("qdi_box", x_qdi_box / 2, y_qdi_box / 2, z_qdi_box / 2);

  // auto *s_RL1hole=new TGeoTube("S_RL1HOLE",radin_RL1hole,radout_RL1hole,high_RL1hole/2);
  // auto *s_irL_hole=new TGeoTube("S_irL_HOLE",radin_ir_railL,radout_ir_railL,high_ir_railL/2);
  // composite shape for rail L

  auto* RL_Shape_0 = new TGeoCompositeShape("RL_Shape_0", " xtru3_RL:tr_vol3_RL + S_XTRU_RL1 + S_XTRU_RL2 + RL_1box:tr_RL_1box - qdi_box:combi_qdi");

  TGeoVolume* rail_L_vol0 = new TGeoVolume("RAIL_L_VOL0", RL_Shape_0, kMedAlu);

  rail_L->AddNode(rail_L_vol0, 1, new TGeoTranslation(0., 0., 1.5));

  // piece 7th RAIL RIGHT
  // auto *rail_R = new TGeoVolumeAssembly("rail_R");

  Double_t x_RR_1box = 3.0;                                                       // dx=15
  Double_t y_RR_1box = 1.2;                                                       // dy=6, -dy=6
  Double_t z_RR_1box = 0.8;                                                       // dz=4     to quit
  TGeoTranslation* tr_RR_1box = new TGeoTranslation("tr_RR_1box", 0, 0.6, 1.825); // 81
  tr_RR_1box->RegisterYourself();

  TGeoXtru* part_RR1 = new TGeoXtru(2);
  part_RR1->SetName("part_RR1");
  // TGeoVolume *vol_RR1 = gGeoManager->MakeXtru("S_part_RR1",kMedAlu,2);
  // TGeoXtru *part_RR1 = (TGeoXtru*)vol_RR1->GetShape();

  Double_t x_RR1[5] = { -1.5, -0.5, -0.5, 1.5, 1.5 }; // C,D,K,L,C' //vertices
  Double_t y_RR1[5] = { 1.2, 2.2, 8.2, 8.2, 1.2 };    // 357.5,357.5,250.78,145.91};

  part_RR1->DefinePolygon(5, x_RR1, y_RR1);
  part_RR1->DefineSection(0, -2.225, 0, 0, 1); // (plane,-zplane/ +zplane, x0, y0,(x/y))
  part_RR1->DefineSection(1, 2.225, 0, 0, 1);

  TGeoXtru* part_RR2 = new TGeoXtru(2);
  part_RR2->SetName("part_RR2");
  // TGeoVolume *vol_RR2 = gGeoManager->MakeXtru("part_RR2",Al,2);
  // TGeoXtru *xtru_RR2 = (TGeoXtru*)vol_RR2->GetShape();

  Double_t x_RR2[8] = { -0.5, -0.5, -9.3, -9.3, -7.3, -7.3, 1.5, 1.5 }; // K,E,F,G,H,I,J,L // vertices
  Double_t y_RR2[8] = { 8.2, 13.863, 24.35, 35.75, 35.75, 25.078, 14.591, 8.2 };

  part_RR2->DefinePolygon(8, x_RR2, y_RR2);
  part_RR2->DefineSection(0, 0.776, 0, 0, 1); // (plane,-zplane/+zplane, x0, y0,(x/y))
  part_RR2->DefineSection(1, 2.225, 0, 0, 1);

  // knee (small)

  TGeoXtru* part_RR3 = new TGeoXtru(2);
  part_RR3->SetName("part_RR3");

  Double_t x_3RR[6] = { 1.0, 1.0, -1.2497, -2.2138, -0.5, -0.5 }; // R,Q,P,O,N.M // vertices
  Double_t y_3RR[6] = { 10.91, 14.41, 17.0911, 15.9421, 13.86, 10.91 };

  part_RR3->DefinePolygon(6, x_3RR, y_3RR);
  part_RR3->DefineSection(0, -0.75, 0, 0, 1); // (plane,-zplane/+zplane, x0, y0,(x/y))
  part_RR3->DefineSection(1, 0.78, 0, 0, 1);

  TGeoTranslation* tr_vol3_RR = new TGeoTranslation("tr_vol3_RR", -0.25, 12.66, 0); //
  tr_vol3_RR->RegisterYourself();

  //  quit diagona-> qdi
  Double_t x_qdi_Rbox = 3.1;   // dx=1.5
  Double_t y_qdi_Rbox = 7.159; //
  Double_t z_qdi_Rbox = 3.005; //

  TGeoRotation* rot_Rqdi = new TGeoRotation("rot_Rqdi", 0, 24.775, 0);
  rot_Rqdi->RegisterYourself();
  TGeoCombiTrans* combi_Rqdi = new TGeoCombiTrans(0, 5.579, -2.087, rot_Rqdi); // y=
  combi_Rqdi->SetName("combi_Rqdi");
  combi_Rqdi->RegisterYourself();

  // holes   circular hole_a. diameter=6.5 (a(6,22)); hole_midle d=6.5 H11
  Double_t radin_a_rail = 0.;
  Double_t radout_a_rail = 0.325; // diameter 3.5
  Double_t high_a_rail = 0.82;    //

  TGeoTranslation* tr_a_RR = new TGeoTranslation("tr_a_RR", -0.7, 0.6, 1.825); // right
  tr_a_RR->RegisterYourself();
  // circular hole_ir. diameter=M3 (3 mm)) prof trou:8, tar:6mm
  Double_t radin_ir_rail = 0.;
  Double_t radout_ir_rail = 0.15; // diameter 3
  Double_t high_ir_rail = 3.2;    // 19
  TGeoRotation* rot_ir_RR = new TGeoRotation("rot_ir_RR", 90, 90, 0);
  rot_ir_RR->RegisterYourself();
  // in y = l_253.5 - 6. center in (0,6,0)
  TGeoCombiTrans* combi_ir_RR = new TGeoCombiTrans(-8.62, 24.75, 1.5, rot_ir_RR);
  combi_ir_RR->SetName("combi_ir_RR");
  combi_ir_RR->RegisterYourself();

  TGeoCombiTrans* combi_ir2_RR = new TGeoCombiTrans(-8.6, 33.15, 1.5, rot_ir_RR);
  combi_ir2_RR->SetName("combi_ir2_RR");
  combi_ir2_RR->RegisterYourself();

  TGeoCombiTrans* combi_rail_R = new TGeoCombiTrans(24.1, -1.825, 0, rot_90x); // y=
  combi_rail_R->SetName("combi_rail_R");
  combi_rail_R->RegisterYourself();
  TGeoCombiTrans* combi_rail_L = new TGeoCombiTrans(-24.1, -1.825, 0, rot_90x); // y=
  combi_rail_L->SetName("combi_rail_L");
  combi_rail_L->RegisterYourself();

  // trasl L and R
  TGeoTranslation* tr_sr_l = new TGeoTranslation("tr_sr_l", -15.01, 0, 0); //
  tr_sr_l->RegisterYourself();
  TGeoTranslation* tr_sr_r = new TGeoTranslation("tr_sr_r", 15.01, 0, 0); //
  tr_sr_r->RegisterYourself();

  // shape for rail R
  new TGeoBBox("RR_1box", x_RR_1box / 2, y_RR_1box / 2, z_RR_1box / 2);

  // auto *s_qdi_Rbox =new TGeoBBox("S_QDI_RBOX", x_qdi_Rbox/2,y_qdi_Rbox/2,z_qdi_Rbox/2);

  // auto *s_ir_hole=new TGeoTube("S_ir_HOLE",radin_ir_rail,radout_ir_rail,high_ir_rail/2);

  // auto *s_cc_hole=new TGeoTube("S_CC_HOLE",radin_cc_rail,radout_cc_rail,high_cc_rail/2);

  // composite shape for rail R
  new TGeoCompositeShape("RR_Shape_0", "RR_1box:tr_RR_1box + part_RR1 + part_RR2 + part_RR3 - qdi_box:combi_qdi ");

  // auto * RR_Shape_0 = new TGeoCompositeShape("RR_Shape_0","RR_1box:tr_RR_1box+ S_part_RR1  + part_RR2 +part_RR3- qdi_box:combi_qdi + S_ir_HOLE:combi_ir_RR +S_ir_HOLE:combi_ir2_RR     "); //-RR_1box:tr_RL_1box- S_b_HOLE:tr_b_RR -S_CC_HOLE:combi_cc2_RR

  // JOIN only for show L and R parts
  auto* rail_L_R_Shape = new TGeoCompositeShape("RAIL_L_R_Shape", "  RL_Shape_0:combi_rail_L + RR_Shape_0:combi_rail_R");

  TGeoVolume* rail_L_R_vol0 = new TGeoVolume("RAIL_L_R_VOL0", rail_L_R_Shape, kMedAlu);

  TGeoRotation* rot_rLR = new TGeoRotation("rot_rLR", 180, 180, 0);
  rot_rLR->RegisterYourself();
  TGeoCombiTrans* combi_rLR = new TGeoCombiTrans(0, -6.9, -0.5, rot_rLR); // 0,-6.9,-0.5-80
  combi_rLR->SetName("combi_rLR");
  combi_rLR->RegisterYourself();

  rail_L_R->AddNode(rail_L_R_vol0, 2, combi_rLR);

  // piece 8th support rail MB \_

  auto* sup_rail_MBL = new TGeoVolumeAssembly("sup_rail_MBL");

  TGeoXtru* part_MBL_0 = new TGeoXtru(2);
  part_MBL_0->SetName("part_MBL_0"); // V-MBL_0

  // vertices a,b,c,d,e,f,g,h
  Double_t x[8] = { 0., 0, 6.1, 31.55, 34.55, 34.55, 31.946, 6.496 };
  Double_t y[8] = { -0.4, 0.4, 0.4, 13.0, 13.0, 12.2, 12.2, -0.4 };

  part_MBL_0->DefinePolygon(8, x, y);
  part_MBL_0->DefineSection(0, -0.4, 0, 0, 1); // (plane, -zplane/ +zplane,x0,y0,(x/y))
  part_MBL_0->DefineSection(1, 0.4, 0, 0, 1);

  TGeoRotation* rot1_MBL_0 = new TGeoRotation("rot1_MBL_0", -90, -90, 90);
  rot1_MBL_0->RegisterYourself();

  // quit box in diag
  Double_t x_mb_box = 0.8;                                                       // dx=4
  Double_t y_mb_box = 0.8;                                                       // dy=4
  Double_t z_mb_box = 0.81;                                                      // dz=4 to quit
  TGeoTranslation* tr_mb_box = new TGeoTranslation("tr_mb_box", 24.05, 9.55, 0); // 240.5
  tr_mb_box->RegisterYourself();

  // lateral hole-box
  Double_t x_lat_box = 0.7;                                                         //dx=0.35
  Double_t y_lat_box = 1.8;                                                         // dy=0.9
  Double_t z_lat_box = 0.2;                                                         // dz=0.1
  TGeoTranslation* tr_lat1L_box = new TGeoTranslation("tr_lat1L_box", 4.6, 0, 0.4); //
  tr_lat1L_box->RegisterYourself();
  TGeoTranslation* tr_lat2L_box = new TGeoTranslation("tr_lat2L_box", 9.6, 1.65, 0.4); //
  tr_lat2L_box->RegisterYourself();
  TGeoTranslation* tr_lat3L_box = new TGeoTranslation("tr_lat3L_box", 18.53, 6.1, 0.4); //
  tr_lat3L_box->RegisterYourself();
  TGeoTranslation* tr_lat4L_box = new TGeoTranslation("tr_lat4L_box", 26.45, 10, 0.4); //
  tr_lat4L_box->RegisterYourself();
  TGeoTranslation* tr_lat5L_box = new TGeoTranslation("tr_lat5L_box", 29.9, 11.6, 0.4); //
  tr_lat5L_box->RegisterYourself();

  TGeoTranslation* tr_lat1R_box = new TGeoTranslation("tr_lat1R_box", 4.6, 0, -0.4); //
  tr_lat1R_box->RegisterYourself();
  TGeoTranslation* tr_lat2R_box = new TGeoTranslation("tr_lat2R_box", 9.6, 1.65, -0.4); //
  tr_lat2R_box->RegisterYourself();
  TGeoTranslation* tr_lat3R_box = new TGeoTranslation("tr_lat3R_box", 18.53, 6.1, -0.4); //
  tr_lat3R_box->RegisterYourself();
  TGeoTranslation* tr_lat4R_box = new TGeoTranslation("tr_lat4R_box", 26.45, 10, -0.4); //
  tr_lat4R_box->RegisterYourself();
  TGeoTranslation* tr_lat5R_box = new TGeoTranslation("tr_lat5R_box", 29.9, 11.6, -0.4); //
  tr_lat5R_box->RegisterYourself();

  // circular hole_1mbl. diameter=3.5 H9
  Double_t radin_1mb = 0.;
  Double_t radout_1mb = 0.175;                                             // diameter 3.5mm _0.35 cm
  Double_t high_1mb = 2.825;                                               //  dh=+/- 4
  TGeoTranslation* tr1_mb = new TGeoTranslation("tr1_mb", 18.48, 6.1, 0.); // right
  tr1_mb->RegisterYourself();

  TGeoTranslation* tr2_mb = new TGeoTranslation("tr2_mb", 24.15, 8.9, 0.); // right
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
  TGeoCombiTrans* combi_hup_mb = new TGeoCombiTrans(32.5, 12.6, 0, rot_90x); //y=
  combi_hup_mb->SetName("combi_hup_mb");
  combi_hup_mb->RegisterYourself();

  // shape for rail MB
  new TGeoBBox("mb_box", x_mb_box / 2, y_mb_box / 2, z_mb_box / 2);
  new TGeoTube("hole_1mbl", radin_1mb, radout_1mb, high_1mb / 2); // d3.5
  new TGeoTube("hole_2mbl", radin_2mb, radout_2mb, high_2mb / 2); // d3
  new TGeoBBox("lat_box", x_lat_box / 2, y_lat_box / 2, z_lat_box / 2);

  // composite shape for rail_MB R + L

  // auto * MB_Shape_0 = new TGeoCompositeShape("MB_Shape_0","  V_MBL_0 - mb_box:tr_mb_box - hole_1mbl:tr1_mb + hole_1mbl:tr2_mb -hole_2mbl:combi_hup_mb  ");
  new TGeoCompositeShape("MB_Shape_0", "part_MBL_0 - mb_box:tr_mb_box - hole_1mbl:tr1_mb - hole_2mbl:combi_hup_mb");

  new TGeoCompositeShape("MB_Shape_0L", "MB_Shape_0 - lat_box:tr_lat1L_box - lat_box:tr_lat2L_box - lat_box:tr_lat3L_box - lat_box:tr_lat4L_box - lat_box:tr_lat5L_box");

  new TGeoCompositeShape("MB_Shape_0R", "MB_Shape_0 - lat_box:tr_lat1R_box - lat_box:tr_lat2R_box - lat_box:tr_lat3R_box - lat_box:tr_lat4R_box - lat_box:tr_lat5R_box");

  new TGeoCompositeShape("MB_Shape_1L", "MB_Shape_0L:rot1_MBL_0 - hole_2mbl"); // one piece "completed"
  // left and right
  new TGeoCompositeShape("MB_Shape_1R", "MB_Shape_0R:rot1_MBL_0 - hole_2mbl");

  auto* MB_Shape_2 = new TGeoCompositeShape("MB_Shape_2", " MB_Shape_1L:tr_mbl +  MB_Shape_1R:tr_mbr ");

  // TGeoVolume *sup_rail_MBL_vol0 = new TGeoVolume("SUPPORT_MBL_VOL0",MB_Shape_0,Al);
  TGeoVolume* sup_rail_MBL_vol = new TGeoVolume("SUPPORT_MBL_VOL", MB_Shape_2, kMedAlu);

  sup_rail_MBL->AddNode(sup_rail_MBL_vol, 1, rot_halfR);

  auto* stair = new TGeoVolumeAssembly("stair");

  stair->AddNode(sup_rail_MBL, 1, new TGeoTranslation(0, 0 - 28.8, 0 + 0.675));
  stair->AddNode(Cross_mft, 2, new TGeoTranslation(0, -28.8, 4.55 + 0.675));
  stair->AddNode(Cross_mb0, 3, new TGeoTranslation(0, 1.65 - 28.8, 9.55 + 0.675));
  stair->AddNode(Cross_mb0, 4, new TGeoTranslation(0, 6.1 - 28.8, 18.48 + 0.675));
  stair->AddNode(Cross_mft, 6, new TGeoTranslation(0, 10.0 - 28.8, 26.4 + 0.675));
  stair->AddNode(Cross_mft, 7, new TGeoTranslation(0, 11.6 - 28.8, 29.85 + 0.675));

  Double_t t_final_x;
  Double_t t_final_y;
  Double_t t_final_z;

  Double_t r_final_x;
  Double_t r_final_y;
  Double_t r_final_z;

  if (half == 0) {
    t_final_x = 0;
    t_final_y = 0;
    t_final_z = -80;

    r_final_x = 0;
    r_final_y = 0;
    r_final_z = 0;
  }

  if (half == 1) {
    t_final_x = 0;
    t_final_y = 0;
    t_final_z = -80;

    r_final_x = 0;
    r_final_y = 0;
    r_final_z = 180;
  }

  auto* t_final = new TGeoTranslation("t_final", t_final_x, t_final_y, t_final_z);
  auto* r_final = new TGeoRotation("r_final", r_final_x, r_final_y, r_final_z);
  auto* c_final = new TGeoCombiTrans(*t_final, *r_final);

  auto* Half_3 = new TGeoVolumeAssembly("Half_3");

  // Shell radii
  Float_t Shell_rmax = 60.6 + .7;
  Float_t Shell_rmin = 37.5 + .7;

  // Rotations and translations
  auto* tShell_0 = new TGeoTranslation("tShell_0", 0., 0., 3.1 + (25.15 + 1.) / 2.);
  auto* tShell_1 = new TGeoTranslation("tShell_1", 0., 0., -1.6 - (25.15 + 1.) / 2.);
  auto* tShellHole = new TGeoTranslation("tShellHole", 0., 0., 2. / 2. + (25.15 + 1.) / 2.);
  auto* tShellHole_0 = new TGeoTranslation("tShellHole_0", 0., -6.9, -26.1 / 2. - 6.2 / 2. - .1);
  auto* tShellHole_1 = new TGeoTranslation("tShellHole_1", 0., 0., -26.1 / 2. - 6.2 / 2. - .1);
  auto* tShell_Cut = new TGeoTranslation("tShell_Cut", 0., 25. / 2., 0.);
  auto* tShell_Cut_1 = new TGeoTranslation("tShell_Cut_1", -23., 0., -8.);
  auto* tShell_Cut_1_inv = new TGeoTranslation("tShell_Cut_1_inv", 23., 0., -8.);
  auto* Rz = new TGeoRotation("Rz", 50., 0., 0.);
  auto* Rz_inv = new TGeoRotation("Rz_inv", -50., 0., 0.);
  auto* RShell_Cut = new TGeoRotation("RShell_Cut", 90., 90. - 24., -7.5);
  auto* RShell_Cut_inv = new TGeoRotation("RShell_Cut_inv", 90., 90. + 24., -7.5);

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
  TGeoShape* Shell_0 = new TGeoTubeSeg("Shell_0", Shell_rmax / 2. - .1, Shell_rmax / 2., 6.2 / 2., 12., 168.);
  TGeoShape* Shell_1 = new TGeoTubeSeg("Shell_1", Shell_rmin / 2. - .1, Shell_rmin / 2., 3.2 / 2., 0., 180.);
  new TGeoConeSeg("Shell_2", (25.15 + 1.0) / 2., Shell_rmin / 2. - .1, Shell_rmin / 2., Shell_rmax / 2. - .1, Shell_rmax / 2., 0., 180.);
  TGeoShape* Shell_3 = new TGeoTube("Shell_3", 0., Shell_rmin / 2. + .1, .1 / 2.);
  TGeoShape* ShellHole_0 = new TGeoTrd1("ShellHole_0", 17.5 / 4., 42.5 / 4., 80. / 2., (25.15 + 1.) / 2.);
  TGeoShape* ShellHole_1 = new TGeoBBox("ShellHole_1", 42.5 / 4., 80. / 2., 2. / 2. + 0.00001);
  TGeoShape* ShellHole_2 = new TGeoBBox("ShellHole_2", 58.9 / 4., (Shell_rmin - 2.25) / 2., .4 / 2. + 0.00001);
  TGeoShape* ShellHole_3 = new TGeoBBox("ShellHole_3", 80. / 4., (Shell_rmin - 11.6) / 2., .4 / 2. + 0.00001);

  // For the extra cut, not sure if this is the shape (apprx. distances)
  TGeoShape* Shell_Cut_0 = new TGeoTube("Shell_Cut_0", 0., 3.5, 5. / 2.);
  TGeoShape* Shell_Cut_1 = new TGeoBBox("Shell_Cut_1", 7. / 2., 25. / 2., 5. / 2.);

  // Composite shapes for Half_3
  auto* Half_3_Shape_0 = new TGeoCompositeShape("Half_3_Shape_0", "Shell_Cut_0+Shell_Cut_1:tShell_Cut");
  new TGeoCompositeShape("Half_3_Shape_1", "Shell_2 - Half_3_Shape_0:cShell_Cut - Half_3_Shape_0:cShell_Cut_inv");
  auto* Half_3_Shape_2 = new TGeoCompositeShape("Half_3_Shape_2", "ShellHole_0+ShellHole_1:tShellHole");
  new TGeoCompositeShape("Half_3_Shape_3", "Shell_3:tShellHole_1 -(ShellHole_2:tShellHole_1 + ShellHole_3:tShellHole_0)");
  auto* Half_3_Shape_4 = new TGeoCompositeShape("Half_3_Shape_4",
                                                "(Shell_0:tShell_0 + Half_3_Shape_1+ Shell_1:tShell_1) - (Half_3_Shape_2 + "
                                                "Half_3_Shape_2:Rz + Half_3_Shape_2:Rz_inv)+Half_3_Shape_3");

  auto* Half_3_Volume = new TGeoVolume("Half_3_Volume", Half_3_Shape_4, kMedAlu);
  // Position of the piece relative to the origin which for this code is the center of the the Framework piece (See
  // Half_2)
  // Half_3->AddNode(Half_3_Volume, 1, new TGeoTranslation(0., 0., -19.));

  TGeoRotation* rot_z180 = new TGeoRotation("rot_z180", 0, 180, 0); // orig: (180,0,0)
  rot_z180->RegisterYourself();
  // in y = l_253.5 - 6. center in (0,6,0)
  TGeoCombiTrans* combi_coat = new TGeoCombiTrans(0, 0, 19.5 - 0.45, rot_z180); // TGeoCombiTrans(0,0, -19.5,rot_z180)  // -0.5 ->0.45
  combi_coat->SetName("combi_coat");
  combi_coat->RegisterYourself();

  Half_3->AddNode(Half_3_Volume, 1, combi_coat);
  // Half_3_Volume->SetLineColor(1);

  HalfConeVolume->AddNode(stair, 1, c_final); //
  HalfConeVolume->AddNode(base, 2, c_final);
  HalfConeVolume->AddNode(rail_L_R, 3, c_final); // R&L
  HalfConeVolume->AddNode(Fra_front, 4, c_final);
  HalfConeVolume->AddNode(midle, 5, c_final); //
  HalfConeVolume->AddNode(Half_3, 6, c_final);
  // HalfConeVolume->AddNode(Half_3,8, new TGeoCombiTrans(0,0,0-0.5,rot_halfR)); //-0.675

  return HalfConeVolume;
}
