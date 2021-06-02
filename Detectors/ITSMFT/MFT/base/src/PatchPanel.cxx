// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file PatchPanel.cxx
/// \brief Class building the MFT Patch-Panel

#include "MFTBase/PatchPanel.h"
#include "TGeoBBox.h"
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

using namespace o2::mft;

ClassImp(o2::mft::PatchPanel);

//_____________________________________________________________________________
PatchPanel::PatchPanel() //: TNamed(), mPatchPanel(nullptr)
{
  createPatchPanel();
  // default constructor
}

//_____________________________________________________________________________
// PatchPanel::~PatchPanel() = default;

//_____________________________________________________________________________
TGeoVolumeAssembly* PatchPanel::createPatchPanel()
{

  auto* PatchPanelVolume = new TGeoVolumeAssembly("PatchPanelVolume");

  TGeoMedium* kMedAlu = gGeoManager->GetMedium("MFT_Alu$");
  TGeoMedium* kMedCu = gGeoManager->GetMedium("MFT_Cu$");

  /////////////////////////////////////   A ////////////////////////////

  // auto* face_A = new TGeoVolumeAssembly("face_A");
  // seg tub  disc ;)
  Double_t radin_disc = 28.0;  ////// 27.5;
  Double_t radout_disc = 50.3; // cm
  Double_t high_disc = 0.4;    // cm
  Double_t ang_in_disc = 180.;
  Double_t ang_fin_disc = 360;

  // seg tub 2  SCUT1 ;)
  Double_t radin_scut1 = 0;
  Double_t radout_scut1 = 15.5; //// 29.87; //cm
  Double_t high_scut1 = 2;      // cm
  Double_t ang_in_scut1 = 180;
  Double_t ang_fin_scut1 = 270;

  TGeoTranslation* tr_discL =
    new TGeoTranslation("tr_discL", -12.5, -10.4, 0); // left
  tr_discL->RegisterYourself();

  TGeoTranslation* tr_discR = new TGeoTranslation("tr_discR", 12.5, -10.4, 0);
  tr_discR->RegisterYourself();

  // seg tub 3  SCUT2 ;)
  Double_t radin_scut2 = 0;     ////// 29.87;
  Double_t radout_scut2 = 15.5; // cm
  Double_t high_scut2 = 2;      // cm
  Double_t ang_in_scut2 = 270;
  Double_t ang_fin_scut2 = 360.;

  // holes tub  1hole tranversal o3.5
  Double_t radin_holeB = 0.;
  Double_t radout_holeB = 0.175; // diameter 3.5 H11
  Double_t high_holeB = 1.5;     ///
  TGeoTranslation* tr1_holeB = new TGeoTranslation("tr1_holeB", -7.5, -28.8, 0);
  tr1_holeB->RegisterYourself();

  TGeoTranslation* tr2_holeB = new TGeoTranslation("tr2_holeB", 7.5, -28.8, 0);
  tr2_holeB->RegisterYourself();

  // box 1 |==|
  Double_t x_1box = 105.0;
  Double_t y_1box = 11.81; // 5.9 distanc
  Double_t z_1box = 1.4;

  // box 2   ;)
  Double_t x_2box = 20;   // cm
  Double_t y_2box = 60.6; // from origin 30.0397 cm , 60.0784
  Double_t z_2box = 2;

  /// triangular border right to cut partA and partB
  TGeoXtru* tria_cut1 = new TGeoXtru(2);
  tria_cut1->SetName("S_TRIA_CUT1");

  Double_t x_tria1[6] = {52, 45.6, 45.6, 37.17, 36.77, 52};
  Double_t y_tria1[6] = {-21.62, -21.62, -11.478, -5.9, -3, -3};
  tria_cut1->DefinePolygon(6, x_tria1, y_tria1);
  tria_cut1->DefineSection(0, -2.4, 0., 0.,
                           1); //(plane,-zplane/ +zplane, x0, y0,(x/y))
  tria_cut1->DefineSection(1, 2.4, 0., 0., 1);

  // triangular border left
  TGeoXtru* tria_cut2 = new TGeoXtru(2);
  tria_cut2->SetName("S_TRIA_CUT2");

  Double_t x_tria2[6] = {-52, -45.6, -45.6, -37.17, -38.06, -52};
  Double_t y_tria2[6] = {-21.62, -21.62, -11.478, -5.9, -3, -3};
  tria_cut2->DefinePolygon(6, x_tria2, y_tria2);
  tria_cut2->DefineSection(0, -2.45, 0., 0.,
                           1); //(plane,-zplane/ +zplane, x0, y0,(x/y))
  tria_cut2->DefineSection(1, 2.45, 0., 0., 1);

  // hole A
  Double_t radin_A1 = 0;      // diam 0
  Double_t radout_A1 = 0.223; // dia 0.246
  Double_t high_A1 = 10;      // dz 6

  TGeoTranslation* tr_A1 =
    new TGeoTranslation("tr_A1", -18.627, -24.278, 0); // A hole
  tr_A1->RegisterYourself();

  TGeoTranslation* tr_B = new TGeoTranslation("tr_B", 18.627, -24.278, 0);
  tr_B->RegisterYourself();

  TGeoTranslation* tr_II = new TGeoTranslation("tr_II", -25.25, -6.65, 0);
  tr_II->RegisterYourself();

  //
  TGeoTranslation* tr_H = new TGeoTranslation("tr_H", -26.092, -34.042, 0);
  tr_H->RegisterYourself();

  // shoulder
  TGeoXtru* shoulder = new TGeoXtru(2);
  shoulder->SetName("S_shoulder");

  /// auto* arms = new TGeoVolumeAssembly("arms");

  Double_t x_shoulder[4] = {-13.9, -13.9, 13.8,
                            13.8};                         // vertices to coincide with hone
  Double_t y_shoulder[4] = {-24.4, -26.45, -26.45, -24.4}; //

  shoulder->DefinePolygon(4, x_shoulder, y_shoulder);
  shoulder->DefineSection(0, 0., 0., 0., 1);
  shoulder->DefineSection(1, 4.8, 0., 0., 1); //

  // HANDS
  TGeoXtru* hand_L = new TGeoXtru(2);
  hand_L->SetName("hand_L"); // S_HANDL

  Double_t x_handL[12] = {-44.5, -35.89, -31.38, -30.53, -30, -30.,
                          -26.2, -24.98, -24.5, -24.5, -37.17, -45.8};
  Double_t y_handL[12] = {-13.42, -7.45, -7.45, -8.03, -8.91, -10.5,
                          -10.5, -9.76, -9.01, -5.9, -5.9, -11.5};
  hand_L->DefinePolygon(12, x_handL, y_handL);
  hand_L->DefineSection(0, 0., 0., 0.,
                        1); //(plane,-zplane/ +zplane, x0, y0,(x/y))
  hand_L->DefineSection(1, 4.8, 0., 0., 1);
  ///
  TGeoXtru* part_handL = new TGeoXtru(2);
  part_handL->SetName("S_PART_HAND_L");

  Double_t x_part_HL[4] = {-43.5, -43.5, -45.8, -45.8};
  Double_t y_part_HL[4] = {-21.6, -11.49, -11.49, -21.6};
  part_handL->DefinePolygon(4, x_part_HL, y_part_HL);
  part_handL->DefineSection(0, 0., 0., 0.,
                            1); //(plane,-zplane/ +zplane, x0, y0,(x/y))
  part_handL->DefineSection(1, 4.8, 0., 0., 1);
  //////////////
  TGeoRotation* rot_y180 = new TGeoRotation("rot_y180", 0, 180, 180);
  rot_y180->RegisterYourself();
  ///// right /////
  TGeoXtru* hand_R = new TGeoXtru(2);
  hand_R->SetName("hand_R");

  Double_t x_handR[12] = {44.5, 35.89, 31.38, 30.53, 30, 30.,
                          26.2, 24.98, 24.5, 24.5, 37.17, 45.8};
  Double_t y_handR[12] = {-13.42, -7.45, -7.45, -8.03, -8.91, -10.5,
                          -10.5, -9.76, -9.01, -5.9, -5.9, -11.5};
  hand_R->DefinePolygon(12, x_handR, y_handR);
  hand_R->DefineSection(0, 0., 0., 0., 1);
  hand_R->DefineSection(1, 4.8, 0., 0., 1);
  //////////////

  TGeoXtru* part_handR = new TGeoXtru(2);
  part_handR->SetName("part_handR");

  Double_t x_part_HR[4] = {43.5, 43.5, 45.8, 45.8};
  Double_t y_part_HR[4] = {-21.6, -11.75, -11.75, -21.6};
  part_handR->DefinePolygon(4, x_part_HR, y_part_HR);
  part_handR->DefineSection(0, 0., 0., 0., 1);
  part_handR->DefineSection(1, 4.8, 0., 0., 1);

  //////////////// horns
  Double_t radin_hornL = 7.0;
  Double_t radout_hornL = 9.05;
  Double_t high_hornL = 4.8;
  Double_t angin_hornL = 230;
  Double_t angfin_hornL = 270;

  TGeoTranslation* tr_hornl =
    new TGeoTranslation("tr_hornl", -13.8, -17.4, 2.2);
  tr_hornl->RegisterYourself();

  Double_t radin_hornR = 7.0;
  Double_t radout_hornR = 9.05;
  Double_t high_hornR = 4.8;
  Double_t angin_hornR = 270;
  Double_t angfin_hornR = 310;

  TGeoTranslation* tr_hornR = new TGeoTranslation("tr_hornR", 13.8, -17.4, 2.2);
  tr_hornR->RegisterYourself();

  // arm box
  Double_t x_Abox = 15.6;
  Double_t y_Abox = 1.5;
  Double_t z_Abox = 4.8;

  TGeoRotation* rot_zp54 = new TGeoRotation("rot2_zp54", 54, 0, 0);
  rot_zp54->RegisterYourself();
  TGeoCombiTrans* combi_rotzp54 = new TGeoCombiTrans(
    -22.46, -29.4, 2.3, rot_zp54); // to start in the lamine
  combi_rotzp54->SetName("combi_zp54");
  combi_rotzp54->RegisterYourself();

  TGeoRotation* rot_zn54 = new TGeoRotation("rot2_zn54", -54, 0, 0);
  rot_zn54->RegisterYourself();
  TGeoCombiTrans* combi_rotzn54 =
    new TGeoCombiTrans(22.46, -29.4, 2.3, rot_zn54); // y=
  combi_rotzn54->SetName("combi_zn54");
  combi_rotzn54->RegisterYourself();

  ///// smile  //seg tub
  Double_t radin_sm = 28.;
  Double_t radout_sm = 32;
  Double_t high_sm = 2;
  Double_t angin_sm = 251.79;
  Double_t angfin_sm = 288.21;

  ///// ext //seg tub U
  Double_t radin_cext = 49.6;
  Double_t radout_cext = 50.3;
  Double_t high_cext = 4.8;
  Double_t angin_cext = 227;
  Double_t angfin_cext = 313;

  TGeoTranslation* tr_cext =
    new TGeoTranslation("tr_cext", 0, 0, 2.3); // to put over the disc
  tr_cext->RegisterYourself();

  /// kiro_c //seg tub
  Double_t radin_kiroc = 48.35;
  Double_t radout_kiroc = 50.3;
  Double_t high_kiroc = 4.8;
  Double_t angin_kiroc = 256.5;
  Double_t angfin_kiroc = 283.5;

  // seg tub 1 hole
  Double_t radin_1hole = 23.0;
  Double_t radout_1hole = 25.5;
  Double_t high_1hole = 1.4;
  Double_t ang_in_1hole = 207.83;
  Double_t ang_fin_1hole = 249.998;

  //// circular central hole1 to conexion with other parts
  Double_t radin_hole1 = 0;
  Double_t radout_hole1 = 0.4;
  Double_t high_hole1 = 1.36;

  // circular hole2 ; hole2 r=6.7
  Double_t radin_hole2 = 0;
  Double_t radout_hole2 = 0.335;
  Double_t high_hole2 = 1.36;

  // box 4 lamine 1
  Double_t x_labox = 60.0;
  Double_t y_labox = 30.3;
  Double_t z_labox = 0.305;
  TGeoTranslation* tr_la =
    new TGeoTranslation("tr_la", 0, -y_labox / 2 - 9.3, high_disc / 2); //
  tr_la->RegisterYourself();

  TGeoTranslation* tr_2la =
    new TGeoTranslation("tr_2la", 0, -8.1, high_disc / 2); //
  tr_2la->RegisterYourself();

  /////box 5   lamin 2
  Double_t x_2labox = 51.2;
  Double_t y_2labox = 2.8; //
  Double_t z_2labox = 0.303;

  // cut lateral Left
  Double_t radin_cutlatL = 48.;
  Double_t radout_cutlatL = 51.0;
  Double_t high_cutlatL = 3;
  Double_t angin_cutlatL = 208;
  Double_t angfin_cutlatL = 227.;

  // cut lateral Right
  Double_t radin_cutlatR = 48.;
  Double_t radout_cutlatR = 51.0;
  Double_t high_cutlatR = 3;
  Double_t angin_cutlatR = 313.;
  Double_t angfin_cutlatR = 332;

  // small section of disc
  Double_t radin_slimdisL = 48.;
  Double_t radout_slimdisL = 50.3;
  Double_t high_slimdisL = 4.79;
  Double_t angin_slimdisL = 205;
  Double_t angfin_slimdisL = 208.;

  TGeoTranslation* tr_slimL =
    new TGeoTranslation("tr_slimL", 0, 0, high_slimdisL / 2 - 0.1); //
  tr_slimL->RegisterYourself();

  // small section of disc R
  Double_t radin_slimdisR = 48.;
  Double_t radout_slimdisR = 50.3;
  Double_t high_slimdisR = 4.79;
  Double_t angin_slimdisR = 332;
  Double_t angfin_slimdisR = 335.;

  // piramide
  TGeoXtru* pyramid = new TGeoXtru(2);
  pyramid->SetName("pyramid");

  Double_t x_pyramid[4] = {-1.2, 1.2, 1.4, -1.4};
  Double_t y_pyramid[4] = {-26.4 - 1.05, -26.4 - 1.05, -26.4, -26.4};
  pyramid->DefinePolygon(4, x_pyramid, y_pyramid);
  pyramid->DefineSection(0, 0., 0., 0.,
                         1); //(plane,-zplane/ +zplane, x0, y0,(x/y))
  pyramid->DefineSection(1, 4.8, 0., 0., 1);
  ////////////////

  TGeoXtru* tanqL = new TGeoXtru(2);
  tanqL->SetName("tanqL");

  Double_t x_tanqL[6] = {-29., -26.78, -25.57, -26.2, -27.6, -27.25};
  Double_t y_tanqL[6] = {-41., -41.9, -39.534, -36., -35.2, -37.8}; //
  tanqL->DefinePolygon(6, x_tanqL, y_tanqL);
  tanqL->DefineSection(0, 0., 0., 0., 1);
  tanqL->DefineSection(1, 1.65, 0., 0., 1);
  ////////////////

  TGeoXtru* tanqR = new TGeoXtru(2);
  tanqR->SetName("tanqR");

  Double_t x_tanqR[6] = {29., 26.78, 25.57, 26.2, 27.6, 27.25};
  Double_t y_tanqR[6] = {-41., -41.9, -39.534, -36., -35.2, -37.8}; //
  tanqR->DefinePolygon(6, x_tanqR, y_tanqR);
  tanqR->DefineSection(0, 0., 0., 0., 1);
  tanqR->DefineSection(1, 1.65, 0., 0., 1);

  // eyess L
  TGeoXtru* frog_eyeL = new TGeoXtru(2);
  frog_eyeL->SetName("frog_eyeL");

  Double_t x_frog_eyeL[4] = {-13.33, -10.72, -11.11, -11.89};
  Double_t y_frog_eyeL[4] = {-47.78, -48.61, -45.41, -45.41};
  frog_eyeL->DefinePolygon(4, x_frog_eyeL, y_frog_eyeL);
  frog_eyeL->DefineSection(0, 0., 0., 0., 1);
  frog_eyeL->DefineSection(1, 1.65, 0., 0., 1);
  ////////////////
  // eyess R
  TGeoXtru* frog_eyeR = new TGeoXtru(2);
  frog_eyeR->SetName("frog_eyeR");

  Double_t x_frog_eyeR[4] = {13.33, 10.72, 11.11, 11.89};
  Double_t y_frog_eyeR[4] = {-47.78, -48.61, -45.41, -45.41};
  frog_eyeR->DefinePolygon(4, x_frog_eyeR, y_frog_eyeR);
  frog_eyeR->DefineSection(0, 0., 0., 0., 1);
  frog_eyeR->DefineSection(1, 1.65, 0., 0., 1);

  TGeoRotation* rot_A = new TGeoRotation("rot_A", 180, 180, 0);
  rot_A->SetName("rot_A");
  rot_A->RegisterYourself();

  //////////////// new cut border slide
  Double_t radin_slideL = 48;
  Double_t radout_slideL = 53; //
  Double_t high_slideL = 3.15; //
  Double_t angin_slideL = 226.5;
  Double_t angfin_slideL = 228.8;

  TGeoTranslation* tr_slide =
    new TGeoTranslation("tr_slide", 0, 0, 4.832); //  2.39
  tr_slide->RegisterYourself();

  //////////////// new cut border slide
  Double_t radin_slideR = 48;
  Double_t radout_slideR = 53; /// s8.5;
  Double_t high_slideR = 3.15; // 31.44
  Double_t angin_slideR = 311.2;
  Double_t angfin_slideR = 313.5;

  //////backear
  TGeoXtru* earL = new TGeoXtru(2);
  earL->SetName("earL");

  Double_t x_earL[8] = {-44., -47.5, -47.5, -44.61,
                        -44.08, -44.08, -43.59, -43.59};
  Double_t y_earL[8] = {-21.39, -21.39, -14.74, -14.74,
                        -15.14, -19.47, -20.08, -20.97}; //
  earL->DefinePolygon(8, x_earL, y_earL);
  earL->DefineSection(0, 0.5, 0., 0.,
                      1); //(plane,-zplane/ +zplane, x0, y0,(x/y))
  earL->DefineSection(1, 1.2, 0., 0., 1);

  TGeoXtru* earR = new TGeoXtru(2);
  earR->SetName("earR");

  Double_t x_earR[8] = {44., 47.5, 47.5, 44.61, 44.08, 44.08, 43.59, 43.59};
  Double_t y_earR[8] = {-21.39, -21.39, -14.74, -14.74,
                        -15.14, -19.47, -20.08, -20.97}; //
  earR->DefinePolygon(8, x_earR, y_earR);
  earR->DefineSection(0, -0.5, 0., 0.,
                      1); //(plane,-zplane/ +zplane, x0, y0,(x/y))
  earR->DefineSection(1, 1.2, 0., 0., 1);

  ///////// shape for base --new

  new TGeoTube("S_CIRC_AHOLE", radin_A1, radout_A1, high_A1 / 2);

  new TGeoTubeSeg("S_DISC", radin_disc, radout_disc, high_disc / 2, ang_in_disc,
                  ang_fin_disc);

  new TGeoTubeSeg("S_SCUT1", radin_scut1, radout_scut1, high_scut1 / 2,
                  ang_in_scut1, ang_fin_scut1);

  new TGeoTubeSeg("S_SCUT2", radin_scut2, radout_scut2, high_scut2 / 2,
                  ang_in_scut2, ang_fin_scut2);

  // new TGeoTubeSeg("S_ARM", radin_arm, radout_arm, high_arm / 2, angin_arm,
  // angfin_arm); new TGeoTubeSeg("S_ARM_R", radin_armR, radout_armR, high_armR /
  // 2, angin_armR, angfin_armR);

  new TGeoBBox("Abox", x_Abox / 2, y_Abox / 2, z_Abox / 2);

  new TGeoTubeSeg("smile", radin_sm, radout_sm, high_sm / 2, angin_sm,
                  angfin_sm);

  new TGeoTubeSeg("c_ext", radin_cext, radout_cext, high_cext / 2, angin_cext,
                  angfin_cext);

  new TGeoTubeSeg("kiroc", radin_kiroc, radout_kiroc, high_kiroc / 2,
                  angin_kiroc, angfin_kiroc);

  new TGeoTubeSeg("cutlatL", radin_cutlatL, radout_cutlatL, high_cutlatL / 2,
                  angin_cutlatL, angfin_cutlatL);

  new TGeoTubeSeg("cutlatR", radin_cutlatR, radout_cutlatR, high_cutlatR / 2,
                  angin_cutlatR, angfin_cutlatR);

  new TGeoTubeSeg("slimdisL", radin_slimdisL, radout_slimdisL,
                  high_slimdisL / 2, angin_slimdisL, angfin_slimdisL);

  new TGeoTubeSeg("slimdisR", radin_slimdisR, radout_slimdisR,
                  high_slimdisR / 2, angin_slimdisR, angfin_slimdisR);

  new TGeoBBox("BOX1", x_1box / 2, y_1box / 2, z_1box / 2);
  new TGeoBBox("BOX2", x_2box / 2, y_2box / 2, z_2box / 2);

  new TGeoBBox("LA_BOX", x_labox / 2, y_labox / 2, z_labox / 2);

  new TGeoBBox("LA_2BOX", x_2labox / 2, y_2labox / 2, z_2labox / 2);

  new TGeoTubeSeg("SEG_1HOLE", radin_1hole, radout_1hole, high_1hole / 2,
                  ang_in_1hole, ang_fin_1hole);

  new TGeoTubeSeg("S_SEG_HORNL", radin_hornL, radout_hornL, high_hornL / 2,
                  angin_hornL, angfin_hornL);

  new TGeoTubeSeg("S_SEG_HORNR", radin_hornR, radout_hornR, high_hornR / 2,
                  angin_hornR, angfin_hornR);

  new TGeoTube("S_CIRC_HOLE1", radin_hole1, radout_hole1, high_hole1 / 2);

  new TGeoTube("S_CIRC_HOLE2", radin_hole2, radout_hole2, high_hole2 / 2);

  new TGeoTube("S_CIRC_HOLEB", radin_holeB, radout_holeB, high_holeB / 2);

  new TGeoTubeSeg("s_slideL", radin_slideL, radout_slideL, high_slideL / 2,
                  angin_slideL, angfin_slideL);

  new TGeoTubeSeg("s_slideR", radin_slideR, radout_slideR, high_slideR / 2,
                  angin_slideR, angfin_slideR);

  //// composite shape for base new

  new TGeoCompositeShape(
    "baseA_Shape_0",
    "S_DISC -BOX1 -S_TRIA_CUT1 -S_TRIA_CUT2  -BOX2 "
    "-SEG_1HOLE -S_SCUT1:tr_discL -S_SCUT2:tr_discR  -smile "
    "-cutlatL - cutlatR ");

  new TGeoCompositeShape(
    "baseA_Shape_1",
    "S_shoulder  - S_CIRC_AHOLE:tr_H  + S_SEG_HORNL:tr_hornl +  "
    "S_SEG_HORNR:tr_hornR +Abox:combi_zp54 +Abox:combi_zn54 +pyramid "
    " ");

  new TGeoCompositeShape(
    "baseA_Shape_2",
    " S_PART_HAND_L +hand_L +hand_R +part_handR  -S_CIRC_AHOLE:tr_B "
    "- S_CIRC_AHOLE:tr_II -S_CIRC_AHOLE:tr_A1 +c_ext:tr_cext  "
    "+kiroc:tr_cext +slimdisL:tr_slimL +slimdisR:tr_slimL + tanqL "
    "+tanqR +frog_eyeL +frog_eyeR - s_slideL:tr_slide - "
    "s_slideR:tr_slide");

  new TGeoCompositeShape("baseA_Shape_3",
                         " (baseA_Shape_0  + baseA_Shape_2 - earL - "
                         "earR):rot_A + baseA_Shape_1:rot_A");

  ////////////////////////////////////////   B   /////////////////////////

  // auto* face_B = new TGeoVolumeAssembly("face_B");
  // principal  disc ;)
  Double_t radin_discB = 0.;    ////// 27.5;f
  Double_t radout_discB = 50.3; // cm
  Double_t high_discB = 0.2;    // cm
  Double_t angin_discB = 180.;
  Double_t angfin_discB = 360;

  /// central cut
  TGeoXtru* central_cut = new TGeoXtru(2);
  central_cut->SetName("central_cut");
  /// M,N,L,O,P,Q
  Double_t x_central[6] = {-24.5, -16.5, 16.5, 24.5, 24.5, -24.5};
  Double_t y_central[6] = {-18.33, -24.28, -24.28, -18.33, 1, -1};
  central_cut->DefinePolygon(6, x_central, y_central);
  central_cut->DefineSection(0, -2.4, 0., 0.,
                             1); //(plane,-zplane/ +zplane, x0, y0,(x/y))
  central_cut->DefineSection(1, 2.4, 0., 0., 1);

  // nhawi box
  Double_t x_wibox = 14.628;
  Double_t y_wibox = 0.5;
  Double_t z_wibox = 0.7;

  TGeoTranslation* tr_wiL = new TGeoTranslation(
    "tr_wiL", -11.013, -27.5, 0.35 - 0.09); // z_wibox/2 - high_discB/2)
  tr_wiL->RegisterYourself();

  TGeoTranslation* tr_wiR =
    new TGeoTranslation("tr_wiR", 11.013, -27.5, 0.35 - 0.09);
  tr_wiR->RegisterYourself();

  // vertical_ box
  Double_t x_vbox = 0.5; //
  Double_t y_vbox = 2.1; //
  Double_t z_vbox = 0.7;

  TGeoTranslation* tr_vboxL =
    new TGeoTranslation("tr_vboxL", -26.55, -12.4, 0.35 - 0.09);
  tr_vboxL->RegisterYourself();

  TGeoTranslation* tr_vboxR =
    new TGeoTranslation("tr_vboxR", 26.55, -12.4, 0.35 - 0.09);
  tr_vboxR->RegisterYourself();

  // eyebrow
  TGeoXtru* eyebrowL = new TGeoXtru(2);
  eyebrowL->SetName("eyebrowL"); // S_HANDL

  Double_t x_eyebrowL[8] = {-43.45, -42.95, -42.943, -35.806,
                            -32.566, -32.566, -35.76, -43.45};
  Double_t y_eyebrowL[8] = {-16.61, -16.61, -12.59, -7.99,
                            -7.760, -7.26, -7.41, -12.47}; //
  eyebrowL->DefinePolygon(8, x_eyebrowL, y_eyebrowL);
  eyebrowL->DefineSection(0, -0.09, 0., 0.,
                          1); //(plane,-zplane/ +zplane, x0, y0,(x/y))
  eyebrowL->DefineSection(1, 0.5, 0., 0., 1);
  // eyebrow
  TGeoXtru* eyebrowR = new TGeoXtru(2);
  eyebrowR->SetName("eyebrowR");

  Double_t x_eyebrowR[8] = {43.45, 42.95, 42.943, 35.806,
                            32.566, 32.566, 35.76, 43.45};
  Double_t y_eyebrowR[8] = {-16.61, -16.61, -12.59, -7.99,
                            -7.760, -7.26, -7.41, -12.47}; //
  eyebrowR->DefinePolygon(8, x_eyebrowR, y_eyebrowR);
  eyebrowR->DefineSection(0, -0.09, 0., 0.,
                          1); //(plane,-zplane/ +zplane, x0, y0,(x/y))
  eyebrowR->DefineSection(1, 0.5, 0., 0., 1);

  // axe
  TGeoXtru* axeR = new TGeoXtru(2);
  axeR->SetName("axeR");
  Double_t x_axeR[8] = {33.25, 32.26, 29.86, 29.53, 26.52, 23.29, 33.6, 34.35};
  Double_t y_axeR[8] = {-29.35, -30.42, -33.30, -33.10,
                        -35.76, -30.98, -27.27, -27.3};
  axeR->DefinePolygon(8, x_axeR, y_axeR);
  axeR->DefineSection(0, -0.09, 0., 0., 1);
  axeR->DefineSection(1, 1.6, 0., 0., 1);
  //
  TGeoXtru* axeL = new TGeoXtru(2);
  axeL->SetName("axeL");
  Double_t x_axeL[8] = {-33.25, -32.26, -29.86, -29.53,
                        -26.52, -23.29, -33.6, -34.35};
  Double_t y_axeL[8] = {-29.35, -30.42, -33.30, -33.10,
                        -35.76, -30.98, -27.27, -27.3};
  axeL->DefinePolygon(8, x_axeL, y_axeL);
  axeL->DefineSection(0, -0.09, 0., 0., 1);
  axeL->DefineSection(1, 1.6, 0., 0., 1);
  ////////////////

  //////shark
  TGeoXtru* sharkL = new TGeoXtru(2);
  sharkL->SetName("sharkL");
  Double_t x_sharkL[13] = {-44.64, -43.62, -34.95, -37.79, -37.39,
                           -35.47, -33.86, -27.97, -29.49, -34.92,
                           -37.89, -41.48, -44.68};
  Double_t y_sharkL[13] = {-20.61, -22.89, -31.7, -26.54, -26.05,
                           -26.98, -28.15, -9.31, -9.31, -24.58,
                           -21.85, -19.13, -17.2};
  sharkL->DefinePolygon(13, x_sharkL, y_sharkL);
  sharkL->DefineSection(0, -0.09, 0., 0., 1);
  sharkL->DefineSection(1, 1.6, 0., 0., 1);

  TGeoXtru* sharkR = new TGeoXtru(2);
  sharkR->SetName("sharkR");
  Double_t x_sharkR[13] = {44.64, 43.62, 34.9, 37.79, 37.39, 35.47, 33.86,
                           27.97, 29.49, 34.92, 37.89, 41.48, 44.68};
  Double_t y_sharkR[13] = {-20.61, -22.89, -31.7, -26.54, -26.05,
                           -26.98, -28.15, -9.31, -9.31, -24.58,
                           -21.85, -19.13, -17.2};
  sharkR->DefinePolygon(13, x_sharkR, y_sharkR);
  sharkR->DefineSection(0, -0.09, 0., 0., 1);
  sharkR->DefineSection(1, 1.6, 0., 0., 1);

  //////////////
  Double_t radin_boatL = 47.2;  //
  Double_t radout_boatL = 50.3; // cm
  Double_t high_boatL = 1.6;    // cm
  Double_t angin_boatL = 208.;
  Double_t angfin_boatL = 228.75; // 48.75

  TGeoTranslation* tr_boatL = new TGeoTranslation("tr_boatL", 0, 0, 0.79);
  tr_boatL->RegisterYourself();

  // bo
  Double_t radin_boatR = 47.2;  ////// 27.5;f
  Double_t radout_boatR = 50.3; // cm
  Double_t high_boatR = 1.6;    // cm
  Double_t angin_boatR = 311.25;
  Double_t angfin_boatR = 332.;

  TGeoTranslation* tr_boatR = new TGeoTranslation("tr_boatR", 0, 0, 0.79);
  tr_boatR->RegisterYourself();

  ////////////// \//arc_cut
  Double_t radin_arcutL = 46.;
  Double_t radout_arcutL = 50.5;
  Double_t high_arcutL = 1.6;
  Double_t angin_arcutL = 230.65; // angY 55.65
  Double_t angfin_arcutL = 256.7; // angY 76.5  (256.5)

  TGeoTranslation* tr_arcutL = new TGeoTranslation("tr_arcutL", 0, 0, 0.79);
  tr_arcutL->RegisterYourself();

  ////////////// \//arc_cut
  Double_t radin_arcutR = 46.;     ////// 27.5;
  Double_t radout_arcutR = 50.5;   // cm
  Double_t high_arcutR = 1.6;      // cm
  Double_t angin_arcutR = 283.4;   // angY 50.65
  Double_t angfin_arcutR = 309.35; // angY 76.5
  ///////////////////

  /// canine left  to cut
  TGeoXtru* canine_cutL = new TGeoXtru(2);
  canine_cutL->SetName("canine_cutL");
  Double_t x_canine_cutL[6] = {-33.06, -28.51, -26.38,
                               -26.41, -28.61, -31.62}; //
  Double_t y_canine_cutL[6] = {-38.93, -41.94, -37.85, -36.83, -34.9, -38.67};
  canine_cutL->DefinePolygon(6, x_canine_cutL, y_canine_cutL);
  canine_cutL->DefineSection(0, -2.4, 0., 0., 1);
  canine_cutL->DefineSection(1, 2.4, 0., 0., 1);

  /// canine right  to cut
  TGeoXtru* canine_cutR = new TGeoXtru(2);
  canine_cutR->SetName("canine_cutR");
  Double_t x_canine_cutR[6] = {33.06, 28.51, 26.38, 26.41, 28.61, 31.62}; //
  Double_t y_canine_cutR[6] = {-38.93, -41.94, -37.85, -36.83, -34.9, -38.67};
  canine_cutR->DefinePolygon(6, x_canine_cutR, y_canine_cutR);
  canine_cutR->DefineSection(0, -2.4, 0., 0., 1);
  canine_cutR->DefineSection(1, 2.4, 0., 0., 1);

  /// triangle dawn left
  TGeoXtru* triacut_downL = new TGeoXtru(2);
  triacut_downL->SetName("triacut_downL");
  Double_t x_triacut_downL[3] = {-16.52, -10.57, -10.57}; //
  Double_t y_triacut_downL[3] = {-50, -50, -44.96};       //
  triacut_downL->DefinePolygon(3, x_triacut_downL, y_triacut_downL);
  triacut_downL->DefineSection(0, -2.4, 0., 0., 1);
  triacut_downL->DefineSection(1, 2.4, 0., 0., 1);

  /// triangle dawn rigth
  TGeoXtru* triacut_downR = new TGeoXtru(2);
  triacut_downR->SetName("triacut_downR");
  Double_t x_triacut_downR[3] = {16.52, 10.57, 10.57}; //
  Double_t y_triacut_downR[3] = {-50, -50, -44.96};    //
  triacut_downR->DefinePolygon(3, x_triacut_downR, y_triacut_downR);
  triacut_downR->DefineSection(0, -2.4, 0., 0., 1);
  triacut_downR->DefineSection(1, 2.4, 0., 0., 1);

  //////////////lip
  Double_t radin_lip = 47.7;  //////
  Double_t radout_lip = 48.2; // mm
  Double_t high_lip = 0.69;   // mm
  Double_t angin_lip = 258.;  // angY 12
  Double_t angfin_lip = 282.; // angY   282

  TGeoTranslation* tr_lip = new TGeoTranslation("tr_lip", 0, 0, 0.35 - 0.09);
  tr_lip->RegisterYourself();

  /// part lip left
  TGeoXtru* lip_cornerL = new TGeoXtru(2);
  lip_cornerL->SetName("lip_cornerL");

  Double_t x_lip_cornerL[7] = {-10.59, -10.02, -9.64, -10.09,
                               -10.09, -10.76, -10.59}; //
  Double_t y_lip_cornerL[7] = {-46.48, -47.15, -46.88, -46.32,
                               -44.86, -44.79, -45.26};
  lip_cornerL->DefinePolygon(7, x_lip_cornerL, y_lip_cornerL);
  lip_cornerL->DefineSection(0, -0.09, 0., 0., 1);
  lip_cornerL->DefineSection(1, 0.6, 0., 0., 1);

  /// part lip right
  TGeoXtru* lip_cornerR = new TGeoXtru(2);
  lip_cornerR->SetName("lip_cornerR");

  Double_t x_lip_cornerR[6] = {10.59, 10.02, 9.64, 10.09, 10.09, 10.59}; //
  Double_t y_lip_cornerR[6] = {-46.48, -47.15, -46.88, -46.32, -44.51, -44.51};
  lip_cornerR->DefinePolygon(6, x_lip_cornerR, y_lip_cornerR);
  lip_cornerR->DefineSection(0, -0.09, 0., 0., 1);
  lip_cornerR->DefineSection(1, 0.6, 0., 0., 1);

  /// tears left
  TGeoXtru* tear_L = new TGeoXtru(2);
  tear_L->SetName("tear_L");

  Double_t x_tear_L[4] = {-24.12, -23.71, -20.16, -20.55}; //
  Double_t y_tear_L[4] = {-29.69, -29.99, -24.97, -24.78};
  tear_L->DefinePolygon(4, x_tear_L, y_tear_L);
  tear_L->DefineSection(0, -0.09, 0., 0., 1);
  tear_L->DefineSection(1, 0.6, 0., 0., 1);

  /// tears R
  TGeoXtru* tear_R = new TGeoXtru(2);
  tear_R->SetName("tear_R");

  Double_t x_tear_R[4] = {24.12, 23.71, 20.16, 20.55}; //
  Double_t y_tear_R[4] = {-29.69, -29.99, -24.97, -24.78};
  tear_R->DefinePolygon(4, x_tear_R, y_tear_R);
  tear_R->DefineSection(0, -0.09, 0., 0.,
                        1); //(plane,-zplane/ +zplane, x0, y0,(x/y))
  tear_R->DefineSection(1, 0.6, 0., 0., 1);

  TGeoTranslation* tra_B = new TGeoTranslation("tra_B", 0, 0, -4.7); //-4.8
  tra_B->RegisterYourself();

  new TGeoTubeSeg("S_DISCB", radin_discB, radout_discB, high_discB / 2,
                  angin_discB, angfin_discB);

  new TGeoBBox("nhawi_box", x_wibox / 2, y_wibox / 2, z_wibox / 2);

  new TGeoBBox("vert_box", x_vbox / 2, y_vbox / 2, z_vbox / 2);

  new TGeoTubeSeg("boatL", radin_boatL, radout_boatL, high_boatL / 2,
                  angin_boatL, angfin_boatL);

  new TGeoTubeSeg("boatR", radin_boatR, radout_boatR, high_boatR / 2,
                  angin_boatR, angfin_boatR);

  new TGeoTubeSeg("arcutL", radin_arcutL, radout_arcutL, high_arcutL / 2,
                  angin_arcutL, angfin_arcutL);

  new TGeoTubeSeg("arcutR", radin_arcutR, radout_arcutR, high_arcutR / 2,
                  angin_arcutR, angfin_arcutR);

  new TGeoTubeSeg("lip", radin_lip, radout_lip, high_lip / 2, angin_lip,
                  angfin_lip);

  //// composite shape for base ----

  new TGeoCompositeShape(
    "baseB_Shape_0",
    "S_DISCB -BOX1 -S_TRIA_CUT1 -S_TRIA_CUT2 -central_cut - "
    "S_CIRC_AHOLE:tr_B - arcutL -arcutR - canine_cutL "
    "-canine_cutR - triacut_downL - triacut_downR");

  new TGeoCompositeShape(
    "baseB_Shape_1",
    " nhawi_box:tr_wiL +nhawi_box:tr_wiR + vert_box:tr_vboxL + "
    "vert_box:tr_vboxR  +eyebrowL +eyebrowR  + axeR + axeL + sharkL + sharkR "
    "+ boatL:tr_boatL + boatR:tr_boatR + lip:tr_lip + lip_cornerL + "
    "lip_cornerR + tear_L + tear_R");

  new TGeoCompositeShape("baseB_Shape_2",
                         " baseB_Shape_0:tra_B + baseB_Shape_1:tra_B");

  auto* patchpanel_Shape = new TGeoCompositeShape(
    "patchpanel_Shape", "  baseA_Shape_3 + baseB_Shape_2");

  auto* patchpanel_Volume =
    new TGeoVolume("patchpanel_Volume", patchpanel_Shape, kMedAlu);

  //====== Contents of the patch panel (cables, pipes, cards) coded as plates ======
  Double_t radin_pl0 = 33;   // inner radius
  Double_t radout_pl0 = 45;  // outer radius
  Double_t high_pl0 = 0.15;  // thickness
  Double_t angin_pl0 = 20.;  // theta min
  Double_t angfin_pl0 = 30.; // theta max

  Double_t radin_pl1 = 33;
  Double_t radout_pl1 = 49;
  Double_t high_pl1 = 0.15;
  Double_t angin_pl1 = 31.;
  Double_t angfin_pl1 = 49.;

  //=== Central part with high density of materials ==
  Double_t radin_pl2 = 32;
  Double_t radout_pl2 = 49;
  Double_t high_pl2 = 0.3;
  Double_t angin_pl2 = 57.;
  Double_t angfin_pl2 = 75.;

  Double_t radin_pl3 = 29;
  Double_t radout_pl3 = 47;
  Double_t high_pl3 = 0.3;
  Double_t angin_pl3 = 75.5;
  Double_t angfin_pl3 = 104.5;

  Double_t radin_pl4 = 32;
  Double_t radout_pl4 = 49;
  Double_t high_pl4 = 0.3;
  Double_t angin_pl4 = 105.;
  Double_t angfin_pl4 = 122.;
  //===================================================

  Double_t radin_pl5 = 33;
  Double_t radout_pl5 = 49;
  Double_t high_pl5 = 0.15;
  Double_t angin_pl5 = 131;
  Double_t angfin_pl5 = 149;

  Double_t radin_pl6 = 33;
  Double_t radout_pl6 = 45;
  Double_t high_pl6 = 0.15;
  Double_t angin_pl6 = 150;
  Double_t angfin_pl6 = 160;

  auto* plate_0 = new TGeoTubeSeg("plate_0", radin_pl0, radout_pl0, high_pl0 / 2, 180. + angin_pl0, 180. + angfin_pl0);
  auto* plate_1 = new TGeoTubeSeg("plate_1", radin_pl1, radout_pl1, high_pl1 / 2, 180. + angin_pl1, 180. + angfin_pl1);
  auto* plate_2 = new TGeoTubeSeg("plate_2", radin_pl2, radout_pl2, high_pl2 / 2, 180. + angin_pl2, 180. + angfin_pl2);
  auto* plate_3 = new TGeoTubeSeg("plate_3", radin_pl3, radout_pl3, high_pl3 / 2, 180. + angin_pl3, 180. + angfin_pl3);
  auto* plate_4 = new TGeoTubeSeg("plate_4", radin_pl4, radout_pl4, high_pl4 / 2, 180. + angin_pl4, 180. + angfin_pl4);
  auto* plate_5 = new TGeoTubeSeg("plate_5", radin_pl5, radout_pl5, high_pl5 / 2, 180. + angin_pl5, 180. + angfin_pl5);
  auto* plate_6 = new TGeoTubeSeg("plate_6", radin_pl6, radout_pl6, high_pl6 / 2, 180. + angin_pl6, 180. + angfin_pl6);

  auto* plate_Shape = new TGeoCompositeShape("plate_Shape", "plate_0 + plate_1 + plate_2 + plate_3 + plate_4 + plate_5 + plate_6");
  auto* plate_Volume = new TGeoVolume("plate_Volume", plate_Shape, kMedCu);
  auto* tr_pl = new TGeoTranslation("tr_pl", 0, 0, -2.4);
  tr_pl->RegisterYourself();

  auto* tr_fin = new TGeoTranslation("tr_fin", 0, 0, -0.2);
  tr_fin->RegisterYourself();

  patchpanel_Volume->SetLineColor(kGreen - 9);
  PatchPanelVolume->AddNode(patchpanel_Volume, 1, tr_fin);
  PatchPanelVolume->AddNode(plate_Volume, 2, tr_pl);

  return PatchPanelVolume;
}
