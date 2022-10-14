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

/// \file PowerSupplyUnit.cxx
/// \brief Class building the MFT Power Supply Unit
/// \author Satoshi Yano

#include "TMath.h"
#include "TGeoManager.h"
#include "TGeoCompositeShape.h"
#include "TGeoTube.h"
#include "TGeoTorus.h"
#include "TGeoCone.h"
#include "TGeoBoolNode.h"
#include "TGeoBBox.h"
#include "TGeoSphere.h"
#include "TGeoVolume.h"
#include <fairlogger/Logger.h>
#include "MFTBase/Constants.h"
#include "MFTBase/PowerSupplyUnit.h"
#include "MFTBase/Geometry.h"

using namespace o2::mft;

ClassImp(o2::mft::PowerSupplyUnit);

//_____________________________________________________________________________
PowerSupplyUnit::PowerSupplyUnit()
{
  create();
}
//_____________________________________________________________________________

TGeoVolumeAssembly* PowerSupplyUnit::create()
{

  //TGeoManager *gManager = new TGeoManager();

  TGeoMedium* kMedPeek = gGeoManager->GetMedium("MFT_PEEK$");
  TGeoMedium* kMed_Water = gGeoManager->GetMedium("MFT_Water$");
  TGeoMedium* kMedAlu = gGeoManager->GetMedium("MFT_Alu$");
  TGeoMedium* kMedPolyPipe = gGeoManager->GetMedium("MFT_Polyimide$");

  TGeoVolumeAssembly* mHalfPSU = new TGeoVolumeAssembly("PSU");

  Double_t delta_thickness = 0.2;

  Double_t water_pipe_main_position_radius = (18.62 + 18.18) / 2.;

  Double_t block_widest_angle = 126.07;
  Double_t block_radius = (19.5 + 17.7) / 2.;
  Double_t block_angle_index[] = {126.08, 114.03, 101.91, 89.96, 77.88, 66.06, 54.16, 42.06, 30, 17.99, 6.07};

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Middle Spacer
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  std::string name_middle_spacer_main_cover = "";

  //main
  Double_t middle_spacer_main_thickness = 0.50;
  Double_t middle_spacer_main_min_radius1 = 17.0;
  Double_t middle_spacer_main_max_radius1 = 21.7;
  Double_t middle_spacer_main_sub_rectangle_top_height = 5 + 2.787;
  Double_t middle_spacer_main_sub_rectangle_top_width = middle_spacer_main_max_radius1 * 2;

  TGeoTubeSeg* middle_spacer_main_arc = new TGeoTubeSeg("middle_spacer_main_arc", middle_spacer_main_min_radius1, middle_spacer_main_max_radius1, middle_spacer_main_thickness / 2, 180, 0);
  TGeoBBox* middle_spacer_main_sub_rectangle_top = new TGeoBBox("middle_spacer_main_sub_rectangle_top", middle_spacer_main_sub_rectangle_top_width / 2., middle_spacer_main_sub_rectangle_top_height / 2., middle_spacer_main_thickness / 2. + delta_thickness);
  TGeoTranslation* trans_middle_spacer_main_sub_rectangle_top = new TGeoTranslation("trans_middle_spacer_main_sub_rectangle_top", 0, -middle_spacer_main_sub_rectangle_top_height / 2, 0);
  trans_middle_spacer_main_sub_rectangle_top->RegisterYourself();

  Double_t middle_spacer_main_add_rectangle_side_height = 4.5 + 1.5;
  Double_t middle_spacer_main_add_rectangle_side_width = 9.0;
  TGeoBBox* middle_spacer_main_add_rectangle_side = new TGeoBBox("middle_spacer_main_add_rectangle_side", middle_spacer_main_add_rectangle_side_width / 2., middle_spacer_main_add_rectangle_side_height / 2., middle_spacer_main_thickness / 2.);
  TGeoTranslation* trans_middle_spacer_main_add_rectangle_side_left = new TGeoTranslation("trans_middle_spacer_main_add_rectangle_side_left", 28.4 / 2. + middle_spacer_main_add_rectangle_side_width / 2., -5 - middle_spacer_main_add_rectangle_side_height / 2., 0);
  TGeoTranslation* trans_middle_spacer_main_add_rectangle_side_right = new TGeoTranslation("trans_middle_spacer_main_add_rectangle_side_right", -(28.4 / 2. + middle_spacer_main_add_rectangle_side_width / 2.), -5 - middle_spacer_main_add_rectangle_side_height / 2., 0);
  trans_middle_spacer_main_add_rectangle_side_left->RegisterYourself();
  trans_middle_spacer_main_add_rectangle_side_right->RegisterYourself();

  Double_t middle_spacer_main_add_rectangle_side_small_thickness = 1.18;
  Double_t middle_spacer_main_add_rectangle_side_small_height = 1.5;
  Double_t middle_spacer_main_add_rectangle_side_small_width = 2.4;
  TGeoBBox* middle_spacer_main_add_rectangle_side_small = new TGeoBBox("middle_spacer_main_add_rectangle_side_small", middle_spacer_main_add_rectangle_side_small_width / 2., middle_spacer_main_add_rectangle_side_small_height / 2., middle_spacer_main_add_rectangle_side_small_thickness / 2.);
  TGeoTranslation* trans_middle_spacer_main_add_rectangle_side_small_left = new TGeoTranslation("trans_middle_spacer_main_add_rectangle_side_small_left", 28.4 / 2. + middle_spacer_main_add_rectangle_side_width + middle_spacer_main_add_rectangle_side_small_width / 2, -5 - middle_spacer_main_add_rectangle_side_small_height / 2., 0);
  TGeoTranslation* trans_middle_spacer_main_add_rectangle_side_small_right = new TGeoTranslation("trans_middle_spacer_main_add_rectangle_side_small_right", -(28.4 / 2. + middle_spacer_main_add_rectangle_side_width + middle_spacer_main_add_rectangle_side_small_width / 2), -5 - middle_spacer_main_add_rectangle_side_small_height / 2., 0);
  trans_middle_spacer_main_add_rectangle_side_small_left->RegisterYourself();
  trans_middle_spacer_main_add_rectangle_side_small_right->RegisterYourself();

  TGeoCompositeShape* middle_spacer_main = new TGeoCompositeShape("middle_spacer_main",
                                                                  "middle_spacer_main_arc - middle_spacer_main_sub_rectangle_top:trans_middle_spacer_main_sub_rectangle_top"
                                                                  " + middle_spacer_main_add_rectangle_side:trans_middle_spacer_main_add_rectangle_side_left + middle_spacer_main_add_rectangle_side:trans_middle_spacer_main_add_rectangle_side_right"
                                                                  " + middle_spacer_main_add_rectangle_side_small:trans_middle_spacer_main_add_rectangle_side_small_left + middle_spacer_main_add_rectangle_side_small:trans_middle_spacer_main_add_rectangle_side_small_right");

  //Cover

  Double_t middle_spacer_cover_thickness = 0.15;
  Double_t middle_spacer_cover_arc_max_radius = 19.5;
  Double_t middle_spacer_cover_arc_min_radius = 17.0;
  TGeoTubeSeg* middle_spacer_cover_arc = new TGeoTubeSeg("middle_spacer_cover_arc", middle_spacer_cover_arc_min_radius, middle_spacer_cover_arc_max_radius, middle_spacer_cover_thickness / 2, 180, 0);
  TGeoTranslation* trans_middle_spacer_cover_arc = new TGeoTranslation("trans_middle_spacer_cover_arc", 0, 0, middle_spacer_main_thickness / 2 + middle_spacer_cover_thickness / 2);
  trans_middle_spacer_cover_arc->RegisterYourself();

  Double_t middle_spacer_cover_sub_rectangle_top_height = 5.0 + 1.5 + 1.9;
  Double_t middle_spacer_cover_sub_rectangle_top_width = middle_spacer_cover_arc_max_radius * 2;
  TGeoBBox* middle_spacer_cover_sub_rectangle_top = new TGeoBBox("middle_spacer_cover_sub_rectangle_top", middle_spacer_cover_sub_rectangle_top_width / 2., middle_spacer_cover_sub_rectangle_top_height, middle_spacer_cover_thickness / 2. + delta_thickness);
  TGeoTranslation* trans_middle_spacer_cover_sub_rectangle_top = new TGeoTranslation("trans_middle_spacer_cover_sub_rectangle_top", 0, 0, middle_spacer_main_thickness / 2 + middle_spacer_cover_thickness / 2);
  trans_middle_spacer_cover_sub_rectangle_top->RegisterYourself();

  Double_t middle_spacer_cover_rectangle_side1_height = 1.9;
  Double_t middle_spacer_cover_rectangle_side1_width_left = 4.2;  //rahgh
  Double_t middle_spacer_cover_rectangle_side1_width_right = 8.8; //rahgh
  TGeoBBox* middle_spacer_cover_rectangle_side1_left = new TGeoBBox("middle_spacer_cover_rectangle_side1_left", middle_spacer_cover_rectangle_side1_width_left / 2., middle_spacer_cover_rectangle_side1_height / 2., middle_spacer_cover_thickness / 2.);
  TGeoBBox* middle_spacer_cover_rectangle_side1_right = new TGeoBBox("middle_spacer_cover_rectangle_side1_right", middle_spacer_cover_rectangle_side1_width_right / 2., middle_spacer_cover_rectangle_side1_height / 2., middle_spacer_cover_thickness / 2.);
  TGeoTranslation* trans_middle_spacer_cover_rectangle_side1_left = new TGeoTranslation("trans_middle_spacer_cover_rectangle_side1_left", 28.4 / 2 + middle_spacer_cover_rectangle_side1_width_left / 2, -5 - 1.5 - 1.9 + middle_spacer_cover_rectangle_side1_height / 2, middle_spacer_main_thickness / 2 + middle_spacer_cover_thickness / 2);
  TGeoTranslation* trans_middle_spacer_cover_rectangle_side1_right = new TGeoTranslation("trans_middle_spacer_cover_rectangle_side1_right", -(28.4 / 2 + middle_spacer_cover_rectangle_side1_width_right / 2), -5 - 1.5 - 1.9 + middle_spacer_cover_rectangle_side1_height / 2, middle_spacer_main_thickness / 2 + middle_spacer_cover_thickness / 2);
  trans_middle_spacer_cover_rectangle_side1_left->RegisterYourself();
  trans_middle_spacer_cover_rectangle_side1_right->RegisterYourself();

  Double_t middle_spacer_cover_rectangle_side2_height = 1.5;
  Double_t middle_spacer_cover_rectangle_side2_width = 2.0; //rahgh
  TGeoBBox* middle_spacer_cover_rectangle_side2 = new TGeoBBox("middle_spacer_cover_rectangle_side2", middle_spacer_cover_rectangle_side2_width / 2., middle_spacer_cover_rectangle_side2_height / 2., middle_spacer_cover_thickness / 2.);
  TGeoTranslation* trans_middle_spacer_cover_rectangle_side2_left = new TGeoTranslation("trans_middle_spacer_cover_rectangle_side2_left", 28.4 / 2 + middle_spacer_cover_rectangle_side2_width / 2, -5 - 1.5 - 2.787 + middle_spacer_cover_rectangle_side2_height / 2, middle_spacer_main_thickness / 2 + middle_spacer_cover_thickness / 2);
  TGeoTranslation* trans_middle_spacer_cover_rectangle_side2_right = new TGeoTranslation("trans_middle_spacer_cover_rectangle_side2_right", -(28.4 / 2 + middle_spacer_cover_rectangle_side2_width / 2), -5 - 1.5 - 2.787 + middle_spacer_cover_rectangle_side2_height / 2, middle_spacer_main_thickness / 2 + middle_spacer_cover_thickness / 2);
  trans_middle_spacer_cover_rectangle_side2_left->RegisterYourself();
  trans_middle_spacer_cover_rectangle_side2_right->RegisterYourself();

  Double_t middle_spacer_cover_rectangle_side3_height = 1.9;
  Double_t middle_spacer_cover_rectangle_side3_width = 2.1; //rahgh
  TGeoBBox* middle_spacer_cover_rectangle_side3 = new TGeoBBox("middle_spacer_cover_rectangle_side3", middle_spacer_cover_rectangle_side3_width / 2., middle_spacer_cover_rectangle_side3_height / 2., middle_spacer_cover_thickness / 2.);
  TGeoTranslation* trans_middle_spacer_cover_rectangle_side3 = new TGeoTranslation("trans_middle_spacer_cover_rectangle_side3", 28.4 / 2 + 8.8 - middle_spacer_cover_rectangle_side3_width / 2, -5 - 1.5 - middle_spacer_cover_rectangle_side3_height / 2, middle_spacer_main_thickness / 2 + middle_spacer_cover_thickness / 2);
  trans_middle_spacer_cover_rectangle_side3->RegisterYourself();

  name_middle_spacer_main_cover +=
    " middle_spacer_cover_arc:trans_middle_spacer_cover_arc - middle_spacer_cover_sub_rectangle_top:trans_middle_spacer_cover_sub_rectangle_top"
    " + middle_spacer_cover_rectangle_side1_left:trans_middle_spacer_cover_rectangle_side1_left + middle_spacer_cover_rectangle_side1_right:trans_middle_spacer_cover_rectangle_side1_right"
    " + middle_spacer_cover_rectangle_side2:trans_middle_spacer_cover_rectangle_side2_left + middle_spacer_cover_rectangle_side2:trans_middle_spacer_cover_rectangle_side2_right"
    " + middle_spacer_cover_rectangle_side3:trans_middle_spacer_cover_rectangle_side3";

  Double_t middle_spacer_cover_block_thickness = 0.19;
  Double_t middle_spacer_cover_block_height = 1.0;
  Double_t middle_spacer_cover_block_width = 1.0;
  Double_t middle_spacer_cover_block_position_z = middle_spacer_main_thickness / 2. + middle_spacer_cover_thickness + middle_spacer_cover_block_thickness / 2.;

  TGeoBBox* middle_spacer_cover_block = new TGeoBBox("middle_spacer_cover_block", middle_spacer_cover_block_width / 2, middle_spacer_cover_block_height / 2, middle_spacer_cover_block_thickness / 2);

  TGeoCompositeShape* rotated_middle_spacer_cover_block[24];
  for (Int_t iB = 0; iB < 11; ++iB) {

    Double_t block_angle = (180. - block_angle_index[iB]) / 2. * TMath::Pi() / 180.;

    TGeoRotation* rotate_electric_board_sub_box_left = new TGeoRotation(Form("rotate_middle_spacer_cover_block_No%d", iB * 2), 180 - block_angle * 180. / TMath::Pi(), 0, 0);
    TGeoRotation* rotate_electric_board_sub_box_right = new TGeoRotation(Form("rotate_middle_spacer_cover_block_No%d", iB * 2 + 1), block_angle * 180. / TMath::Pi(), 0, 0);
    rotate_electric_board_sub_box_left->RegisterYourself();
    rotate_electric_board_sub_box_right->RegisterYourself();

    Double_t cent_block_left[] = {block_radius * TMath::Cos(block_angle), -block_radius * TMath::Sin(block_angle), middle_spacer_cover_block_position_z};
    Double_t cent_block_right[] = {block_radius * TMath::Cos(TMath::Pi() - block_angle), -block_radius * TMath::Sin(TMath::Pi() - block_angle), middle_spacer_cover_block_position_z};

    TGeoCombiTrans* combtrans_electric_board_sub_box_left = new TGeoCombiTrans(Form("combtrans_spacer_cover_block_No%d", 2 * iB), cent_block_left[0], cent_block_left[1], cent_block_left[2], rotate_electric_board_sub_box_left);
    TGeoCombiTrans* combtrans_electric_board_sub_box_right = new TGeoCombiTrans(Form("combtrans_spacer_cover_block_No%d", 2 * iB + 1), cent_block_right[0], cent_block_right[1], cent_block_right[2], rotate_electric_board_sub_box_right);
    combtrans_electric_board_sub_box_left->RegisterYourself();
    combtrans_electric_board_sub_box_right->RegisterYourself();

    name_middle_spacer_main_cover += Form("+middle_spacer_cover_block:combtrans_spacer_cover_block_No%d", 2 * iB);
    name_middle_spacer_main_cover += Form("+middle_spacer_cover_block:combtrans_spacer_cover_block_No%d", 2 * iB + 1);
  }

  TGeoTranslation* trans_spacer_cover_block_left = new TGeoTranslation("trans_spacer_cover_block_left", 22.259 - middle_spacer_cover_block_width / 2, -8.305 + middle_spacer_cover_block_height / 2, middle_spacer_cover_block_position_z);
  trans_spacer_cover_block_left->RegisterYourself();
  name_middle_spacer_main_cover += "+middle_spacer_cover_block:trans_spacer_cover_block_left";

  TGeoTranslation* trans_spacer_cover_block_right = new TGeoTranslation("trans_spacer_cover_block_right", -22.247 + middle_spacer_cover_block_width / 2, -8.305 + middle_spacer_cover_block_height / 2, middle_spacer_cover_block_position_z);
  trans_spacer_cover_block_right->RegisterYourself();
  name_middle_spacer_main_cover += "+middle_spacer_cover_block:trans_spacer_cover_block_right";

  TGeoCompositeShape* middle_spacer_cover = new TGeoCompositeShape("middle_spacer_cover", name_middle_spacer_main_cover.c_str());

  TGeoRotation* rotate_middle_spacer_cover_back = new TGeoRotation("rotate_middle_spacer_cover_back", 180, 180, 0);
  rotate_middle_spacer_cover_back->RegisterYourself();

  TGeoCompositeShape* middle_spacer_cover_bothside = new TGeoCompositeShape("middle_spacer_cover_bothside", "middle_spacer_cover + middle_spacer_cover:rotate_middle_spacer_cover_back");

  //Water pipe hole

  Double_t water_pipe_inner_radius = 0.3 / 2.;
  Double_t water_pipe_outer_radius = 0.4 / 2.;

  Double_t water_pipe_main_angle = 126.19;

  Double_t water_pipe_side_position_radius = (1.8 + 1.4) / 2;
  Double_t water_pipe_side_angle = 63.1;

  Double_t water_pipe_straight_tube_side1_length = 2.865 / 2.;
  Double_t water_pipe_straight_tube_side2_length = 8.950 / 2;

  TGeoTorus* middle_spacer_sub_water_pipe_main_torus = new TGeoTorus("middle_spacer_sub_water_pipe_main_torus", water_pipe_main_position_radius, 0., water_pipe_outer_radius, 180 + (180 - water_pipe_main_angle) / 2., water_pipe_main_angle);

  TGeoTorus* middle_spacer_sub_water_pipe_side_torus_left1 = new TGeoTorus("middle_spacer_sub_water_pipe_side_torus_left1", water_pipe_side_position_radius, 0., water_pipe_outer_radius, 90, water_pipe_side_angle);
  TGeoTorus* middle_spacer_sub_water_pipe_side_torus_right1 = new TGeoTorus("middle_spacer_sub_water_pipe_side_torus_right1", water_pipe_side_position_radius, 0., water_pipe_outer_radius, 90 - water_pipe_side_angle, water_pipe_side_angle);

  TGeoTranslation* trans_middle_spacer_sub_water_pipe_side_torus_left1 = new TGeoTranslation("trans_middle_spacer_sub_water_pipe_side_torus_left1", (water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Cos((90 - water_pipe_side_angle) * TMath::Pi() / 180.), -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Sin((90 - water_pipe_side_angle) * TMath::Pi() / 180.), 0);
  TGeoTranslation* trans_middle_spacer_sub_water_pipe_side_torus_right1 = new TGeoTranslation("trans_middle_spacer_sub_water_pipe_side_torus_right1", -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Cos((90 - water_pipe_side_angle) * TMath::Pi() / 180.), -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Sin((90 - water_pipe_side_angle) * TMath::Pi() / 180.), 0);
  trans_middle_spacer_sub_water_pipe_side_torus_left1->RegisterYourself();
  trans_middle_spacer_sub_water_pipe_side_torus_right1->RegisterYourself();

  TGeoTubeSeg* middle_spacer_sub_water_pipe_straight_tube_side1 = new TGeoTubeSeg("middle_spacer_sub_water_pipe_straight_tube_side1", 0., water_pipe_outer_radius, water_pipe_straight_tube_side1_length, 0, 360);
  TGeoRotation* rotate_middle_spacer_sub_water_pipe_straight_tube_side1 = new TGeoRotation("rotate_middle_spacer_sub_water_pipe_straight_tube_side1", 90, 90, 0);
  rotate_middle_spacer_sub_water_pipe_straight_tube_side1->RegisterYourself();

  TGeoCombiTrans* combtrans_rotated_middle_spacer_sub_water_pipe_straight_tube_side1_left = new TGeoCombiTrans("combtrans_rotated_middle_spacer_sub_water_pipe_straight_tube_side1_left", (water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Cos((90 - water_pipe_side_angle) * TMath::Pi() / 180.) + water_pipe_straight_tube_side1_length,
                                                                                                               -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Sin((90 - water_pipe_side_angle) * TMath::Pi() / 180.) + water_pipe_side_position_radius, 0, rotate_middle_spacer_sub_water_pipe_straight_tube_side1);
  TGeoCombiTrans* combtrans_rotated_middle_spacer_sub_water_pipe_straight_tube_side1_right = new TGeoCombiTrans("combtrans_rotated_middle_spacer_sub_water_pipe_straight_tube_side1_right", -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Cos((90 - water_pipe_side_angle) * TMath::Pi() / 180.) - water_pipe_straight_tube_side1_length,
                                                                                                                -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Sin((90 - water_pipe_side_angle) * TMath::Pi() / 180.) + water_pipe_side_position_radius, 0, rotate_middle_spacer_sub_water_pipe_straight_tube_side1);
  combtrans_rotated_middle_spacer_sub_water_pipe_straight_tube_side1_left->RegisterYourself();
  combtrans_rotated_middle_spacer_sub_water_pipe_straight_tube_side1_right->RegisterYourself();

  TGeoTorus* middle_spacer_sub_water_pipe_side_torus_left2 = new TGeoTorus("middle_spacer_sub_water_pipe_side_torus_left2", water_pipe_side_position_radius, 0., water_pipe_outer_radius, 0, 90);
  TGeoTorus* middle_spacer_sub_water_pipe_side_torus_right2 = new TGeoTorus("middle_spacer_sub_water_pipe_side_torus_right2", water_pipe_side_position_radius, 0., water_pipe_outer_radius, 90, 90);

  TGeoTranslation* trans_middle_spacer_sub_water_pipe_side_torus_left2 = new TGeoTranslation("trans_middle_spacer_sub_water_pipe_side_torus_left2", +(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Cos((90 - water_pipe_side_angle) * TMath::Pi() / 180.) + water_pipe_straight_tube_side1_length * 2,
                                                                                             -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Sin((90 - water_pipe_side_angle) * TMath::Pi() / 180.) + water_pipe_side_position_radius - water_pipe_side_position_radius, 0);
  TGeoTranslation* trans_middle_spacer_sub_water_pipe_side_torus_right2 = new TGeoTranslation("trans_middle_spacer_sub_water_pipe_side_torus_right2", -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Cos((90 - water_pipe_side_angle) * TMath::Pi() / 180.) - water_pipe_straight_tube_side1_length * 2,
                                                                                              -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Sin((90 - water_pipe_side_angle) * TMath::Pi() / 180.) + water_pipe_side_position_radius - water_pipe_side_position_radius, 0);
  trans_middle_spacer_sub_water_pipe_side_torus_left2->RegisterYourself();
  trans_middle_spacer_sub_water_pipe_side_torus_right2->RegisterYourself();

  TGeoTubeSeg* middle_spacer_sub_water_pipe_straight_tube_side2 = new TGeoTubeSeg("middle_spacer_sub_water_pipe_straight_tube_side2", 0., water_pipe_outer_radius, water_pipe_straight_tube_side2_length, 0, 360);

  TGeoRotation* rotate_middle_spacer_sub_water_pipe_straight_tube_side2 = new TGeoRotation("rotate_middle_spacer_sub_water_pipe_straight_tube_side2", 0, 90, 0);
  rotate_middle_spacer_sub_water_pipe_straight_tube_side2->RegisterYourself();

  TGeoCombiTrans* combtrans_middle_spacer_sub_water_pipe_straight_tube_side_left2 = new TGeoCombiTrans("combtrans_middle_spacer_sub_water_pipe_straight_tube_side_left2", +(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Cos((90 - water_pipe_side_angle) * TMath::Pi() / 180.) + water_pipe_straight_tube_side1_length * 2 + water_pipe_side_position_radius,
                                                                                                       -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Sin((90 - water_pipe_side_angle) * TMath::Pi() / 180.) + water_pipe_side_position_radius - water_pipe_side_position_radius - water_pipe_straight_tube_side2_length, 0, rotate_middle_spacer_sub_water_pipe_straight_tube_side2);

  TGeoCombiTrans* combtrans_middle_spacer_sub_water_pipe_straight_tube_side_right2 = new TGeoCombiTrans("combtrans_middle_spacer_sub_water_pipe_straight_tube_side_right2", -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Cos((90 - water_pipe_side_angle) * TMath::Pi() / 180.) - water_pipe_straight_tube_side1_length * 2 - water_pipe_side_position_radius,
                                                                                                        -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Sin((90 - water_pipe_side_angle) * TMath::Pi() / 180.) + water_pipe_side_position_radius - water_pipe_side_position_radius - water_pipe_straight_tube_side2_length, 0, rotate_middle_spacer_sub_water_pipe_straight_tube_side2);

  combtrans_middle_spacer_sub_water_pipe_straight_tube_side_left2->RegisterYourself();
  combtrans_middle_spacer_sub_water_pipe_straight_tube_side_right2->RegisterYourself();

  TGeoCompositeShape* middle_spacer_shape = new TGeoCompositeShape("middle_spacer_shape",
                                                                   "middle_spacer_main + middle_spacer_cover_bothside"
                                                                   " - middle_spacer_sub_water_pipe_main_torus - middle_spacer_sub_water_pipe_side_torus_left1:trans_middle_spacer_sub_water_pipe_side_torus_left1 - middle_spacer_sub_water_pipe_side_torus_right1:trans_middle_spacer_sub_water_pipe_side_torus_right1"
                                                                   " - middle_spacer_sub_water_pipe_straight_tube_side1:combtrans_rotated_middle_spacer_sub_water_pipe_straight_tube_side1_left - middle_spacer_sub_water_pipe_straight_tube_side1:combtrans_rotated_middle_spacer_sub_water_pipe_straight_tube_side1_right"
                                                                   " - middle_spacer_sub_water_pipe_side_torus_left2:trans_middle_spacer_sub_water_pipe_side_torus_left2 - middle_spacer_sub_water_pipe_side_torus_right2:trans_middle_spacer_sub_water_pipe_side_torus_right2"
                                                                   " - middle_spacer_sub_water_pipe_straight_tube_side2:combtrans_middle_spacer_sub_water_pipe_straight_tube_side_left2 - middle_spacer_sub_water_pipe_straight_tube_side2:combtrans_middle_spacer_sub_water_pipe_straight_tube_side_right2");

  TGeoVolume* middle_spacer = new TGeoVolume("middle_spacer", middle_spacer_shape, kMedAlu);
  middle_spacer->SetLineColor(kTeal - 4);
  mHalfPSU->AddNode(middle_spacer, 0, nullptr);

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Electoric board
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  Double_t electric_board_thickness = 0.171;
  Double_t electric_board_max_radius1 = 26.706;
  Double_t electric_board_min_radius1 = 17.0;
  Double_t electric_board_sub_rectangle_middle_height = 5 + 4.347;
  Double_t electric_board_sub_rectangle_middle_width = electric_board_max_radius1 * 2;
  Double_t electric_board_sub_rectangle_bottom_height = 10.00; //laugh
  Double_t electric_board_sub_rectangle_bottom_width = 20.80;
  Double_t electric_board_position_z = middle_spacer_main_thickness / 2. + middle_spacer_cover_thickness + electric_board_thickness / 2;

  TGeoTubeSeg* electric_board_main = new TGeoTubeSeg("electric_board_main", electric_board_min_radius1, electric_board_max_radius1, electric_board_thickness / 2, 180, 0);
  TGeoTranslation* trans_electric_board_main = new TGeoTranslation("trans_electric_board_main", 0, 0, electric_board_position_z);
  trans_electric_board_main->RegisterYourself();

  TGeoBBox* electric_board_sub_rectangle_middle = new TGeoBBox("electric_board_sub_rectangle_middle", electric_board_sub_rectangle_middle_width / 2., electric_board_sub_rectangle_middle_height / 2., electric_board_thickness / 2. + delta_thickness);
  TGeoBBox* electric_board_sub_rectangle_bottom = new TGeoBBox("electric_board_sub_rectangle_bottom", electric_board_sub_rectangle_bottom_width / 2., electric_board_sub_rectangle_bottom_height / 2., electric_board_thickness / 2. + delta_thickness);

  TGeoTranslation* trans_electric_board_sub_rectangle_middle = new TGeoTranslation("trans_electric_board_sub_rectangle_middle", 0, -electric_board_sub_rectangle_middle_height / 2., electric_board_position_z);
  TGeoTranslation* trans_electric_board_sub_rectangle_bottom = new TGeoTranslation("trans_electric_board_sub_rectangle_bottom", 0, -(24.6 + electric_board_sub_rectangle_bottom_height / 2.), electric_board_position_z);
  trans_electric_board_sub_rectangle_middle->RegisterYourself();
  trans_electric_board_sub_rectangle_bottom->RegisterYourself();

  Double_t electric_board_add_rectangle_side_height = 4.347;
  Double_t electric_board_add_rectangle_side_width = 8.800;

  TGeoBBox* electric_board_add_rectangle_side = new TGeoBBox("electric_board_add_rectangle_side", electric_board_add_rectangle_side_width / 2., electric_board_add_rectangle_side_height / 2., electric_board_thickness / 2.);
  TGeoTranslation* trans_electric_board_add_rectangle_side_left = new TGeoTranslation("trans_electric_board_add_rectangle_side_left", 28.4 / 2. + electric_board_add_rectangle_side_width / 2., -5.0 - electric_board_add_rectangle_side_height / 2., electric_board_position_z);
  TGeoTranslation* trans_electric_board_add_rectangle_side_right = new TGeoTranslation("trans_electric_board_add_rectangle_side_right", -(28.4 / 2. + electric_board_add_rectangle_side_width / 2.), -5.0 - electric_board_add_rectangle_side_height / 2., electric_board_position_z);
  trans_electric_board_add_rectangle_side_left->RegisterYourself();
  trans_electric_board_add_rectangle_side_right->RegisterYourself();

  Double_t electric_board_sub_rectangle_side_width = 10.; //raugh
  Double_t electric_board_sub_rectangle_side_height = 8.576;

  TGeoBBox* electric_board_sub_rectangle_side = new TGeoBBox("electric_board_sub_rectangle_side", electric_board_sub_rectangle_side_width / 2., electric_board_sub_rectangle_side_height / 2., electric_board_thickness / 2. + delta_thickness);
  TGeoTranslation* trans_electric_board_sub_rectangle_side_left = new TGeoTranslation("trans_electric_board_sub_rectangle_side_left", 28.4 / 2 + 8.8 + electric_board_sub_rectangle_side_width / 2., -5.0 - electric_board_sub_rectangle_side_height / 2., electric_board_position_z);
  TGeoTranslation* trans_electric_board_sub_rectangle_side_right = new TGeoTranslation("trans_electric_board_sub_rectangle_side_right", -(28.4 / 2 + 8.8 + electric_board_sub_rectangle_side_width / 2.), -5.0 - electric_board_sub_rectangle_side_height / 2., electric_board_position_z);
  trans_electric_board_sub_rectangle_side_left->RegisterYourself();
  trans_electric_board_sub_rectangle_side_right->RegisterYourself();

  Double_t electric_board_sub_box_height = 1.1;
  Double_t electric_board_sub_box_width = 1.1;

  TGeoBBox* electric_board_sub_box = new TGeoBBox("electric_board_sub_box", electric_board_sub_box_width / 2, electric_board_sub_box_height / 2, electric_board_thickness / 2 + delta_thickness);

  TGeoCompositeShape* rotated_electric_board_sub_box[24];

  std::string name_electric_board_shape =
    "electric_board_main:trans_electric_board_main - electric_board_sub_rectangle_middle:trans_electric_board_sub_rectangle_middle - electric_board_sub_rectangle_bottom:trans_electric_board_sub_rectangle_bottom"
    " + electric_board_add_rectangle_side:trans_electric_board_add_rectangle_side_left + electric_board_add_rectangle_side:trans_electric_board_add_rectangle_side_right"
    " - electric_board_sub_rectangle_side:trans_electric_board_sub_rectangle_side_left - electric_board_sub_rectangle_side:trans_electric_board_sub_rectangle_side_right";

  for (Int_t iB = 0; iB < 11; ++iB) {

    Double_t block_angle = (180. - block_angle_index[iB]) / 2. * TMath::Pi() / 180.;

    TGeoRotation* rotate_electric_board_sub_box_left = new TGeoRotation(Form("rotate_electric_board_sub_box_No%d", iB * 2), 180 - block_angle * 180. / TMath::Pi(), 0, 0);
    TGeoRotation* rotate_electric_board_sub_box_right = new TGeoRotation(Form("rotate_electric_board_sub_box_No%d", iB * 2 + 1), block_angle * 180. / TMath::Pi(), 0, 0);
    rotate_electric_board_sub_box_left->RegisterYourself();
    rotate_electric_board_sub_box_right->RegisterYourself();

    Double_t cent_block_left[] = {block_radius * TMath::Cos(block_angle), -block_radius * TMath::Sin(block_angle), electric_board_position_z};
    Double_t cent_block_right[] = {block_radius * TMath::Cos(TMath::Pi() - block_angle), -block_radius * TMath::Sin(TMath::Pi() - block_angle), electric_board_position_z};

    TGeoCombiTrans* combtrans_electric_board_sub_box_left = new TGeoCombiTrans(Form("combtrans_electric_board_sub_box_No%d", 2 * iB), cent_block_left[0], cent_block_left[1], cent_block_left[2], rotate_electric_board_sub_box_left);
    TGeoCombiTrans* combtrans_electric_board_sub_box_right = new TGeoCombiTrans(Form("combtrans_electric_board_sub_box_No%d", 2 * iB + 1), cent_block_right[0], cent_block_right[1], cent_block_right[2], rotate_electric_board_sub_box_right);
    combtrans_electric_board_sub_box_left->RegisterYourself();
    combtrans_electric_board_sub_box_right->RegisterYourself();

    name_electric_board_shape += Form("-electric_board_sub_box:combtrans_electric_board_sub_box_No%d", 2 * iB);
    name_electric_board_shape += Form("-electric_board_sub_box:combtrans_electric_board_sub_box_No%d", 2 * iB + 1);
  }

  TGeoTranslation* trans_electric_board_sub_box_left = new TGeoTranslation("trans_electric_board_sub_box_left", 22.259 - electric_board_sub_box_width / 2, -8.305 + electric_board_sub_box_height / 2, electric_board_position_z);
  trans_electric_board_sub_box_left->RegisterYourself();
  name_electric_board_shape += "-electric_board_sub_box:trans_electric_board_sub_box_left";

  TGeoTranslation* trans_electric_board_sub_box_right = new TGeoTranslation("trans_electric_board_sub_box_right", -22.247 + electric_board_sub_box_width / 2, -8.305 + electric_board_sub_box_height / 2, electric_board_position_z);
  trans_electric_board_sub_box_right->RegisterYourself();
  name_electric_board_shape += "-electric_board_sub_box:trans_electric_board_sub_box_right";

  TGeoCompositeShape* electric_board_shape = new TGeoCompositeShape("electric_board_shape", name_electric_board_shape.c_str());

  TGeoRotation* trans_electric_board_front = new TGeoRotation("trans_electric_board_front", 0, 0, 0);
  TGeoRotation* trans_electric_board_back = new TGeoRotation("trans_electric_board_back", 180, 180, 0);

  TGeoVolume* electric_board = new TGeoVolume("electric_board", electric_board_shape, kMedPeek);
  electric_board->SetLineColor(kGreen + 2);
  mHalfPSU->AddNode(electric_board, 0, trans_electric_board_front);
  mHalfPSU->AddNode(electric_board, 0, trans_electric_board_back);

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Water pipe
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  TGeoTorus* water_pipe_main_torus = new TGeoTorus("water_pipe_main_torus", water_pipe_main_position_radius, water_pipe_inner_radius, water_pipe_outer_radius, 180 + (180 - water_pipe_main_angle) / 2., water_pipe_main_angle);

  TGeoTorus* water_pipe_side_torus_left1 = new TGeoTorus("water_pipe_side_torus_left1", water_pipe_side_position_radius, water_pipe_inner_radius, water_pipe_outer_radius, 90, water_pipe_side_angle);
  TGeoTorus* water_pipe_side_torus_right1 = new TGeoTorus("water_pipe_side_torus_right1", water_pipe_side_position_radius, water_pipe_inner_radius, water_pipe_outer_radius, 90 - water_pipe_side_angle, water_pipe_side_angle);

  TGeoTranslation* trans_water_pipe_side_torus_left1 = new TGeoTranslation("trans_water_pipe_side_torus_left1", (water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Cos((90 - water_pipe_side_angle) * TMath::Pi() / 180.), -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Sin((90 - water_pipe_side_angle) * TMath::Pi() / 180.), 0);
  TGeoTranslation* trans_water_pipe_side_torus_right1 = new TGeoTranslation("trans_water_pipe_side_torus_right1", -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Cos((90 - water_pipe_side_angle) * TMath::Pi() / 180.), -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Sin((90 - water_pipe_side_angle) * TMath::Pi() / 180.), 0);
  trans_water_pipe_side_torus_left1->RegisterYourself();
  trans_water_pipe_side_torus_right1->RegisterYourself();

  TGeoTubeSeg* water_pipe_straight_tube_side1 = new TGeoTubeSeg("water_pipe_straight_tube_side1", water_pipe_inner_radius, water_pipe_outer_radius, water_pipe_straight_tube_side1_length, 0, 360);

  TGeoRotation* rotate_water_pipe_straight_tube_side1 = new TGeoRotation("rotate_water_pipe_straight_tube_side1", 90, 90, 0);
  rotate_water_pipe_straight_tube_side1->RegisterYourself();

  TGeoCombiTrans* combtrans_rotated_water_pipe_straight_tube_side1_left = new TGeoCombiTrans("combtrans_rotated_water_pipe_straight_tube_side1_left", (water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Cos((90 - water_pipe_side_angle) * TMath::Pi() / 180.) + water_pipe_straight_tube_side1_length,
                                                                                             -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Sin((90 - water_pipe_side_angle) * TMath::Pi() / 180.) + water_pipe_side_position_radius, 0, rotate_water_pipe_straight_tube_side1);
  TGeoCombiTrans* combtrans_rotated_water_pipe_straight_tube_side1_right = new TGeoCombiTrans("combtrans_rotated_water_pipe_straight_tube_side1_right", -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Cos((90 - water_pipe_side_angle) * TMath::Pi() / 180.) - water_pipe_straight_tube_side1_length,
                                                                                              -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Sin((90 - water_pipe_side_angle) * TMath::Pi() / 180.) + water_pipe_side_position_radius, 0, rotate_water_pipe_straight_tube_side1);
  combtrans_rotated_water_pipe_straight_tube_side1_left->RegisterYourself();
  combtrans_rotated_water_pipe_straight_tube_side1_right->RegisterYourself();

  TGeoTorus* water_pipe_side_torus_left2 = new TGeoTorus("water_pipe_side_torus_left2", water_pipe_side_position_radius, water_pipe_inner_radius, water_pipe_outer_radius, 0, 90);
  TGeoTorus* water_pipe_side_torus_right2 = new TGeoTorus("water_pipe_side_torus_right2", water_pipe_side_position_radius, water_pipe_inner_radius, water_pipe_outer_radius, 90, 90);
  TGeoTranslation* trans_water_pipe_side_torus_left2 = new TGeoTranslation("trans_water_pipe_side_torus_left2", +(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Cos((90 - water_pipe_side_angle) * TMath::Pi() / 180.) + water_pipe_straight_tube_side1_length * 2, -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Sin((90 - water_pipe_side_angle) * TMath::Pi() / 180.) + water_pipe_side_position_radius - water_pipe_side_position_radius, 0);
  TGeoTranslation* trans_water_pipe_side_torus_right2 = new TGeoTranslation("trans_water_pipe_side_torus_right2", -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Cos((90 - water_pipe_side_angle) * TMath::Pi() / 180.) - water_pipe_straight_tube_side1_length * 2, -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Sin((90 - water_pipe_side_angle) * TMath::Pi() / 180.) + water_pipe_side_position_radius - water_pipe_side_position_radius, 0);
  trans_water_pipe_side_torus_left2->RegisterYourself();
  trans_water_pipe_side_torus_right2->RegisterYourself();

  TGeoTubeSeg* water_pipe_straight_tube_side2 = new TGeoTubeSeg("water_pipe_straight_tube_side2", water_pipe_inner_radius, water_pipe_outer_radius, water_pipe_straight_tube_side2_length, 0, 360);
  TGeoRotation* rotate_water_pipe_straight_tube_side2 = new TGeoRotation("rotate_water_pipe_straight_tube_side2", 0, 90, 0);

  rotate_water_pipe_straight_tube_side2->RegisterYourself();

  TGeoCombiTrans* combtrans_water_pipe_straight_tube_side_left2 = new TGeoCombiTrans("combtrans_water_pipe_straight_tube_side_left2", +(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Cos((90 - water_pipe_side_angle) * TMath::Pi() / 180.) + water_pipe_straight_tube_side1_length * 2 + water_pipe_side_position_radius,
                                                                                     -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Sin((90 - water_pipe_side_angle) * TMath::Pi() / 180.) + water_pipe_side_position_radius - water_pipe_side_position_radius - water_pipe_straight_tube_side2_length, 0, rotate_water_pipe_straight_tube_side2);

  TGeoCombiTrans* combtrans_water_pipe_straight_tube_side_right2 = new TGeoCombiTrans("combtrans_water_pipe_straight_tube_side_right2", -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Cos((90 - water_pipe_side_angle) * TMath::Pi() / 180.) - water_pipe_straight_tube_side1_length * 2 - water_pipe_side_position_radius,
                                                                                      -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Sin((90 - water_pipe_side_angle) * TMath::Pi() / 180.) + water_pipe_side_position_radius - water_pipe_side_position_radius - water_pipe_straight_tube_side2_length, 0, rotate_water_pipe_straight_tube_side2);

  combtrans_water_pipe_straight_tube_side_left2->RegisterYourself();
  combtrans_water_pipe_straight_tube_side_right2->RegisterYourself();

  //==================== PIPE - connection to the patch panel ======================
  TGeoTorus* pipe_side_torus_left3 = new TGeoTorus("pipe_side_torus_left3", water_pipe_side_position_radius, water_pipe_inner_radius, water_pipe_outer_radius, 0, 90);
  TGeoRotation* rotate_water_torus_left3 = new TGeoRotation("rotate_water_torus_left3", -90, 90, 0);
  TGeoCombiTrans* combtrans_water_torus_left3 = new TGeoCombiTrans("combtrans_water_torus_left3", +(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Cos((90 - water_pipe_side_angle) * TMath::Pi() / 180.) + water_pipe_straight_tube_side1_length * 2 + water_pipe_side_position_radius,
                                                                   -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Sin((90 - water_pipe_side_angle) * TMath::Pi() / 180.) - 2 * water_pipe_straight_tube_side2_length, -water_pipe_side_position_radius, rotate_water_torus_left3);
  combtrans_water_torus_left3->RegisterYourself();
  Float_t water_pipe_straight_tube_side3_length = 5;
  TGeoTubeSeg* pipe_straight_tube_left3 = new TGeoTubeSeg("pipe_straight_tube_left3", water_pipe_inner_radius, water_pipe_outer_radius, water_pipe_straight_tube_side3_length, 0, 360);
  TGeoRotation* rotate_water_straight_tube_left3 = new TGeoRotation("rotate_water_straight_tube_left3", 0, 0, 0);
  TGeoCombiTrans* combtrans_water_straight_tube_side_left3 = new TGeoCombiTrans("combtrans_water_straight_tube_side_left3", +(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Cos((90 - water_pipe_side_angle) * TMath::Pi() / 180.) + water_pipe_straight_tube_side1_length * 2 + water_pipe_side_position_radius,
                                                                                -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Sin((90 - water_pipe_side_angle) * TMath::Pi() / 180.) - 2 * water_pipe_straight_tube_side2_length - water_pipe_side_position_radius, -water_pipe_straight_tube_side3_length - water_pipe_side_position_radius, rotate_water_straight_tube_left3);
  combtrans_water_straight_tube_side_left3->RegisterYourself();

  TGeoTorus* pipe_side_torus_rigth3 = new TGeoTorus("pipe_side_torus_rigth3", water_pipe_side_position_radius, water_pipe_inner_radius, water_pipe_outer_radius, 0, 90);
  TGeoRotation* rotate_water_torus_rigth3 = new TGeoRotation("rotate_water_torus_rigth3", -90, 90, 0);
  TGeoCombiTrans* combtrans_water_torus_rigth3 = new TGeoCombiTrans("combtrans_water_torus_rigth3", -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Cos((90 - water_pipe_side_angle) * TMath::Pi() / 180.) - water_pipe_straight_tube_side1_length * 2 - water_pipe_side_position_radius,
                                                                    -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Sin((90 - water_pipe_side_angle) * TMath::Pi() / 180.) - 2 * water_pipe_straight_tube_side2_length, -water_pipe_side_position_radius, rotate_water_torus_rigth3);
  combtrans_water_torus_rigth3->RegisterYourself();

  TGeoTubeSeg* pipe_straight_tube_rigth3 = new TGeoTubeSeg("pipe_straight_tube_rigth3", water_pipe_inner_radius, water_pipe_outer_radius, water_pipe_straight_tube_side3_length, 0, 360);
  TGeoRotation* rotate_water_straight_tube_rigth3 = new TGeoRotation("rotate_water_straight_tube_rigth3", 0, 0, 0);
  TGeoCombiTrans* combtrans_water_straight_tube_side_rigth3 = new TGeoCombiTrans("combtrans_water_straight_tube_side_rigth3", -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Cos((90 - water_pipe_side_angle) * TMath::Pi() / 180.) - water_pipe_straight_tube_side1_length * 2 - water_pipe_side_position_radius,
                                                                                 -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Sin((90 - water_pipe_side_angle) * TMath::Pi() / 180.) - 2 * water_pipe_straight_tube_side2_length - water_pipe_side_position_radius, -water_pipe_straight_tube_side3_length - water_pipe_side_position_radius, rotate_water_straight_tube_rigth3);
  combtrans_water_straight_tube_side_rigth3->RegisterYourself();

  TGeoCompositeShape* water_pipe_toPatchPanel = new TGeoCompositeShape("water_pipe_toPatchPanel",
                                                                       "pipe_straight_tube_left3:combtrans_water_straight_tube_side_left3"
                                                                       " + pipe_side_torus_left3:combtrans_water_torus_left3"
                                                                       " + pipe_straight_tube_rigth3:combtrans_water_straight_tube_side_rigth3"
                                                                       " + pipe_side_torus_rigth3:combtrans_water_torus_rigth3");
  TGeoVolume* poly_pipe = new TGeoVolume("poly_pipe_toPatchPanel", water_pipe_toPatchPanel, kMedPolyPipe);
  poly_pipe->SetLineColor(kGray);
  mHalfPSU->AddNode(poly_pipe, 1, nullptr);

  //==================================================

  TGeoCompositeShape* water_pipe_shape = new TGeoCompositeShape("water_pipe_shape",
                                                                "water_pipe_main_torus + water_pipe_side_torus_left1:trans_water_pipe_side_torus_left1 + water_pipe_side_torus_right1:trans_water_pipe_side_torus_right1"
                                                                " + water_pipe_straight_tube_side1:combtrans_rotated_water_pipe_straight_tube_side1_left + water_pipe_straight_tube_side1:combtrans_rotated_water_pipe_straight_tube_side1_right"
                                                                " + water_pipe_side_torus_left2:trans_water_pipe_side_torus_left2 + water_pipe_side_torus_right2:trans_water_pipe_side_torus_right2"
                                                                " + water_pipe_straight_tube_side2:combtrans_water_pipe_straight_tube_side_left2 + water_pipe_straight_tube_side2:combtrans_water_pipe_straight_tube_side_right2"
                                                                " + pipe_straight_tube_left3:combtrans_water_straight_tube_side_left3"
                                                                " + pipe_side_torus_left3:combtrans_water_torus_left3"
                                                                " + pipe_straight_tube_rigth3:combtrans_water_straight_tube_side_rigth3"
                                                                " + pipe_side_torus_rigth3:combtrans_water_torus_rigth3");

  TGeoVolume* water_pipe = new TGeoVolume("water_pipe", water_pipe_shape, kMedAlu);
  water_pipe->SetLineColor(kGray);
  mHalfPSU->AddNode(water_pipe, 1, nullptr);

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Water
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  TGeoTorus* water_main_torus = new TGeoTorus("water_main_torus", water_pipe_main_position_radius, 0, water_pipe_inner_radius, 180 + (180 - water_pipe_main_angle) / 2., water_pipe_main_angle);

  TGeoTorus* water_side_torus_left1 = new TGeoTorus("water_side_torus_left1", water_pipe_side_position_radius, 0, water_pipe_inner_radius, 90, water_pipe_side_angle);
  TGeoTorus* water_side_torus_right1 = new TGeoTorus("water_side_torus_right1", water_pipe_side_position_radius, 0, water_pipe_inner_radius, 90 - water_pipe_side_angle, water_pipe_side_angle);

  TGeoTranslation* trans_water_side_torus_left1 = new TGeoTranslation("trans_water_side_torus_left1", (water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Cos((90 - water_pipe_side_angle) * TMath::Pi() / 180.), -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Sin((90 - water_pipe_side_angle) * TMath::Pi() / 180.), 0);
  TGeoTranslation* trans_water_side_torus_right1 = new TGeoTranslation("trans_water_side_torus_right1", -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Cos((90 - water_pipe_side_angle) * TMath::Pi() / 180.), -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Sin((90 - water_pipe_side_angle) * TMath::Pi() / 180.), 0);
  trans_water_side_torus_left1->RegisterYourself();
  trans_water_side_torus_right1->RegisterYourself();

  TGeoTubeSeg* water_straight_tube_side1 = new TGeoTubeSeg("water_straight_tube_side1", 0, water_pipe_inner_radius, water_pipe_straight_tube_side1_length, 0, 360);
  TGeoRotation* rotate_water_straight_tube_side1 = new TGeoRotation("rotate_water_straight_tube_side1", 90, 90, 0);
  rotate_water_straight_tube_side1->RegisterYourself();

  TGeoCombiTrans* combtrans_rotated_water_straight_tube_side1_left = new TGeoCombiTrans("combtrans_rotated_water_straight_tube_side1_left", (water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Cos((90 - water_pipe_side_angle) * TMath::Pi() / 180.) + water_pipe_straight_tube_side1_length, -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Sin((90 - water_pipe_side_angle) * TMath::Pi() / 180.) + water_pipe_side_position_radius, 0, rotate_water_straight_tube_side1);
  TGeoCombiTrans* combtrans_rotated_water_straight_tube_side1_right = new TGeoCombiTrans("combtrans_rotated_water_straight_tube_side1_right", -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Cos((90 - water_pipe_side_angle) * TMath::Pi() / 180.) - water_pipe_straight_tube_side1_length, -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Sin((90 - water_pipe_side_angle) * TMath::Pi() / 180.) + water_pipe_side_position_radius, 0, rotate_water_straight_tube_side1);
  combtrans_rotated_water_straight_tube_side1_left->RegisterYourself();
  combtrans_rotated_water_straight_tube_side1_right->RegisterYourself();

  TGeoTorus* water_side_torus_left2 = new TGeoTorus("water_side_torus_left2", water_pipe_side_position_radius, 0, water_pipe_inner_radius, 0, 90);
  TGeoTorus* water_side_torus_right2 = new TGeoTorus("water_side_torus_right2", water_pipe_side_position_radius, 0, water_pipe_inner_radius, 90, 90);
  TGeoTranslation* trans_water_side_torus_left2 = new TGeoTranslation("trans_water_side_torus_left2", +(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Cos((90 - water_pipe_side_angle) * TMath::Pi() / 180.) + water_pipe_straight_tube_side1_length * 2, -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Sin((90 - water_pipe_side_angle) * TMath::Pi() / 180.) + water_pipe_side_position_radius - water_pipe_side_position_radius, 0);
  TGeoTranslation* trans_water_side_torus_right2 = new TGeoTranslation("trans_water_side_torus_right2", -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Cos((90 - water_pipe_side_angle) * TMath::Pi() / 180.) - water_pipe_straight_tube_side1_length * 2, -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Sin((90 - water_pipe_side_angle) * TMath::Pi() / 180.) + water_pipe_side_position_radius - water_pipe_side_position_radius, 0);
  trans_water_side_torus_left2->RegisterYourself();
  trans_water_side_torus_right2->RegisterYourself();

  TGeoTubeSeg* water_straight_tube_side2 = new TGeoTubeSeg("water_straight_tube_side2", 0, water_pipe_inner_radius, water_pipe_straight_tube_side2_length, 0, 360);
  TGeoRotation* rotate_water_straight_tube_side2 = new TGeoRotation("rotate_water_straight_tube_side2", 0, 90, 0);
  rotate_water_straight_tube_side2->RegisterYourself();

  TGeoCombiTrans* combtrans_water_straight_tube_side_left2 = new TGeoCombiTrans("combtrans_water_straight_tube_side_left2", +(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Cos((90 - water_pipe_side_angle) * TMath::Pi() / 180.) + water_pipe_straight_tube_side1_length * 2 + water_pipe_side_position_radius,
                                                                                -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Sin((90 - water_pipe_side_angle) * TMath::Pi() / 180.) + water_pipe_side_position_radius - water_pipe_side_position_radius - water_pipe_straight_tube_side2_length, 0, rotate_water_straight_tube_side2);

  TGeoCombiTrans* combtrans_water_straight_tube_side_right2 = new TGeoCombiTrans("combtrans_water_straight_tube_side_right2", -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Cos((90 - water_pipe_side_angle) * TMath::Pi() / 180.) - water_pipe_straight_tube_side1_length * 2 - water_pipe_side_position_radius,
                                                                                 -(water_pipe_main_position_radius + water_pipe_side_position_radius) * TMath::Sin((90 - water_pipe_side_angle) * TMath::Pi() / 180.) + water_pipe_side_position_radius - water_pipe_side_position_radius - water_pipe_straight_tube_side2_length, 0, rotate_water_straight_tube_side2);

  combtrans_water_straight_tube_side_left2->RegisterYourself();
  combtrans_water_straight_tube_side_right2->RegisterYourself();

  //============ WATER, Connecting to the Patch Panel  ===============
  TGeoTorus* water_side_torus_left3 = new TGeoTorus("water_side_torus_left3", water_pipe_side_position_radius, 0, water_pipe_inner_radius, 0, 90);
  TGeoTubeSeg* water_straight_tube_left3 = new TGeoTubeSeg("water_straight_tube_left3", 0, water_pipe_inner_radius, water_pipe_straight_tube_side3_length, 0, 360);
  TGeoTorus* water_side_torus_rigth3 = new TGeoTorus("water_side_torus_rigth3", water_pipe_side_position_radius, 0, water_pipe_inner_radius, 0, 90);
  TGeoTubeSeg* water_straight_tube_rigth3 = new TGeoTubeSeg("water_straight_tube_rigth3", 0, water_pipe_inner_radius, water_pipe_straight_tube_side3_length, 0, 360);
  //==================================================================

  TGeoCompositeShape* water_shape = new TGeoCompositeShape("water_shape",
                                                           "water_main_torus + water_side_torus_left1:trans_water_side_torus_left1 + water_side_torus_right1:trans_water_side_torus_right1"
                                                           " + water_straight_tube_side1:combtrans_rotated_water_straight_tube_side1_left + water_straight_tube_side1:combtrans_rotated_water_straight_tube_side1_right"
                                                           " + water_side_torus_left2:trans_water_side_torus_left2 + water_side_torus_right2:trans_water_side_torus_right2"
                                                           " + water_straight_tube_side2:combtrans_water_straight_tube_side_left2 + water_straight_tube_side2:combtrans_water_straight_tube_side_right2"
                                                           " + water_straight_tube_left3:combtrans_water_straight_tube_side_left3"
                                                           " + water_side_torus_left3:combtrans_water_torus_left3"
                                                           " + water_straight_tube_rigth3:combtrans_water_straight_tube_side_rigth3"
                                                           " + water_side_torus_rigth3:combtrans_water_torus_rigth3");

  TGeoVolume* water = new TGeoVolume("water", water_shape, kMed_Water);
  water->SetLineColor(kBlue);
  mHalfPSU->AddNode(water, 1, nullptr);

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //DCDC cobverter
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  //Sheet1

  Double_t DCDC_sheet1_thickness = 0.040;
  Double_t DCDC_sheet1_height = 1.694;
  Double_t DCDC_sheet1_width = 4.348;
  Double_t DCDC_sheet1_radius = (block_radius + DCDC_sheet1_width / 2. - 0.673 - 1.85 + 1.85 / 2.);

  Double_t DCDC_sheet1_position_z = middle_spacer_main_thickness / 2. + middle_spacer_cover_thickness + middle_spacer_cover_block_thickness + DCDC_sheet1_thickness / 2;
  //Double_t DCDC_sheet1_position_z = electric_board_position_z + electric_board_thickness/2. + DCDC_sheet1_thickness/2.;

  TGeoBBox* DCDC_sheet1_box = new TGeoBBox("DCDC_sheet1_box", DCDC_sheet1_width / 2., DCDC_sheet1_height / 2., DCDC_sheet1_thickness / 2.);

  TGeoCompositeShape* rotated_DCDC_sheet1_box[24];

  std::string name_DCDC_sheet1_box = "";

  for (Int_t iB = 0; iB < 11; ++iB) {

    Double_t block_angle = (180. - block_angle_index[iB]) / 2. * TMath::Pi() / 180.;

    TGeoRotation* rotate_DCDC_sheet1_box_left = new TGeoRotation(Form("rotate_DCDC_sheet1_box_No%d", iB * 2), 180 - block_angle * 180. / TMath::Pi(), 0, 0);
    TGeoRotation* rotate_DCDC_sheet1_box_right = new TGeoRotation(Form("rotate_DCDC_sheet1_box_No%d", iB * 2 + 1), block_angle * 180. / TMath::Pi(), 0, 0);
    rotate_DCDC_sheet1_box_left->RegisterYourself();
    rotate_DCDC_sheet1_box_right->RegisterYourself();

    Double_t cent_block_left[] = {DCDC_sheet1_radius * TMath::Cos(block_angle), -DCDC_sheet1_radius * TMath::Sin(block_angle), DCDC_sheet1_position_z};
    Double_t cent_block_right[] = {DCDC_sheet1_radius * TMath::Cos(TMath::Pi() - block_angle), -DCDC_sheet1_radius * TMath::Sin(TMath::Pi() - block_angle), DCDC_sheet1_position_z};

    TGeoCombiTrans* combtrans_DCDC_sheet1_box_left = new TGeoCombiTrans(Form("combtrans_DCDC_sheet1_box_No%d", 2 * iB), cent_block_left[0], cent_block_left[1], cent_block_left[2], rotate_DCDC_sheet1_box_left);
    TGeoCombiTrans* combtrans_DCDC_sheet1_box_right = new TGeoCombiTrans(Form("combtrans_DCDC_sheet1_box_No%d", 2 * iB + 1), cent_block_right[0], cent_block_right[1], cent_block_right[2], rotate_DCDC_sheet1_box_right);
    combtrans_DCDC_sheet1_box_left->RegisterYourself();
    combtrans_DCDC_sheet1_box_right->RegisterYourself();

    if (iB == 0) {
      name_DCDC_sheet1_box += Form("DCDC_sheet1_box:combtrans_DCDC_sheet1_box_No%d", 2 * iB);
    } else {
      name_DCDC_sheet1_box += Form("+DCDC_sheet1_box:combtrans_DCDC_sheet1_box_No%d", 2 * iB);
    }

    name_DCDC_sheet1_box += Form("+DCDC_sheet1_box:combtrans_DCDC_sheet1_box_No%d", 2 * iB + 1);
  }

  TGeoBBox* DCDC_sheet1_box_side = new TGeoBBox("DCDC_sheet1_box_side", DCDC_sheet1_height / 2., DCDC_sheet1_width / 2., DCDC_sheet1_thickness / 2.);

  TGeoTranslation* trans_DCDC_sheet1_box_left = new TGeoTranslation("trans_DCDC_sheet1_box_left", 22.259 - middle_spacer_cover_block_width / 2, -8.305 + middle_spacer_cover_block_height / 2 - 0.576, DCDC_sheet1_position_z);
  trans_DCDC_sheet1_box_left->RegisterYourself();
  name_DCDC_sheet1_box += "+DCDC_sheet1_box_side:trans_DCDC_sheet1_box_left";

  TGeoTranslation* trans_DCDC_sheet1_box_right = new TGeoTranslation("trans_DCDC_sheet1_box_right", -22.247 + middle_spacer_cover_block_width / 2, -8.305 + middle_spacer_cover_block_height / 2 - 0.576, DCDC_sheet1_position_z);
  trans_DCDC_sheet1_box_right->RegisterYourself();
  name_DCDC_sheet1_box += "+DCDC_sheet1_box_side:trans_DCDC_sheet1_box_right";

  TGeoCompositeShape* DCDC_sheet1_shape = new TGeoCompositeShape("DCDC_sheet1_shape", name_DCDC_sheet1_box.c_str());

  TGeoVolume* DCDC_sheet1 = new TGeoVolume("DCDC_sheet1", DCDC_sheet1_shape, kMedPeek);
  DCDC_sheet1->SetLineColor(kGreen);

  TGeoRotation* trans_DCDC_sheet1_front = new TGeoRotation("trans_DCDC_sheet1_front", 0, 0, 0);
  TGeoRotation* trans_DCDC_sheet1_back = new TGeoRotation("trans_DCDC_sheet1_back", 180, 180, 0);

  mHalfPSU->AddNode(DCDC_sheet1, 0, trans_DCDC_sheet1_front);
  mHalfPSU->AddNode(DCDC_sheet1, 0, trans_DCDC_sheet1_back);

  //Cover box

  Double_t DCDC_cover_thickness = 0.800;
  Double_t DCDC_cover_outer_height = 1.400;
  Double_t DCDC_cover_outer_width = 1.85;
  Double_t DCDC_cover_depth = 0.05;
  Double_t DCDC_cover_inner_width = DCDC_cover_outer_width - 2 * DCDC_cover_depth;
  Double_t DCDC_cover_inner_height = DCDC_cover_outer_height - 2 * DCDC_cover_depth;

  Double_t DCDC_cover_position_z = DCDC_sheet1_position_z + DCDC_sheet1_thickness / 2. + DCDC_cover_thickness / 2.;

  TGeoBBox* DCDC_cover_outer_box = new TGeoBBox("DCDC_cover_outer_box", DCDC_cover_outer_width / 2., DCDC_cover_outer_height / 2., DCDC_cover_thickness / 2.);
  TGeoBBox* DCDC_cover_innner_box = new TGeoBBox("DCDC_cover_inner_box", DCDC_cover_inner_width / 2., DCDC_cover_inner_height / 2., DCDC_cover_thickness / 2.);

  TGeoCompositeShape* rotated_DCDC_cover_outer_box[23];
  TGeoCompositeShape* rotated_DCDC_cover_inner_box[23];

  std::string name_DCDC_cover_box = "";

  for (Int_t iB = 0; iB < 11; ++iB) {

    Double_t block_angle = (180. - block_angle_index[iB]) / 2. * TMath::Pi() / 180.;

    TGeoRotation* rotate_DCDC_cover_box_left = new TGeoRotation(Form("rotate_DCDC_cover_box_No%d", iB * 2), 180 - block_angle * 180. / TMath::Pi(), 0, 0);
    TGeoRotation* rotate_DCDC_cover_box_right = new TGeoRotation(Form("rotate_DCDC_cover_box_No%d", iB * 2 + 1), block_angle * 180. / TMath::Pi(), 0, 0);
    rotate_DCDC_cover_box_left->RegisterYourself();
    rotate_DCDC_cover_box_right->RegisterYourself();

    Double_t cent_block_left[] = {block_radius * TMath::Cos(block_angle), -block_radius * TMath::Sin(block_angle), DCDC_cover_position_z};
    Double_t cent_block_right[] = {block_radius * TMath::Cos(TMath::Pi() - block_angle), -block_radius * TMath::Sin(TMath::Pi() - block_angle), DCDC_cover_position_z};

    TGeoCombiTrans* combtrans_DCDC_cover_outer_box_left = new TGeoCombiTrans(Form("combtrans_DCDC_cover_outer_box_No%d", 2 * iB), cent_block_left[0], cent_block_left[1], cent_block_left[2], rotate_DCDC_cover_box_left);
    TGeoCombiTrans* combtrans_DCDC_cover_outer_box_right = new TGeoCombiTrans(Form("combtrans_DCDC_cover_outer_box_No%d", 2 * iB + 1), cent_block_right[0], cent_block_right[1], cent_block_right[2], rotate_DCDC_cover_box_right);
    combtrans_DCDC_cover_outer_box_left->RegisterYourself();
    combtrans_DCDC_cover_outer_box_right->RegisterYourself();

    TGeoCombiTrans* combtrans_DCDC_cover_inner_box_left = new TGeoCombiTrans(Form("combtrans_DCDC_cover_inner_box_No%d", 2 * iB), cent_block_left[0], cent_block_left[1], cent_block_left[2] - 2 * DCDC_cover_depth, rotate_DCDC_cover_box_left);
    TGeoCombiTrans* combtrans_DCDC_cover_inner_box_right = new TGeoCombiTrans(Form("combtrans_DCDC_cover_inner_box_No%d", 2 * iB + 1), cent_block_right[0], cent_block_right[1], cent_block_right[2] - 2 * DCDC_cover_depth, rotate_DCDC_cover_box_right);
    combtrans_DCDC_cover_inner_box_left->RegisterYourself();
    combtrans_DCDC_cover_inner_box_right->RegisterYourself();

    if (iB == 0) {
      name_DCDC_cover_box += Form("DCDC_cover_outer_box:combtrans_DCDC_cover_outer_box_No%d - DCDC_cover_inner_box:combtrans_DCDC_cover_inner_box_No%d", 2 * iB, 2 * iB);
    } else {
      name_DCDC_cover_box += Form("+DCDC_cover_outer_box:combtrans_DCDC_cover_outer_box_No%d - DCDC_cover_inner_box:combtrans_DCDC_cover_inner_box_No%d", 2 * iB, 2 * iB);
    }
    name_DCDC_cover_box += Form("+DCDC_cover_outer_box:combtrans_DCDC_cover_outer_box_No%d - DCDC_cover_inner_box:combtrans_DCDC_cover_inner_box_No%d", 2 * iB + 1, 2 * iB + 1);
  }

  TGeoBBox* DCDC_cover_outer_box_side = new TGeoBBox("DCDC_cover_outer_box_side", DCDC_cover_outer_height / 2., DCDC_cover_outer_width / 2., DCDC_cover_thickness / 2.);
  TGeoBBox* DCDC_cover_innner_box_side = new TGeoBBox("DCDC_cover_inner_box_side", DCDC_cover_inner_height / 2., DCDC_cover_inner_width / 2., DCDC_cover_thickness / 2.);

  TGeoTranslation* trans_DCDC_cover_outer_box_left = new TGeoTranslation("trans_DCDC_cover_outer_box_left", 22.259 - middle_spacer_cover_block_width / 2, -8.305 + middle_spacer_cover_block_height / 2, DCDC_cover_position_z);
  TGeoTranslation* trans_DCDC_cover_inner_box_left = new TGeoTranslation("trans_DCDC_cover_inner_box_left", 22.259 - middle_spacer_cover_block_width / 2, -8.305 + middle_spacer_cover_block_height / 2, DCDC_cover_position_z - 2 * DCDC_cover_depth);
  trans_DCDC_cover_outer_box_left->RegisterYourself();
  trans_DCDC_cover_inner_box_left->RegisterYourself();
  name_DCDC_cover_box += "+DCDC_cover_outer_box_side:trans_DCDC_cover_outer_box_left - DCDC_cover_inner_box_side:trans_DCDC_cover_inner_box_left";

  TGeoTranslation* trans_DCDC_cover_outer_box_right = new TGeoTranslation("trans_DCDC_cover_outer_box_right", -22.247 + middle_spacer_cover_block_width / 2, -8.305 + middle_spacer_cover_block_height / 2, DCDC_cover_position_z);
  TGeoTranslation* trans_DCDC_cover_inner_box_right = new TGeoTranslation("trans_DCDC_cover_inner_box_right", -22.247 + middle_spacer_cover_block_width / 2, -8.305 + middle_spacer_cover_block_height / 2, DCDC_cover_position_z - 2 * DCDC_cover_depth);
  trans_DCDC_cover_outer_box_right->RegisterYourself();
  trans_DCDC_cover_inner_box_right->RegisterYourself();
  name_DCDC_cover_box += "+DCDC_cover_outer_box_side:trans_DCDC_cover_outer_box_right - DCDC_cover_inner_box_side:trans_DCDC_cover_inner_box_right";

  TGeoCompositeShape* DCDC_cover_shape = new TGeoCompositeShape("DCDC_cover_shape", name_DCDC_cover_box.c_str());
  TGeoVolume* DCDC_cover = new TGeoVolume("DCDC_cover", DCDC_cover_shape, kMedAlu);
  DCDC_cover->SetLineColor(kGray);

  TGeoRotation* trans_DCDC_cover_front = new TGeoRotation("trans_DCDC_cover_front", 0, 0, 0);
  TGeoRotation* trans_DCDC_cover_back = new TGeoRotation("trans_DCDC_cover_back", 180, 180, 0);

  mHalfPSU->AddNode(DCDC_cover, 0, trans_DCDC_cover_front);
  mHalfPSU->AddNode(DCDC_cover, 0, trans_DCDC_cover_back);

  //DCDC converter connector

  Double_t DCDC_connector_thickness = 0.225;
  Double_t DCDC_connector_height = 1.44;
  Double_t DCDC_connector_width = 0.305;
  Double_t DCDC_connector_radius = (block_radius + DCDC_sheet1_width / 2. - 0.673 - 1.85 + 1.85 / 2. + DCDC_sheet1_width / 2 - 0.55 - 0.305 + 0.305 / 2.);
  Double_t DCDC_connector_position_z = DCDC_sheet1_position_z + DCDC_sheet1_thickness / 2. + DCDC_connector_thickness / 2;

  TGeoBBox* DCDC_connector_box = new TGeoBBox("DCDC_connector_box", DCDC_connector_width / 2., DCDC_connector_height / 2., DCDC_connector_thickness / 2.);

  TGeoCompositeShape* rotated_DCDC_connector_box[23];

  std::string name_DCDC_connector_box = "";

  for (Int_t iB = 0; iB < 11; ++iB) {

    Double_t block_angle = (180. - block_angle_index[iB]) / 2. * TMath::Pi() / 180.;

    TGeoRotation* rotate_DCDC_connector_box_left = new TGeoRotation(Form("rotate_DCDC_connector_box_No%d", iB * 2), 180 - block_angle * 180. / TMath::Pi(), 0, 0);
    TGeoRotation* rotate_DCDC_connector_box_right = new TGeoRotation(Form("rotate_DCDC_connector_box_No%d", iB * 2 + 1), block_angle * 180. / TMath::Pi(), 0, 0);
    rotate_DCDC_connector_box_left->RegisterYourself();
    rotate_DCDC_connector_box_right->RegisterYourself();
    Double_t cent_block_left[] = {DCDC_connector_radius * TMath::Cos(block_angle), -DCDC_connector_radius * TMath::Sin(block_angle), DCDC_connector_position_z};
    Double_t cent_block_right[] = {DCDC_connector_radius * TMath::Cos(TMath::Pi() - block_angle), -DCDC_connector_radius * TMath::Sin(TMath::Pi() - block_angle), DCDC_connector_position_z};

    TGeoCombiTrans* combtrans_DCDC_connector_box_left = new TGeoCombiTrans(Form("combtrans_DCDC_connector_box_No%d", 2 * iB), cent_block_left[0], cent_block_left[1], cent_block_left[2], rotate_DCDC_connector_box_left);
    TGeoCombiTrans* combtrans_DCDC_connector_box_right = new TGeoCombiTrans(Form("combtrans_DCDC_connector_box_No%d", 2 * iB + 1), cent_block_right[0], cent_block_right[1], cent_block_right[2], rotate_DCDC_connector_box_right);
    combtrans_DCDC_connector_box_left->RegisterYourself();
    combtrans_DCDC_connector_box_right->RegisterYourself();

    if (iB == 0) {
      name_DCDC_connector_box += Form("DCDC_connector_box:combtrans_DCDC_connector_box_No%d", 2 * iB);
    } else {
      name_DCDC_connector_box += Form("+DCDC_connector_box:combtrans_DCDC_connector_box_No%d", 2 * iB);
    }
    name_DCDC_connector_box += Form("+DCDC_connector_box:combtrans_DCDC_connector_box_No%d", 2 * iB + 1);
  }

  TGeoBBox* DCDC_connector_box_side = new TGeoBBox("DCDC_connector_box_side", DCDC_connector_height / 2., DCDC_connector_width / 2., DCDC_connector_thickness / 2.);

  TGeoTranslation* trans_DCDC_connector_box_left = new TGeoTranslation("trans_DCDC_connector_box_left", 22.259 - middle_spacer_cover_block_width / 2, -8.305 + middle_spacer_cover_block_height / 2 - 2.019, DCDC_connector_position_z);
  trans_DCDC_connector_box_left->RegisterYourself();
  name_DCDC_connector_box += "+DCDC_connector_box_side:trans_DCDC_connector_box_left";

  TGeoTranslation* trans_DCDC_connector_box_right = new TGeoTranslation("trans_DCDC_connector_box_right", -22.247 + middle_spacer_cover_block_width / 2, -8.305 + middle_spacer_cover_block_height / 2 - 2.019, DCDC_connector_position_z);
  trans_DCDC_connector_box_right->RegisterYourself();
  name_DCDC_connector_box += "+DCDC_connector_box_side:trans_DCDC_connector_box_right";

  TGeoCompositeShape* DCDC_connector_shape = new TGeoCompositeShape("DCDC_connector_shape", name_DCDC_connector_box.c_str());

  TGeoVolume* DCDC_connector = new TGeoVolume("DCDC_connector", DCDC_connector_shape, kMedPeek);
  DCDC_connector->SetLineColor(kGray + 2);

  TGeoRotation* trans_DCDC_connector_front = new TGeoRotation("trans_DCDC_connector_front", 0, 0, 0);
  TGeoRotation* trans_DCDC_connector_back = new TGeoRotation("trans_DCDC_connector_back", 180, 180, 0);

  mHalfPSU->AddNode(DCDC_connector, 0, trans_DCDC_connector_front);
  mHalfPSU->AddNode(DCDC_connector, 0, trans_DCDC_connector_back);

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Mezzanine
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  //prop

  Double_t mezzanine_prop_main_length = 1.615;
  Double_t mezzanine_prop_main_radius = 0.476 / 2.;
  Double_t mezzanine_prop_small_length = 0.16;
  Double_t mezzanine_prop_small_radius = 0.3 / 2.;
  Double_t mezzanine_prop_lid_radius = 0.57 / 2.;

  Double_t mezzanine_prop_main_position_z = middle_spacer_main_thickness / 2. + middle_spacer_cover_thickness + electric_board_thickness + mezzanine_prop_main_length / 2;
  Double_t mezzanine_prop_small_position_z = middle_spacer_main_thickness / 2. + middle_spacer_cover_thickness + electric_board_thickness + mezzanine_prop_main_length + mezzanine_prop_small_length / 2.;
  Double_t mezzanine_prop_lid_position_z = middle_spacer_main_thickness / 2. + middle_spacer_cover_thickness + electric_board_thickness + mezzanine_prop_main_length + mezzanine_prop_small_length;

  TGeoTubeSeg* mezzanine_prop_main_tube = new TGeoTubeSeg("mezzanine_prop_main_tube", 0, mezzanine_prop_main_radius, mezzanine_prop_main_length / 2., 0, 360);
  TGeoTubeSeg* mezzanine_prop_small_tube = new TGeoTubeSeg("mezzanine_prop_small_tube", 0, mezzanine_prop_small_radius, mezzanine_prop_small_length / 2., 0, 360);
  TGeoSphere* mezzanine_prop_lid_sphere = new TGeoSphere("mezzanine_prop_lid_sphere", 0, mezzanine_prop_lid_radius, 0, 90, 0, 360);
  TGeoTranslation* trans_mezzanine_prop_main_tube = new TGeoTranslation("trans_mezzanine_prop_main_tube", 0, 0, mezzanine_prop_main_position_z);
  TGeoTranslation* trans_mezzanine_prop_small_tube = new TGeoTranslation("trans_mezzanine_prop_small_tube", 0, 0, mezzanine_prop_small_position_z);
  TGeoTranslation* trans_mezzanine_prop_lid_sphere = new TGeoTranslation("trans_mezzanine_prop_lid_sphere", 0, 0, mezzanine_prop_lid_position_z);
  trans_mezzanine_prop_main_tube->RegisterYourself();
  trans_mezzanine_prop_small_tube->RegisterYourself();
  trans_mezzanine_prop_lid_sphere->RegisterYourself();

  TGeoCompositeShape* mazzanine_prop_shape = new TGeoCompositeShape("mazzanine_prop_shape", "mezzanine_prop_main_tube:trans_mezzanine_prop_main_tube + mezzanine_prop_small_tube:trans_mezzanine_prop_small_tube + mezzanine_prop_lid_sphere:trans_mezzanine_prop_lid_sphere");
  TGeoTranslation* trans_mezzanine_prop_left = new TGeoTranslation("trans_mezzanine_prop_left", +8, -21.5, 0);
  TGeoTranslation* trans_mezzanine_prop_right = new TGeoTranslation("trans_mezzanine_prop_right", -8, -21.5, 0);
  trans_mezzanine_prop_left->RegisterYourself();
  trans_mezzanine_prop_right->RegisterYourself();
  TGeoCompositeShape* mazzanine_prop_shape_bothside = new TGeoCompositeShape("mazzanine_prop_shape_bothside", "mazzanine_prop_shape:trans_mezzanine_prop_left+mazzanine_prop_shape:trans_mezzanine_prop_right");

  TGeoVolume* mazzanine_prop = new TGeoVolume("mazzanine_prop", mazzanine_prop_shape_bothside, kMedAlu);
  mazzanine_prop->SetLineColor(kAzure - 3);

  TGeoRotation* trans_mazzanine_prop_front = new TGeoRotation("trans_mazzanine_prop_front", 0, 0, 0);
  TGeoRotation* trans_mazzanine_prop_back = new TGeoRotation("trans_mazzanine_prop_back", 180, 180, 0);

  mHalfPSU->AddNode(mazzanine_prop, 0, trans_mazzanine_prop_front);
  mHalfPSU->AddNode(mazzanine_prop, 0, trans_mazzanine_prop_back);

  //main

  Double_t mezzanine_main_thickness = mezzanine_prop_small_length;
  Double_t mezzanine_main_width = 20.7;
  Double_t mezzanine_main_height = 9.5;
  Double_t mezzanine_main_position_z = middle_spacer_main_thickness / 2. + middle_spacer_cover_thickness + electric_board_thickness + mezzanine_prop_main_length + mezzanine_main_thickness / 2.;

  TGeoBBox* mezzanine_main_box = new TGeoBBox("mezzanine_main_box", mezzanine_main_width / 2., mezzanine_main_height / 2., mezzanine_main_thickness / 2.);
  TGeoTubeSeg* mezzanine_main_sub_arc = new TGeoTubeSeg("mezzanine_main_sub_arc", 0, electric_board_min_radius1, mezzanine_main_thickness / 2 + delta_thickness, 180, 0);
  TGeoTubeSeg* mezzanine_main_sub_hole = new TGeoTubeSeg("mezzanine_main_sub_hole", 0, mezzanine_prop_small_radius, mezzanine_main_thickness / 2 + delta_thickness, 0, 360);

  TGeoTranslation* trans_mezzanine_main_box = new TGeoTranslation("trans_mezzanine_main_box", 0, -15.1 - mezzanine_main_height / 2., mezzanine_main_position_z);
  TGeoTranslation* trans_mezzanine_main_sub_arc = new TGeoTranslation("trans_mezzanine_main_sub_arc", 0, 0, mezzanine_main_position_z);
  TGeoTranslation* trans_mezzanine_main_sub_hole_left = new TGeoTranslation("trans_mezzanine_main_sub_hole_left", +8, -21.5, mezzanine_main_position_z);
  TGeoTranslation* trans_mezzanine_main_sub_hole_right = new TGeoTranslation("trans_mezzanine_main_sub_hole_right", +8, -21.5, mezzanine_main_position_z);
  trans_mezzanine_main_box->RegisterYourself();
  trans_mezzanine_main_sub_arc->RegisterYourself();
  trans_mezzanine_main_sub_hole_right->RegisterYourself();
  trans_mezzanine_main_sub_hole_left->RegisterYourself();

  TGeoCompositeShape* mezzanine_shape = new TGeoCompositeShape("mezzanine_shape", "mezzanine_main_box:trans_mezzanine_main_box - mezzanine_main_sub_arc:trans_mezzanine_main_sub_arc - mezzanine_main_sub_hole:trans_mezzanine_main_sub_hole_left - mezzanine_main_sub_hole:trans_mezzanine_main_sub_hole_right");
  TGeoVolume* mezzanine = new TGeoVolume("mezzanine", mezzanine_shape, kMedPeek);
  mezzanine->SetLineColor(kGreen + 2);

  TGeoRotation* trans_mezzanine_front = new TGeoRotation("trans_mazzanine_front", 0, 0, 0);
  TGeoRotation* trans_mazzanine_back = new TGeoRotation("trans_mazzanine_back", 180, 180, 0);

  mHalfPSU->AddNode(mezzanine, 0, trans_mezzanine_front);
  mHalfPSU->AddNode(mezzanine, 0, trans_mazzanine_back);

  //connector

  Double_t mezzanine_connector_base_box_thickness = 1.186;
  Double_t mezzanine_connector_base_box_width = 6.778;
  Double_t mezzanine_connector_base_box_height = 1.595;
  Double_t mezzanine_connector_base_box_position_z = middle_spacer_main_thickness / 2. + middle_spacer_cover_thickness + electric_board_thickness + mezzanine_connector_base_box_thickness / 2;

  Double_t mezzanine_connector_base_sub_box_thickness = mezzanine_connector_base_box_thickness - 0.307;
  Double_t mezzanine_connector_base_sub_box_width = 6.778;
  Double_t mezzanine_connector_base_sub_box_height = 1.468;

  TGeoBBox* mezzanine_connector_base_box = new TGeoBBox("mezzanine_connector_base", mezzanine_connector_base_box_width / 2., mezzanine_connector_base_box_height / 2., mezzanine_connector_base_box_thickness / 2.);
  TGeoBBox* mezzanine_connector_base_sub_box = new TGeoBBox("mezzanine_connector_base_sub_box", mezzanine_connector_base_sub_box_width / 2. + delta_thickness, mezzanine_connector_base_sub_box_height / 2., mezzanine_connector_base_sub_box_thickness / 2.);

  TGeoTranslation* trans_mezzanine_connector_base_box = new TGeoTranslation("trans_mezzanine_connector_base_box", 0, -22.421 - mezzanine_connector_base_box_height / 2.0, mezzanine_connector_base_box_position_z);
  TGeoTranslation* trans_mezzanine_connector_base_sub_box = new TGeoTranslation("trans_mezzanine_connector_base_sub_box", 0, -22.421 - mezzanine_connector_base_box_height / 2.0, mezzanine_connector_base_box_position_z + (mezzanine_connector_base_box_thickness - mezzanine_connector_base_sub_box_thickness) / 2.);
  trans_mezzanine_connector_base_box->RegisterYourself();
  trans_mezzanine_connector_base_sub_box->RegisterYourself();

  TGeoCompositeShape* mezzanine_connector_base_shape = new TGeoCompositeShape("mezzanine_connector_base_shape", "mezzanine_connector_base:trans_mezzanine_connector_base_box - mezzanine_connector_base_sub_box:trans_mezzanine_connector_base_sub_box");
  TGeoVolume* mezzanine_connector_base = new TGeoVolume("mezzanine_connector_base", mezzanine_connector_base_shape, kMedPeek);
  mezzanine_connector_base->SetLineColor(kOrange + 7);

  TGeoRotation* trans_mezzanine_connector_base_front = new TGeoRotation("trans_mezzanine_connector_base_front", 0, 0, 0);
  TGeoRotation* trans_mezzanine_connector_base_back = new TGeoRotation("trans_mezzanine_connector_base_back", 180, 180, 0);
  mHalfPSU->AddNode(mezzanine_connector_base, 0, trans_mezzanine_connector_base_front);
  mHalfPSU->AddNode(mezzanine_connector_base, 0, trans_mezzanine_connector_base_back);

  Double_t mezzanine_connector_lid_bottom_box_thickness = 0.112;
  Double_t mezzanine_connector_lid_bottom_box_width = 6.778;
  Double_t mezzanine_connector_lid_bottom_box_height = 1.468;
  Double_t mezzanine_connector_lid_bottom_box_position_z = middle_spacer_main_thickness / 2. + middle_spacer_cover_thickness + electric_board_thickness + 0.967 + mezzanine_connector_lid_bottom_box_thickness / 2.;

  TGeoBBox* mezzanine_connector_lid_bottom_box = new TGeoBBox("mezzanine_connector_lid_bottom_box", mezzanine_connector_lid_bottom_box_width / 2., mezzanine_connector_lid_bottom_box_height / 2., mezzanine_connector_lid_bottom_box_thickness / 2.);
  TGeoVolume* mezzanine_connector_lid_bottom = new TGeoVolume("mezzanine_connector_lid_bottom", mezzanine_connector_lid_bottom_box, kMedPeek);
  mezzanine_connector_lid_bottom->SetLineColor(kGray + 2);

  TGeoTranslation* trans_mezzanine_connector_lid_bottom_front = new TGeoTranslation("trans_mezzanine_connector_lid_bottom_front", 0, 0 - 22.421 - mezzanine_connector_base_box_height / 2., mezzanine_connector_lid_bottom_box_position_z);
  TGeoTranslation* trans_mezzanine_connector_lid_bottom_back = new TGeoTranslation("trans_mezzanine_connector_lid_bottom_back", 0, 0 - 22.421 - mezzanine_connector_base_box_height / 2., -mezzanine_connector_lid_bottom_box_position_z);
  mHalfPSU->AddNode(mezzanine_connector_lid_bottom, 0, trans_mezzanine_connector_lid_bottom_front);
  mHalfPSU->AddNode(mezzanine_connector_lid_bottom, 0, trans_mezzanine_connector_lid_bottom_back);

  Double_t mezzanine_connector_lid_top_box_thickness = mezzanine_main_position_z - mezzanine_main_thickness / 2 - mezzanine_connector_lid_bottom_box_position_z - mezzanine_connector_lid_bottom_box_thickness / 2.;
  Double_t mezzanine_connector_lid_top_box_width = 6.660;
  Double_t mezzanine_connector_lid_top_box_height = 1.328;
  Double_t mezzanine_connector_lid_top_box_position_z = middle_spacer_main_thickness / 2. + middle_spacer_cover_thickness + electric_board_thickness + 0.967 + mezzanine_connector_lid_bottom_box_thickness + mezzanine_connector_lid_top_box_thickness / 2.;
  TGeoBBox* mezzanine_connector_lid_top_box = new TGeoBBox("mezzanine_connector_lid_top_box", mezzanine_connector_lid_top_box_width / 2., mezzanine_connector_lid_top_box_height / 2., mezzanine_connector_lid_top_box_thickness / 2.);
  TGeoVolume* mezzanine_connector_lid_top = new TGeoVolume("mezzanine_connector_lid_top", mezzanine_connector_lid_top_box, kMedPeek);
  mezzanine_connector_lid_top->SetLineColor(kGray + 2);

  TGeoTranslation* trans_mezzanine_connector_lid_top_front = new TGeoTranslation("trans_mezzanine_connector_lid_top_front", 0, 0 - 22.421 - mezzanine_connector_base_box_height / 2., mezzanine_connector_lid_top_box_position_z);
  TGeoTranslation* trans_mezzanine_connector_lid_top_back = new TGeoTranslation("trans_mezzanine_connector_lid_top_back", 0, 0 - 22.421 - mezzanine_connector_base_box_height / 2., -mezzanine_connector_lid_top_box_position_z);
  mHalfPSU->AddNode(mezzanine_connector_lid_top, 0, trans_mezzanine_connector_lid_top_front);
  mHalfPSU->AddNode(mezzanine_connector_lid_top, 0, trans_mezzanine_connector_lid_top_back);

  //DCDC converter on mezzanine

  Double_t spacer_sheet1_mezzanine_box_thickness = 0.086;
  Double_t spacer_sheet1_mezzanine_box_width = 1.397;
  Double_t spacer_sheet1_mezzanine_box_height = 0.343;
  Double_t spacer_sheet1_mezzanine_box_position_z = middle_spacer_main_thickness / 2. + middle_spacer_cover_thickness + electric_board_thickness + mezzanine_prop_main_length - spacer_sheet1_mezzanine_box_thickness / 2.;
  Double_t gap_sheet1_on_mezzanine = 0.1;

  TGeoBBox* spacer_sheet1_mezzanine_box = new TGeoBBox("spacer_sheet1_mezzanine_box", spacer_sheet1_mezzanine_box_width / 2., spacer_sheet1_mezzanine_box_height / 2., spacer_sheet1_mezzanine_box_thickness / 2.);
  TGeoTranslation* trans_spacer_sheet1_mezzanine_box_front[5];
  trans_spacer_sheet1_mezzanine_box_front[0] = new TGeoTranslation("trans_spacer_sheet1_mezzanine_box_front_No0", +2 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (spacer_sheet1_mezzanine_box_width / 2. + 0.531), spacer_sheet1_mezzanine_box_position_z);
  trans_spacer_sheet1_mezzanine_box_front[1] = new TGeoTranslation("trans_spacer_sheet1_mezzanine_box_front_No1", +1 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (spacer_sheet1_mezzanine_box_width / 2. + 0.531), spacer_sheet1_mezzanine_box_position_z);
  trans_spacer_sheet1_mezzanine_box_front[2] = new TGeoTranslation("trans_spacer_sheet1_mezzanine_box_front_No2", 0 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (spacer_sheet1_mezzanine_box_width / 2. + 0.531), spacer_sheet1_mezzanine_box_position_z);
  trans_spacer_sheet1_mezzanine_box_front[3] = new TGeoTranslation("trans_spacer_sheet1_mezzanine_box_front_No3", -1 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (spacer_sheet1_mezzanine_box_width / 2. + 0.531), spacer_sheet1_mezzanine_box_position_z);
  trans_spacer_sheet1_mezzanine_box_front[4] = new TGeoTranslation("trans_spacer_sheet1_mezzanine_box_front_No4", -2 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (spacer_sheet1_mezzanine_box_width / 2. + 0.531), spacer_sheet1_mezzanine_box_position_z);
  TGeoTranslation* trans_spacer_sheet1_mezzanine_box_back[5];
  trans_spacer_sheet1_mezzanine_box_back[0] = new TGeoTranslation("trans_spacer_sheet1_mezzanine_box_back_No0", +2 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (spacer_sheet1_mezzanine_box_width / 2. + 0.531), -spacer_sheet1_mezzanine_box_position_z);
  trans_spacer_sheet1_mezzanine_box_back[1] = new TGeoTranslation("trans_spacer_sheet1_mezzanine_box_back_No1", +1 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (spacer_sheet1_mezzanine_box_width / 2. + 0.531), -spacer_sheet1_mezzanine_box_position_z);
  trans_spacer_sheet1_mezzanine_box_back[2] = new TGeoTranslation("trans_spacer_sheet1_mezzanine_box_back_No2", 0 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (spacer_sheet1_mezzanine_box_width / 2. + 0.531), -spacer_sheet1_mezzanine_box_position_z);
  trans_spacer_sheet1_mezzanine_box_back[3] = new TGeoTranslation("trans_spacer_sheet1_mezzanine_box_back_No3", -1 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (spacer_sheet1_mezzanine_box_width / 2. + 0.531), -spacer_sheet1_mezzanine_box_position_z);
  trans_spacer_sheet1_mezzanine_box_back[4] = new TGeoTranslation("trans_spacer_sheet1_mezzanine_box_back_No4", -2 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (spacer_sheet1_mezzanine_box_width / 2. + 0.531), -spacer_sheet1_mezzanine_box_position_z);

  TGeoVolume* spacer_sheet1_mezzanine = new TGeoVolume("spacer_sheet1_mezzanine", spacer_sheet1_mezzanine_box, kMedPeek);
  spacer_sheet1_mezzanine->SetLineColor(kGray + 2);
  mHalfPSU->AddNode(spacer_sheet1_mezzanine, 0, trans_spacer_sheet1_mezzanine_box_front[0]);
  mHalfPSU->AddNode(spacer_sheet1_mezzanine, 0, trans_spacer_sheet1_mezzanine_box_front[1]);
  mHalfPSU->AddNode(spacer_sheet1_mezzanine, 0, trans_spacer_sheet1_mezzanine_box_front[2]);
  mHalfPSU->AddNode(spacer_sheet1_mezzanine, 0, trans_spacer_sheet1_mezzanine_box_front[3]);
  mHalfPSU->AddNode(spacer_sheet1_mezzanine, 0, trans_spacer_sheet1_mezzanine_box_front[4]);
  mHalfPSU->AddNode(spacer_sheet1_mezzanine, 0, trans_spacer_sheet1_mezzanine_box_back[0]);
  mHalfPSU->AddNode(spacer_sheet1_mezzanine, 0, trans_spacer_sheet1_mezzanine_box_back[1]);
  mHalfPSU->AddNode(spacer_sheet1_mezzanine, 0, trans_spacer_sheet1_mezzanine_box_back[2]);
  mHalfPSU->AddNode(spacer_sheet1_mezzanine, 0, trans_spacer_sheet1_mezzanine_box_back[3]);
  mHalfPSU->AddNode(spacer_sheet1_mezzanine, 0, trans_spacer_sheet1_mezzanine_box_back[4]);

  Double_t sheet1_on_mezzanine_box_position_z = middle_spacer_main_thickness / 2. + middle_spacer_cover_thickness + electric_board_thickness + mezzanine_prop_main_length - spacer_sheet1_mezzanine_box_thickness - DCDC_sheet1_thickness / 2.;

  TGeoTranslation* trans_sheet1_on_mezzanine_box_front[5];
  trans_sheet1_on_mezzanine_box_front[0] = new TGeoTranslation("trans_sheet1_on_mezzanine_box_front_No0", +2 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (DCDC_sheet1_width / 2.), sheet1_on_mezzanine_box_position_z);
  trans_sheet1_on_mezzanine_box_front[1] = new TGeoTranslation("trans_sheet1_on_mezzanine_box_front_No1", +1 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (DCDC_sheet1_width / 2.), sheet1_on_mezzanine_box_position_z);
  trans_sheet1_on_mezzanine_box_front[2] = new TGeoTranslation("trans_sheet1_on_mezzanine_box_front_No2", 0 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (DCDC_sheet1_width / 2.), sheet1_on_mezzanine_box_position_z);
  trans_sheet1_on_mezzanine_box_front[3] = new TGeoTranslation("trans_sheet1_on_mezzanine_box_front_No3", -1 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (DCDC_sheet1_width / 2.), sheet1_on_mezzanine_box_position_z);
  trans_sheet1_on_mezzanine_box_front[4] = new TGeoTranslation("trans_sheet1_on_mezzanine_box_front_No4", -2 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (DCDC_sheet1_width / 2.), sheet1_on_mezzanine_box_position_z);
  TGeoTranslation* trans_sheet1_on_mezzanine_box_back[5];
  trans_sheet1_on_mezzanine_box_back[0] = new TGeoTranslation("trans_sheet1_on_mezzanine_box_back_No0", +2 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (DCDC_sheet1_width / 2.), -sheet1_on_mezzanine_box_position_z);
  trans_sheet1_on_mezzanine_box_back[1] = new TGeoTranslation("trans_sheet1_on_mezzanine_box_back_No1", +1 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (DCDC_sheet1_width / 2.), -sheet1_on_mezzanine_box_position_z);
  trans_sheet1_on_mezzanine_box_back[2] = new TGeoTranslation("trans_sheet1_on_mezzanine_box_back_No2", 0 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (DCDC_sheet1_width / 2.), -sheet1_on_mezzanine_box_position_z);
  trans_sheet1_on_mezzanine_box_back[3] = new TGeoTranslation("trans_sheet1_on_mezzanine_box_back_No3", -1 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (DCDC_sheet1_width / 2.), -sheet1_on_mezzanine_box_position_z);
  trans_sheet1_on_mezzanine_box_back[4] = new TGeoTranslation("trans_sheet1_on_mezzanine_box_back_No4", -2 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (DCDC_sheet1_width / 2.), -sheet1_on_mezzanine_box_position_z);

  TGeoVolume* sheet1_on_mezzanine = new TGeoVolume("sheet1_on_mezzanine", DCDC_sheet1_box_side, kMedPeek);
  sheet1_on_mezzanine->SetLineColor(kGreen);

  mHalfPSU->AddNode(sheet1_on_mezzanine, 0, trans_sheet1_on_mezzanine_box_front[0]);
  mHalfPSU->AddNode(sheet1_on_mezzanine, 0, trans_sheet1_on_mezzanine_box_front[1]);
  mHalfPSU->AddNode(sheet1_on_mezzanine, 0, trans_sheet1_on_mezzanine_box_front[2]);
  mHalfPSU->AddNode(sheet1_on_mezzanine, 0, trans_sheet1_on_mezzanine_box_front[3]);
  mHalfPSU->AddNode(sheet1_on_mezzanine, 0, trans_sheet1_on_mezzanine_box_front[4]);
  mHalfPSU->AddNode(sheet1_on_mezzanine, 0, trans_sheet1_on_mezzanine_box_back[0]);
  mHalfPSU->AddNode(sheet1_on_mezzanine, 0, trans_sheet1_on_mezzanine_box_back[1]);
  mHalfPSU->AddNode(sheet1_on_mezzanine, 0, trans_sheet1_on_mezzanine_box_back[2]);
  mHalfPSU->AddNode(sheet1_on_mezzanine, 0, trans_sheet1_on_mezzanine_box_back[3]);
  mHalfPSU->AddNode(sheet1_on_mezzanine, 0, trans_sheet1_on_mezzanine_box_back[4]);

  Double_t DCDC_connector_on_mezzanine_position_z = middle_spacer_main_thickness / 2. + middle_spacer_cover_thickness + electric_board_thickness + mezzanine_prop_main_length - spacer_sheet1_mezzanine_box_thickness - DCDC_sheet1_thickness - DCDC_connector_thickness / 2.;

  TGeoVolume* DCDC_connector_on_mezzanine = new TGeoVolume("DCDC_connector_on_mezzanine", DCDC_connector_box_side, kMedPeek);
  DCDC_connector_on_mezzanine->SetLineColor(kGray + 2);
  TGeoTranslation* trans_DCDC_connector_on_mezzanine_front[5];
  trans_DCDC_connector_on_mezzanine_front[0] = new TGeoTranslation("trans_DCDC_connector_on_mezzanine_front_No0", +2 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (spacer_sheet1_mezzanine_box_width / 2. + 0.531), DCDC_connector_on_mezzanine_position_z);
  trans_DCDC_connector_on_mezzanine_front[1] = new TGeoTranslation("trans_DCDC_connector_on_mezzanine_front_No1", +1 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (spacer_sheet1_mezzanine_box_width / 2. + 0.531), DCDC_connector_on_mezzanine_position_z);
  trans_DCDC_connector_on_mezzanine_front[2] = new TGeoTranslation("trans_DCDC_connector_on_mezzanine_front_No2", 0 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (spacer_sheet1_mezzanine_box_width / 2. + 0.531), DCDC_connector_on_mezzanine_position_z);
  trans_DCDC_connector_on_mezzanine_front[3] = new TGeoTranslation("trans_DCDC_connector_on_mezzanine_front_No3", -1 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (spacer_sheet1_mezzanine_box_width / 2. + 0.531), DCDC_connector_on_mezzanine_position_z);
  trans_DCDC_connector_on_mezzanine_front[4] = new TGeoTranslation("trans_DCDC_connector_on_mezzanine_front_No4", -2 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (spacer_sheet1_mezzanine_box_width / 2. + 0.531), DCDC_connector_on_mezzanine_position_z);
  TGeoTranslation* trans_DCDC_connector_on_mezzanine_back[5];
  trans_DCDC_connector_on_mezzanine_back[0] = new TGeoTranslation("trans_DCDC_connector_on_mezzanine_back_No0", +2 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (spacer_sheet1_mezzanine_box_width / 2. + 0.531), -DCDC_connector_on_mezzanine_position_z);
  trans_DCDC_connector_on_mezzanine_back[1] = new TGeoTranslation("trans_DCDC_connector_on_mezzanine_back_No1", +1 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (spacer_sheet1_mezzanine_box_width / 2. + 0.531), -DCDC_connector_on_mezzanine_position_z);
  trans_DCDC_connector_on_mezzanine_back[2] = new TGeoTranslation("trans_DCDC_connector_on_mezzanine_back_No2", 0 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (spacer_sheet1_mezzanine_box_width / 2. + 0.531), -DCDC_connector_on_mezzanine_position_z);
  trans_DCDC_connector_on_mezzanine_back[3] = new TGeoTranslation("trans_DCDC_connector_on_mezzanine_back_No3", -1 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (spacer_sheet1_mezzanine_box_width / 2. + 0.531), -DCDC_connector_on_mezzanine_position_z);
  trans_DCDC_connector_on_mezzanine_back[4] = new TGeoTranslation("trans_DCDC_connector_on_mezzanine_back_No4", -2 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (spacer_sheet1_mezzanine_box_width / 2. + 0.531), -DCDC_connector_on_mezzanine_position_z);
  mHalfPSU->AddNode(DCDC_connector_on_mezzanine, 0, trans_DCDC_connector_on_mezzanine_front[0]);
  mHalfPSU->AddNode(DCDC_connector_on_mezzanine, 0, trans_DCDC_connector_on_mezzanine_front[1]);
  mHalfPSU->AddNode(DCDC_connector_on_mezzanine, 0, trans_DCDC_connector_on_mezzanine_front[2]);
  mHalfPSU->AddNode(DCDC_connector_on_mezzanine, 0, trans_DCDC_connector_on_mezzanine_front[3]);
  mHalfPSU->AddNode(DCDC_connector_on_mezzanine, 0, trans_DCDC_connector_on_mezzanine_front[4]);
  mHalfPSU->AddNode(DCDC_connector_on_mezzanine, 0, trans_DCDC_connector_on_mezzanine_back[0]);
  mHalfPSU->AddNode(DCDC_connector_on_mezzanine, 0, trans_DCDC_connector_on_mezzanine_back[1]);
  mHalfPSU->AddNode(DCDC_connector_on_mezzanine, 0, trans_DCDC_connector_on_mezzanine_back[2]);
  mHalfPSU->AddNode(DCDC_connector_on_mezzanine, 0, trans_DCDC_connector_on_mezzanine_back[3]);
  mHalfPSU->AddNode(DCDC_connector_on_mezzanine, 0, trans_DCDC_connector_on_mezzanine_back[4]);

  Double_t DCDC_cover_on_mezzanine_position_z = middle_spacer_main_thickness / 2. + middle_spacer_cover_thickness + electric_board_thickness + mezzanine_prop_main_length - spacer_sheet1_mezzanine_box_thickness - DCDC_sheet1_thickness - DCDC_cover_thickness / 2.;

  TGeoTranslation* trans_DCDC_cover_innner_box_back = new TGeoTranslation("trans_DCDC_cover_innner_box_back", 0, 0, -2 * DCDC_cover_depth);
  TGeoTranslation* trans_DCDC_cover_innner_box_front = new TGeoTranslation("trans_DCDC_cover_innner_box_front", 0, 0, +2 * DCDC_cover_depth);
  trans_DCDC_cover_innner_box_back->RegisterYourself();
  trans_DCDC_cover_innner_box_front->RegisterYourself();

  TGeoRotation* rotate_DCDC_cover_on_mezzanine_shape = new TGeoRotation("rotate_DCDC_cover_on_mezzanine_shape", 0, 180, 180);
  rotate_DCDC_cover_on_mezzanine_shape->RegisterYourself();

  TGeoCompositeShape* DCDC_cover_on_mezzanine_shape_back = new TGeoCompositeShape("DCDC_cover_on_mezzanine_shape_back", "DCDC_cover_outer_box_side - DCDC_cover_inner_box_side:trans_DCDC_cover_innner_box_back");
  TGeoCompositeShape* DCDC_cover_on_mezzanine_shape_front = new TGeoCompositeShape("DCDC_cover_on_mezzanine_shape_front", "DCDC_cover_outer_box_side - DCDC_cover_inner_box_side:trans_DCDC_cover_innner_box_front");

  TGeoVolume* DCDC_cover_on_mezzanine_front = new TGeoVolume("DCDC_cover_on_mezzanine_front", DCDC_cover_on_mezzanine_shape_front, kMedPeek);
  TGeoVolume* DCDC_cover_on_mezzanine_back = new TGeoVolume("DCDC_cover_on_mezzanine_back", DCDC_cover_on_mezzanine_shape_back, kMedPeek);
  DCDC_cover_on_mezzanine_front->SetLineColor(kGray);
  DCDC_cover_on_mezzanine_back->SetLineColor(kGray);

  TGeoTranslation* trans_DCDC_cover_on_mezzanine_front[5];
  trans_DCDC_cover_on_mezzanine_front[0] = new TGeoTranslation("trans_DCDC_cover_on_mezzanine_front_No0", +2 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (DCDC_cover_outer_width / 2. + 1.825), DCDC_cover_on_mezzanine_position_z);
  trans_DCDC_cover_on_mezzanine_front[1] = new TGeoTranslation("trans_DCDC_cover_on_mezzanine_front_No1", +1 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (DCDC_cover_outer_width / 2. + 1.825), DCDC_cover_on_mezzanine_position_z);
  trans_DCDC_cover_on_mezzanine_front[2] = new TGeoTranslation("trans_DCDC_cover_on_mezzanine_front_No2", 0 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (DCDC_cover_outer_width / 2. + 1.825), DCDC_cover_on_mezzanine_position_z);
  trans_DCDC_cover_on_mezzanine_front[3] = new TGeoTranslation("trans_DCDC_cover_on_mezzanine_front_No3", -1 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (DCDC_cover_outer_width / 2. + 1.825), DCDC_cover_on_mezzanine_position_z);
  trans_DCDC_cover_on_mezzanine_front[4] = new TGeoTranslation("trans_DCDC_cover_on_mezzanine_front_No4", -2 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (DCDC_cover_outer_width / 2. + 1.825), DCDC_cover_on_mezzanine_position_z);
  TGeoTranslation* trans_DCDC_cover_on_mezzanine_back[5];
  trans_DCDC_cover_on_mezzanine_back[0] = new TGeoTranslation("trans_DCDC_cover_on_mezzanine_back_No0", +2 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (DCDC_cover_outer_width / 2. + 1.825), -DCDC_cover_on_mezzanine_position_z);
  trans_DCDC_cover_on_mezzanine_back[1] = new TGeoTranslation("trans_DCDC_cover_on_mezzanine_back_No1", +1 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (DCDC_cover_outer_width / 2. + 1.825), -DCDC_cover_on_mezzanine_position_z);
  trans_DCDC_cover_on_mezzanine_back[2] = new TGeoTranslation("trans_DCDC_cover_on_mezzanine_back_No2", 0 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (DCDC_cover_outer_width / 2. + 1.825), -DCDC_cover_on_mezzanine_position_z);
  trans_DCDC_cover_on_mezzanine_back[3] = new TGeoTranslation("trans_DCDC_cover_on_mezzanine_back_No3", -1 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (DCDC_cover_outer_width / 2. + 1.825), -DCDC_cover_on_mezzanine_position_z);
  trans_DCDC_cover_on_mezzanine_back[4] = new TGeoTranslation("trans_DCDC_cover_on_mezzanine_back_No4", -2 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (DCDC_cover_outer_width / 2. + 1.825), -DCDC_cover_on_mezzanine_position_z);

  mHalfPSU->AddNode(DCDC_cover_on_mezzanine_front, 0, trans_DCDC_cover_on_mezzanine_front[0]);
  mHalfPSU->AddNode(DCDC_cover_on_mezzanine_front, 0, trans_DCDC_cover_on_mezzanine_front[1]);
  mHalfPSU->AddNode(DCDC_cover_on_mezzanine_front, 0, trans_DCDC_cover_on_mezzanine_front[2]);
  mHalfPSU->AddNode(DCDC_cover_on_mezzanine_front, 0, trans_DCDC_cover_on_mezzanine_front[3]);
  mHalfPSU->AddNode(DCDC_cover_on_mezzanine_front, 0, trans_DCDC_cover_on_mezzanine_front[4]);
  mHalfPSU->AddNode(DCDC_cover_on_mezzanine_back, 0, trans_DCDC_cover_on_mezzanine_back[0]);
  mHalfPSU->AddNode(DCDC_cover_on_mezzanine_back, 0, trans_DCDC_cover_on_mezzanine_back[1]);
  mHalfPSU->AddNode(DCDC_cover_on_mezzanine_back, 0, trans_DCDC_cover_on_mezzanine_back[2]);
  mHalfPSU->AddNode(DCDC_cover_on_mezzanine_back, 0, trans_DCDC_cover_on_mezzanine_back[3]);
  mHalfPSU->AddNode(DCDC_cover_on_mezzanine_back, 0, trans_DCDC_cover_on_mezzanine_back[4]);

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Connector
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  Double_t main_connector1_thickness = 0.752;
  Double_t main_connector1_width = 2.350;
  Double_t main_connector1_height = 0.160;
  TGeoBBox* connector1_box = new TGeoBBox("connector1_box", main_connector1_width / 2., main_connector1_height / 2., main_connector1_thickness / 2.);

  Double_t main_connector2_thickness = 0.564;
  Double_t main_connector2_width = 2.086;
  Double_t main_connector2_height = 0.499;
  TGeoBBox* connector2_box = new TGeoBBox("connector2_box", main_connector2_width / 2., main_connector2_height / 2., main_connector2_thickness / 2.);

  Double_t main_connector3_thickness = 0.742;
  Double_t main_connector3_width = 2.567;
  Double_t main_connector3_height = 0.579;
  TGeoBBox* connector3_box = new TGeoBBox("connector3_box", main_connector3_width / 2., main_connector3_height / 2., main_connector3_thickness / 2.);

  TGeoVolume* main_connector1 = new TGeoVolume("main_connector1", connector1_box, kMedPeek);
  main_connector1->SetLineColor(kGray + 2);
  TGeoVolume* main_connector2 = new TGeoVolume("main_connector2", connector2_box, kMedPeek);
  main_connector2->SetLineColor(kGray + 2);
  TGeoVolume* main_connector3 = new TGeoVolume("main_connector3", connector3_box, kMedPeek);
  main_connector3->SetLineColor(kGray + 2);

  TGeoTranslation* trans_main_connector1_front[10];
  trans_main_connector1_front[0] = new TGeoTranslation("trans_main_connector1_front_No0", 14.462 + main_connector2_width / 2., -4.276 - main_connector1_height / 2., middle_spacer_main_add_rectangle_side_small_thickness / 2. + main_connector3_thickness / 2);
  trans_main_connector1_front[1] = new TGeoTranslation("trans_main_connector1_front_No1", 14.462 + main_connector2_width / 2. + (0.766 + main_connector2_width) * 1, -4.276 - main_connector1_height / 2., middle_spacer_main_add_rectangle_side_small_thickness / 2. + main_connector3_thickness / 2);
  trans_main_connector1_front[2] = new TGeoTranslation("trans_main_connector1_front_No2", 14.462 + main_connector2_width / 2. + (0.766 + main_connector2_width) * 2, -4.276 - main_connector1_height / 2., middle_spacer_main_add_rectangle_side_small_thickness / 2. + main_connector3_thickness / 2);
  trans_main_connector1_front[3] = new TGeoTranslation("trans_main_connector1_front_No3", -(14.462 + main_connector2_width / 2.), -4.276 - main_connector1_height / 2., middle_spacer_main_add_rectangle_side_small_thickness / 2. + main_connector3_thickness / 2);
  trans_main_connector1_front[4] = new TGeoTranslation("trans_main_connector1_front_No4", -(14.462 + main_connector2_width / 2. + (0.766 + main_connector2_width) * 1), -4.276 - main_connector1_height / 2., middle_spacer_main_add_rectangle_side_small_thickness / 2. + main_connector3_thickness / 2);
  trans_main_connector1_front[5] = new TGeoTranslation("trans_main_connector1_front_No5", -(14.462 + main_connector2_width / 2. + (0.766 + main_connector2_width) * 2), -4.276 - main_connector1_height / 2., middle_spacer_main_add_rectangle_side_small_thickness / 2. + main_connector3_thickness / 2);
  TGeoTranslation* trans_main_connector1_back[10];
  trans_main_connector1_back[0] = new TGeoTranslation("trans_main_connector1_back_No0", 14.462 + main_connector2_width / 2., -4.276 - main_connector1_height / 2., -(middle_spacer_main_add_rectangle_side_small_thickness / 2. + main_connector3_thickness / 2));
  trans_main_connector1_back[1] = new TGeoTranslation("trans_main_connector1_back_No1", 14.462 + main_connector2_width / 2. + (0.766 + main_connector2_width) * 1, -4.276 - main_connector1_height / 2., -(middle_spacer_main_add_rectangle_side_small_thickness / 2. + main_connector3_thickness / 2));
  trans_main_connector1_back[2] = new TGeoTranslation("trans_main_connector1_back_No2", 14.462 + main_connector2_width / 2. + (0.766 + main_connector2_width) * 2, -4.276 - main_connector1_height / 2., -(middle_spacer_main_add_rectangle_side_small_thickness / 2. + main_connector3_thickness / 2));
  trans_main_connector1_back[3] = new TGeoTranslation("trans_main_connector1_back_No3", -(14.462 + main_connector2_width / 2.), -4.276 - main_connector1_height / 2., -(middle_spacer_main_add_rectangle_side_small_thickness / 2. + main_connector3_thickness / 2));
  trans_main_connector1_back[4] = new TGeoTranslation("trans_main_connector1_back_No4", -(14.462 + main_connector2_width / 2. + (0.766 + main_connector2_width) * 1), -4.276 - main_connector1_height / 2., -(middle_spacer_main_add_rectangle_side_small_thickness / 2. + main_connector3_thickness / 2));
  trans_main_connector1_back[5] = new TGeoTranslation("trans_main_connector1_back_No5", -(14.462 + main_connector2_width / 2. + (0.766 + main_connector2_width) * 2), -4.276 - main_connector1_height / 2., -(middle_spacer_main_add_rectangle_side_small_thickness / 2. + main_connector3_thickness / 2));

  TGeoTranslation* trans_main_connector2_front[10];
  trans_main_connector2_front[0] = new TGeoTranslation("trans_main_connector2_front_No0", 14.462 + main_connector2_width / 2., -4.276 - main_connector1_height - main_connector2_height / 2., middle_spacer_main_add_rectangle_side_small_thickness / 2. + main_connector3_thickness / 2);
  trans_main_connector2_front[1] = new TGeoTranslation("trans_main_connector2_front_No1", 14.462 + main_connector2_width / 2. + (0.766 + main_connector2_width) * 1, -4.276 - main_connector1_height - main_connector2_height / 2., middle_spacer_main_add_rectangle_side_small_thickness / 2. + main_connector3_thickness / 2);
  trans_main_connector2_front[2] = new TGeoTranslation("trans_main_connector2_front_No2", 14.462 + main_connector2_width / 2. + (0.766 + main_connector2_width) * 2, -4.276 - main_connector1_height - main_connector2_height / 2., middle_spacer_main_add_rectangle_side_small_thickness / 2. + main_connector3_thickness / 2);
  trans_main_connector2_front[3] = new TGeoTranslation("trans_main_connector2_front_No3", -(14.462 + main_connector2_width / 2.), -4.276 - main_connector1_height - main_connector2_height / 2., middle_spacer_main_add_rectangle_side_small_thickness / 2. + main_connector3_thickness / 2);
  trans_main_connector2_front[4] = new TGeoTranslation("trans_main_connector2_front_No4", -(14.462 + main_connector2_width / 2. + (0.766 + main_connector2_width) * 1), -4.276 - main_connector1_height - main_connector2_height / 2., middle_spacer_main_add_rectangle_side_small_thickness / 2. + main_connector3_thickness / 2);
  trans_main_connector2_front[5] = new TGeoTranslation("trans_main_connector2_front_No5", -(14.462 + main_connector2_width / 2. + (0.766 + main_connector2_width) * 2), -4.276 - main_connector1_height - main_connector2_height / 2., middle_spacer_main_add_rectangle_side_small_thickness / 2. + main_connector3_thickness / 2);
  TGeoTranslation* trans_main_connector2_back[10];
  trans_main_connector2_back[0] = new TGeoTranslation("trans_main_connector2_back_No0", 14.462 + main_connector2_width / 2., -4.276 - main_connector1_height - main_connector2_height / 2., -(middle_spacer_main_add_rectangle_side_small_thickness / 2. + main_connector3_thickness / 2));
  trans_main_connector2_back[1] = new TGeoTranslation("trans_main_connector2_back_No1", 14.462 + main_connector2_width / 2. + (0.766 + main_connector2_width) * 1, -4.276 - main_connector1_height - main_connector2_height / 2., -(middle_spacer_main_add_rectangle_side_small_thickness / 2. + main_connector3_thickness / 2));
  trans_main_connector2_back[2] = new TGeoTranslation("trans_main_connector2_back_No2", 14.462 + main_connector2_width / 2. + (0.766 + main_connector2_width) * 2, -4.276 - main_connector1_height - main_connector2_height / 2., -(middle_spacer_main_add_rectangle_side_small_thickness / 2. + main_connector3_thickness / 2));
  trans_main_connector2_back[3] = new TGeoTranslation("trans_main_connector2_back_No3", -(14.462 + main_connector2_width / 2.), -4.276 - main_connector1_height - main_connector2_height / 2., -(middle_spacer_main_add_rectangle_side_small_thickness / 2. + main_connector3_thickness / 2));
  trans_main_connector2_back[4] = new TGeoTranslation("trans_main_connector2_back_No4", -(14.462 + main_connector2_width / 2. + (0.766 + main_connector2_width) * 1), -4.276 - main_connector1_height - main_connector2_height / 2., -(middle_spacer_main_add_rectangle_side_small_thickness / 2. + main_connector3_thickness / 2));
  trans_main_connector2_back[5] = new TGeoTranslation("trans_main_connector2_back_No5", -(14.462 + main_connector2_width / 2. + (0.766 + main_connector2_width) * 2), -4.276 - main_connector1_height - main_connector2_height / 2., -(middle_spacer_main_add_rectangle_side_small_thickness / 2. + main_connector3_thickness / 2));

  TGeoTranslation* trans_main_connector3_front[10];
  trans_main_connector3_front[0] = new TGeoTranslation("trans_main_connector3_front_No0", 14.462 + main_connector2_width / 2., -4.436 - main_connector2_height - main_connector3_height / 2., electric_board_position_z + electric_board_thickness / 2 + main_connector3_thickness / 2);
  trans_main_connector3_front[1] = new TGeoTranslation("trans_main_connector3_front_No1", 14.462 + main_connector2_width / 2. + (0.766 + main_connector2_width) * 1, -4.436 - main_connector2_height - main_connector3_height / 2., electric_board_position_z + electric_board_thickness / 2 + main_connector3_thickness / 2);
  trans_main_connector3_front[2] = new TGeoTranslation("trans_main_connector3_front_No2", 14.462 + main_connector2_width / 2. + (0.766 + main_connector2_width) * 2, -4.436 - main_connector2_height - main_connector3_height / 2., electric_board_position_z + electric_board_thickness / 2 + main_connector3_thickness / 2);
  trans_main_connector3_front[3] = new TGeoTranslation("trans_main_connector3_front_No3", -(14.462 + main_connector2_width / 2.), -4.436 - main_connector2_height - main_connector3_height / 2., electric_board_position_z + electric_board_thickness / 2 + main_connector3_thickness / 2);
  trans_main_connector3_front[4] = new TGeoTranslation("trans_main_connector3_front_No4", -(14.462 + main_connector2_width / 2. + (0.766 + main_connector2_width) * 1), -4.436 - main_connector2_height - main_connector3_height / 2., electric_board_position_z + electric_board_thickness / 2 + main_connector3_thickness / 2);
  trans_main_connector3_front[5] = new TGeoTranslation("trans_main_connector3_front_No5", -(14.462 + main_connector2_width / 2. + (0.766 + main_connector2_width) * 2), -4.436 - main_connector2_height - main_connector3_height / 2., electric_board_position_z + electric_board_thickness / 2 + main_connector3_thickness / 2);
  TGeoTranslation* trans_main_connector3_back[10];
  trans_main_connector3_back[0] = new TGeoTranslation("trans_main_connector3_back_No0", 14.462 + main_connector2_width / 2., -4.436 - main_connector2_height - main_connector3_height / 2., -(electric_board_position_z + electric_board_thickness / 2 + main_connector3_thickness / 2));
  trans_main_connector3_back[1] = new TGeoTranslation("trans_main_connector3_back_No1", 14.462 + main_connector2_width / 2. + (0.766 + main_connector2_width) * 1, -4.436 - main_connector2_height - main_connector3_height / 2., -(electric_board_position_z + electric_board_thickness / 2 + main_connector3_thickness / 2));
  trans_main_connector3_back[2] = new TGeoTranslation("trans_main_connector3_back_No2", 14.462 + main_connector2_width / 2. + (0.766 + main_connector2_width) * 2, -4.436 - main_connector2_height - main_connector3_height / 2., -(electric_board_position_z + electric_board_thickness / 2 + main_connector3_thickness / 2));
  trans_main_connector3_back[3] = new TGeoTranslation("trans_main_connector3_back_No3", -(14.462 + main_connector2_width / 2.), -4.436 - main_connector2_height - main_connector3_height / 2., -(electric_board_position_z + electric_board_thickness / 2 + main_connector3_thickness / 2));
  trans_main_connector3_back[4] = new TGeoTranslation("trans_main_connector3_back_No4", -(14.462 + main_connector2_width / 2. + (0.766 + main_connector2_width) * 1), -4.436 - main_connector2_height - main_connector3_height / 2., -(electric_board_position_z + electric_board_thickness / 2 + main_connector3_thickness / 2));
  trans_main_connector3_back[5] = new TGeoTranslation("trans_main_connector3_back_No5", -(14.462 + main_connector2_width / 2. + (0.766 + main_connector2_width) * 2), -4.436 - main_connector2_height - main_connector3_height / 2., -(electric_board_position_z + electric_board_thickness / 2 + main_connector3_thickness / 2));

  mHalfPSU->AddNode(main_connector1, 0, trans_main_connector1_front[0]);
  mHalfPSU->AddNode(main_connector1, 0, trans_main_connector1_front[1]);
  mHalfPSU->AddNode(main_connector1, 0, trans_main_connector1_front[2]);
  mHalfPSU->AddNode(main_connector1, 0, trans_main_connector1_front[3]);
  mHalfPSU->AddNode(main_connector1, 0, trans_main_connector1_front[4]);
  mHalfPSU->AddNode(main_connector1, 0, trans_main_connector1_front[5]);

  mHalfPSU->AddNode(main_connector2, 0, trans_main_connector2_front[0]);
  mHalfPSU->AddNode(main_connector2, 0, trans_main_connector2_front[1]);
  mHalfPSU->AddNode(main_connector2, 0, trans_main_connector2_front[2]);
  mHalfPSU->AddNode(main_connector2, 0, trans_main_connector2_front[3]);
  mHalfPSU->AddNode(main_connector2, 0, trans_main_connector2_front[4]);
  mHalfPSU->AddNode(main_connector2, 0, trans_main_connector2_front[5]);

  mHalfPSU->AddNode(main_connector3, 0, trans_main_connector3_front[0]);
  mHalfPSU->AddNode(main_connector3, 0, trans_main_connector3_front[1]);
  mHalfPSU->AddNode(main_connector3, 0, trans_main_connector3_front[2]);
  mHalfPSU->AddNode(main_connector3, 0, trans_main_connector3_front[3]);
  mHalfPSU->AddNode(main_connector3, 0, trans_main_connector3_front[4]);
  mHalfPSU->AddNode(main_connector3, 0, trans_main_connector3_front[5]);

  mHalfPSU->AddNode(main_connector1, 0, trans_main_connector1_back[0]);
  mHalfPSU->AddNode(main_connector1, 0, trans_main_connector1_back[1]);
  mHalfPSU->AddNode(main_connector1, 0, trans_main_connector1_back[2]);
  mHalfPSU->AddNode(main_connector1, 0, trans_main_connector1_back[3]);
  mHalfPSU->AddNode(main_connector1, 0, trans_main_connector1_back[4]);
  mHalfPSU->AddNode(main_connector1, 0, trans_main_connector1_back[5]);
  mHalfPSU->AddNode(main_connector2, 0, trans_main_connector2_back[0]);
  mHalfPSU->AddNode(main_connector2, 0, trans_main_connector2_back[1]);
  mHalfPSU->AddNode(main_connector2, 0, trans_main_connector2_back[2]);
  mHalfPSU->AddNode(main_connector2, 0, trans_main_connector2_back[3]);
  mHalfPSU->AddNode(main_connector2, 0, trans_main_connector2_back[4]);
  mHalfPSU->AddNode(main_connector2, 0, trans_main_connector2_back[5]);
  mHalfPSU->AddNode(main_connector3, 0, trans_main_connector3_back[0]);
  mHalfPSU->AddNode(main_connector3, 0, trans_main_connector3_back[1]);
  mHalfPSU->AddNode(main_connector3, 0, trans_main_connector3_back[2]);
  mHalfPSU->AddNode(main_connector3, 0, trans_main_connector3_back[3]);
  mHalfPSU->AddNode(main_connector3, 0, trans_main_connector3_back[4]);
  mHalfPSU->AddNode(main_connector3, 0, trans_main_connector3_back[5]);

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Rough coil shape part
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  Double_t coil_torus_inner_radius1 = TMath::Min(DCDC_cover_inner_width, DCDC_cover_inner_height) / 10.0;
  Double_t coil_torus_outer_radius1 = TMath::Min(DCDC_cover_inner_width, DCDC_cover_inner_height) / 8.0;
  Double_t coil_radius1 = TMath::Min(DCDC_cover_inner_width, DCDC_cover_inner_height) / 3.0 - coil_torus_outer_radius1;
  Double_t coil_position_z = DCDC_sheet1_position_z + DCDC_sheet1_thickness / 2. + coil_torus_outer_radius1;

  TGeoTorus* coil_torus = new TGeoTorus("coil_torus", coil_radius1, coil_torus_inner_radius1, coil_torus_outer_radius1, 0, 360);

  std::string name_coil = "";

  TGeoCompositeShape* rotated_coil_torus[23];

  for (Int_t iB = 0; iB < 11; ++iB) {

    Double_t block_angle = (180. - block_angle_index[iB]) / 2. * TMath::Pi() / 180.;

    TGeoRotation* rotate_coil_torus_left = new TGeoRotation(Form("rotate_coil_torus_No%d", iB * 2), 180 - block_angle * 180. / TMath::Pi(), 0, 0);
    TGeoRotation* rotate_coil_torus_right = new TGeoRotation(Form("rotate_coil_torus_No%d", iB * 2 + 1), block_angle * 180. / TMath::Pi(), 0, 0);
    rotate_coil_torus_left->RegisterYourself();
    rotate_coil_torus_right->RegisterYourself();

    Double_t cent_block_left[] = {block_radius * TMath::Cos(block_angle), -block_radius * TMath::Sin(block_angle), coil_position_z};
    Double_t cent_block_right[] = {block_radius * TMath::Cos(TMath::Pi() - block_angle), -block_radius * TMath::Sin(TMath::Pi() - block_angle), coil_position_z};

    TGeoCombiTrans* combtrans_coil_torus_left = new TGeoCombiTrans(Form("combtrans_coil_torus_No%d", 2 * iB), cent_block_left[0], cent_block_left[1], cent_block_left[2], rotate_coil_torus_left);
    TGeoCombiTrans* combtrans_coil_torus_right = new TGeoCombiTrans(Form("combtrans_coil_torus_No%d", 2 * iB + 1), cent_block_right[0], cent_block_right[1], cent_block_right[2], rotate_coil_torus_right);
    combtrans_coil_torus_left->RegisterYourself();
    combtrans_coil_torus_right->RegisterYourself();

    if (iB == 0) {
      name_coil += Form("coil_torus:combtrans_coil_torus_No%d", 2 * iB);
    } else {
      name_coil += Form("+coil_torus:combtrans_coil_torus_No%d", 2 * iB);
    }
    name_coil += Form("+coil_torus:combtrans_coil_torus_No%d", 2 * iB + 1);
  }

  TGeoTorus* coil_torus_side = new TGeoTorus("coil_torus_side", coil_radius1, coil_torus_inner_radius1, coil_torus_outer_radius1, 0, 360);

  TGeoTranslation* trans_coil_torus_side_left = new TGeoTranslation("trans_coil_torus_side_left", 22.259 - middle_spacer_cover_block_width / 2, -8.305 + middle_spacer_cover_block_height / 2, coil_position_z);
  trans_coil_torus_side_left->RegisterYourself();
  name_coil += "+coil_torus_side:trans_coil_torus_side_left";

  TGeoTranslation* trans_coil_torus_side_right = new TGeoTranslation("trans_coil_torus_side_right", -22.247 + middle_spacer_cover_block_width / 2, -8.305 + middle_spacer_cover_block_height / 2, coil_position_z);
  trans_coil_torus_side_right->RegisterYourself();
  name_coil += "+coil_torus_side:trans_coil_torus_side_right";

  TGeoCompositeShape* coil_shape = new TGeoCompositeShape("coil_shape", name_coil.c_str());

  TGeoVolume* coil = new TGeoVolume("coil", coil_shape, kMedAlu);
  coil->SetLineColor(kYellow);

  TGeoRotation* trans_coil_front = new TGeoRotation("trans_coil_front", 0, 0, 0);
  TGeoRotation* trans_coil_back = new TGeoRotation("trans_coil_back", 180, 180, 0);

  mHalfPSU->AddNode(coil, 0, trans_coil_front);
  mHalfPSU->AddNode(coil, 0, trans_coil_back);

  //On Mazzenine

  Double_t coil_on_mezzanine_position_z = middle_spacer_main_thickness / 2. + middle_spacer_cover_thickness + electric_board_thickness + mezzanine_prop_main_length - spacer_sheet1_mezzanine_box_thickness - DCDC_sheet1_thickness - coil_torus_outer_radius1;

  TGeoTranslation* trans_coil_on_mezzanine_front[5];
  trans_coil_on_mezzanine_front[0] = new TGeoTranslation("trans_coil_on_mezzanine_front_No0", +2 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (DCDC_cover_outer_width / 2. + 1.825), coil_on_mezzanine_position_z);
  trans_coil_on_mezzanine_front[1] = new TGeoTranslation("trans_coil_on_mezzanine_front_No1", +1 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (DCDC_cover_outer_width / 2. + 1.825), coil_on_mezzanine_position_z);
  trans_coil_on_mezzanine_front[2] = new TGeoTranslation("trans_coil_on_mezzanine_front_No2", 0 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (DCDC_cover_outer_width / 2. + 1.825), coil_on_mezzanine_position_z);
  trans_coil_on_mezzanine_front[3] = new TGeoTranslation("trans_coil_on_mezzanine_front_No3", -1 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (DCDC_cover_outer_width / 2. + 1.825), coil_on_mezzanine_position_z);
  trans_coil_on_mezzanine_front[4] = new TGeoTranslation("trans_coil_on_mezzanine_front_No4", -2 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (DCDC_cover_outer_width / 2. + 1.825), coil_on_mezzanine_position_z);
  TGeoTranslation* trans_coil_on_mezzanine_back[5];
  trans_coil_on_mezzanine_back[0] = new TGeoTranslation("trans_coil_on_mezzanine_back_No0", +2 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (DCDC_cover_outer_width / 2. + 1.825), -coil_on_mezzanine_position_z);
  trans_coil_on_mezzanine_back[1] = new TGeoTranslation("trans_coil_on_mezzanine_back_No1", +1 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (DCDC_cover_outer_width / 2. + 1.825), -coil_on_mezzanine_position_z);
  trans_coil_on_mezzanine_back[2] = new TGeoTranslation("trans_coil_on_mezzanine_back_No2", 0 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (DCDC_cover_outer_width / 2. + 1.825), -coil_on_mezzanine_position_z);
  trans_coil_on_mezzanine_back[3] = new TGeoTranslation("trans_coil_on_mezzanine_back_No3", -1 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (DCDC_cover_outer_width / 2. + 1.825), -coil_on_mezzanine_position_z);
  trans_coil_on_mezzanine_back[4] = new TGeoTranslation("trans_coil_on_mezzanine_back_No4", -2 * (DCDC_sheet1_height + gap_sheet1_on_mezzanine), -17.952 - (DCDC_cover_outer_width / 2. + 1.825), -coil_on_mezzanine_position_z);

  TGeoVolume* coil_on_mezzanine = new TGeoVolume("coil_on_mezzanine", coil_torus, kMedAlu);
  coil_on_mezzanine->SetLineColor(kYellow);

  mHalfPSU->AddNode(coil_on_mezzanine, 0, trans_coil_on_mezzanine_front[0]);
  mHalfPSU->AddNode(coil_on_mezzanine, 0, trans_coil_on_mezzanine_front[1]);
  mHalfPSU->AddNode(coil_on_mezzanine, 0, trans_coil_on_mezzanine_front[2]);
  mHalfPSU->AddNode(coil_on_mezzanine, 0, trans_coil_on_mezzanine_front[3]);
  mHalfPSU->AddNode(coil_on_mezzanine, 0, trans_coil_on_mezzanine_front[4]);
  mHalfPSU->AddNode(coil_on_mezzanine, 0, trans_coil_on_mezzanine_back[0]);
  mHalfPSU->AddNode(coil_on_mezzanine, 0, trans_coil_on_mezzanine_back[1]);
  mHalfPSU->AddNode(coil_on_mezzanine, 0, trans_coil_on_mezzanine_back[2]);
  mHalfPSU->AddNode(coil_on_mezzanine, 0, trans_coil_on_mezzanine_back[3]);
  mHalfPSU->AddNode(coil_on_mezzanine, 0, trans_coil_on_mezzanine_back[4]);

  //small connector borrom

  Double_t main_connector_angle1 = 52 * TMath::Pi() / 180.;
  Double_t main_connector_angle2 = 45 * TMath::Pi() / 180.;

  //front

  //left

  TGeoRotation* rotate_main_connector_box_angle1_left = new TGeoRotation("rotate_main_connector_box_angle1_left", 0, 0, 90 - main_connector_angle1 * 180 / TMath::Pi());
  TGeoRotation* rotate_main_connector_box_angle2_left = new TGeoRotation("rotate_main_connector_box_angle2_left", 0, 0, 90 - main_connector_angle2 * 180 / TMath::Pi());
  TGeoRotation* rotate_main_connector_box_angle1_right = new TGeoRotation("rotate_main_connector_box_angle1_right", 0, 0, -90 + main_connector_angle1 * 180 / TMath::Pi());
  TGeoRotation* rotate_main_connector_box_angle2_right = new TGeoRotation("rotate_main_connector_box_angle2_right", 0, 0, -90 + main_connector_angle2 * 180 / TMath::Pi());
  rotate_main_connector_box_angle1_left->RegisterYourself();
  rotate_main_connector_box_angle2_left->RegisterYourself();
  rotate_main_connector_box_angle1_right->RegisterYourself();
  rotate_main_connector_box_angle2_right->RegisterYourself();

  TGeoTranslation* trans_connector1_box = new TGeoTranslation("trans_connector1_box", 0, -main_connector3_height / 2. - main_connector2_height - main_connector1_height / 2., 0);
  TGeoTranslation* trans_connector2_box = new TGeoTranslation("trans_connector2_box", 0, -main_connector3_height / 2. - main_connector2_height / 2., 0);
  TGeoTranslation* trans_connector3_box = new TGeoTranslation("trans_connector3_box", 0, 0, 0);
  trans_connector1_box->RegisterYourself();
  trans_connector2_box->RegisterYourself();
  trans_connector3_box->RegisterYourself();

  TGeoCompositeShape* comp_connector_box = new TGeoCompositeShape("comp_connector_box", "connector1_box:trans_connector1_box+connector2_box:trans_connector2_box+connector3_box:trans_connector3_box");

  TGeoVolume* connector = new TGeoVolume("connector", comp_connector_box, kMedPeek);
  connector->SetLineColor(kGray + 2);

  TGeoCombiTrans* trans_connector1_front = new TGeoCombiTrans("trans_connector1_front", +(17.064 + 15.397) / 2., -(21.795 + 19.758) / 2, electric_board_position_z + electric_board_thickness / 2 + main_connector3_thickness / 2, rotate_main_connector_box_angle1_left);
  TGeoCombiTrans* trans_connector2_front = new TGeoCombiTrans("trans_connector2_front", +(19.345 + 17.941) / 2, -(19.757 + 17.531) / 2, electric_board_position_z + electric_board_thickness / 2 + main_connector3_thickness / 2, rotate_main_connector_box_angle2_left);
  TGeoCombiTrans* trans_connector3_front = new TGeoCombiTrans("trans_connector1_front", -(17.064 + 15.397) / 2., -(21.795 + 19.758) / 2, electric_board_position_z + electric_board_thickness / 2 + main_connector3_thickness / 2, rotate_main_connector_box_angle1_right);
  TGeoCombiTrans* trans_connector4_front = new TGeoCombiTrans("trans_connector2_front", -(19.345 + 17.941) / 2, -(19.757 + 17.531) / 2, electric_board_position_z + electric_board_thickness / 2 + main_connector3_thickness / 2, rotate_main_connector_box_angle2_right);

  mHalfPSU->AddNode(connector, 0, trans_connector1_front);
  mHalfPSU->AddNode(connector, 0, trans_connector2_front);
  mHalfPSU->AddNode(connector, 0, trans_connector3_front);
  mHalfPSU->AddNode(connector, 0, trans_connector4_front);

  TGeoCombiTrans* trans_connector1_back = new TGeoCombiTrans("trans_connector1_back", +(17.064 + 15.397) / 2., -(21.795 + 19.758) / 2, -(electric_board_position_z + electric_board_thickness / 2 + main_connector3_thickness / 2), rotate_main_connector_box_angle1_left);
  TGeoCombiTrans* trans_connector2_back = new TGeoCombiTrans("trans_connector2_back", +(19.345 + 17.941) / 2, -(19.757 + 17.531) / 2, -(electric_board_position_z + electric_board_thickness / 2 + main_connector3_thickness / 2), rotate_main_connector_box_angle2_left);
  TGeoCombiTrans* trans_connector3_back = new TGeoCombiTrans("trans_connector1_back", -(17.064 + 15.397) / 2., -(21.795 + 19.758) / 2, -(electric_board_position_z + electric_board_thickness / 2 + main_connector3_thickness / 2), rotate_main_connector_box_angle1_right);
  TGeoCombiTrans* trans_connector4_back = new TGeoCombiTrans("trans_connector2_back", -(19.345 + 17.941) / 2, -(19.757 + 17.531) / 2, -(electric_board_position_z + electric_board_thickness / 2 + main_connector3_thickness / 2), rotate_main_connector_box_angle2_right);

  mHalfPSU->AddNode(connector, 0, trans_connector1_back);
  mHalfPSU->AddNode(connector, 0, trans_connector2_back);
  mHalfPSU->AddNode(connector, 0, trans_connector3_back);
  mHalfPSU->AddNode(connector, 0, trans_connector4_back);

  //large connector bottom

  Double_t main_large_connector_bottom_angle1 = 36 * TMath::Pi() / 180.;

  Double_t main_large_connector1_thickness = 0.752;
  Double_t main_large_connector1_width = 3.95;
  Double_t main_large_connector1_height = 0.16;

  Double_t main_large_connector2_thickness = 0.56;
  Double_t main_large_connector2_width = 3.95;
  Double_t main_large_connector2_height = 0.536;

  Double_t main_large_connector3_thickness = 0.626;
  Double_t main_large_connector3_width = 4.167;
  Double_t main_large_connector3_height = 0.579;

  TGeoBBox* large_connector_bottom1_box = new TGeoBBox("large_connector1_box", main_large_connector1_width / 2., main_large_connector1_height / 2., main_large_connector1_thickness / 2.);
  TGeoBBox* large_connector_bottom2_box = new TGeoBBox("large_connector2_box", main_large_connector2_width / 2., main_large_connector2_height / 2., main_large_connector2_thickness / 2.);
  TGeoBBox* large_connector_bottom3_box = new TGeoBBox("large_connector3_box", main_large_connector3_width / 2., main_large_connector3_height / 2., main_large_connector3_thickness / 2.);

  TGeoTranslation* trans_large_connector1_box = new TGeoTranslation("trans_large_connector1_box", 0, -main_large_connector3_height / 2. - main_large_connector2_height - main_large_connector1_height / 2., 0);
  TGeoTranslation* trans_large_connector2_box = new TGeoTranslation("trans_large_connector2_box", 0, -main_large_connector3_height / 2. - main_large_connector2_height / 2., 0);
  TGeoTranslation* trans_large_connector3_box = new TGeoTranslation("trans_large_connector3_box", 0, 0, 0);
  trans_large_connector1_box->RegisterYourself();
  trans_large_connector2_box->RegisterYourself();
  trans_large_connector3_box->RegisterYourself();

  TGeoCompositeShape* comp_large_connector_box = new TGeoCompositeShape("comp_large_connector_box", "large_connector1_box:trans_large_connector1_box+large_connector2_box:trans_large_connector2_box+large_connector3_box:trans_large_connector3_box");

  TGeoRotation* rotate_large_connector_bottom_box_front = new TGeoRotation("rotate_large_connector_bottom_box_front", 0, 0, -(90 - main_large_connector_bottom_angle1 * 180 / TMath::Pi()));
  TGeoRotation* rotate_large_connector_bottom_box_back = new TGeoRotation("rotate_large_connector_bottom_box_back", 0, 0, (90 - main_large_connector_bottom_angle1 * 180 / TMath::Pi()));
  rotate_large_connector_bottom_box_front->RegisterYourself();
  rotate_large_connector_bottom_box_back->RegisterYourself();

  TGeoCombiTrans* combtrans_rotated_large_connector_front = new TGeoCombiTrans("combtrans_rotated_large_connector_front", -(22.268 + 20.287) / 2, (-17.315 - 13.603) / 2, +(electric_board_position_z + electric_board_thickness / 2 + main_connector3_thickness / 2), rotate_large_connector_bottom_box_front);
  TGeoCombiTrans* combtrans_rotated_large_connector_back = new TGeoCombiTrans("combtrans_rotated_large_connector_back", +(22.268 + 20.287) / 2, (-17.315 - 13.603) / 2, -(electric_board_position_z + electric_board_thickness / 2 + main_connector3_thickness / 2), rotate_large_connector_bottom_box_back);

  TGeoVolume* large_connector = new TGeoVolume("large_connector", comp_large_connector_box, kMedPeek);
  large_connector->SetLineColor(kGray + 2);

  mHalfPSU->AddNode(large_connector, 0, combtrans_rotated_large_connector_front);
  mHalfPSU->AddNode(large_connector, 0, combtrans_rotated_large_connector_back);

  return mHalfPSU;
}
