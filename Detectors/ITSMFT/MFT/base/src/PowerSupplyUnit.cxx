// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file PowerSupplyUnit.cxx
/// \brief Class building the MFT heat exchanger
/// \author P. Demongodin, Raphael Tieulent <raphael.tieulent@cern.ch>

#include "TMath.h"
#include "TGeoManager.h"
#include "TGeoCompositeShape.h"
#include "TGeoTube.h"
#include "TGeoTorus.h"
#include "TGeoCone.h"
#include "TGeoBoolNode.h"
#include "TGeoBBox.h"
#include "TGeoVolume.h"
#include "FairLogger.h"
#include "MFTBase/Constants.h"
#include "MFTBase/PowerSupplyUnit.h"
#include "MFTBase/Geometry.h"

using namespace o2::mft;

ClassImp(o2::mft::PowerSupplyUnit);

//_____________________________________________________________________________
PowerSupplyUnit::PowerSupplyUnit() : TNamed()
{
  create();
}
/*
//_____________________________________________________________________________
PowerSupplyUnit::PowerSupplyUnit(Double_t rWater, Double_t dRPipe, Double_t heatExchangerThickness,
				 Double_t carbonThickness)
  : TNamed(),mPSU(nullptr)
{
  create(1,1);
}
*/
//_____________________________________________________________________________
TGeoVolumeAssembly* PowerSupplyUnit::create()
{

  //fm auto* mPowerSupplyUnit = new TGeoVolumeAssembly();

  new TGeoBBox("dummy", 0, 0, 0);

  TGeoMedium* kMedPeek = gGeoManager->GetMedium("MFT_PEEK$");
  TGeoMedium* kMed_Water = gGeoManager->GetMedium("MFT_Water$");
  TGeoMedium* kMedAlu = gGeoManager->GetMedium("MFT_Alu$");
  TGeoMedium* kMedCu = gGeoManager->GetMedium("MFT_Cu$");

  TGeoVolumeAssembly* mHalfPSU = new TGeoVolumeAssembly("PSU");

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Middle peek part
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  Double_t middle_board_thickness = 0.8;
  Double_t middle_board_max_radius1 = 26.432;
  Double_t middle_board_min_radius1 = 17.0;
  Double_t middleBox_sub_angle2 = 112.26 * TMath::Pi() / 180.;
  Double_t middleBox_sub_width1 = 20.8;

  Double_t middleBox_sub_height1 = middle_board_max_radius1 * (1 - sqrt(1 - pow(middleBox_sub_width1 / 2. / middle_board_max_radius1, 2)));

  Double_t middleBox_sub_width2 = 2 * middle_board_max_radius1;
  Double_t middleBox_sub_height2 = 2 * middle_board_min_radius1 * TMath::Cos(middleBox_sub_angle2 / 2);

  new TGeoTubeSeg("middleTubeSeg1", middle_board_min_radius1, middle_board_max_radius1, middle_board_thickness / 2, 180, 0);
  new TGeoBBox("middleBox_sub1", middleBox_sub_width1 / 2, middleBox_sub_height1 / 2, middle_board_thickness);
  new TGeoBBox("middleBox_sub2", middleBox_sub_width2 / 2, middleBox_sub_height2 / 2, middle_board_thickness);

  auto* tmiddleBox_sub1 = new TGeoTranslation("tmiddleBox_sub1", 0, -(middle_board_max_radius1 - middleBox_sub_height1 / 2), 0);
  auto* tmiddleBox_sub2 = new TGeoTranslation("tmiddleBox_sub2", 0., 0., 0.);
  tmiddleBox_sub1->RegisterYourself();
  tmiddleBox_sub2->RegisterYourself();

  Double_t middleBox_edge_width1 = 11.4;
  Double_t middleBox_edge_height1 = 4.996;

  Double_t middleBox_edge_sub_width1 = 1.8 + 0.5 * 2;
  Double_t middleBox_edge_sub_height1 = 6.025 + 0.5 * 2;

  Double_t middleBox_edge_sub_width2 = 1.1;
  Double_t middleBox_edge_sub_height2 = 1.0;

  Double_t middleBox_edge_sub_height3 = 1.3;

  new TGeoBBox("middleBox_edge1", middleBox_edge_width1 / 2, middleBox_edge_height1 / 2, middle_board_thickness / 2);
  new TGeoBBox("middleBox_edge_sub1", middleBox_edge_sub_width1 / 2, middleBox_edge_sub_height1 / 2, middle_board_thickness);
  new TGeoBBox("middleBox_edge_sub2", middleBox_edge_sub_width2 / 2, middleBox_edge_sub_height2 / 2, middle_board_thickness);

  auto* tmiddleBox_edge_right1 = new TGeoTranslation("tmiddleBox_edge_right1",
                                                     middle_board_min_radius1 * TMath::Sin(middleBox_sub_angle2 / 2) + middleBox_edge_width1 / 2,
                                                     -(middle_board_min_radius1 * TMath::Cos(middleBox_sub_angle2 / 2) - middleBox_edge_height1 / 2), 0);
  auto* tmiddleBox_edge_sub_right1 = new TGeoTranslation("tmiddleBox_edge_sub_right1",
                                                         middle_board_min_radius1 * TMath::Sin(middleBox_sub_angle2 / 2) + middleBox_edge_width1 - middleBox_edge_sub_width1 / 2,
                                                         -middle_board_min_radius1 * TMath::Cos(middleBox_sub_angle2 / 2) + middleBox_edge_height1 - middleBox_edge_sub_height2 - middleBox_edge_sub_height3 - middleBox_edge_sub_height1 / 2,
                                                         0);
  auto* tmiddleBox_edge_sub_right2 = new TGeoTranslation("tmiddleBox_edge_sub_right2",
                                                         middle_board_min_radius1 * TMath::Sin(middleBox_sub_angle2 / 2) + middleBox_edge_width1 - middleBox_edge_sub_width2 / 2,
                                                         -(middle_board_min_radius1 * TMath::Cos(middleBox_sub_angle2 / 2) - middleBox_edge_height1 / 2) + middleBox_edge_height1 / 2 - middleBox_edge_sub_height2 / 2,
                                                         0);

  auto* tmiddleBox_edge_left1 = new TGeoTranslation("tmiddleBox_edge_left1",
                                                    -(middle_board_min_radius1 * TMath::Sin(middleBox_sub_angle2 / 2) + middleBox_edge_width1 / 2),
                                                    -(middle_board_min_radius1 * TMath::Cos(middleBox_sub_angle2 / 2) - middleBox_edge_height1 / 2), 0);
  auto* tmiddleBox_edge_sub_left1 = new TGeoTranslation("tmiddleBox_edge_sub_left1",
                                                        -(middle_board_min_radius1 * TMath::Sin(middleBox_sub_angle2 / 2) + middleBox_edge_width1 - middleBox_edge_sub_width1 / 2),
                                                        -middle_board_min_radius1 * TMath::Cos(middleBox_sub_angle2 / 2) + middleBox_edge_height1 - middleBox_edge_sub_height2 - middleBox_edge_sub_height3 - middleBox_edge_sub_height1 / 2,
                                                        0);
  auto* tmiddleBox_edge_sub_left2 = new TGeoTranslation("tmiddleBox_edge_sub_left2",
                                                        -(middle_board_min_radius1 * TMath::Sin(middleBox_sub_angle2 / 2) + middleBox_edge_width1 - middleBox_edge_sub_width2 / 2),
                                                        -(middle_board_min_radius1 * TMath::Cos(middleBox_sub_angle2 / 2) - middleBox_edge_height1 / 2) + middleBox_edge_height1 / 2 - middleBox_edge_sub_height2 / 2,
                                                        0);

  tmiddleBox_edge_right1->RegisterYourself();
  tmiddleBox_edge_sub_right1->RegisterYourself();
  tmiddleBox_edge_sub_right2->RegisterYourself();

  tmiddleBox_edge_left1->RegisterYourself();
  tmiddleBox_edge_sub_left1->RegisterYourself();
  tmiddleBox_edge_sub_left2->RegisterYourself();

  Double_t middleBox_side_lack_radius1 = 25.0;
  Double_t middleBox_side_lack_height1 = 0.937 + 0.5;
  Double_t middleBox_side_lack_angle1 = 24.75 + (2 * 0.5 * 360) / (2 * TMath::Pi() * middleBox_side_lack_radius1);

  Double_t middleBox_side_lack_position_angle_min_right1 = 270 - middleBox_side_lack_angle1 - 9.83 - TMath::ASin(middleBox_sub_width1 / 2 / middle_board_max_radius1) * 180 / TMath::Pi();
  Double_t middleBox_side_lack_position_angle_max_right1 = 270 - 9.83 - TMath::ASin(middleBox_sub_width1 / 2 / middle_board_max_radius1) * 180 / TMath::Pi();

  new TGeoTubeSeg("middleBox_side_lack_right1", middleBox_side_lack_radius1, middle_board_max_radius1, middle_board_thickness, middleBox_side_lack_position_angle_min_right1, middleBox_side_lack_position_angle_max_right1);

  Double_t middleBox_side_lack_position_angle_min_left1 = 360 - middleBox_side_lack_angle1 - 9.83 - TMath::ASin(middleBox_sub_width1 / 2 / middle_board_max_radius1) * 180 / TMath::Pi();
  Double_t middleBox_side_lack_position_angle_max_left1 = 360 - 9.83 - TMath::ASin(middleBox_sub_width1 / 2 / middle_board_max_radius1) * 180 / TMath::Pi();

  new TGeoTubeSeg("middleBox_side_lack_left1", middleBox_side_lack_radius1, middle_board_max_radius1, middle_board_thickness, middleBox_side_lack_position_angle_min_left1, middleBox_side_lack_position_angle_max_left1);

  new TGeoCompositeShape("middle_board_shape",
                         "middleTubeSeg1"
                         "-middleBox_sub1:tmiddleBox_sub1"
                         "-middleBox_sub2:tmiddleBox_sub2"
                         "+middleBox_edge1:tmiddleBox_edge_right1 - middleBox_edge_sub1:tmiddleBox_edge_sub_right1 - middleBox_edge_sub2:tmiddleBox_edge_sub_right2"
                         "+middleBox_edge1:tmiddleBox_edge_left1 - middleBox_edge_sub1:tmiddleBox_edge_sub_left1 - middleBox_edge_sub2:tmiddleBox_edge_sub_left2"
                         "-middleBox_side_lack_right1 - middleBox_side_lack_left1");

  Double_t pipe_sub_radius1 = 0.2;
  Double_t center_pipe_sub_torus_radius1 = 18.7 - pipe_sub_radius1;
  Double_t center_pipe_sub_torus_angle1 = 114.84;

  new TGeoTorus("middleCenterWaterPipeTorus_sub1", center_pipe_sub_torus_radius1, pipe_sub_radius1, pipe_sub_radius1, 0, center_pipe_sub_torus_angle1);
  auto* rmiddleCenterWaterPipeTorus_sub1 = new TGeoRotation("rmiddleCenterWaterPipeTorus_sub1", -(180 - 0.5 * (180 - center_pipe_sub_torus_angle1)), 0, 0);
  rmiddleCenterWaterPipeTorus_sub1->RegisterYourself();

  Double_t side_pipe_sub_torus_radius1 = 1.2 - pipe_sub_radius1;
  Double_t side_pipe_sub_torus_angle1 = 57.42;

  new TGeoTorus("middleSideWaterPipeTorus_sub1", side_pipe_sub_torus_radius1, pipe_sub_radius1, pipe_sub_radius1, 0, side_pipe_sub_torus_angle1);

  auto* rmiddleSideWaterPipeTorus_sub_right1 = new TGeoRotation("rmiddleSideWaterPipeTorus_sub_right1", 90, 0, 0);
  auto* rmiddleSideWaterPipeTorus_sub_left1 = new TGeoRotation("rmiddleSideWaterPipeTorus_sub_left1", 90 - side_pipe_sub_torus_angle1, 0, 0);
  rmiddleSideWaterPipeTorus_sub_right1->RegisterYourself();
  rmiddleSideWaterPipeTorus_sub_left1->RegisterYourself();

  new TGeoCompositeShape("rotated_middleSideWaterPipeTorus_sub_right1",
                         "dummy + middleSideWaterPipeTorus_sub1:rmiddleSideWaterPipeTorus_sub_right1");
  new TGeoCompositeShape("rotated_middleSideWaterPipeTorus_sub_left1",
                         "dummy + middleSideWaterPipeTorus_sub1:rmiddleSideWaterPipeTorus_sub_left1");

  auto* tmiddleSideWaterPipeTorus_sub_right1 = new TGeoTranslation("tmiddleSideWaterPipeTorus_sub_right1",
                                                                   center_pipe_sub_torus_radius1 * TMath::Sin(center_pipe_sub_torus_angle1 / 2 * TMath::Pi() / 180.) + side_pipe_sub_torus_radius1 * TMath::Sin(side_pipe_sub_torus_angle1 * TMath::Pi() / 180.),
                                                                   -center_pipe_sub_torus_radius1 * TMath::Cos(center_pipe_sub_torus_angle1 / 2 * TMath::Pi() / 180.) - side_pipe_sub_torus_radius1 * TMath::Cos(side_pipe_sub_torus_angle1 * TMath::Pi() / 180.), 0);
  tmiddleSideWaterPipeTorus_sub_right1->RegisterYourself();

  auto* tmiddleSideWaterPipeTorus_sub_left1 = new TGeoTranslation("tmiddleSideWaterPipeTorus_sub_left1",
                                                                  -(center_pipe_sub_torus_radius1 * TMath::Sin(center_pipe_sub_torus_angle1 / 2 * TMath::Pi() / 180.) + side_pipe_sub_torus_radius1 * TMath::Sin(side_pipe_sub_torus_angle1 * TMath::Pi() / 180.)),
                                                                  -center_pipe_sub_torus_radius1 * TMath::Cos(center_pipe_sub_torus_angle1 / 2 * TMath::Pi() / 180.) - side_pipe_sub_torus_radius1 * TMath::Cos(side_pipe_sub_torus_angle1 * TMath::Pi() / 180.), 0);
  tmiddleSideWaterPipeTorus_sub_left1->RegisterYourself();

  Double_t side_pipe_sub_tube_length1 = 6.568;

  new TGeoTube("middleSideWaterPipeTube_sub1", 0, pipe_sub_radius1, side_pipe_sub_tube_length1 / 2);

  auto* rmiddleSideWaterPipeTube_sub1 = new TGeoRotation("rmiddleSideWaterPipeTube_sub1", 90, 90, 0);
  rmiddleSideWaterPipeTube_sub1->RegisterYourself();

  new TGeoCompositeShape("rotated_middleSideWaterPipeTube_sub1", "dummy+middleSideWaterPipeTube_sub1:rmiddleSideWaterPipeTube_sub1");

  TGeoTranslation* tmiddleSideWaterPipeTube_sub_right1 = new TGeoTranslation("tmiddleSideWaterPipeTube_sub_right1",
                                                                             center_pipe_sub_torus_radius1 * TMath::Sin(center_pipe_sub_torus_angle1 / 2 * TMath::Pi() / 180) + side_pipe_sub_torus_radius1 * TMath::Sin(side_pipe_sub_torus_angle1 * TMath::Pi() / 180) + side_pipe_sub_tube_length1 / 2,
                                                                             -center_pipe_sub_torus_radius1 * TMath::Cos(center_pipe_sub_torus_angle1 / 2 * TMath::Pi() / 180) + side_pipe_sub_torus_radius1 * (1 - TMath::Sin((90 - side_pipe_sub_torus_angle1) * TMath::Pi() / 180)),
                                                                             0);
  TGeoTranslation* tmiddleSideWaterPipeTube_sub_left1 = new TGeoTranslation("tmiddleSideWaterPipeTube_sub_left1",
                                                                            -(center_pipe_sub_torus_radius1 * TMath::Sin(center_pipe_sub_torus_angle1 / 2 * TMath::Pi() / 180) + side_pipe_sub_torus_radius1 * TMath::Sin(side_pipe_sub_torus_angle1 * TMath::Pi() / 180) + side_pipe_sub_tube_length1 / 2),
                                                                            -center_pipe_sub_torus_radius1 * TMath::Cos(center_pipe_sub_torus_angle1 / 2 * TMath::Pi() / 180) + side_pipe_sub_torus_radius1 * (1 - TMath::Sin((90 - side_pipe_sub_torus_angle1) * TMath::Pi() / 180)),
                                                                            0);
  tmiddleSideWaterPipeTube_sub_right1->RegisterYourself();
  tmiddleSideWaterPipeTube_sub_left1->RegisterYourself();

  new TGeoCompositeShape("middleWaterPipeTorus_sub_shape",
                         "dummy + middleCenterWaterPipeTorus_sub1:rmiddleCenterWaterPipeTorus_sub1"
                         "+rotated_middleSideWaterPipeTorus_sub_right1:tmiddleSideWaterPipeTorus_sub_right1"
                         "+rotated_middleSideWaterPipeTube_sub1:tmiddleSideWaterPipeTube_sub_right1"
                         "+rotated_middleSideWaterPipeTorus_sub_left1:tmiddleSideWaterPipeTorus_sub_left1"
                         "+rotated_middleSideWaterPipeTube_sub1:tmiddleSideWaterPipeTube_sub_left1");

  TGeoTranslation* tmiddleWaterPipeTorus_sub_shape1 = new TGeoTranslation("tmiddleWaterPipeTorus_sub_shape1", 0, 0, middle_board_thickness / 2);
  TGeoTranslation* tmiddleWaterPipeTorus_sub_shape2 = new TGeoTranslation("tmiddleWaterPipeTorus_sub_shape2", 0, 0, -middle_board_thickness / 2);
  tmiddleWaterPipeTorus_sub_shape1->RegisterYourself();
  tmiddleWaterPipeTorus_sub_shape2->RegisterYourself();

  TGeoCompositeShape* middle_board_pipeSubtracted_shape = new TGeoCompositeShape("middle_board_pipeSubtracted_shape",
                                                                                 "middle_board_shape-middleWaterPipeTorus_sub_shape:tmiddleWaterPipeTorus_sub_shape1"
                                                                                 "-middleWaterPipeTorus_sub_shape:tmiddleWaterPipeTorus_sub_shape2");

  TGeoVolume* middle_peek_board = new TGeoVolume("middle_peek_board", middle_board_pipeSubtracted_shape, kMedPeek);

  mHalfPSU->AddNode(middle_peek_board, 0, nullptr);

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////
  //water pipe part
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////

  Double_t pipe_min_radius1 = 0.075;
  Double_t pipe_max_radius1 = 0.100;
  Double_t center_pipe_torus_radius1 = 18.5;
  Double_t center_pipe_torus_angle1 = 114.84;

  new TGeoTorus("middleCenterWaterPipeTorus1", center_pipe_torus_radius1, pipe_min_radius1, pipe_max_radius1, 0, center_pipe_torus_angle1);

  auto* rmiddleCenterWaterPipeTorus1 = new TGeoRotation("rmiddleCenterWaterPipeTorus1", -(180 - 0.5 * (180 - center_pipe_torus_angle1)), 0, 0);
  rmiddleCenterWaterPipeTorus1->RegisterYourself();

  Double_t side_pipe_torus_radius1 = 1.0;
  Double_t side_pipe_torus_angle1 = 57.42;

  new TGeoTorus("middleSideWaterPipeTorus1", side_pipe_torus_radius1, pipe_min_radius1, pipe_max_radius1, 0, side_pipe_torus_angle1);

  auto* rmiddleSideWaterPipeTorus_right1 = new TGeoRotation("rmiddleSideWaterPipeTorus_right1", 90, 0, 0);
  auto* rmiddleSideWaterPipeTorus_left1 = new TGeoRotation("rmiddleSideWaterPipeTorus_left1", 90 - side_pipe_torus_angle1, 0, 0);
  rmiddleSideWaterPipeTorus_right1->RegisterYourself();
  rmiddleSideWaterPipeTorus_left1->RegisterYourself();

  new TGeoCompositeShape("rotated_middleSideWaterPipeTorus_right1",
                         "dummy + middleSideWaterPipeTorus1:rmiddleSideWaterPipeTorus_right1");
  new TGeoCompositeShape("rotated_middleSideWaterPipeTorus_left1",
                         "dummy + middleSideWaterPipeTorus1:rmiddleSideWaterPipeTorus_left1");

  auto* tmiddleSideWaterPipeTorus_right1 = new TGeoTranslation("tmiddleSideWaterPipeTorus_right1",
                                                               center_pipe_torus_radius1 * TMath::Sin(center_pipe_torus_angle1 / 2 * TMath::Pi() / 180.) + side_pipe_torus_radius1 * TMath::Sin(side_pipe_torus_angle1 * TMath::Pi() / 180.),
                                                               -center_pipe_torus_radius1 * TMath::Cos(center_pipe_torus_angle1 / 2 * TMath::Pi() / 180.) - side_pipe_torus_radius1 * TMath::Cos(side_pipe_torus_angle1 * TMath::Pi() / 180.), 0);
  auto* tmiddleSideWaterPipeTorus_left1 = new TGeoTranslation("tmiddleSideWaterPipeTorus_left1",
                                                              -(center_pipe_torus_radius1 * TMath::Sin(center_pipe_torus_angle1 / 2 * TMath::Pi() / 180.) + side_pipe_torus_radius1 * TMath::Sin(side_pipe_torus_angle1 * TMath::Pi() / 180.)),
                                                              -center_pipe_torus_radius1 * TMath::Cos(center_pipe_torus_angle1 / 2 * TMath::Pi() / 180.) - side_pipe_torus_radius1 * TMath::Cos(side_pipe_torus_angle1 * TMath::Pi() / 180.), 0);
  tmiddleSideWaterPipeTorus_right1->RegisterYourself();
  tmiddleSideWaterPipeTorus_left1->RegisterYourself();

  Double_t side_pipe_tube_length1 = 6.868;

  new TGeoTube("middleSideWaterPipeTube1", pipe_min_radius1, pipe_max_radius1, side_pipe_tube_length1 / 2);
  auto* rmiddleSideWaterPipeTube1 = new TGeoRotation("rmiddleSideWaterPipeTube1", 90, 90, 0);
  rmiddleSideWaterPipeTube1->RegisterYourself();

  new TGeoCompositeShape("rotated_middleSideWaterPipeTube1", "dummy+middleSideWaterPipeTube1:rmiddleSideWaterPipeTube1");

  TGeoTranslation* tmiddleSideWaterPipeTube_right1 = new TGeoTranslation("tmiddleSideWaterPipeTube_right1",
                                                                         center_pipe_torus_radius1 * TMath::Sin(center_pipe_torus_angle1 / 2 * TMath::Pi() / 180) + side_pipe_torus_radius1 * TMath::Sin(side_pipe_torus_angle1 * TMath::Pi() / 180) + side_pipe_tube_length1 / 2,
                                                                         -center_pipe_torus_radius1 * TMath::Cos(center_pipe_torus_angle1 / 2 * TMath::Pi() / 180) + side_pipe_torus_radius1 * (1 - TMath::Sin((90 - side_pipe_torus_angle1) * TMath::Pi() / 180)),
                                                                         0);
  TGeoTranslation* tmiddleSideWaterPipeTube_left1 = new TGeoTranslation("tmiddleSideWaterPipeTube_left1",
                                                                        -(center_pipe_torus_radius1 * TMath::Sin(center_pipe_torus_angle1 / 2 * TMath::Pi() / 180) + side_pipe_torus_radius1 * TMath::Sin(side_pipe_torus_angle1 * TMath::Pi() / 180) + side_pipe_tube_length1 / 2),
                                                                        -center_pipe_torus_radius1 * TMath::Cos(center_pipe_torus_angle1 / 2 * TMath::Pi() / 180) + side_pipe_torus_radius1 * (1 - TMath::Sin((90 - side_pipe_torus_angle1) * TMath::Pi() / 180)),
                                                                        0);
  tmiddleSideWaterPipeTube_right1->RegisterYourself();
  tmiddleSideWaterPipeTube_left1->RegisterYourself();

  new TGeoCompositeShape("water_pipe_shape",
                         "middleCenterWaterPipeTorus1:rmiddleCenterWaterPipeTorus1"
                         " + rotated_middleSideWaterPipeTorus_right1:tmiddleSideWaterPipeTorus_right1 + rotated_middleSideWaterPipeTorus_left1:tmiddleSideWaterPipeTorus_left1"
                         " + rotated_middleSideWaterPipeTube1:tmiddleSideWaterPipeTube_right1 + rotated_middleSideWaterPipeTube1:tmiddleSideWaterPipeTube_left1");

  TGeoTranslation* tmiddleWaterPipeTorus_shape1 = new TGeoTranslation("tmiddleWaterPipeTorus_shape1", 0, 0, middle_board_thickness / 2 - pipe_max_radius1);
  TGeoTranslation* tmiddleWaterPipeTorus_shape2 = new TGeoTranslation("tmiddleWaterPipeTorus_shape2", 0, 0, -middle_board_thickness / 2 + pipe_max_radius1);
  tmiddleWaterPipeTorus_shape1->RegisterYourself();
  tmiddleWaterPipeTorus_shape2->RegisterYourself();

  TGeoCompositeShape* water_pipe_2side_shape = new TGeoCompositeShape("water_pipe_2side_shape", "water_pipe_shape:tmiddleWaterPipeTorus_shape1+water_pipe_shape:tmiddleWaterPipeTorus_shape2");

  TGeoVolume* water_pipe = new TGeoVolume("water_pipe", water_pipe_2side_shape, kMedAlu);
  water_pipe->SetLineColor(kGray);

  mHalfPSU->AddNode(water_pipe, 1, nullptr);

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////
  //water in the pipe part
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////

  new TGeoTorus("middleCenterWaterTorus1", center_pipe_torus_radius1, 0, pipe_min_radius1, 0, center_pipe_torus_angle1);
  auto* rmiddleCenterWaterTorus1 = new TGeoRotation("rmiddleCenterWaterTorus1", -(180 - 0.5 * (180 - center_pipe_torus_angle1)), 0, 0);
  rmiddleCenterWaterTorus1->RegisterYourself();

  new TGeoTorus("middleSideWaterTorus1", side_pipe_torus_radius1, 0, pipe_min_radius1, 0, side_pipe_torus_angle1);

  auto* rmiddleSideWaterTorus_right1 = new TGeoRotation("rmiddleSideWaterTorus_right1", 90, 0, 0);
  auto* rmiddleSideWaterTorus_left1 = new TGeoRotation("rmiddleSideWaterTorus_left1", 90 - side_pipe_torus_angle1, 0, 0);
  rmiddleSideWaterTorus_right1->RegisterYourself();
  rmiddleSideWaterTorus_left1->RegisterYourself();

  new TGeoCompositeShape("rotated_middleSideWaterTorus_right1", "dummy + middleSideWaterTorus1:rmiddleSideWaterTorus_right1");
  new TGeoCompositeShape("rotated_middleSideWaterTorus_left1", "dummy + middleSideWaterTorus1:rmiddleSideWaterTorus_left1");

  auto* tmiddleSideWaterTorus_right1 = new TGeoTranslation("tmiddleSideWaterTorus_right1",
                                                           center_pipe_torus_radius1 * TMath::Sin(center_pipe_torus_angle1 / 2 * TMath::Pi() / 180.) + side_pipe_torus_radius1 * TMath::Sin(side_pipe_torus_angle1 * TMath::Pi() / 180.),
                                                           -center_pipe_torus_radius1 * TMath::Cos(center_pipe_torus_angle1 / 2 * TMath::Pi() / 180.) - side_pipe_torus_radius1 * TMath::Cos(side_pipe_torus_angle1 * TMath::Pi() / 180.), 0);
  auto* tmiddleSideWaterTorus_left1 = new TGeoTranslation("tmiddleSideWaterTorus_left1",
                                                          -(center_pipe_torus_radius1 * TMath::Sin(center_pipe_torus_angle1 / 2 * TMath::Pi() / 180.) + side_pipe_torus_radius1 * TMath::Sin(side_pipe_torus_angle1 * TMath::Pi() / 180.)),
                                                          -center_pipe_torus_radius1 * TMath::Cos(center_pipe_torus_angle1 / 2 * TMath::Pi() / 180.) - side_pipe_torus_radius1 * TMath::Cos(side_pipe_torus_angle1 * TMath::Pi() / 180.), 0);
  tmiddleSideWaterTorus_right1->RegisterYourself();
  tmiddleSideWaterTorus_left1->RegisterYourself();

  new TGeoTube("middleSideWaterTube1", 0, pipe_min_radius1, side_pipe_tube_length1 / 2);

  auto* rmiddleSideWaterTube1 = new TGeoRotation("rmiddleSideWaterTube1", 90, 90, 0);
  rmiddleSideWaterTube1->RegisterYourself();

  new TGeoCompositeShape("rotated_middleSideWaterTube1", "dummy+middleSideWaterTube1:rmiddleSideWaterTube1");

  TGeoTranslation* tmiddleSideWaterTube_right1 = new TGeoTranslation("tmiddleSideWaterTube_right1",
                                                                     center_pipe_torus_radius1 * TMath::Sin(center_pipe_torus_angle1 / 2 * TMath::Pi() / 180) + side_pipe_torus_radius1 * TMath::Sin(side_pipe_torus_angle1 * TMath::Pi() / 180) + side_pipe_tube_length1 / 2,
                                                                     -center_pipe_torus_radius1 * TMath::Cos(center_pipe_torus_angle1 / 2 * TMath::Pi() / 180) + side_pipe_torus_radius1 * (1 - TMath::Sin((90 - side_pipe_torus_angle1) * TMath::Pi() / 180)),
                                                                     0);
  TGeoTranslation* tmiddleSideWaterTube_left1 = new TGeoTranslation("tmiddleSideWaterTube_left1",
                                                                    -(center_pipe_torus_radius1 * TMath::Sin(center_pipe_torus_angle1 / 2 * TMath::Pi() / 180) + side_pipe_torus_radius1 * TMath::Sin(side_pipe_torus_angle1 * TMath::Pi() / 180) + side_pipe_tube_length1 / 2),
                                                                    -center_pipe_torus_radius1 * TMath::Cos(center_pipe_torus_angle1 / 2 * TMath::Pi() / 180) + side_pipe_torus_radius1 * (1 - TMath::Sin((90 - side_pipe_torus_angle1) * TMath::Pi() / 180)),
                                                                    0);
  tmiddleSideWaterTube_right1->RegisterYourself();
  tmiddleSideWaterTube_left1->RegisterYourself();

  new TGeoCompositeShape("water_shape",
                         "middleCenterWaterTorus1:rmiddleCenterWaterTorus1"
                         " + rotated_middleSideWaterTorus_right1:tmiddleSideWaterTorus_right1 + rotated_middleSideWaterTorus_left1:tmiddleSideWaterTorus_left1"
                         " + rotated_middleSideWaterTube1:tmiddleSideWaterTube_right1 + rotated_middleSideWaterTube1:tmiddleSideWaterTube_left1");

  TGeoTranslation* tmiddleWaterTorus_shape1 = new TGeoTranslation("tmiddleWaterTorus_shape1", 0, 0, middle_board_thickness / 2 - pipe_max_radius1);
  TGeoTranslation* tmiddleWaterTorus_shape2 = new TGeoTranslation("tmiddleWaterTorus_shape2", 0, 0, -middle_board_thickness / 2 + pipe_max_radius1);
  tmiddleWaterTorus_shape1->RegisterYourself();
  tmiddleWaterTorus_shape2->RegisterYourself();

  TGeoCompositeShape* water_2side_shape = new TGeoCompositeShape("water_2side_shape", "water_shape:tmiddleWaterTorus_shape1+water_shape:tmiddleWaterTorus_shape2");

  TGeoVolume* water = new TGeoVolume("water", water_2side_shape, kMed_Water);
  water->SetLineColor(kBlue);
  mHalfPSU->AddNode(water, 1, nullptr);

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////
  //PSU surface
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////

  Double_t surface_board_thickness = 0.16;
  Double_t surface_board_max_radius1 = 26.432;
  Double_t surface_board_min_radius1 = 17.0;
  Double_t surfaceBox_sub_angle2 = 113.29 * TMath::Pi() / 180.;
  Double_t surfaceBox_sub_width1 = 20.8;

  Double_t surfaceBox_sub_height1 = surface_board_max_radius1 * (1 - sqrt(1 - pow(surfaceBox_sub_width1 / 2. / surface_board_max_radius1, 2)));

  Double_t surfaceBox_sub_width2 = 2 * surface_board_max_radius1;
  Double_t surfaceBox_sub_height2 = 2 * surface_board_min_radius1 * TMath::Cos(surfaceBox_sub_angle2 / 2);

  new TGeoTubeSeg("surfaceTubeSeg1", surface_board_min_radius1, surface_board_max_radius1, surface_board_thickness / 2, 180, 0);
  new TGeoBBox("surfaceBox_sub1", surfaceBox_sub_width1 / 2, surfaceBox_sub_height1 / 2, surface_board_thickness);
  new TGeoBBox("surfaceBox_sub2", surfaceBox_sub_width2 / 2, surfaceBox_sub_height2 / 2, surface_board_thickness);

  auto* tsurfaceBox_sub1 = new TGeoTranslation("tsurfaceBox_sub1", 0, -(surface_board_max_radius1 - surfaceBox_sub_height1 / 2), 0);
  auto* tsurfaceBox_sub2 = new TGeoTranslation("tsurfaceBox_sub2", 0., 0., 0.);
  tsurfaceBox_sub1->RegisterYourself();
  tsurfaceBox_sub2->RegisterYourself();

  Double_t surfaceBox_edge_width1 = 8.8;
  Double_t surfaceBox_edge_height1 = 4.347;

  Double_t surfaceBox_edge_sub_width1 = 5.;
  Double_t surfaceBox_edge_sub_height1 = 8.025;

  new TGeoBBox("surfaceBox_edge1", surfaceBox_edge_width1 / 2, surfaceBox_edge_height1 / 2, surface_board_thickness / 2);
  new TGeoBBox("surfaceBox_sub_edge1", surfaceBox_edge_sub_width1 / 2, surfaceBox_edge_sub_height1 / 2, surface_board_thickness);

  auto* tsurfaceBox_edge_right1 = new TGeoTranslation("tsurfaceBox_edge_right1",
                                                      surface_board_min_radius1 * TMath::Sin(surfaceBox_sub_angle2 / 2) + surfaceBox_edge_width1 / 2,
                                                      -surface_board_min_radius1 * TMath::Cos(surfaceBox_sub_angle2 / 2) + surfaceBox_edge_height1 / 2, 0);
  auto* tsurfaceBox_edge_sub_right1 = new TGeoTranslation("tsurfaceBox_edge_sub_right1",
                                                          surface_board_min_radius1 * TMath::Sin(surfaceBox_sub_angle2 / 2) + surfaceBox_edge_width1 + surfaceBox_edge_sub_width1 / 2,
                                                          -surface_board_min_radius1 * TMath::Cos(surfaceBox_sub_angle2 / 2) + surfaceBox_edge_height1 - surfaceBox_edge_sub_height1 / 2, 0);
  auto* tsurfaceBox_edge_left1 = new TGeoTranslation("tsurfaceBox_edge_left1",
                                                     -(surface_board_min_radius1 * TMath::Sin(surfaceBox_sub_angle2 / 2) + surfaceBox_edge_width1 / 2),
                                                     -surface_board_min_radius1 * TMath::Cos(surfaceBox_sub_angle2 / 2) + surfaceBox_edge_height1 / 2, 0);
  auto* tsurfaceBox_edge_sub_left1 = new TGeoTranslation("tsurfaceBox_edge_sub_left1",
                                                         -(surface_board_min_radius1 * TMath::Sin(surfaceBox_sub_angle2 / 2) + surfaceBox_edge_width1 + surfaceBox_edge_sub_width1 / 2),
                                                         -surface_board_min_radius1 * TMath::Cos(surfaceBox_sub_angle2 / 2) + surfaceBox_edge_height1 - surfaceBox_edge_sub_height1 / 2, 0);

  tsurfaceBox_edge_right1->RegisterYourself();
  tsurfaceBox_edge_sub_right1->RegisterYourself();

  tsurfaceBox_edge_left1->RegisterYourself();
  tsurfaceBox_edge_sub_left1->RegisterYourself();

  Double_t surfaceBox_center_sub_min_radius1 = 18.0;
  Double_t surfaceBox_center_sub_max_radius1 = 19.0;
  Double_t surfaceBox_center_sub_max_angle1 = 130;

  new TGeoTubeSeg("surfaceCenterTubeSeg_sub1", surfaceBox_center_sub_min_radius1, surfaceBox_center_sub_max_radius1, middle_board_thickness,
                  180 + (180 - surfaceBox_center_sub_max_angle1) / 2, 180 + (180 - surfaceBox_center_sub_max_angle1) / 2 + surfaceBox_center_sub_max_angle1);

  Double_t surfaceBox_edge_sub_width2 = 1.;
  Double_t surfaceBox_edge_sub_height2 = 1.;
  Double_t surfaceBox_edge_sub_refposi_x = 0.76;
  Double_t surfaceBox_edge_sub_refposi_y = 4.004;

  new TGeoBBox("surfaceBox_square_sub2", surfaceBox_edge_sub_width2 / 2, surfaceBox_edge_sub_height2 / 2, surface_board_thickness);
  auto* tsurfaceBox_square_sub_right2 = new TGeoTranslation("tsurfaceBox_square_sub_right2",
                                                            surface_board_min_radius1 * TMath::Sin(surfaceBox_sub_angle2 / 2) + surfaceBox_edge_width1 - surfaceBox_edge_sub_refposi_x - surfaceBox_edge_sub_width2 / 2,
                                                            -surface_board_min_radius1 * TMath::Cos(surfaceBox_sub_angle2 / 2) + surfaceBox_edge_height1 - surfaceBox_edge_sub_refposi_y - surfaceBox_edge_sub_height2 / 2, 0);
  auto* tsurfaceBox_square_sub_left2 = new TGeoTranslation("tsurfaceBox_square_sub_left2",
                                                           -(surface_board_min_radius1 * TMath::Sin(surfaceBox_sub_angle2 / 2) + surfaceBox_edge_width1 - surfaceBox_edge_sub_refposi_x - surfaceBox_edge_sub_width2 / 2),
                                                           -surface_board_min_radius1 * TMath::Cos(surfaceBox_sub_angle2 / 2) + surfaceBox_edge_height1 - surfaceBox_edge_sub_refposi_y - surfaceBox_edge_sub_height2 / 2, 0);
  tsurfaceBox_square_sub_right2->RegisterYourself();
  tsurfaceBox_square_sub_left2->RegisterYourself();

  new TGeoCompositeShape("surface_board_shape",
                         "surfaceTubeSeg1 - surfaceBox_sub1:tsurfaceBox_sub1 - surfaceBox_sub2:tsurfaceBox_sub2"
                         "+surfaceBox_edge1:tsurfaceBox_edge_right1 - surfaceBox_sub_edge1:tsurfaceBox_edge_sub_right1"
                         "+surfaceBox_edge1:tsurfaceBox_edge_left1 - surfaceBox_sub_edge1:tsurfaceBox_edge_sub_left1"
                         "-surfaceCenterTubeSeg_sub1 - surfaceBox_square_sub2:tsurfaceBox_square_sub_right2"
                         "-surfaceCenterTubeSeg_sub1 - surfaceBox_square_sub2:tsurfaceBox_square_sub_left2");

  auto* tsurface_board_shape1 = new TGeoTranslation("tsurface_board_shape1", 0, 0, middle_board_thickness / 2 + surface_board_thickness / 2);
  auto* tsurface_board_shape2 = new TGeoTranslation("tsurface_board_shape2", 0, 0, -(middle_board_thickness / 2 + surface_board_thickness / 2));
  tsurface_board_shape1->RegisterYourself();
  tsurface_board_shape2->RegisterYourself();

  TGeoCompositeShape* surface_board_2side_shape = new TGeoCompositeShape("surface_board_2side_shape", "surface_board_shape:tsurface_board_shape1 + surface_board_shape:tsurface_board_shape2");

  TGeoVolume* surface_board = new TGeoVolume("surface_board", surface_board_2side_shape, kMedPeek);
  surface_board->SetLineColor(kGreen + 3);

  mHalfPSU->AddNode(surface_board, 1, nullptr);

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////
  //PSU DCDC
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////

  Int_t nDCDC = 22;

  Double_t DCDC_main_width1 = 1.4;
  Double_t DCDC_main_height1 = 1.85;
  Double_t DCDC_main_thickness1 = 0.8;

  new TGeoBBox("DCDC_main_shape1", DCDC_main_width1 / 2, DCDC_main_height1 / 2, DCDC_main_thickness1 / 2);

  Double_t DCDC_width2 = 1.694;
  Double_t DCDC_height2 = 4.348;
  Double_t DCDC_thickness2 = 0.04;

  new TGeoBBox("DCDC_shape2", DCDC_width2 / 2, DCDC_height2 / 2, DCDC_thickness2 / 2);

  Double_t DCDC_sub_edge_thickness1 = 0.02;
  Double_t DCDC_sub_width1 = DCDC_main_width1 - DCDC_sub_edge_thickness1 * 2;
  Double_t DCDC_sub_height1 = DCDC_main_height1 - DCDC_sub_edge_thickness1 * 2;
  Double_t DCDC_sub_thickness1 = DCDC_main_thickness1;

  new TGeoBBox("DCDC_sub_shape1", DCDC_sub_width1 / 2, DCDC_sub_height1 / 2, DCDC_sub_thickness1 / 2);

  TGeoTranslation* tDCDC_sub_shape1 = new TGeoTranslation("tDCDC_sub_shape1", 0, 0, DCDC_sub_edge_thickness1);
  tDCDC_sub_shape1->RegisterYourself();

  new TGeoCompositeShape("DCDC_shape1", "DCDC_main_shape1 - DCDC_sub_shape1:tDCDC_sub_shape1");

  Double_t full_angle = 126;
  Double_t one_angle = 126. / (nDCDC - 1);
  Double_t start_DCDC_angle = 180 + (180 - full_angle) / 2;

  Double_t position_radius = 18.5;

  TGeoRotation* rDCDC_shape1[22];
  TGeoTranslation* tDCDC_shape1[22];
  TGeoCompositeShape* rotated_DCDC_shape1[22];

  TGeoRotation* rDCDC_shape2[22];
  TGeoTranslation* tDCDC_shape2[22];
  TGeoCompositeShape* rotated_DCDC_shape2[22];

  TString string_all_DCDC_shape1 = "";
  TString string_all_DCDC_shape2 = "";

  for (Int_t iDCDC = 0; iDCDC < nDCDC; ++iDCDC) {
    rDCDC_shape1[iDCDC] = new TGeoRotation(Form("rDCDC_shape1_angle%d", iDCDC + 1), full_angle / 2 - one_angle * iDCDC, 0, 0);
    rDCDC_shape2[iDCDC] = new TGeoRotation(Form("rDCDC_shape2_angle%d", iDCDC + 1), full_angle / 2 - one_angle * iDCDC, 0, 0);
    rDCDC_shape1[iDCDC]->RegisterYourself();
    rDCDC_shape2[iDCDC]->RegisterYourself();

    rotated_DCDC_shape1[iDCDC] = new TGeoCompositeShape(Form("rotated_DCDC_shape1_angle%d", iDCDC + 1), Form("dummy+DCDC_shape1:rDCDC_shape1_angle%d", iDCDC + 1));
    rotated_DCDC_shape2[iDCDC] = new TGeoCompositeShape(Form("rotated_DCDC_shape2_angle%d", iDCDC + 1), Form("dummy+DCDC_shape2:rDCDC_shape2_angle%d", iDCDC + 1));

    tDCDC_shape1[iDCDC] = new TGeoTranslation(Form("tDCDC_shape1_pos%d", iDCDC + 1),
                                              -position_radius * TMath::Cos((start_DCDC_angle + one_angle * iDCDC) * TMath::Pi() / 180.),
                                              position_radius * TMath::Sin((start_DCDC_angle + one_angle * iDCDC) * TMath::Pi() / 180.), 0);
    tDCDC_shape2[iDCDC] = new TGeoTranslation(Form("tDCDC_shape2_pos%d", iDCDC + 1),
                                              -(position_radius + DCDC_height2 / 5) * TMath::Cos((start_DCDC_angle + one_angle * iDCDC) * TMath::Pi() / 180.),
                                              (position_radius + DCDC_height2 / 5) * TMath::Sin((start_DCDC_angle + one_angle * iDCDC) * TMath::Pi() / 180.), 0);
    tDCDC_shape1[iDCDC]->RegisterYourself();
    tDCDC_shape2[iDCDC]->RegisterYourself();

    if (iDCDC + 1 == nDCDC) {
      string_all_DCDC_shape1 += Form("rotated_DCDC_shape1_angle%d:tDCDC_shape1_pos%d", iDCDC + 1, iDCDC + 1);
      string_all_DCDC_shape2 += Form("rotated_DCDC_shape2_angle%d:tDCDC_shape2_pos%d", iDCDC + 1, iDCDC + 1);
    } else {
      string_all_DCDC_shape1 += Form("rotated_DCDC_shape1_angle%d:tDCDC_shape1_pos%d+", iDCDC + 1, iDCDC + 1);
      string_all_DCDC_shape2 += Form("rotated_DCDC_shape2_angle%d:tDCDC_shape2_pos%d+", iDCDC + 1, iDCDC + 1);
    }
  }

  string_all_DCDC_shape1 += "+ DCDC_shape1:tsurfaceBox_square_sub_right2 + DCDC_shape1:tsurfaceBox_square_sub_left2";

  auto* tDCDC_shape_side_right2 = new TGeoTranslation("tDCDC_shape_side_right2",
                                                      surface_board_min_radius1 * TMath::Sin(surfaceBox_sub_angle2 / 2) + surfaceBox_edge_width1 - surfaceBox_edge_sub_refposi_x - surfaceBox_edge_sub_width2 / 2,
                                                      -surface_board_min_radius1 * TMath::Cos(surfaceBox_sub_angle2 / 2) + surfaceBox_edge_height1 - surfaceBox_edge_sub_refposi_y - surfaceBox_edge_sub_height2 / 2 + DCDC_height2 / 5.,
                                                      0);

  auto* tDCDC_shape_side_left2 = new TGeoTranslation("tDCDC_shape_side_left2",
                                                     -(surface_board_min_radius1 * TMath::Sin(surfaceBox_sub_angle2 / 2) + surfaceBox_edge_width1 - surfaceBox_edge_sub_refposi_x - surfaceBox_edge_sub_width2 / 2),
                                                     -surface_board_min_radius1 * TMath::Cos(surfaceBox_sub_angle2 / 2) + surfaceBox_edge_height1 - surfaceBox_edge_sub_refposi_y - surfaceBox_edge_sub_height2 / 2 + DCDC_height2 / 5.,
                                                     0);
  tDCDC_shape_side_right2->RegisterYourself();
  tDCDC_shape_side_left2->RegisterYourself();

  string_all_DCDC_shape2 += "+ DCDC_shape2:tDCDC_shape_side_right2 + DCDC_shape2:tDCDC_shape_side_left2";

  new TGeoCompositeShape("all_DCDC_shape1", string_all_DCDC_shape1);

  TGeoRotation* rall_DCDC_shape1 = new TGeoRotation("rall_DCDC_shape1", 0, 180, 180);
  rall_DCDC_shape1->RegisterYourself();

  new TGeoCompositeShape("all_DCDC_shape_opposit", "dummy+all_DCDC_shape1:rall_DCDC_shape1");

  new TGeoCompositeShape("all_DCDC_base_shape", string_all_DCDC_shape2);

  auto* tall_DCDC_shape_shape1 = new TGeoTranslation("tall_DCDC_shape_shape1", 0, 0, middle_board_thickness / 2 + surface_board_thickness + DCDC_thickness2 + DCDC_main_thickness1 / 2);
  auto* tall_DCDC_shape_shape2 = new TGeoTranslation("tall_DCDC_shape_shape2", 0, 0, -(middle_board_thickness / 2 + surface_board_thickness + DCDC_thickness2 + DCDC_main_thickness1 / 2));
  tall_DCDC_shape_shape1->RegisterYourself();
  tall_DCDC_shape_shape2->RegisterYourself();

  auto* tall_DCDC_base_shape_shape1 = new TGeoTranslation("tall_DCDC_base_shape_shape1", 0, 0, middle_board_thickness / 2 + surface_board_thickness + DCDC_thickness2 / 2);
  auto* tall_DCDC_base_shape_shape2 = new TGeoTranslation("tall_DCDC_base_shape_shape2", 0, 0, -(middle_board_thickness / 2 + surface_board_thickness + DCDC_thickness2 / 2));
  tall_DCDC_base_shape_shape1->RegisterYourself();
  tall_DCDC_base_shape_shape2->RegisterYourself();

  TGeoCompositeShape* two_side_all_DCDC_shape = new TGeoCompositeShape("two_side_all_DCDC_shape", "all_DCDC_shape_opposit:tall_DCDC_shape_shape1 + all_DCDC_shape1:tall_DCDC_shape_shape2");

  TGeoCompositeShape* two_side_all_DCDC_base_shape = new TGeoCompositeShape("two_side_all_DCDC_base_shape",
                                                                            "all_DCDC_base_shape:tall_DCDC_base_shape_shape1 + all_DCDC_base_shape:tall_DCDC_base_shape_shape2");

  TGeoVolume* DCDC1 = new TGeoVolume("DCDC1", two_side_all_DCDC_shape, kMedAlu);
  DCDC1->SetLineColor(kGray);

  TGeoVolume* DCDC2 = new TGeoVolume("DCDC2", two_side_all_DCDC_base_shape, kMedAlu);
  DCDC2->SetLineColor(kSpring);

  mHalfPSU->AddNode(DCDC1, 1, nullptr);
  mHalfPSU->AddNode(DCDC2, 1, nullptr);

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // DC-DC coil part
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Very detaild coil shape part
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  Double_t DCDC_coil_radius1 = 0.1;
  Double_t DCDC_coil_min_radius1 = 0;
  Double_t DCDC_coil_max_radius1 = DCDC_coil_radius1 - 0.075;
  Double_t DCDC_coil_straight_gap = DCDC_coil_radius1 / 20.;
  Double_t DCDC_coil_torus_part_radius = 2. / 5. * DCDC_width2 / 2.;

  new TGeoTorus("DCDC_coil_main_shape", DCDC_coil_radius1, 0, DCDC_coil_max_radius1, 0, 360);

  TGeoTranslation* tDCDC_coil_straight[9];
  tDCDC_coil_straight[0] = new TGeoTranslation("tDCDC_coil_straight_No0", 0, 0, 0);
  tDCDC_coil_straight[1] = new TGeoTranslation("tDCDC_coil_straight_No1", 0, 0, (DCDC_coil_max_radius1 * 2 + DCDC_coil_straight_gap) * 1);
  tDCDC_coil_straight[2] = new TGeoTranslation("tDCDC_coil_straight_No2", 0, 0, (DCDC_coil_max_radius1 * 2 + DCDC_coil_straight_gap) * 2);
  tDCDC_coil_straight[3] = new TGeoTranslation("tDCDC_coil_straight_No3", 0, 0, (DCDC_coil_max_radius1 * 2 + DCDC_coil_straight_gap) * 3);
  tDCDC_coil_straight[4] = new TGeoTranslation("tDCDC_coil_straight_No4", 0, 0, (DCDC_coil_max_radius1 * 2 + DCDC_coil_straight_gap) * 4);
  tDCDC_coil_straight[5] = new TGeoTranslation("tDCDC_coil_straight_No5", 0, 0, (DCDC_coil_max_radius1 * 2 + DCDC_coil_straight_gap) * -1);
  tDCDC_coil_straight[6] = new TGeoTranslation("tDCDC_coil_straight_No6", 0, 0, (DCDC_coil_max_radius1 * 2 + DCDC_coil_straight_gap) * -2);
  tDCDC_coil_straight[7] = new TGeoTranslation("tDCDC_coil_straight_No7", 0, 0, (DCDC_coil_max_radius1 * 2 + DCDC_coil_straight_gap) * -3);
  tDCDC_coil_straight[8] = new TGeoTranslation("tDCDC_coil_straight_No8", 0, 0, (DCDC_coil_max_radius1 * 2 + DCDC_coil_straight_gap) * -4);

  for (Int_t i = 0; i < 9; ++i)
    tDCDC_coil_straight[i]->RegisterYourself();

  new TGeoCompositeShape("DCDC_coil_straight_shape1",
                         "DCDC_coil_main_shape:tDCDC_coil_straight_No0"
                         "+ DCDC_coil_main_shape:tDCDC_coil_straight_No1 + DCDC_coil_main_shape:tDCDC_coil_straight_No2"
                         "+ DCDC_coil_main_shape:tDCDC_coil_straight_No3 + DCDC_coil_main_shape:tDCDC_coil_straight_No4"
                         "+ DCDC_coil_main_shape:tDCDC_coil_straight_No5 + DCDC_coil_main_shape:tDCDC_coil_straight_No6"
                         "+ DCDC_coil_main_shape:tDCDC_coil_straight_No7 + DCDC_coil_main_shape:tDCDC_coil_straight_No8");
  TGeoRotation* rDCDC_coil_torus[7];
  Double_t DCDC_coil_torus_part_angle[] = { 15, 30, 60, 90, 120, 150, 165 };

  TGeoCompositeShape* rotated_DCDC_coil_main_shape[7];
  for (Int_t i = 0; i < 7; ++i) {
    rDCDC_coil_torus[i] = new TGeoRotation(Form("rDCDC_coil_torus_No%d", i), -90, DCDC_coil_torus_part_angle[i], 0);
    rDCDC_coil_torus[i]->RegisterYourself();
    rotated_DCDC_coil_main_shape[i] = new TGeoCompositeShape(Form("rotated_DCDC_coil_main_shape_No%d", i), Form("dummy + DCDC_coil_main_shape:rDCDC_coil_torus_No%d", i));
  }

  TGeoTranslation* trotated_DCDC_coil_main_shape[7];

  for (Int_t i = 0; i < 7; ++i) {
    trotated_DCDC_coil_main_shape[i] = new TGeoTranslation(Form("trotated_DCDC_coil_main_shape_No%d", i),
                                                           DCDC_coil_torus_part_radius * TMath::Cos(DCDC_coil_torus_part_angle[i] * TMath::Pi() / 180.),
                                                           0,
                                                           DCDC_coil_torus_part_radius * TMath::Sin(DCDC_coil_torus_part_angle[i] * TMath::Pi() / 180.));
    trotated_DCDC_coil_main_shape[i]->RegisterYourself();
  }

  new TGeoCompositeShape("DCDC_coil_torus_shape1",
                         "rotated_DCDC_coil_main_shape_No0:trotated_DCDC_coil_main_shape_No0"
                         "+ rotated_DCDC_coil_main_shape_No1:trotated_DCDC_coil_main_shape_No1"
                         "+ rotated_DCDC_coil_main_shape_No2:trotated_DCDC_coil_main_shape_No2"
                         "+ rotated_DCDC_coil_main_shape_No3:trotated_DCDC_coil_main_shape_No3"
                         "+ rotated_DCDC_coil_main_shape_No4:trotated_DCDC_coil_main_shape_No4"
                         "+ rotated_DCDC_coil_main_shape_No5:trotated_DCDC_coil_main_shape_No5"
                         "+ rotated_DCDC_coil_main_shape_No6:trotated_DCDC_coil_main_shape_No6");

  TGeoTranslation* tDCDC_coil_straight_shape1[2];
  tDCDC_coil_straight_shape1[0] = new TGeoTranslation("tDCDC_coil_straight_shape1_right", DCDC_coil_torus_part_radius, 0, 0);
  tDCDC_coil_straight_shape1[1] = new TGeoTranslation("tDCDC_coil_straight_shape1_left", -DCDC_coil_torus_part_radius, 0, 0);
  tDCDC_coil_straight_shape1[0]->RegisterYourself();
  tDCDC_coil_straight_shape1[1]->RegisterYourself();

  TGeoRotation* rDCDC_coil_torus_shape1[2];
  rDCDC_coil_torus_shape1[0] = new TGeoRotation("rDCDC_coil_torus_shape1_top", 0, 0, 0);
  rDCDC_coil_torus_shape1[1] = new TGeoRotation("rDCDC_coil_torus_shape1_bottom", 0, 180, 0);
  rDCDC_coil_torus_shape1[0]->RegisterYourself();
  rDCDC_coil_torus_shape1[1]->RegisterYourself();

  TGeoCompositeShape* rotated_DCDC_coil_torus_shape1[2];
  rotated_DCDC_coil_torus_shape1[0] = new TGeoCompositeShape("rotated_DCDC_coil_torus_shape1_top", "dummy+DCDC_coil_torus_shape1:rDCDC_coil_torus_shape1_top");
  rotated_DCDC_coil_torus_shape1[1] = new TGeoCompositeShape("rotated_DCDC_coil_torus_shape1_bottom", "dummy+DCDC_coil_torus_shape1:rDCDC_coil_torus_shape1_bottom");

  TGeoTranslation* trotated_DCDC_coil_torus_shape1[2];
  trotated_DCDC_coil_torus_shape1[0] = new TGeoTranslation("trotated_DCDC_coil_torus_shape1_top", 0, 0, (DCDC_coil_max_radius1 + DCDC_coil_straight_gap / 4.) * 9);
  trotated_DCDC_coil_torus_shape1[1] = new TGeoTranslation("trotated_DCDC_coil_torus_shape1_bottom", 0, 0, -(DCDC_coil_max_radius1 + DCDC_coil_straight_gap / 4.) * 9);
  trotated_DCDC_coil_torus_shape1[0]->RegisterYourself();
  trotated_DCDC_coil_torus_shape1[1]->RegisterYourself();
  /*
  TGeoCompositeShape *DCDC_coil_shape1 = new TGeoCompositeShape("DCDC_coil_shape1",
								"DCDC_coil_straight_shape1:tDCDC_coil_straight_shape1_right + DCDC_coil_straight_shape1:tDCDC_coil_straight_shape1_left"
								"+rotated_DCDC_coil_torus_shape1_top:trotated_DCDC_coil_torus_shape1_top"
								"+rotated_DCDC_coil_torus_shape1_bottom:trotated_DCDC_coil_torus_shape1_bottom");
  */
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Rough coil shape part
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //
  new TGeoTorus("DCDC_rough_coil_torus_shape", DCDC_coil_torus_part_radius, DCDC_coil_radius1 - DCDC_coil_max_radius1,
                DCDC_coil_radius1 + DCDC_coil_max_radius1, 0, 180);
  new TGeoTube("DCDC_rough_coil_straight_shape", DCDC_coil_radius1 - DCDC_coil_max_radius1, DCDC_coil_radius1 + DCDC_coil_max_radius1,
               DCDC_coil_max_radius1 * 9 + DCDC_coil_straight_gap * 4);

  TGeoRotation* rDCDC_rough_coil_torus_shape1 = new TGeoRotation("rDCDC_rough_coil_torus_shape1", 0, 90, 0);
  TGeoRotation* rDCDC_rough_coil_torus_shape2 = new TGeoRotation("rDCDC_rough_coil_torus_shape2", 0, -90, 0);
  rDCDC_rough_coil_torus_shape1->RegisterYourself();
  rDCDC_rough_coil_torus_shape2->RegisterYourself();

  new TGeoCompositeShape("rotated_DCDC_rough_coil_torus_shape1", "dummy+DCDC_rough_coil_torus_shape:rDCDC_rough_coil_torus_shape1");
  new TGeoCompositeShape("rotated_DCDC_rough_coil_torus_shape2", "dummy+DCDC_rough_coil_torus_shape:rDCDC_rough_coil_torus_shape2");

  TGeoTranslation* tDCDC_rough_coil_straight_shape1 = new TGeoTranslation("tDCDC_rough_coil_straight_shape1", DCDC_coil_torus_part_radius, 0, 0);
  TGeoTranslation* tDCDC_rough_coil_straight_shape2 = new TGeoTranslation("tDCDC_rough_coil_straight_shape2", -DCDC_coil_torus_part_radius, 0, 0);
  tDCDC_rough_coil_straight_shape1->RegisterYourself();
  tDCDC_rough_coil_straight_shape2->RegisterYourself();
  TGeoTranslation* tDCDC_rough_coil_torus_shape1 = new TGeoTranslation("tDCDC_rough_coil_torus_shape1", 0, 0, DCDC_coil_max_radius1 * 9 + DCDC_coil_straight_gap * 4);
  TGeoTranslation* tDCDC_rough_coil_torus_shape2 = new TGeoTranslation("tDCDC_rough_coil_torus_shape2", 0, 0, -(DCDC_coil_max_radius1 * 9 + DCDC_coil_straight_gap * 4));
  tDCDC_rough_coil_torus_shape1->RegisterYourself();
  tDCDC_rough_coil_torus_shape2->RegisterYourself();

  new TGeoCompositeShape("DCDC_coil_shape1",
                         "DCDC_rough_coil_straight_shape:tDCDC_rough_coil_straight_shape1+"
                         "DCDC_rough_coil_straight_shape:tDCDC_rough_coil_straight_shape2+"
                         "rotated_DCDC_rough_coil_torus_shape1:tDCDC_rough_coil_torus_shape1+"
                         "rotated_DCDC_rough_coil_torus_shape2:tDCDC_rough_coil_torus_shape2");
  TGeoRotation* rDCDC_coil_shape1[nDCDC];
  TGeoTranslation* tDCDC_coil_shape1[nDCDC];
  TGeoCompositeShape* rotated_DCDC_coil_shape1[nDCDC];
  TString string_all_DCDC_coil_shape1 = "";

  for (Int_t iDCDC = 0; iDCDC < nDCDC; ++iDCDC) {
    rDCDC_coil_shape1[iDCDC] = new TGeoRotation(Form("rDCDC_coil_shape1_angle%d", iDCDC + 1), full_angle / 2 - one_angle * iDCDC, 90, 0);
    rDCDC_coil_shape1[iDCDC]->RegisterYourself();

    rotated_DCDC_coil_shape1[iDCDC] = new TGeoCompositeShape(Form("rotated_DCDC_coil_shape1_angle%d", iDCDC + 1), Form("dummy+DCDC_coil_shape1:rDCDC_coil_shape1_angle%d", iDCDC + 1));

    tDCDC_shape1[iDCDC] = new TGeoTranslation(Form("tDCDC_coil_shape1_pos%d", iDCDC + 1),
                                              -position_radius * TMath::Cos((start_DCDC_angle + one_angle * iDCDC) * TMath::Pi() / 180.),
                                              position_radius * TMath::Sin((start_DCDC_angle + one_angle * iDCDC) * TMath::Pi() / 180.), 0);
    tDCDC_shape1[iDCDC]->RegisterYourself();

    if (iDCDC + 1 == nDCDC) {
      string_all_DCDC_coil_shape1 += Form("rotated_DCDC_coil_shape1_angle%d:tDCDC_shape1_pos%d", iDCDC + 1, iDCDC + 1);
    } else {
      string_all_DCDC_coil_shape1 += Form("rotated_DCDC_coil_shape1_angle%d:tDCDC_shape1_pos%d+", iDCDC + 1, iDCDC + 1);
    }
  }

  TGeoRotation* rDCDC_side_coil_shape1 = new TGeoRotation("rDCDC_side_coil_shape1", 0, 90, 0);
  rDCDC_side_coil_shape1->RegisterYourself();
  new TGeoCompositeShape("rotated_DCDC_side_coil_shape", "dummy+DCDC_coil_shape1:rDCDC_side_coil_shape1");

  string_all_DCDC_coil_shape1 += "+rotated_DCDC_side_coil_shape:tsurfaceBox_square_sub_right2 + rotated_DCDC_side_coil_shape:tsurfaceBox_square_sub_left2";

  new TGeoCompositeShape("all_DCDC_coil_shape1", string_all_DCDC_coil_shape1);

  auto* tall_DCDC_coil_shape1 = new TGeoTranslation("tall_DCDC_coil_shape1", 0, 0, middle_board_thickness / 2 + surface_board_thickness + DCDC_thickness2 + DCDC_coil_radius1);
  auto* tall_DCDC_coil_shape2 = new TGeoTranslation("tall_DCDC_coil_shape2", 0, 0, -(middle_board_thickness / 2 + surface_board_thickness + DCDC_thickness2 + DCDC_coil_radius1));
  tall_DCDC_coil_shape1->RegisterYourself();
  tall_DCDC_coil_shape2->RegisterYourself();

  TGeoCompositeShape* two_side_all_DCDC_coil_shape1 = new TGeoCompositeShape("two_side_all_DCDC_coil_shape1",
                                                                             "all_DCDC_coil_shape1:tall_DCDC_coil_shape1+"
                                                                             "all_DCDC_coil_shape1:tall_DCDC_coil_shape2");
  TGeoVolume* DCDC_coil1 = new TGeoVolume("DCDC_coil1", two_side_all_DCDC_coil_shape1, kMedCu);

  mHalfPSU->AddNode(DCDC_coil1, 1, nullptr);
  //TGeoTranslation* tHalfPSU = new TGeoTranslation("tHalfPSU",0, -4.2 - (middleBox_sub_height1/2-surfaceBox_edge_height1/2), -72.6 + 46.0);
  //TGeoTranslation* tHalfPSU = new TGeoTranslation("tHalfPSU",0,0, -72.6 + 46.0);
  //tHalfPSU->RegisterYourself();
  //mHalfVolume->AddNode(mHalfPSU,0,tHalfPSU);

  return mHalfPSU;
}
