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

#include "MFTBase/Barrel.h"

using namespace o2::mft;

Barrel::Barrel()
{
  createBarrel();
}

TGeoVolumeAssembly* Barrel::createBarrel()
{

  auto* BarrelVolume = new TGeoVolumeAssembly("BarrelVolume");

  //
  TGeoMedium* kMeAl = gGeoManager->GetMedium("MFT_Alu$");
  TGeoMedium* mCu = gGeoManager->GetMedium("MFT_Cu$");
  TGeoMedium* mCarbon = gGeoManager->GetMedium("MFT_CarbonFiberM46J$");
  TGeoMedium* mRohacell = gGeoManager->GetMedium("MFT_Rohacell$");
  TGeoMedium* mKapton = gGeoManager->GetMedium("MFT_Kapton$");
  TGeoMedium* mWater = gGeoManager->GetMedium("MFT_Water$");
  TGeoMedium* mAir = gGeoManager->GetMedium("MFT_Air$");
  TGeoMedium* mPolypropylene = gGeoManager->GetMedium("MFT_Polypropylene$");
  TGeoMedium* mPolyurethane = gGeoManager->GetMedium("MFT_Polyurethane$");
  TGeoMedium* mPolyimide = gGeoManager->GetMedium("MFT_Polyimide$");
  Double_t angle_open = 26.5;
  const Float_t kDegRad = TMath::Pi() / 180.;
  Double_t angle_open_rad = 26.5 * kDegRad;
  Double_t rin = 49.90;
  Double_t rout = 50.30;
  Double_t no_overlap = 0.5;
  Double_t shift_y = 0.4;
  // define shape of joints
  TGeoVolume* BarrelJoint0 = gGeoManager->MakeTubs("Barrel_joint0", kMeAl, rin - 0.25, rin, 1.05, 180 + (angle_open + no_overlap), 360 - (angle_open + no_overlap));
  TGeoVolume* BarrelJoint1 = gGeoManager->MakeTubs("Barrel_joint1", kMeAl, rin, rin + 0.45, 0.525, 180 + (angle_open + no_overlap), 360 - (angle_open + no_overlap));
  TGeoVolume* BoxJoint0 = gGeoManager->MakeBox("BoxJoint0", kMeAl, 0.2375, 2.750, 0.5125);
  TGeoVolume* BoxJoint1 = gGeoManager->MakeBox("BoxJoint1", kMeAl, 0.75, 2.750, 0.5125);
  // define shape of cyl vol
  TGeoVolume* BarrelTube0 =
    gGeoManager->MakeTubs("Barrel_Cylinder0", mCarbon, rin, rout, 20.325, 180 + (angle_open + no_overlap), 360 - (angle_open + no_overlap));
  TGeoVolume* BarrelTube1 =
    gGeoManager->MakeTubs("Barrel_Cylinder1", mCarbon, rin, rout, 61.00, 180 + (angle_open + no_overlap), 360 - (angle_open + no_overlap));

  // defining rails 1st cyl
  TGeoVolume* barrel_rail0 = gGeoManager->MakeBox("RailBox0", mCarbon, 0.5, 0.45, 19.20);
  TGeoVolume* cylrail = gGeoManager->MakeTubs("cylrail0", mCarbon, 0, .20, 19.20, 0, 180);
  TGeoVolume* barrel_rail1 = gGeoManager->MakeBox("RailBox1", mCarbon, 0.2, .45, 19.20);
  TGeoCompositeShape* com_rail1;
  TGeoTranslation* transcyl = new TGeoTranslation(0, 0.45, 0);
  transcyl->SetName("transcyl");
  transcyl->RegisterYourself();

  com_rail1 =
    new TGeoCompositeShape("Composite_rail1", "RailBox1 + cylrail0:transcyl");
  TGeoVolume* comp_volrail0 = new TGeoVolume("Comp_rail0_vol", com_rail1, mCarbon);
  TGeoVolume* BoxCavity0 = gGeoManager->MakeBox("BoxCavity0", mCarbon, 0.50, 2.50, 19.20);
  TGeoVolume* BoxCavity1 = gGeoManager->MakeBox("BoxCavity1", mCarbon, 0.40, 2.30, 19.20);
  TGeoVolume* BoxCavity2 = gGeoManager->MakeBox("BoxCavity2", mCarbon, 0.50, 0.10, 19.20);
  TGeoVolume* Joincyl0 = gGeoManager->MakeTubs("Joincyl0", mCarbon, 0.7500, 0.95, 19.20, 105.4660, 180);
  TGeoVolume* Joincyl1 = gGeoManager->MakeTubs("Joincyl1", mCarbon, 10.000, 10.20, 19.20, 0.0, 5.1);
  TGeoCompositeShape* com_box0;
  TGeoTranslation* cstr = new TGeoTranslation(0, -2.40, 0);
  TGeoTranslation* jcyl0 = new TGeoTranslation(0.7, -3.25, 0);
  TGeoTranslation* jcyl1 = new TGeoTranslation(-10.50, -3.25, 0);
  cstr->SetName("cstr");
  jcyl0->SetName("jcyltr0");
  jcyl1->SetName("jcyltr1");
  cstr->RegisterYourself();
  jcyl0->RegisterYourself();
  jcyl1->RegisterYourself();
  com_box0 = new TGeoCompositeShape("CompBox", "BoxCavity0 - BoxCavity1 - BoxCavity2:cstr + Joincyl0:jcyltr0 + Joincyl1:jcyltr1");

  TGeoVolume* comp_volBox = new TGeoVolume("Comp_Box_Vol0", com_box0, mCarbon);
  // definig pos_box0
  TGeoRotation rotrail0;
  TGeoRotation rotrail1;
  rotrail0.SetAngles(-127.7, 0, 0); // inversion du signe car le positionnement était incorrect
  rotrail1.SetAngles(127.7, 0, 0);  // inversion du signe

  Double_t rin_shift = rin - 0.9;
  Double_t xcr = rin_shift * TMath::Sin(127.7 * kDegRad);
  Double_t ycr = rin_shift * TMath::Cos(127.7 * kDegRad);
  TGeoTranslation transrail0(-xcr, -ycr, 20.350); // central rail R pos= 49.90; H= 0.90; W = 1.0 (cm)
  TGeoTranslation transrail1(xcr, -ycr, 20.350);  // les fameux 127.7°
  TGeoCombiTrans Combirail0(transrail0, rotrail0);
  TGeoCombiTrans Combirail1(transrail1, rotrail1);
  TGeoHMatrix* pos_rail0 = new TGeoHMatrix(Combirail0);
  TGeoHMatrix* pos_rail1 = new TGeoHMatrix(Combirail1);

  TGeoRotation rotrail2;
  TGeoRotation rotrail3;
  rotrail2.SetAngles(-61.28, 0, 0); // right
  rotrail3.SetAngles(61.28, 0, 0);  // left
  Double_t xrail_2_3 = rin_shift * TMath::Sin(61.28 * kDegRad);
  Double_t yrail_2_3 = rin_shift * TMath::Cos(61.28 * kDegRad);

  TGeoTranslation transrail2(-xrail_2_3, -yrail_2_3, 20.350); // R = 498.5; H = 11 (mm)
  TGeoTranslation transrail3(xrail_2_3, -yrail_2_3, 20.350);

  TGeoCombiTrans Combirail2(transrail2, rotrail2);
  TGeoCombiTrans Combirail3(transrail3, rotrail3);
  TGeoHMatrix* pos_rail2 = new TGeoHMatrix(Combirail2);
  TGeoHMatrix* pos_rail3 = new TGeoHMatrix(Combirail3);

  TGeoRotation rotrail4;
  TGeoRotation rotrail5;
  rotrail4.SetAngles(-43.32, 0, 0); // right
  rotrail5.SetAngles(43.32, 0, 0);  // left
  Double_t xrail_4_5 = rin_shift * TMath::Sin(43.32 * kDegRad);
  Double_t yrail_4_5 = rin_shift * TMath::Cos(43.32 * kDegRad);

  TGeoTranslation transrail4(-xrail_4_5, -yrail_4_5, 20.350); // R = 498.5
  TGeoTranslation transrail5(xrail_4_5, -yrail_4_5, 20.350);
  TGeoCombiTrans Combirail4(transrail4, rotrail4);
  TGeoCombiTrans Combirail5(transrail5, rotrail5);
  TGeoHMatrix* pos_rail4 = new TGeoHMatrix(Combirail4);
  TGeoHMatrix* pos_rail5 = new TGeoHMatrix(Combirail5);
  // rotating 2nd Box

  TGeoRotation rotbox0;
  rotbox0.SetAngles(180, 180, 0);
  TGeoTranslation transbox0(-45.30, -19.85 + shift_y, 20.35); // ??? 23.66°
  TGeoCombiTrans Combibox0(transbox0, rotbox0);
  TGeoHMatrix* pos_box0 = new TGeoHMatrix(Combibox0);

  // rails 2nd cyl

  TGeoVolume* barrel_rail02 = gGeoManager->MakeBox("RailBox02", mCarbon, 0.5, 0.45, 60.4750);
  TGeoVolume* cylrail2 = gGeoManager->MakeTubs("cylrail2", mCarbon, 0, 0.20, 60.4750, 0, 180);
  TGeoVolume* barrel_rail12 = gGeoManager->MakeBox("RailBox12", mCarbon, 0.2, 0.45, 60.4750);
  TGeoCompositeShape* com_rail12;
  TGeoTranslation* transcyl2 = new TGeoTranslation(0, 0.45, 0);
  transcyl2->SetName("transcyl2");
  transcyl2->RegisterYourself();
  com_rail12 = new TGeoCompositeShape("Composite_rail1", "RailBox12 + cylrail2:transcyl2");
  TGeoVolume* comp_volrail02 = new TGeoVolume("Comp_rail0_vol2", com_rail12, mCarbon);
  TGeoVolume* BoxCavity02 = gGeoManager->MakeBox("BoxCavity02", mCarbon, 0.50, 2.50, 60.4750);
  TGeoVolume* BoxCavity12 = gGeoManager->MakeBox("BoxCavity12", mCarbon, 0.40, 2.30, 60.4750);
  TGeoVolume* BoxCavity22 = gGeoManager->MakeBox("BoxCavity22", mCarbon, 0.50, 0.10, 60.4750);
  TGeoVolume* Joincyl02 = gGeoManager->MakeTubs("Joincyl02", mCarbon, 0.7500, 0.95, 60.4750, 105.4660, 180);
  TGeoVolume* Joincyl12 = gGeoManager->MakeTubs("Joincyl12", mCarbon, 10.000, 10.20, 60.4750, 0.0, 5.1);
  TGeoCompositeShape* com_box02;
  TGeoTranslation* cstr2 = new TGeoTranslation(0, -2.40, 0);
  TGeoTranslation* jcyl02 = new TGeoTranslation(0.7, -3.25, 0);
  TGeoTranslation* jcyl12 = new TGeoTranslation(-10.50, -3.25, 0);
  cstr2->SetName("cstr2");
  jcyl02->SetName("jcyltr02");
  jcyl12->SetName("jcyltr12");
  cstr2->RegisterYourself();
  jcyl02->RegisterYourself();
  jcyl12->RegisterYourself();
  com_box02 = new TGeoCompositeShape("CompBox2", "BoxCavity02 - BoxCavity12 - BoxCavity22:cstr2 + Joincyl02:jcyltr02 + Joincyl12:jcyltr12");

  TGeoVolume* comp_volBox2 = new TGeoVolume("Comp_Box_Vol02", com_box02, mCarbon);
  TGeoRotation rotrail02;
  TGeoRotation rotrail12;
  rotrail02.SetAngles(-127.7, 0, 0);
  rotrail12.SetAngles(127.7, 0, 0);

  TGeoTranslation transrail02(-xcr, -ycr, 104.425); // central rail R pos= 499.0; H= 9.0; W = 10.0
  TGeoTranslation transrail12(xcr, -ycr, 104.425);
  TGeoCombiTrans Combirail02(transrail02, rotrail02);
  TGeoCombiTrans Combirail12(transrail12, rotrail12);
  TGeoHMatrix* pos_rail02 = new TGeoHMatrix(Combirail02);
  TGeoHMatrix* pos_rail12 = new TGeoHMatrix(Combirail12);

  TGeoRotation rotrail22;
  TGeoRotation rotrail32;
  rotrail22.SetAngles(-4.62, 0, 0);
  rotrail32.SetAngles(4.62, 0, 0);

  TGeoTranslation transrail22(-rin_shift * TMath::Cos(28.76 * kDegRad), -rin_shift * TMath::Sin(28.76 * kDegRad), 104.425); // 28.76°
  TGeoTranslation transrail32(rin_shift * TMath::Cos(28.76 * kDegRad), -rin_shift * TMath::Sin(28.76 * kDegRad), 104.425);
  TGeoCombiTrans Combirail22(transrail22, rotrail2);
  TGeoCombiTrans Combirail32(transrail32, rotrail3);
  TGeoHMatrix* pos_rail22 = new TGeoHMatrix(Combirail22);
  TGeoHMatrix* pos_rail32 = new TGeoHMatrix(Combirail32);

  TGeoRotation rotrail42;
  TGeoRotation rotrail52;
  rotrail42.SetAngles(-46.3, 0, 0); // right
  rotrail52.SetAngles(46.3, 0, 0);  // left

  TGeoTranslation transrail42(-xrail_4_5, -yrail_4_5, 104.425);
  TGeoTranslation transrail52(xrail_4_5, -yrail_4_5, 104.425);
  TGeoCombiTrans Combirail42(transrail42, rotrail42);
  TGeoCombiTrans Combirail52(transrail52, rotrail52);
  TGeoHMatrix* pos_rail42 = new TGeoHMatrix(Combirail42);
  TGeoHMatrix* pos_rail52 = new TGeoHMatrix(Combirail52);
  // rotating 2nd Box

  TGeoRotation rotbox02;
  rotbox02.SetAngles(180, 180, 0);
  TGeoTranslation transbox02(-45.30, -19.85 + shift_y, 104.425);
  TGeoCombiTrans Combibox02(transbox02, rotbox02);
  TGeoHMatrix* pos_box02 = new TGeoHMatrix(Combibox02);
  // defining SideRails
  TGeoVolume* SideRail0 = gGeoManager->MakeBox("siderail0", kMeAl, 0.4, 2.5, 23.975); // SideRail H= 50 mm, W = 8 mm, L 479.5 mm
  TGeoVolume* SideRail1 = gGeoManager->MakeBox("siderail1", kMeAl, 0.4, 2.5, 61.525); // SideRail H= 50 mm, W = 8 mm, L 1230.5 mm

  // defining pipes
  TGeoVolume* BarrelPipes = gGeoManager->MakeTube("Barrel_Pipes", mPolyurethane, 0.3, 0.4, 82.375);
  TGeoVolume* WPipes3mm = gGeoManager->MakeTube("Barrel_PipesFill3mm", mWater, 0.0, 0.3, 82.375);
  BarrelPipes->SetLineColor(kBlue);

  TGeoVolume* ConePipes = gGeoManager->MakeTube("Cone_Pipes", kMeAl, 0.3, 0.4, 82.25); //   ?????
  TGeoVolume* PipeBox0 = gGeoManager->MakeBox("PipeBoxOut", mPolyurethane, 1.75, 0.45, 82.375);
  TGeoVolume* PipeBox1 = gGeoManager->MakeBox("PipeBoxInn", mPolyurethane, 1.65, 0.35, 82.375);
  TGeoVolume* PipeBoxFill = gGeoManager->MakeBox("PipeBoxFill", mAir, 1.65, 0.35, 82.375);
  TGeoCompositeShape* ParallelPipeBox = new TGeoCompositeShape("ParallelPipeBox", "PipeBoxOut - PipeBoxInn");
  TGeoVolume* PipeBox = new TGeoVolume("PipeBox", ParallelPipeBox, mPolyurethane);
  PipeBox->SetLineColor(kWhite);
  // rotation+translation pipes
  TGeoRotation rotpipe0;
  rotpipe0.SetAngles(-46.0, 0, 0);

  Double_t xpipe = rin_shift * TMath::Sin(46 * kDegRad);
  Double_t ypipe = rin_shift * TMath::Cos(46 * kDegRad);

  TGeoTranslation transpipe0(-xpipe, -ypipe, 82.375);
  TGeoCombiTrans Combipipe0(transpipe0, rotpipe0);
  TGeoHMatrix* pos_pipe0 = new TGeoHMatrix(Combipipe0);

  TGeoRotation rotpiper;
  rotpiper.SetAngles(46.0, 0, 0);
  TGeoTranslation transpipe2(xpipe, -ypipe, 82.375);
  TGeoCombiTrans Combipipe2(transpipe2, rotpiper);
  TGeoHMatrix* pos_pipe2 = new TGeoHMatrix(Combipipe2);

  Double_t xboxjoint = rin * TMath::Cos(angle_open_rad);
  Double_t yboxjoint = rin * TMath::Sin(angle_open_rad);

  BarrelJoint0->SetLineColor(kRed - 9);
  BarrelVolume->AddNode(BarrelJoint0, 1);
  BarrelVolume->AddNode(BarrelJoint1, 1, new TGeoTranslation(0.0, 0.0, -0.525));
  BarrelVolume->AddNode(BoxJoint0, 1, new TGeoTranslation(xboxjoint, -yboxjoint + 2.75 + shift_y, 0.525));
  BarrelVolume->AddNode(BoxJoint1, 2, new TGeoTranslation(xboxjoint + 0.75 / 2, -yboxjoint + 2.75 + shift_y, -0.535));
  BarrelVolume->AddNode(BoxJoint0, 1, new TGeoTranslation(-xboxjoint, -yboxjoint + 2.75 + shift_y, 0.525));
  BarrelVolume->AddNode(BoxJoint1, 2, new TGeoTranslation(-xboxjoint - 0.75 / 2, -yboxjoint + 2.75 + shift_y, -0.525), "");
  BarrelVolume->AddNode(BarrelTube0, 1, new TGeoTranslation(0.0, 0.0, 20.325));
  BarrelVolume->AddNode(BarrelJoint0, 1, new TGeoTranslation(0.0, 0.0, 40.65));
  BarrelVolume->AddNode(BarrelJoint1, 1, new TGeoTranslation(0.0, 0.0, 41.1875));
  BarrelVolume->AddNode(BoxJoint0, 1, new TGeoTranslation(xboxjoint, -yboxjoint + 2.75 + shift_y, 40.125));
  BarrelVolume->AddNode(BoxJoint1, 1, new TGeoTranslation(xboxjoint + 0.75 / 2, -yboxjoint + 2.75 + shift_y, 41.1875), "");
  BarrelVolume->AddNode(BoxJoint0, 1, new TGeoTranslation(-xboxjoint, -yboxjoint + 2.75 + shift_y, 40.125));
  BarrelVolume->AddNode(BoxJoint1, 2, new TGeoTranslation(-xboxjoint - 0.75 / 2, -yboxjoint + 2.75 + shift_y, 41.1875), "");

  BarrelVolume->AddNode(barrel_rail0, 2, pos_rail0);
  BarrelVolume->AddNode(barrel_rail0, 2, pos_rail1);

  BarrelVolume->AddNode(comp_volrail0, 2, pos_rail2);
  BarrelVolume->AddNode(comp_volrail0, 2, pos_rail3);

  BarrelVolume->AddNode(comp_volrail0, 2, pos_rail4);
  BarrelVolume->AddNode(comp_volrail0, 2, pos_rail5);

  BarrelVolume->AddNode(comp_volBox, 2, pos_box0);

  BarrelVolume->AddNode(comp_volBox, 2, new TGeoTranslation(45.30, -19.84 + shift_y, 20.35));

  // adding siderails
  BarrelVolume->AddNode(SideRail0, 1, new TGeoTranslation(46.200001, -20.0 + shift_y, 17.775));
  BarrelVolume->AddNode(SideRail0, 1, new TGeoTranslation(-46.200001, -20.0 + shift_y, 17.775));

  // start of the 2nd cyl
  BarrelVolume->AddNode(BarrelJoint0, 1, new TGeoTranslation(0.0, 0.0, 42.80));
  BarrelVolume->AddNode(BarrelJoint1, 1, new TGeoTranslation(0.0, 0.0, 42.250));
  BarrelVolume->AddNode(BoxJoint0, 1, new TGeoTranslation(xboxjoint, -yboxjoint + 2.75 + shift_y, 43.325));
  BarrelVolume->AddNode(BoxJoint1, 2, new TGeoTranslation(xboxjoint + 0.75 / 2, -yboxjoint + 2.75 + shift_y, 42.275));
  BarrelVolume->AddNode(BoxJoint0, 1, new TGeoTranslation(-xboxjoint, -yboxjoint + 2.75 + shift_y, 43.325));
  BarrelVolume->AddNode(BoxJoint1, 2, new TGeoTranslation(-xboxjoint + 0.75 / 2, -yboxjoint + 2.75 + shift_y, 42.275));
  BarrelVolume->AddNode(BarrelTube1, 1, new TGeoTranslation(0.0, 0.0, 103.8));

  // adding rails 2nd cyl
  BarrelVolume->AddNode(barrel_rail02, 2, pos_rail02);
  BarrelVolume->AddNode(barrel_rail02, 2, pos_rail12);

  BarrelVolume->AddNode(comp_volrail02, 2, pos_rail22);
  BarrelVolume->AddNode(comp_volrail02, 2, pos_rail32);

  BarrelVolume->AddNode(comp_volrail02, 2, pos_rail42);
  BarrelVolume->AddNode(comp_volrail02, 2, pos_rail52);

  BarrelVolume->AddNode(comp_volBox2, 2, pos_box02);
  BarrelVolume->AddNode(comp_volBox2, 2, new TGeoTranslation(45.30, -19.84 + shift_y, 104.325));

  // adding side rails
  BarrelVolume->AddNode(SideRail1, 1, new TGeoTranslation(46.200001, -20.0 + shift_y, 103.375));
  BarrelVolume->AddNode(SideRail1, 1, new TGeoTranslation(-46.200001, -20.0 + shift_y, 103.375));

  // adding pipes R=490.8 and R 490.0 mm

  BarrelVolume->AddNode(BarrelPipes, 1, new TGeoTranslation(-rin_shift * TMath::Cos(29.55 * kDegRad), -rin_shift * TMath::Sin(29.55 * kDegRad), 82.375));
  BarrelVolume->AddNode(BarrelPipes, 1, new TGeoTranslation(-rin_shift * TMath::Cos(30.62 * kDegRad), -rin_shift * TMath::Sin(30.62 * kDegRad), 82.375));
  BarrelVolume->AddNode(BarrelPipes, 1, new TGeoTranslation(-rin_shift * TMath::Cos(31.68 * kDegRad), -rin_shift * TMath::Sin(31.68 * kDegRad), 82.375));
  BarrelVolume->AddNode(BarrelPipes, 1, new TGeoTranslation(-rin_shift * TMath::Cos(33.06 * kDegRad) + 0.1, -rin_shift * TMath::Sin(33.06 * kDegRad) + 0.1, 82.375));
  BarrelVolume->AddNode(BarrelPipes, 1, new TGeoTranslation(-rin_shift * TMath::Cos(34.70 * kDegRad) + 0.1, -rin_shift * TMath::Sin(34.70 * kDegRad) + 0.1, 82.375));
  BarrelVolume->AddNode(BarrelPipes, 1, new TGeoTranslation(-rin_shift * TMath::Cos(36.34 * kDegRad) + 0.1, -rin_shift * TMath::Sin(36.34 * kDegRad) + 0.1, 82.375));

  BarrelVolume->AddNode(BarrelPipes, 1, new TGeoTranslation(rin_shift * TMath::Cos(29.55 * kDegRad), -rin_shift * TMath::Sin(29.55 * kDegRad), 82.375));
  BarrelVolume->AddNode(BarrelPipes, 1, new TGeoTranslation(rin_shift * TMath::Cos(30.62 * kDegRad), -rin_shift * TMath::Sin(30.62 * kDegRad), 82.375));
  BarrelVolume->AddNode(BarrelPipes, 1, new TGeoTranslation(rin_shift * TMath::Cos(31.68 * kDegRad), -rin_shift * TMath::Sin(31.68 * kDegRad), 82.375));
  BarrelVolume->AddNode(BarrelPipes, 1, new TGeoTranslation(rin_shift * TMath::Cos(33.10 * kDegRad) - 0.1, -rin_shift * TMath::Sin(33.10 * kDegRad) + 0.1, 82.375));
  BarrelVolume->AddNode(BarrelPipes, 1, new TGeoTranslation(rin_shift * TMath::Cos(34.70 * kDegRad) - 0.1, -rin_shift * TMath::Sin(34.70 * kDegRad) + 0.1, 82.375));
  BarrelVolume->AddNode(BarrelPipes, 1, new TGeoTranslation(rin_shift * TMath::Cos(36.34 * kDegRad) - 0.1, -rin_shift * TMath::Sin(36.34 * kDegRad) + 0.1, 82.375));

  BarrelVolume->AddNode(PipeBox, 1, pos_pipe0);
  BarrelVolume->AddNode(PipeBox, 1, pos_pipe2);

  // adding fill to the pipes
  BarrelVolume->AddNode(WPipes3mm, 1, new TGeoTranslation(-rin_shift * TMath::Cos(29.55 * kDegRad), -rin_shift * TMath::Sin(29.55 * kDegRad), 82.375));
  BarrelVolume->AddNode(WPipes3mm, 1, new TGeoTranslation(-rin_shift * TMath::Cos(30.62 * kDegRad), -rin_shift * TMath::Sin(30.62 * kDegRad), 82.375));
  BarrelVolume->AddNode(WPipes3mm, 1, new TGeoTranslation(-rin_shift * TMath::Cos(31.68 * kDegRad), -rin_shift * TMath::Sin(31.68 * kDegRad), 82.375));
  BarrelVolume->AddNode(WPipes3mm, 1, new TGeoTranslation(-rin_shift * TMath::Cos(33.06 * kDegRad) + 0.1, -rin_shift * TMath::Sin(33.06 * kDegRad) + 0.1, 82.375));
  BarrelVolume->AddNode(WPipes3mm, 1, new TGeoTranslation(-rin_shift * TMath::Cos(34.70 * kDegRad) + 0.1, -rin_shift * TMath::Sin(34.70 * kDegRad) + 0.1, 82.375));
  BarrelVolume->AddNode(WPipes3mm, 1, new TGeoTranslation(-rin_shift * TMath::Cos(36.34 * kDegRad) + 0.1, -rin_shift * TMath::Sin(36.34 * kDegRad) + 0.1, 82.375));

  BarrelVolume->AddNode(WPipes3mm, 1, new TGeoTranslation(rin_shift * TMath::Cos(29.55 * kDegRad), -rin_shift * TMath::Sin(29.55 * kDegRad), 82.375));
  BarrelVolume->AddNode(WPipes3mm, 1, new TGeoTranslation(rin_shift * TMath::Cos(30.62 * kDegRad), -rin_shift * TMath::Sin(30.62 * kDegRad), 82.375));
  BarrelVolume->AddNode(WPipes3mm, 1, new TGeoTranslation(rin_shift * TMath::Cos(31.68 * kDegRad), -rin_shift * TMath::Sin(31.68 * kDegRad), 82.375));
  BarrelVolume->AddNode(WPipes3mm, 1, new TGeoTranslation(rin_shift * TMath::Cos(33.10 * kDegRad) - 0.1, -rin_shift * TMath::Sin(33.10 * kDegRad) + 0.1, 82.375));
  BarrelVolume->AddNode(WPipes3mm, 1, new TGeoTranslation(rin_shift * TMath::Cos(34.70 * kDegRad) - 0.1, -rin_shift * TMath::Sin(34.70 * kDegRad) + 0.1, 82.375));
  BarrelVolume->AddNode(WPipes3mm, 1, new TGeoTranslation(rin_shift * TMath::Cos(36.34 * kDegRad) - 0.1, -rin_shift * TMath::Sin(36.34 * kDegRad) + 0.1, 82.375));

  BarrelVolume->AddNode(PipeBoxFill, 2, pos_pipe0);
  BarrelVolume->AddNode(PipeBoxFill, 2, pos_pipe2);

  // ----------------------- set Wires --------------------------------
  Float_t radiia = 0.075;
  Float_t radiib = 0.0317; // HABIA H2419 (Charlotte Riccio)
  Float_t w_pos = 50.35;
  Int_t nwiresa = 10;
  Int_t nwiresb = 16;

  // defining wires shapes
  TGeoVolume* BarrelWiresA = gGeoManager->MakeTube("Barrel_wiresa", mCu, 0.0, radiia, 81.225);
  TGeoVolume* BarrelWiresB = gGeoManager->MakeTube("Barrel_wiresb", mCu, 0.0, radiib, 81.225);
  // defining isolation layers
  TGeoVolume* BarrelWiresIsolA = gGeoManager->MakeTube("Barrel_wiresa_isol", mPolyimide, 7 * radiia, 7 * radiia + 0.05, 81.225);
  TGeoVolume* BarrelWiresIsolB = gGeoManager->MakeTube("Barrel_wiresb_isol", mPolyimide, 7 * radiib, 7 * radiib + 0.05, 81.225);

  Double_t rin_wireA[2] = {rin - 0.82, rin - 0.82 - 2 * radiia};
  Double_t rin_wireB[4] = {rin - 0.82, rin - 0.82 - 2 * radiib, rin - 0.82 - 4 * radiib, rin - 0.82 - 6 * radiib};

  Double_t xPosIniA;
  Double_t xPosIniB;

  // 2 layers of 10 WireA
  for (Int_t l = 0; l < 2; l++) { // 2 layers
    xPosIniA = rin_wireA[l] * TMath::Sin(51.0 * kDegRad);
    for (Int_t k = 0; k < 5; k++) {
      xPosIniA = xPosIniA - radiia;
      Float_t yPosIniA = sqrt(rin_wireA[l] * rin_wireA[l] - xPosIniA * xPosIniA);
      BarrelVolume->AddNode(BarrelWiresA, 1, new TGeoTranslation(xPosIniA, -yPosIniA, 81.325));
      BarrelVolume->AddNode(BarrelWiresA, 1, new TGeoTranslation(-xPosIniA, -yPosIniA, 81.325));
      xPosIniA = xPosIniA - 0.03;
    }
  }
  Double_t xIsol = (rin_wireA[0] - 0.1) * TMath::Sin(50.5 * kDegRad);
  Double_t yIsol = sqrt((rin_wireA[0] - 0.1) * (rin_wireA[0] - 0.1) - xIsol * xIsol);
  BarrelVolume->AddNode(BarrelWiresIsolA, 1, new TGeoTranslation(xIsol, -yIsol, 81.325));
  BarrelVolume->AddNode(BarrelWiresIsolA, 1, new TGeoTranslation(-xIsol, -yIsol, 81.325));

  // 4 layers of 16 WireB
  for (Int_t l = 0; l < 4; l++) { // 4 layers
    xPosIniB = rin_wireB[l] * TMath::Sin(49.2 * kDegRad);
    for (Int_t k = 0; k < 4; k++) {
      xPosIniB = xPosIniB - radiib;
      Float_t yPosIniB = sqrt(rin_wireB[l] * rin_wireB[l] - xPosIniB * xPosIniB);
      BarrelVolume->AddNode(BarrelWiresB, 1, new TGeoTranslation(xPosIniB, -yPosIniB, 81.325));
      BarrelVolume->AddNode(BarrelWiresB, 1, new TGeoTranslation(-xPosIniB, -yPosIniB, 81.325));
      xPosIniB = xPosIniB - 0.02;
    }
  }
  xIsol = (rin_wireB[1] - 0.05) * TMath::Sin(49.0 * kDegRad);
  yIsol = sqrt((rin_wireB[1] - 0.05) * (rin_wireB[1] - 0.05) - xIsol * xIsol);
  BarrelVolume->AddNode(BarrelWiresIsolB, 1, new TGeoTranslation(xIsol, -yIsol, 81.325));
  BarrelVolume->AddNode(BarrelWiresIsolB, 1, new TGeoTranslation(-xIsol, -yIsol, 81.325));

  // fixation services
  Float_t FSThickness = 0.4;
  Float_t FSRad = 47.30;
  TGeoVolume* FixService0 = gGeoManager->MakeTubs("FixService0", mPolypropylene, FSRad, FSRad + FSThickness, 2.00, 313, 331);
  TGeoVolume* FixService1 = gGeoManager->MakeTubs("FixService1", mPolypropylene, FSRad, FSRad + FSThickness, 2.00, 209, 226.7);
  //  coloring
  FixService0->SetLineColor(kWhite);
  FixService1->SetLineColor(kWhite);
  // adding nodes
  BarrelVolume->AddNode(FixService0, 1, new TGeoTranslation(0.00, 0.0, 21.2));
  BarrelVolume->AddNode(FixService1, 1, new TGeoTranslation(0.00, 0.0, 21.2));
  BarrelVolume->AddNode(FixService0, 1, new TGeoTranslation(-0.00, 0.0, 32.7));
  BarrelVolume->AddNode(FixService1, 1, new TGeoTranslation(0.00, 0.0, 32.7));
  BarrelVolume->AddNode(FixService0, 1, new TGeoTranslation(-0.00, 0.0, 44.2));
  BarrelVolume->AddNode(FixService1, 1, new TGeoTranslation(0.00, 0.0, 44.2));
  BarrelVolume->AddNode(FixService0, 1, new TGeoTranslation(-0.00, 0.0, 52.325));
  BarrelVolume->AddNode(FixService1, 1, new TGeoTranslation(0.00, 0.0, 52.325));
  BarrelVolume->AddNode(FixService0, 1, new TGeoTranslation(-0.00, 0.0, 76.325));
  BarrelVolume->AddNode(FixService1, 1, new TGeoTranslation(0.00, 0.0, 76.325));
  BarrelVolume->AddNode(FixService0, 1, new TGeoTranslation(-0.00, 0.0, 100.325));
  BarrelVolume->AddNode(FixService1, 1, new TGeoTranslation(0.00, 0.0, 100.325));
  BarrelVolume->AddNode(FixService0, 1, new TGeoTranslation(-0.00, 0.0, 124.325));
  BarrelVolume->AddNode(FixService1, 1, new TGeoTranslation(0.00, 0.0, 124.325));
  BarrelVolume->AddNode(FixService0, 1, new TGeoTranslation(-0.00, 0.0, 148.325));
  BarrelVolume->AddNode(FixService1, 1, new TGeoTranslation(0.00, 0.0, 148.325));

  /*
  //====================== Need to be rewrited!,  fm july 2023 ================
  //=========================== CONICAL PART, A side ==========================
  //===========================================================================
  // triangular section
  TGeoXtru* trian = new TGeoXtru(2);
  trian->SetName("Tri01");
  Double_t x_tri[3] = {0.0, 164.5, 164.5}; // ini 164.5
  Double_t y_tri[3] = {0.0, 0.0, -31.90};  // init -26.50
  trian->DefinePolygon(3, x_tri, y_tri);
  trian->DefineSection(0, -0.2, 0, 0, 1);
  trian->DefineSection(1, 0.2, 0, 0, 1);
  // creating the volume
  TGeoVolume* comp2 = new TGeoVolume("tri", trian, kMeAl); //   ????????????????????????
  comp2->SetName("tri");
  comp2->RegisterYourself();
  comp2->SetLineColor(kRed);
  comp2->SetLineWidth(2);
  comp2->SetFillColor(4);

  // defining pos
  TGeoRotation rot_tri0;
  rot_tri0.SetAngles(-90, 90, 90);
  TGeoTranslation trans_tri0(46.6, -15.40, -82.25);  // movi  16.40
  TGeoTranslation trans_tri1(-46.6, -15.40, -82.25); // movi
  TGeoCombiTrans combi_tri0(trans_tri0, rot_tri0);
  TGeoCombiTrans combi_tri1(trans_tri1, rot_tri0);
  TGeoHMatrix pos_tri0 = combi_tri0;
  TGeoHMatrix pos_tri1 = combi_tri1;
  TGeoHMatrix* pos_trif0 = new TGeoHMatrix(pos_tri0);
  pos_trif0->SetName("pt0");
  pos_trif0->RegisterYourself();
  TGeoHMatrix* pos_trif1 = new TGeoHMatrix(pos_tri1);

  pos_trif1->SetName("pt1");
  pos_trif1->RegisterYourself();

  TGeoVolume* main_vol2 = gGeoManager->MakeCons("cone", kMeAl, 82.25, 49.059, 49.069 + 0.500, 60.50, 60.51 + 0.500, 198.25, 341.74);     //   ????????????????????????
  TGeoVolume* main_vol2_tt = gGeoManager->MakeCons("cone_tt", kMeAl, 82.35, 49.069, 49.069 + 0.50, 60.51, 70.51 + 0.50, 198.25, 341.74); //   ????????????????????????

  TGeoVolume* main_vol3 = gGeoManager->MakeBox("box", mAir, 46.6, 90.0, 82.25); // movi
  TGeoBBox* main_box = (TGeoBBox*)(main_vol3->GetShape());
  main_vol3->SetLineColor(kRed);
  main_vol3->SetLineWidth(2);
  main_vol3->SetFillColor(4);

  TGeoCompositeShape* cs;
  cs = new TGeoCompositeShape("CS", "(Tri01:pt0 + cone * box + Tri01:pt1) - cone_tt");
  TGeoVolume* comp1 = new TGeoVolume("COMP", cs, kMeAl); // adding missing medium   //   ????????????????????????
  // comp1->SetLineColor(kBlue);
  // comp1->SetLineWidth(2);
  // comp1->SetFillColor(4);

  // defining pipes
  TGeoVolume* BarrelPipes4mmC = gGeoManager->MakeTube("Barrel_PipesC", mPolyurethane, 0.4, 0.45, 83.0);
  TGeoVolume* WPipes4mmC = gGeoManager->MakeTube("Barrel_PipesFill4mmC", mWater, 0.0, 0.4, 83.0);
  TGeoVolume* BarrelPipes6mmC = gGeoManager->MakeTube("Barrel_PipesC", mPolyurethane, 0.6, 0.65, 83.0);
  TGeoVolume* WPipes6mmC = gGeoManager->MakeTube("Barrel_PipesFill6mmC", mWater, 0.0, 0.6, 83.0);
  TGeoVolume* ConePipesC = gGeoManager->MakeTube("Cone_PipesC", kMeAl, 0.3, 0.4, 82.25); //   ????????????????????????
  TGeoVolume* PipeBox0C = gGeoManager->MakeBox("PipeBoxOutC", mPolyurethane, 2.95, 0.45, 82.8);
  TGeoVolume* PipeBox1C = gGeoManager->MakeBox("PipeBoxInnC", mPolyurethane, 2.85, 0.35, 82.8);
  TGeoCompositeShape* ParallelPipeBoxC = new TGeoCompositeShape("ParallelPipeBoxC", "PipeBoxOutC - PipeBoxInnC");
  TGeoVolume* PipeBoxC = new TGeoVolume("PipeBoxC", ParallelPipeBoxC, mPolyurethane);
  BarrelPipes4mmC->SetLineColor(kRed);
  BarrelPipes6mmC->SetLineColor(kRed);

  Float_t y_translation = 2;
  Float_t y_sup = 4;
  Float_t z_translation = 247.2;

  //=============================== Mes rotations ==============================
  auto* rot_R5 = new TGeoRotation("rot_R5", 11.3, 8.4, 130);
  auto* rot_R4 = new TGeoRotation("rot_R4", 12.3, 8.4, 130);
  auto* rot_R3 = new TGeoRotation("rot_R3", 13.3, 8.4, 130);
  auto* rot_R2 = new TGeoRotation("rot_R2", 14.3, 8.4, 130);
  auto* rot_R1 = new TGeoRotation("rot_R1", 15.3, 8.4, 130);
  auto* rot_R0 = new TGeoRotation("rot_R0", 16.3, 8.4, 130);

  auto* rot_Q = new TGeoRotation("rot_Q", 16.3, 8.4, 130);

  BarrelVolume->AddNode(BarrelPipes4mmC, 1, new TGeoCombiTrans(-41.25 + 0.9, -32.75 + y_translation - y_sup - 1.35, z_translation, rot_R5));
  BarrelVolume->AddNode(BarrelPipes4mmC, 2, new TGeoCombiTrans(-40.55 + 1.0, -33.5 + y_translation - y_sup - 1.25, z_translation, rot_R4));
  BarrelVolume->AddNode(BarrelPipes4mmC, 3, new TGeoCombiTrans(-39.95 + 1.1, -34.3 + y_translation - y_sup - 1.2, z_translation, rot_R3));
  BarrelVolume->AddNode(BarrelPipes6mmC, 4, new TGeoCombiTrans(-39.05 + 1.0, -35.11 + y_translation - y_sup - 1.3, z_translation, rot_R2));
  BarrelVolume->AddNode(BarrelPipes6mmC, 5, new TGeoCombiTrans(-37.84 + 0.8, -35.87 + y_translation - y_sup - 1.65, z_translation, rot_R1));
  BarrelVolume->AddNode(BarrelPipes6mmC, 6, new TGeoCombiTrans(-36.79 + 0.9, -36.86 + y_translation - y_sup - 1.6, z_translation, rot_R0));

  auto* rot_L5 = new TGeoRotation("rot_L5", -11.3, 8.4, 130);
  auto* rot_L4 = new TGeoRotation("rot_L4", -12.3, 8.4, 130);
  auto* rot_L3 = new TGeoRotation("rot_L3", -13.3, 8.4, 130);
  auto* rot_L2 = new TGeoRotation("rot_L2", -14.3, 8.4, 130);
  auto* rot_L1 = new TGeoRotation("rot_L1", -15.3, 8.4, 130);
  auto* rot_L0 = new TGeoRotation("rot_L0", -16.3, 8.4, 130);
  auto* rot_B = new TGeoRotation("rot_B", -16.3, 8.4, 130);

  BarrelVolume->AddNode(BarrelPipes4mmC, 7, new TGeoCombiTrans(41.25 - 0.9, -32.75 + y_translation - y_sup - 1.35, z_translation, rot_L5));
  BarrelVolume->AddNode(BarrelPipes4mmC, 8, new TGeoCombiTrans(40.55 - 1.0, -33.5 + y_translation - y_sup - 1.25, z_translation, rot_L4));
  BarrelVolume->AddNode(BarrelPipes4mmC, 9, new TGeoCombiTrans(39.95 - 1.1, -34.3 + y_translation - y_sup - 1.2, z_translation, rot_L3));
  BarrelVolume->AddNode(BarrelPipes6mmC, 10, new TGeoCombiTrans(39.05 - 1.0, -35.11 + y_translation - y_sup - 1.3, z_translation, rot_L2));
  BarrelVolume->AddNode(BarrelPipes6mmC, 11, new TGeoCombiTrans(37.84 - 0.8, -35.87 + y_translation - y_sup - 1.65, z_translation, rot_L1));
  BarrelVolume->AddNode(BarrelPipes6mmC, 12, new TGeoCombiTrans(36.79 - 0.9, -36.86 + y_translation - y_sup - 1.6, z_translation, rot_L0));

  // PIPE rectangular
  //==========================================================
  auto* rot_P = new TGeoRotation("rot_P", 23.0, 8.4, 126); // 16 //box left
  auto* comb_AC = new TGeoCombiTrans(-32.040 + 1.0, -40.275 + y_translation - y_sup - 1.7, z_translation, rot_P);
  auto* rot_A = new TGeoRotation("rot_A", -23.0, 8.4, -126); // box right
  auto* comb_BC = new TGeoCombiTrans(32.040 - 1.0, -40.275 + y_translation - y_sup - 1.7, z_translation, rot_A);
  //==========================================================
  PipeBoxC->SetLineColor(kRed);
  BarrelVolume->AddNode(PipeBoxC, 1, comb_AC); //, pos_pipe0
  BarrelVolume->AddNode(PipeBoxC, 2, comb_BC); //, pos_pipe2

  auto* rotC = new TGeoRotation("rotC", 0.0, 0.0, 0.0);
  auto* combiC = new TGeoCombiTrans(0.0, -y_translation, z_translation, rotC);
  BarrelVolume->AddNode(comp1, 1, combiC);

  //===================================================================
  //========================== Low Voltage ============================
  //===================================================================
  // z_translation = 81.225 + z_translation;
  // defining wires shapes
  TGeoVolume* BarrelWiresAA = gGeoManager->MakeTube("Barrel_wiresaa", mCu, 0.0, radiia, 164.5 / 2);
  TGeoVolume* BarrelWiresBA = gGeoManager->MakeTube("Barrel_wiresba", mCu, 0.0, radiib, 164.5 / 2);
  TGeoVolume* BarrelWiresCA = gGeoManager->MakeTube("Barrel_wiresca", mCu, 0.0, radiic, 164.5 / 2);
  // units cm

  xPosIni = 40.35;
  Float_t correction = 1.0;
  Float_t coeff_inclinaison = 0.75;
  Float_t decalage = -14.325;
  Float_t hypo = 50.35;
  auto* rot_L = new TGeoRotation("rot_R", 11.0, 8.0, 0.0);
  auto* rot_R = new TGeoRotation("rot_L", -11.0, 8.0, 0.0);
  // auto* rot_L = new TGeoRotation("rot_L",  11.3, 8.4, 130);
  // auto* rot_R = new TGeoRotation("rot_L", -11.3, 8.4, 130);
  for (Int_t k = 0; k < 26; k++) {
    switch (wires_array_pos[k]) {

      case 1: {
        xPosIni = xPosIni - radiia;
        Float_t yPosIni = correction * sqrt(hypo * hypo - xPosIni * xPosIni);
        BarrelVolume->AddNode(BarrelWiresAA, 1, new TGeoCombiTrans(xPosIni, -coeff_inclinaison * yPosIni + decalage, z_translation, rot_L3));
        // printf("X %f Y %f \n",xPosIni, -coeff_inclinaison * yPosIni + decalage);
        BarrelVolume->AddNode(BarrelWiresAA, 1, new TGeoCombiTrans(-xPosIni, -coeff_inclinaison * yPosIni + decalage, z_translation, rot_R3));
        xPosIni = xPosIni - 0.02;
        break;
      }

      case 2: {
        xPosIni = xPosIni - radiib - 0.005; // 0.005 to avoid overlap between wires!!!
        Float_t yPosIni = correction * sqrt(hypo * hypo - xPosIni * xPosIni);
        BarrelVolume->AddNode(BarrelWiresBA, 1, new TGeoCombiTrans(xPosIni, -coeff_inclinaison * yPosIni + decalage, z_translation, rot_L3));
        BarrelVolume->AddNode(BarrelWiresBA, 1, new TGeoCombiTrans(-xPosIni, -coeff_inclinaison * yPosIni + decalage, z_translation, rot_R3));
        xPosIni = xPosIni - 0.02;
        break;
      }

      case 3: {
        xPosIni = xPosIni - radiic;
        Float_t yPosIni = correction * sqrt(hypo * hypo - xPosIni * xPosIni);
        BarrelVolume->AddNode(BarrelWiresCA, 1, new TGeoCombiTrans(xPosIni, -coeff_inclinaison * yPosIni + decalage, z_translation, rot_L3));
        BarrelVolume->AddNode(BarrelWiresCA, 1, new TGeoCombiTrans(-xPosIni, -coeff_inclinaison * yPosIni + decalage, z_translation, rot_R3));
        xPosIni = xPosIni - 0.02;
        break;
      }
    }
  }
  //===================================================================
  //===================================================================
  //===================================================================
  */

  return BarrelVolume;
}
