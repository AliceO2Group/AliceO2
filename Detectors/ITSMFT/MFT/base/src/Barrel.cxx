// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

  TGeoMedium* kMeAl = gGeoManager->GetMedium("MFT_Alu$");
  TGeoMedium* mCarbon = gGeoManager->GetMedium("MFT_CarbonFiber$");
  TGeoMedium* mRohacell = gGeoManager->GetMedium("MFT_Rohacell");
  TGeoMedium* mKapton = gGeoManager->GetMedium("MFT_Kapton$");

  // define shape of joints
  TGeoVolume* BarrelJoint0 = gGeoManager->MakeTubs(
    "Barrel_joint0", kMeAl, 49.600, 49.85, 1.05, 207.2660445, 332.7339555);
  TGeoVolume* BarrelJoint1 = gGeoManager->MakeTubs(
    "Barrel_joint1", kMeAl, 49.851, 50.301, 0.525, 207.2660445, 332.7339555);
  TGeoVolume* BoxJoint0 =
    gGeoManager->MakeBox("BoxJoint0", kMeAl, 0.2375, 2.750, 0.5125);
  TGeoVolume* BoxJoint1 =
    gGeoManager->MakeBox("BoxJoint1", kMeAl, 0.75, 2.750, 0.5125);
  // define shape of cyl vol
  TGeoVolume* BarrelTube0 =
    gGeoManager->MakeTubs("Barrel_Cylinder0", mCarbon, 50.400, 50.800, 20.325,
                          207.2660445, 332.7339555);
  TGeoVolume* BarrelTube1 =
    gGeoManager->MakeTubs("Barrel_Cylinder1", mCarbon, 50.400, 50.800, 61.00,
                          207.2660445, 332.7339555);

  // defining rails 1st cyl

  TGeoVolume* barrel_rail0 =
    gGeoManager->MakeBox("RailBox0", mCarbon, 0.5, 0.45, 19.20);
  TGeoVolume* cylrail =
    gGeoManager->MakeTubs("cylrail0", mCarbon, 0, .20, 19.20, 0, 180);
  TGeoVolume* barrel_rail1 =
    gGeoManager->MakeBox("RailBox1", mCarbon, 0.2, .45, 19.20);
  TGeoCompositeShape* com_rail1;
  TGeoTranslation* transcyl = new TGeoTranslation(0, 0.45, 0);
  transcyl->SetName("transcyl");
  transcyl->RegisterYourself();

  com_rail1 =
    new TGeoCompositeShape("Composite_rail1", "RailBox1 + cylrail0:transcyl");
  TGeoVolume* comp_volrail0 =
    new TGeoVolume("Comp_rail0_vol", com_rail1, mCarbon);
  TGeoVolume* BoxCavity0 =
    gGeoManager->MakeBox("BoxCavity0", mCarbon, 0.50, 2.50, 19.20);
  TGeoVolume* BoxCavity1 =
    gGeoManager->MakeBox("BoxCavity1", mCarbon, 0.40, 2.30, 19.20);
  TGeoVolume* BoxCavity2 =
    gGeoManager->MakeBox("BoxCavity2", mCarbon, 0.50, 0.10, 19.20);
  TGeoVolume* Joincyl0 = gGeoManager->MakeTubs("Joincyl0", mCarbon, 0.7500,
                                               0.95, 19.20, 105.4660, 180);
  TGeoVolume* Joincyl1 = gGeoManager->MakeTubs("Joincyl1", mCarbon, 10.000,
                                               10.20, 19.20, 0.0, 5.1);
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
  com_box0 = new TGeoCompositeShape("CompBox",
                                    "BoxCavity0 - BoxCavity1 - BoxCavity2:cstr "
                                    "+ Joincyl0:jcyltr0 + Joincyl1:jcyltr1");

  TGeoVolume* comp_volBox = new TGeoVolume("Comp_Box_Vol0", com_box0, mCarbon);
  // definig pos_box0
  TGeoRotation rotrail0;
  TGeoRotation rotrail1;
  rotrail0.SetAngles(127.7, 0, 0);
  rotrail1.SetAngles(-127.7, 0, 0);

  TGeoTranslation transrail0(
    -39.40, -30.621,
    20.350); // central rail R pos= 49.90; H= 0.90; W = 1.0 (cm)
  TGeoTranslation transrail1(39.40, -30.621, 20.350);
  TGeoCombiTrans Combirail0(transrail0, rotrail0);
  TGeoCombiTrans Combirail1(transrail1, rotrail1);
  TGeoHMatrix* pos_rail0 = new TGeoHMatrix(Combirail0);
  TGeoHMatrix* pos_rail1 = new TGeoHMatrix(Combirail1);

  TGeoRotation rotrail2;
  TGeoRotation rotrail3;
  rotrail2.SetAngles(-61.28, 0, 0); // right
  rotrail3.SetAngles(61.28, 0, 0);  // left

  TGeoTranslation transrail2(-43.70, -23.986, 20.350); // R = 498.5; H = 11 (mm)
  TGeoTranslation transrail3(43.70, -23.986, 20.350);
  TGeoCombiTrans Combirail2(transrail2, rotrail2);
  TGeoCombiTrans Combirail3(transrail3, rotrail3);
  TGeoHMatrix* pos_rail2 = new TGeoHMatrix(Combirail2);
  TGeoHMatrix* pos_rail3 = new TGeoHMatrix(Combirail3);

  TGeoRotation rotrail4;
  TGeoRotation rotrail5;
  rotrail4.SetAngles(-43.32, 0, 0); // right
  rotrail5.SetAngles(43.32, 0, 0);  // left

  TGeoTranslation transrail4(-34.240, -36.23, 20.350); // R = 498.5
  TGeoTranslation transrail5(34.240, -36.23, 20.350);
  TGeoCombiTrans Combirail4(transrail4, rotrail4);
  TGeoCombiTrans Combirail5(transrail5, rotrail5);
  TGeoHMatrix* pos_rail4 = new TGeoHMatrix(Combirail4);
  TGeoHMatrix* pos_rail5 = new TGeoHMatrix(Combirail5);
  // rotating 2nd Box

  TGeoRotation rotbox0;
  rotbox0.SetAngles(180, 180, 0);
  TGeoTranslation transbox0(-45.30, -19.85, 20.35);
  TGeoCombiTrans Combibox0(transbox0, rotbox0);
  TGeoHMatrix* pos_box0 = new TGeoHMatrix(Combibox0);

  // rails 2nd cyl

  TGeoVolume* barrel_rail02 =
    gGeoManager->MakeBox("RailBox02", mCarbon, 0.5, 0.45, 60.4750);
  TGeoVolume* cylrail2 =
    gGeoManager->MakeTubs("cylrail2", mCarbon, 0, 0.20, 60.4750, 0, 180);
  TGeoVolume* barrel_rail12 =
    gGeoManager->MakeBox("RailBox12", mCarbon, 0.2, 0.45, 60.4750);
  TGeoCompositeShape* com_rail12;
  TGeoTranslation* transcyl2 = new TGeoTranslation(0, 0.45, 0);
  transcyl2->SetName("transcyl2");
  transcyl2->RegisterYourself();
  com_rail12 = new TGeoCompositeShape("Composite_rail1",
                                      "RailBox12 + cylrail2:transcyl2");
  TGeoVolume* comp_volrail02 =
    new TGeoVolume("Comp_rail0_vol2", com_rail12, mCarbon);
  TGeoVolume* BoxCavity02 =
    gGeoManager->MakeBox("BoxCavity02", mCarbon, 0.50, 2.50, 60.4750);
  TGeoVolume* BoxCavity12 =
    gGeoManager->MakeBox("BoxCavity12", mCarbon, 0.40, 2.30, 60.4750);
  TGeoVolume* BoxCavity22 =
    gGeoManager->MakeBox("BoxCavity22", mCarbon, 0.50, 0.10, 60.4750);
  TGeoVolume* Joincyl02 = gGeoManager->MakeTubs("Joincyl02", mCarbon, 0.7500,
                                                0.95, 60.4750, 105.4660, 180);
  TGeoVolume* Joincyl12 = gGeoManager->MakeTubs("Joincyl12", mCarbon, 10.000,
                                                10.20, 60.4750, 0.0, 5.1);
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
  com_box02 = new TGeoCompositeShape(
    "CompBox2",
    "BoxCavity02 - BoxCavity12 - BoxCavity22:cstr2 + "
    "Joincyl02:jcyltr02 + Joincyl12:jcyltr12");

  TGeoVolume* comp_volBox2 =
    new TGeoVolume("Comp_Box_Vol02", com_box02, mCarbon);
  // definig pos_box02
  TGeoRotation rotrail02;
  TGeoRotation rotrail12;
  rotrail02.SetAngles(127.7, 0, 0);
  rotrail12.SetAngles(-127.7, 0, 0);

  TGeoTranslation transrail02(
    -39.40, -30.621, 104.425); // central rail R pos= 499.0; H= 9.0; W = 10.0
  TGeoTranslation transrail12(39.40, -30.621, 104.425);
  TGeoCombiTrans Combirail02(transrail02, rotrail02);
  TGeoCombiTrans Combirail12(transrail12, rotrail12);
  TGeoHMatrix* pos_rail02 = new TGeoHMatrix(Combirail02);
  TGeoHMatrix* pos_rail12 = new TGeoHMatrix(Combirail12);

  TGeoRotation rotrail22;
  TGeoRotation rotrail32;
  rotrail22.SetAngles(-4.62, 0, 0);
  rotrail32.SetAngles(4.62, 0, 0);

  TGeoTranslation transrail22(-43.70, -23.986, 104.425);
  TGeoTranslation transrail32(43.70, -23.986, 104.425);
  TGeoCombiTrans Combirail22(transrail22, rotrail2);
  TGeoCombiTrans Combirail32(transrail32, rotrail3);
  TGeoHMatrix* pos_rail22 = new TGeoHMatrix(Combirail22);
  TGeoHMatrix* pos_rail32 = new TGeoHMatrix(Combirail32);

  TGeoRotation rotrail42;
  TGeoRotation rotrail52;
  rotrail42.SetAngles(-46.3, 0, 0); // right
  rotrail52.SetAngles(46.3, 0, 0);  // left

  TGeoTranslation transrail42(-34.240, -36.23, 104.425);
  TGeoTranslation transrail52(34.240, -36.23, 104.425);
  TGeoCombiTrans Combirail42(transrail42, rotrail42);
  TGeoCombiTrans Combirail52(transrail52, rotrail52);
  TGeoHMatrix* pos_rail42 = new TGeoHMatrix(Combirail42);
  TGeoHMatrix* pos_rail52 = new TGeoHMatrix(Combirail52);
  // rotating 2nd Box

  TGeoRotation rotbox02;
  rotbox02.SetAngles(180, 180, 0);
  TGeoTranslation transbox02(-45.30, -19.85, 104.425);
  TGeoCombiTrans Combibox02(transbox02, rotbox02);
  TGeoHMatrix* pos_box02 = new TGeoHMatrix(Combibox02);

  // defining pipes
  TGeoVolume* BarrelPipes =
    gGeoManager->MakeTube("Barrel_Pipes", kMeAl, 0.3, 0.4, 82.375);
  TGeoVolume* ConePipes =
    gGeoManager->MakeTube("Cone_Pipes", kMeAl, 0.3, 0.4, 82.25);
  TGeoVolume* PipeBox0 =
    gGeoManager->MakeBox("PipeBoxOut", kMeAl, 2.95, 0.45, 82.375);
  TGeoVolume* PipeBox1 =
    gGeoManager->MakeBox("PipeBoxInn", kMeAl, 2.85, 0.35, 82.375);
  TGeoCompositeShape* ParallelPipeBox =
    new TGeoCompositeShape("ParallelPipeBox", "PipeBoxOut - PipeBoxInn");
  TGeoVolume* PipeBox = new TGeoVolume("PipeBox", ParallelPipeBox, kMeAl);

  // rotation+translation pipes
  TGeoRotation rotpipe0;
  rotpipe0.SetAngles(-47.3, 0, 0);
  TGeoTranslation transpipe0(-36.040, -33.275, 82.375);
  TGeoTranslation transpipe1(36.55, -33.520, 82.375);
  TGeoCombiTrans Combipipe0(transpipe0, rotpipe0);
  TGeoCombiTrans Combipipe1(transpipe1, rotpipe0);
  TGeoHMatrix* pos_pipe0 = new TGeoHMatrix(Combipipe0);
  TGeoHMatrix* pos_pipe1 = new TGeoHMatrix(Combipipe1);

  TGeoRotation rotpiper;
  rotpiper.SetAngles(47.3, 0, 0);
  TGeoTranslation transpipe2(36.040, -33.275, 82.375);
  TGeoTranslation transpipe3(34.7, -35.85972, 82.375);
  TGeoCombiTrans Combipipe2(transpipe2, rotpiper);
  TGeoCombiTrans Combipipe3(transpipe3, rotpiper);
  TGeoHMatrix* pos_pipe2 = new TGeoHMatrix(Combipipe2);
  TGeoHMatrix* pos_pipe3 = new TGeoHMatrix(Combipipe3);

  // adding nodes

  BarrelVolume->AddNode(BarrelJoint0, 1);
  BarrelVolume->AddNode(BarrelJoint1, 1, new TGeoTranslation(0.0, 0.0, -0.525));
  BarrelVolume->AddNode(BoxJoint0, 1,
                        new TGeoTranslation(44.5375, -20.08, 0.525));
  BarrelVolume->AddNode(BoxJoint1, 1,
                        new TGeoTranslation(45.05, -20.08, -0.535));
  BarrelVolume->AddNode(BoxJoint0, 1,
                        new TGeoTranslation(-44.5375, -20.08, 0.525));
  BarrelVolume->AddNode(BoxJoint1, 1,
                        new TGeoTranslation(-45.05, -20.08, -0.525));
  BarrelVolume->AddNode(BarrelTube0, 1, new TGeoTranslation(0.0, 0.0, 20.325));
  BarrelVolume->AddNode(BarrelJoint0, 1, new TGeoTranslation(0.0, 0.0, 40.65));
  BarrelVolume->AddNode(BarrelJoint1, 1,
                        new TGeoTranslation(0.0, 0.0, 41.1875));
  BarrelVolume->AddNode(BoxJoint0, 1,
                        new TGeoTranslation(44.5375, -20.08, 40.125));
  BarrelVolume->AddNode(BoxJoint1, 1,
                        new TGeoTranslation(45.05, -20.08, 41.1875));
  BarrelVolume->AddNode(BoxJoint0, 1,
                        new TGeoTranslation(-44.5375, -20.08, 40.125));
  BarrelVolume->AddNode(BoxJoint1, 1,
                        new TGeoTranslation(-45.05, -20.08, 41.1875));

  BarrelVolume->AddNode(barrel_rail0, 2, pos_rail0);
  BarrelVolume->AddNode(barrel_rail0, 2, pos_rail1);
  BarrelVolume->AddNode(comp_volrail0, 2, pos_rail2);
  BarrelVolume->AddNode(comp_volrail0, 2, pos_rail3);
  BarrelVolume->AddNode(comp_volrail0, 2, pos_rail4);
  BarrelVolume->AddNode(comp_volrail0, 2, pos_rail5);
  BarrelVolume->AddNode(comp_volBox, 2, pos_box0);
  BarrelVolume->AddNode(comp_volBox, 2,
                        new TGeoTranslation(45.30, -19.84, 20.35));
  // start of the 2nd cyl
  BarrelVolume->AddNode(BarrelJoint0, 1, new TGeoTranslation(0.0, 0.0, 42.80));
  BarrelVolume->AddNode(BarrelJoint1, 1, new TGeoTranslation(0.0, 0.0, 42.250));
  BarrelVolume->AddNode(BoxJoint0, 1,
                        new TGeoTranslation(44.537, -20.08, 43.325));
  BarrelVolume->AddNode(BoxJoint1, 1,
                        new TGeoTranslation(45.05, -20.08, 42.275));
  BarrelVolume->AddNode(BoxJoint0, 1,
                        new TGeoTranslation(-44.537, -20.08, 43.325));
  BarrelVolume->AddNode(BoxJoint1, 1,
                        new TGeoTranslation(-45.05, -20.08, 42.275));
  BarrelVolume->AddNode(BarrelTube1, 1, new TGeoTranslation(0.0, 0.0, 103.8));
  // adding rails 2nd cyl
  BarrelVolume->AddNode(barrel_rail02, 2, pos_rail02);
  BarrelVolume->AddNode(barrel_rail02, 2, pos_rail12);
  BarrelVolume->AddNode(comp_volrail02, 2, pos_rail22);
  BarrelVolume->AddNode(comp_volrail02, 2, pos_rail32);
  BarrelVolume->AddNode(comp_volrail02, 2, pos_rail42);
  BarrelVolume->AddNode(comp_volrail02, 2, pos_rail52);
  BarrelVolume->AddNode(comp_volBox2, 2, pos_box02);
  BarrelVolume->AddNode(comp_volBox2, 2,
                        new TGeoTranslation(45.30, -19.84, 104.325));
  // adding pipes R=490.8
  BarrelVolume->AddNode(BarrelPipes, 1,
                        new TGeoTranslation(-42.67, -24.25113, 82.375));
  BarrelVolume->AddNode(BarrelPipes, 1,
                        new TGeoTranslation(-42.07, -25.27769, 82.375));
  BarrelVolume->AddNode(BarrelPipes, 1,
                        new TGeoTranslation(-41.44, -26.29777, 82.375));
  BarrelVolume->AddNode(BarrelPipes, 1,
                        new TGeoTranslation(-40.78, -27.31000, 82.375));
  BarrelVolume->AddNode(BarrelPipes, 1,
                        new TGeoTranslation(-40.11, -28.28487, 82.375));
  BarrelVolume->AddNode(BarrelPipes, 1,
                        new TGeoTranslation(-39.41, -29.25232, 82.375));
  BarrelVolume->AddNode(BarrelPipes, 1,
                        new TGeoTranslation(-38.25, -30.753, 82.375));
  BarrelVolume->AddNode(BarrelPipes, 1,
                        new TGeoTranslation(42.67, -24.25113, 82.375));
  BarrelVolume->AddNode(BarrelPipes, 1,
                        new TGeoTranslation(42.07, -25.27769, 82.375));
  BarrelVolume->AddNode(BarrelPipes, 1,
                        new TGeoTranslation(41.44, -26.29777, 82.375));
  BarrelVolume->AddNode(BarrelPipes, 1,
                        new TGeoTranslation(40.78, -27.31000, 82.375));
  BarrelVolume->AddNode(BarrelPipes, 1,
                        new TGeoTranslation(40.11, -28.28487, 82.375));
  BarrelVolume->AddNode(BarrelPipes, 1,
                        new TGeoTranslation(39.41, -29.25232, 82.375));
  BarrelVolume->AddNode(BarrelPipes, 1,
                        new TGeoTranslation(38.25, -30.753, 82.375));
  BarrelVolume->AddNode(PipeBox, 1, pos_pipe0);
  BarrelVolume->AddNode(PipeBox, 1, pos_pipe2);

  // set Wires
  Float_t radiia = 0.0375; // case 1
  Float_t radiib = 0.0205; // case 2
  Float_t radiic = 0.0415; // case 3
  Float_t w_pos = 50.35;
  Int_t nwiresa = 19;
  Int_t nwiresb = 4;
  Int_t nwiresc = 3;
  Int_t wires_array_pos[26] = { 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2 };

  // defining wires shapes
  TGeoVolume* BarrelWiresA =
    gGeoManager->MakeTube("Barrel_wiresa", kMeAl, 0.0, radiia, 81.225);
  TGeoVolume* BarrelWiresB =
    gGeoManager->MakeTube("Barrel_wiresb", kMeAl, 0.0, radiib, 81.225);
  TGeoVolume* BarrelWiresC =
    gGeoManager->MakeTube("Barrel_wiresc", kMeAl, 0.0, radiic, 81.225);
  // units cm

  Float_t xPosIni = 43.415;
  for (Int_t k = 0; k < 26; k++) {
    switch (wires_array_pos[k]) {
      case 1: {
        xPosIni = xPosIni - radiia;
        Float_t yPosIni = sqrt(50.35 * 50.35 - xPosIni * xPosIni);
        BarrelVolume->AddNode(
          BarrelWiresA, 1,
          new TGeoTranslation(xPosIni, -1.0 * yPosIni, 81.325));
        BarrelVolume->AddNode(
          BarrelWiresA, 1,
          new TGeoTranslation(-xPosIni, -1.0 * yPosIni, 81.325));
        xPosIni = xPosIni - 0.02;
        break;
      }
      case 2: {
        xPosIni = xPosIni - radiib;
        Float_t yPosIni = sqrt(50.35 * 50.35 - xPosIni * xPosIni);
        BarrelVolume->AddNode(
          BarrelWiresB, 1,
          new TGeoTranslation(xPosIni, -1.0 * yPosIni, 81.325));
        BarrelVolume->AddNode(
          BarrelWiresB, 1,
          new TGeoTranslation(-xPosIni, -1.0 * yPosIni, 81.325));
        xPosIni = xPosIni - 0.02;
        break;
      }

      case 3: {
        xPosIni = xPosIni - radiic;
        Float_t yPosIni = sqrt(50.35 * 50.35 - xPosIni * xPosIni);
        BarrelVolume->AddNode(
          BarrelWiresC, 1,
          new TGeoTranslation(xPosIni, -1.0 * yPosIni, 81.325));
        BarrelVolume->AddNode(
          BarrelWiresC, 1,
          new TGeoTranslation(-xPosIni, -1.0 * yPosIni, 81.325));
        xPosIni = xPosIni - 0.02;
        break;
      }
    }
  }

  // kapton KaptonFoil
  TGeoVolume* KaptonFoil0 = gGeoManager->MakeTubs(
    "Kapton_Foil0", mKapton, 50.301, 50.311, 81.325, 210.46, 212.58);
  TGeoVolume* KaptonFoil1 = gGeoManager->MakeTubs(
    "Kapton_Foil1", mKapton, 50.301, 50.311, 81.325, 327.42, 329.54);
  // coloring
  KaptonFoil0->SetLineColor(kSpring - 4);
  KaptonFoil1->SetLineColor(kSpring - 4);
  // adding nodes
  BarrelVolume->AddNode(KaptonFoil0, 1, new TGeoTranslation(0.0, 0.01, 81.875));
  BarrelVolume->AddNode(KaptonFoil1, 1, new TGeoTranslation(0.0, 0.01, 81.875));

  // fixation services
  TGeoVolume* FixService0 = gGeoManager->MakeTubs("FixService0", kMeAl, 49.40,
                                                  49.60, 2.00, 314.15, 332.78);
  TGeoVolume* FixService1 = gGeoManager->MakeTubs("FixService1", kMeAl, 49.40,
                                                  49.60, 2.00, 207.55, 226.18);
  // coloring
  FixService0->SetLineColor(kRed - 9);
  FixService1->SetLineColor(kRed - 9);
  // adding nodes

  BarrelVolume->AddNode(FixService0, 1, new TGeoTranslation(-2.00, 0.0, 21.2));
  BarrelVolume->AddNode(FixService1, 1, new TGeoTranslation(2.00, 0.0, 21.2));
  BarrelVolume->AddNode(FixService0, 1, new TGeoTranslation(-2.00, 0.0, 32.7));
  BarrelVolume->AddNode(FixService1, 1, new TGeoTranslation(2.00, 0.0, 32.7));
  BarrelVolume->AddNode(FixService0, 1, new TGeoTranslation(-2.00, 0.0, 44.2));
  BarrelVolume->AddNode(FixService1, 1, new TGeoTranslation(2.00, 0.0, 44.2));
  BarrelVolume->AddNode(FixService0, 1,
                        new TGeoTranslation(-2.00, 0.0, 52.325));
  BarrelVolume->AddNode(FixService1, 1, new TGeoTranslation(2.00, 0.0, 52.325));
  BarrelVolume->AddNode(FixService0, 1,
                        new TGeoTranslation(-2.00, 0.0, 76.325));
  BarrelVolume->AddNode(FixService1, 1, new TGeoTranslation(2.00, 0.0, 76.325));
  BarrelVolume->AddNode(FixService0, 1,
                        new TGeoTranslation(-2.00, 0.0, 100.325));
  BarrelVolume->AddNode(FixService1, 1,
                        new TGeoTranslation(2.00, 0.0, 100.325));
  BarrelVolume->AddNode(FixService0, 1,
                        new TGeoTranslation(-2.00, 0.0, 124.325));
  BarrelVolume->AddNode(FixService1, 1,
                        new TGeoTranslation(2.00, 0.0, 124.325));
  BarrelVolume->AddNode(FixService0, 1,
                        new TGeoTranslation(-2.00, 0.0, 148.325));
  BarrelVolume->AddNode(FixService1, 1,
                        new TGeoTranslation(2.00, 0.0, 148.325));
  // pipes cone

  return BarrelVolume;
}
