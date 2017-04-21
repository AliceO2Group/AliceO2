/// \file HalfCone.cxx
/// \brief Class building geometry of one half of one MFT half-cone
/// \author sbest@pucp.pe, eric.endress@gmx.de, franck.manso@clermont.in2p3.fr
/// \date 15/12/2016

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

#include "MFTBase/HalfCone.h"

using namespace o2::MFT;

ClassImp(o2::MFT::HalfCone)

//_____________________________________________________________________________
HalfCone::HalfCone():
TNamed(),
mHalfCone(nullptr)
{
  
  // default constructor
  
}

//_____________________________________________________________________________
HalfCone::~HalfCone() 
= default;

//_____________________________________________________________________________
TGeoVolumeAssembly* HalfCone::createHalfCone(Int_t half)
{
  
  auto *HalfConeVolume = new TGeoVolumeAssembly("HalfConeVolume");
  
  //Left Support Rail
  auto *Half_0 = new TGeoVolumeAssembly("Half_0");
  
  TGeoMedium *kMedAlu = gGeoManager->GetMedium("MFT_Alu$");

  //Dimensions

  //Lower Piece (6 holes)
  //Float_t Lower_x = 2.0;  // to avoid overlap with disks, fm, largeur des barres horizontales
  Float_t Lower_x = 1.6;
  Float_t Lower_y = 1.15;
  Float_t Lower_z = 10.7;

  //Middle Piece
  //Float_t Middle_x = 2.0; // to avoid overlap with disks, fm
  Float_t Middle_x = 1.6;
  Float_t Middle_y = 1.15+0.00001;
  Float_t Middle_z = 14.45;

  //Upper Piece (8 holes)
  //Float_t Upper_x = 2.0; // to avoid overlap with disks, fm
  Float_t Upper_x = 0.2;
  Float_t Upper_y = 1.15;
  Float_t Upper_z = 15.291;

  //Upper piece cut
  Float_t Upper_cut_x = 4.0;
  Float_t Upper_cut_y = 1.15;
  Float_t Upper_cut_z = 3.0;

  //Lower piece cut
  Float_t Lower_cut_x = 2.0;
  Float_t Lower_cut_y = 1.15+0.0001;
  Float_t Lower_cut_z = 0.6;

  //Trapezoid
  Float_t trap_xmax =2.2;
  Float_t trap_xmin =1.2;
  Float_t trap_y =1.15;
  Float_t trap_z =1.0;

  //Angle of the Middle Piece
  auto *MiddleAngle = new TGeoRotation("MiddleAngle",90.,40.,90.);

  //X distance from the center
  Float_t z_Upper_distance = (Upper_z + 9.759)/2.;  //Upper z-length and the distance from Upper to Lower borders
  Float_t z_Lower_distance = -(Lower_z + 9.759)/2.;     //Lower z-length and the distance from Upper to Lower borders
  //Distance from upper to lower piece
  Float_t UpperLowerDistance = 8.8;

  //Upper center Distance
  auto *TUpper = new TGeoTranslation("TUpper",UpperLowerDistance/2.,0.,z_Upper_distance);
  //Lower center Distance
  auto *TLower = new TGeoTranslation("TLower",-UpperLowerDistance/2.,0.,z_Lower_distance);
  //Lower cut distance coordinates
  auto *TCutUpper = new TGeoTranslation("TCutUpper",4.0,.8+0.000001,(z_Upper_distance+Upper_z/2.)+(Upper_cut_z/2.)-1.2);
  //Upper cut distance coordinates
  auto *TCutLower = new TGeoTranslation("TCutLower",-4.4-1.5,0.,(z_Lower_distance+Lower_z/2.)-(Lower_cut_z/2.) );
  //Trapezoid center Distance (the trapezoid for the upper piece is defined in two parts)
  auto *TTrap_0 = new TGeoTranslation("TTrap_0",2.9,0.,z_Upper_distance+Upper_z/2.-(trap_xmax/2.));
  auto *TTrap_1 = new TGeoTranslation("TTrap_1",2.9,0.,z_Upper_distance+Upper_z/2.-(trap_xmax+trap_xmin)/4.);

  //Rotations
  auto *Ry90 = new TGeoRotation("Ry90",90.,90.,90.);

  //Combined transformations for the trapezoid
  auto *cTrap_0 = new TGeoCombiTrans(*TTrap_0, *Ry90);
  auto *cTrap_1 = new TGeoCombiTrans(*TTrap_1, *Ry90);

  MiddleAngle->RegisterYourself();
  TUpper->RegisterYourself();
  TLower->RegisterYourself();
  TCutUpper->RegisterYourself();
  TCutLower->RegisterYourself();
  TTrap_0->RegisterYourself();
  TTrap_1->RegisterYourself();
  Ry90->RegisterYourself();
  cTrap_0->SetName("cTrap_0");
  cTrap_0->RegisterYourself();
  cTrap_1->SetName("cTrap_1");
  cTrap_1->RegisterYourself();

  //Basic Forms for Half_0
  TGeoShape *Half_0_Upper = new TGeoBBox("Half_0_Upper", Upper_x/2., Upper_y/2., Upper_z/2.);
  TGeoShape *Half_0_Lower = new TGeoBBox("Half_0_Lower", Lower_x/2., Lower_y/2., Lower_z/2.);
  TGeoShape *Half_0_Middle = new TGeoBBox("Half_0_Middle", Middle_x/2., Middle_y/2., Middle_z/2.);
  TGeoShape *Half_0_Trap_0 = new TGeoTrd1("Half_0_Trap_0", trap_xmin/2.,trap_xmax/2.,trap_y/2.,trap_z/2.);
  TGeoShape *Half_0_Trap_1 = new TGeoBBox("Half_0_Trap_1", (trap_xmax+trap_xmin)/4.+0.00001,trap_y/2.+0.00001,trap_z/2.+0.000001);
  TGeoShape *Half_0_UpperCut= new TGeoBBox("Half_0_UpperCut", Upper_cut_x/2., Upper_cut_y/2., Upper_cut_z/2.);
  TGeoShape *Half_0_LowerCut = new TGeoBBox("Half_0_LowerCut", Lower_cut_x/2., Lower_cut_y/2., Lower_cut_z/2.);

  //Composite shapes for Half_0
  auto * Half_0_Shape_0 = new TGeoCompositeShape("Half_0_Shape_0","Half_0_Middle:MiddleAngle+Half_0_Lower:TLower+Half_0_Upper:TUpper");
  auto * Half_0_Shape_1 = new TGeoCompositeShape("Half_0_Shape_1","Half_0_Shape_0+Half_0_Trap_0:cTrap_0+Half_0_Trap_1:cTrap_1");
  auto * Half_0_Shape_2 = new TGeoCompositeShape("Half_0_Shape_2","Half_0_Shape_1-(Half_0_UpperCut:TCutUpper+Half_0_LowerCut:TCutLower)");
  /*
  //Holes

  Float_t hole_distance = 250.5-152.91;

  TGeoRotation *Rx90 = new TGeoRotation("Rx90",0.,90.,0.);
  TGeoTranslation *THalf_0_Hole0 = new TGeoTranslation("THalf_0_Hole0",0.,0.,hole_distance-6.);
  TGeoTranslation *THalf_0_Hole1 = new TGeoTranslation("THalf_0_Hole1",0.,0.,hole_distance-6.);
  TGeoTranslation *THalf_0_Hole2 = new TGeoTranslation("THalf_0_Hole2",0.,0.,hole_distance-32.);
  TGeoTranslation *THalf_0_Hole3 = new TGeoTranslation("THalf_0_Hole3",0.,0.,hole_distance-32.-13.5);
  TGeoTranslation *THalf_0_Hole4 = new TGeoTranslation("THalf_0_Hole4",0.,0.,hole_distance-74.);
  TGeoTranslation *THalf_0_Hole5 = new TGeoTranslation("THalf_0_Hole5",0.,0.,hole_distance-74.-13.5);
  TGeoTranslation *THalf_0_Hole6 = new TGeoTranslation("THalf_0_Hole6",0.,0.,hole_distance-116.);
  TGeoTranslation *THalf_0_Hole7 = new TGeoTranslation("THalf_0_Hole7",0.,0.,hole_distance-116.-13.5);

  TGeoTranslation *THalf_0_Hole8 = new TGeoTranslation("THalf_0_Hole8",0.,0.,0.);
  TGeoTranslation *THalf_0_Hole9 = new TGeoTranslation("THalf_0_Hole9",0.,0.,0.);
  TGeoTranslation *THalf_0_Hole10 = new TGeoTranslation("THalf_0_Hole10",0.,0.,0.);
  TGeoTranslation *THalf_0_Hole11 = new TGeoTranslation("THalf_0_Hole11",0.,0.,0.);
  TGeoTranslation *THalf_0_Hole12 = new TGeoTranslation("THalf_0_Hole12",0.,0.,0.);
  TGeoTranslation *THalf_0_Hole13 = new TGeoTranslation("THalf_0_Hole13",0.,0.,0.);

  TGeoCombiTrans *cHalf_0_Hole0 = new TGeoCombiTrans(*THalf_0_Hole0, *Rx90);
  TGeoCombiTrans *cHalf_0_Hole1 = new TGeoCombiTrans(*THalf_0_Hole1, *Rx90);
  TGeoCombiTrans *cHalf_0_Hole2 = new TGeoCombiTrans(*THalf_0_Hole2, *Rx90);
  TGeoCombiTrans *cHalf_0_Hole3 = new TGeoCombiTrans(*THalf_0_Hole3, *Rx90);
  TGeoCombiTrans *cHalf_0_Hole4 = new TGeoCombiTrans(*THalf_0_Hole4, *Rx90);
  TGeoCombiTrans *cHalf_0_Hole5 = new TGeoCombiTrans(*THalf_0_Hole5, *Rx90);
  TGeoCombiTrans *cHalf_0_Hole6 = new TGeoCombiTrans(*THalf_0_Hole6, *Rx90);
  TGeoCombiTrans *cHalf_0_Hole7 = new TGeoCombiTrans(*THalf_0_Hole7, *Rx90);
  TGeoCombiTrans *cHalf_0_Hole8 = new TGeoCombiTrans(*THalf_0_Hole8, *Rx90);
  TGeoCombiTrans *cHalf_0_Hole9 = new TGeoCombiTrans(*THalf_0_Hole9, *Rx90);
  TGeoCombiTrans *cHalf_0_Hole10 = new TGeoCombiTrans(*THalf_0_Hole10, *Rx90);
  TGeoCombiTrans *cHalf_0_Hole11 = new TGeoCombiTrans(*THalf_0_Hole11, *Rx90);
  TGeoCombiTrans *cHalf_0_Hole12 = new TGeoCombiTrans(*THalf_0_Hole12, *Rx90);
  TGeoCombiTrans *cHalf_0_Hole13 = new TGeoCombiTrans(*THalf_0_Hole13, *Rx90);

  cHalf_0_Hole0->SetName("cHalf_0_Hole0");
  cHalf_0_Hole1->SetName("cHalf_0_Hole1");
  cHalf_0_Hole2->SetName("cHalf_0_Hole2");
  cHalf_0_Hole3->SetName("cHalf_0_Hole3");
  cHalf_0_Hole4->SetName("cHalf_0_Hole4");
  cHalf_0_Hole5->SetName("cHalf_0_Hole5");
  cHalf_0_Hole6->SetName("cHalf_0_Hole6");
  cHalf_0_Hole7->SetName("cHalf_0_Hole7");
  cHalf_0_Hole8->SetName("cHalf_0_Hole8");
  cHalf_0_Hole9->SetName("cHalf_0_Hole9");
  cHalf_0_Hole10->SetName("cHalf_0_Hole10");
  cHalf_0_Hole11->SetName("cHalf_0_Hole11");
  cHalf_0_Hole12->SetName("cHalf_0_Hole12");
  cHalf_0_Hole13->SetName("cHalf_0_Hole13");

  cHalf_0_Hole0->RegisterYourself();
  cHalf_0_Hole1->RegisterYourself();
  cHalf_0_Hole2->RegisterYourself();
  cHalf_0_Hole3->RegisterYourself();
  cHalf_0_Hole4->RegisterYourself();
  cHalf_0_Hole5->RegisterYourself();
  cHalf_0_Hole6->RegisterYourself();
  cHalf_0_Hole7->RegisterYourself();
  cHalf_0_Hole8->RegisterYourself();
  cHalf_0_Hole9->RegisterYourself();
  cHalf_0_Hole10->RegisterYourself();
  cHalf_0_Hole11->RegisterYourself();
  cHalf_0_Hole12->RegisterYourself();
  cHalf_0_Hole13->RegisterYourself();

  Float_t Half_0_r0 = 4./2.;
  Float_t Half_0_r1 = 5./2.;
  Float_t Half_0_r2 = 6./2.;
  Float_t Half_0_r3 = 6.5/2.;

  TGeoShape * Half_0_Hole0 = new TGeoTube("Half_0_Hole0",0.,Half_0_r3,Upper_y+0.00001);
  TGeoShape * Half_0_Hole1 = new TGeoTube("Half_0_Hole1",0.,Half_0_r2,Upper_y+0.00001);
  TGeoShape * Half_0_Hole2 = new TGeoTube("Half_0_Hole2",0.,Half_0_r1,Upper_y+0.00001);
  TGeoShape * Half_0_Hole3 = new TGeoTube("Half_0_Hole3",0.,Half_0_r0,Upper_y+0.00001);
  TGeoShape * Half_0_Hole4 = new TGeoTube("Half_0_Hole4",0.,Half_0_r1,Upper_y+0.00001);
  TGeoShape * Half_0_Hole5 = new TGeoTube("Half_0_Hole5",0.,Half_0_r0,Upper_y+0.00001);
  TGeoShape * Half_0_Hole6 = new TGeoTube("Half_0_Hole6",0.,Half_0_r1,Upper_y+0.00001);
  TGeoShape * Half_0_Hole7 = new TGeoTube("Half_0_Hole7",0.,Half_0_r0,Upper_y+0.00001);

  TGeoShape * Half_0_Hole8 = new TGeoTube("Half_
0_Hole8",0.,Half_0_r1,Lower_y+0.00001);
  TGeoShape * Half_0_Hole9 = new TGeoTube("Half_0_Hole9",0.,Half_0_r0,Lower_y+0.00001);
  TGeoShape * Half_0_Hole10 = new TGeoTube("Half_0_Hole10",0.,Half_0_r1,Lower_y+0.00001);
  TGeoShape * Half_0_Hole11 = new TGeoTube("Half_0_Hole11",0.,Half_0_r0,Lower_y+0.00001);
  TGeoShape * Half_0_Hole12 = new TGeoTube("Half_0_Hole12",0.,Half_0_r1,Lower_y+0.00001);
  TGeoShape * Half_0_Hole13 = new TGeoTube("Half_0_Hole13",0.,Half_0_r0,Lower_y+0.00001);

  TGeoCompositeShape * Half_0_Holes = new TGeoCompositeShape("Half_0_Holes","Half_0_Hole0:cHalf_0_Hole0+Half_0_Hole1:cHalf_0_Hole1+Half_0_Hole2:cHalf_0_Hole2+Half_0_Hole3:cHalf_0_Hole3+Half_0_Hole4:cHalf_0_Hole4+Half_0_Hole5:cHalf_0_Hole5+Half_0_Hole6:cHalf_0_Hole6+Half_0_Hole7:cHalf_0_Hole7+Half_0_Hole8:cHalf_0_Hole8+Half_0_Hole9:cHalf_0_Hole9+Half_0_Hole10:cHalf_0_Hole10+Half_0_Hole11:cHalf_0_Hole11+Half_0_Hole12:cHalf_0_Hole12+Half_0_Hole13:cHalf_0_Hole13");
  TGeoCompositeShape * Half_0_Shape_3 = new TGeoCompositeShape("Half_0_Shape_3","Half_0_Shape_2+Half_0_Holes");
  */
  auto * Half_0_Volume = new TGeoVolume("Half_0_Volume",Half_0_Shape_2,kMedAlu);
  //Position of the piece relative to the origin which for this code is the center of the the Framework piece (See Half_2)
  Half_0->AddNode(Half_0_Volume,1,new TGeoTranslation(25.6-5.4-.05,-1.15/2. + 7.3+.35,-34.55/2.-4.591/2.-.25 ));

  //Right Support Rail (the distances and rotations are the same, just mirrored with respect to the yz-plane)
  //See the definition of the distances in the Half_0 piece
  auto *Half_1 = new TGeoVolumeAssembly("Half_1");

  //Angle of the Middle Piece
  auto *MiddleAngle_inv = new TGeoRotation("MiddleAngle_inv",90.,-40.,90.);

  //X distance from the center
  Float_t z_Upper_distance_inv = -(Upper_z + 9.759)/2.;  //Upper x-length and the distance from Upper to Lower borders
  Float_t z_Lower_distance_inv = (Lower_z + 9.759)/2.;     //Lower x-length and the distance from Upper to Lower borders

  //Upper center Distance
  auto *TUpper_inv = new TGeoTranslation("TUpper_inv",-4.4,0.,z_Upper_distance);
  //Lower center Distance
  auto *TLower_inv = new TGeoTranslation("TLower_inv",4.4,0.,z_Lower_distance);
  //Lower cut distance
  auto *TCutUpper_inv = new TGeoTranslation("TCutUpper_inv",-4.0,.8+0.000001,(z_Upper_distance+Upper_z/2.)+(Upper_cut_z/2.)-1.2);
  //Upper cut distance
  auto *TCutLower_inv = new TGeoTranslation("TCutLower_inv",4.4+1.5,0.,(z_Lower_distance+Lower_z/2.)-(Lower_cut_z/2.) );
  //Trapezoid center Distance (the trapezoid for the upper piece is defined in two parts)
  auto *TTrap_0_inv = new TGeoTranslation("TTrap_0_inv",-2.9,0.,z_Upper_distance+Upper_z/2.-(trap_xmax/2.));
  auto *TTrap_1_inv = new TGeoTranslation("TTrap_1_inv",-2.9,0.,z_Upper_distance+Upper_z/2.-(trap_xmax+trap_xmin)/4.);

  //Rotations
  auto *Ry90_inv = new TGeoRotation("Ry90_inv",90.,-90.,90.);

  //Combined transformations for the trapezoid
  auto *cTrap_0_inv = new TGeoCombiTrans(*TTrap_0_inv, *Ry90_inv);
  auto *cTrap_1_inv = new TGeoCombiTrans(*TTrap_1_inv, *Ry90_inv);

  MiddleAngle_inv->RegisterYourself();
  TUpper_inv->RegisterYourself();
  TLower_inv->RegisterYourself();
  TCutUpper_inv->RegisterYourself();
  TCutLower_inv->RegisterYourself();
  TTrap_0_inv->RegisterYourself();
  TTrap_1_inv->RegisterYourself();
  Ry90_inv->RegisterYourself();
  cTrap_0_inv->SetName("cTrap_0_inv");
  cTrap_0_inv->RegisterYourself();
  cTrap_1_inv->SetName("cTrap_1_inv");
  cTrap_1_inv->RegisterYourself();

  //Composite shapes for Half_1
  auto * Half_1_Shape_0 = new TGeoCompositeShape("Half_1_Shape_0","Half_0_Middle:MiddleAngle_inv+Half_0_Lower:TLower_inv+Half_0_Upper:TUpper_inv");
  auto * Half_1_Shape_1 = new TGeoCompositeShape("Half_1_Shape_1","Half_1_Shape_0+Half_0_Trap_0:cTrap_0_inv+Half_0_Trap_1:cTrap_1_inv");
  auto * Half_1_Shape_2 = new TGeoCompositeShape("Half_1_Shape_2","Half_1_Shape_1-(Half_0_UpperCut:TCutUpper_inv+Half_0_LowerCut:TCutLower_inv)");

  auto * Half_1_Volume = new TGeoVolume("Half_1_Volume",Half_1_Shape_2,kMedAlu);
  //Position of the piece relative to the origin which for this code is the center of the the Framework piece (See Half_2)
  Half_1->AddNode(Half_1_Volume,1,new TGeoTranslation(-25.6+5.4+.05,-1.15/2. + 7.3+.35,-34.55/2.-4.591/2.-.25 ));

  //Framework
  auto *Half_2 = new TGeoVolumeAssembly("Half_2");

  //Definitions
  Float_t Framework_rmin=23.6;
  Float_t Framework_rmax=30.3;
  Float_t Framework_z=.6;
  //
  Float_t Framework_Bottom_z=1.35;

  //Holes definition

  //Radii
  Float_t Hole_rmin_0=25.5;
  Float_t Hole_rmax_0=28.0;
  Float_t Hole_rmax_1=27.5;
  Float_t Hole_rmin_2=29.0;
  Float_t Hole_z=1.8;

  //Angles
  Float_t Framework_Angle = 12.3838;
  //Holes maximum and minimum angles
  Float_t angle_0_min =20.;
  Float_t angle_0_max =70.;
  Float_t angle_1_min =80.;
  Float_t angle_1_max =100.;
  Float_t angle_2_min =110.;
  Float_t angle_2_max =160.;
  Float_t angle_3_min =75.;
  Float_t angle_3_max =105.;
  //Middle step
  Float_t Step_Angle =2.; 

  //Translations
  //Distances for the cuts of the upper central hole.
  auto *tHole3 = new TGeoTranslation("tHole3",8.0,30.,0.);
  auto *tHole3m = new TGeoTranslation("tHole3m",-8.0,30.,0.);
  //Distances for the bottom and bottom limit cuts.
  auto *tLimit = new TGeoTranslation("tLimit",0.,9.3+6.5,-(Framework_z+Hole_z+2.)/2.-0.00001); //Half the y-direction of the box
  auto *tLimit_2 = new TGeoTranslation("tLimit_2",0.,9.3,(Framework_z+Hole_z+2.)/2.-0.00001); //Half the y-direction of the box
  auto *tLimitm = new TGeoTranslation("tLimitm",0.,9.3+6.5,(Framework_z+Hole_z+2.)/2.+0.00001); //Half the y-direction of the box
  auto *tBottom = new TGeoTranslation("tBottom",0.,0.,-.175);
  //Distance for the steps
  auto *tStep_0 = new TGeoTranslation("tStep_0",0.,0.,-.3);
  auto *tStep_1 = new TGeoTranslation("tStep_1",0.,0.,.15/2.);

  tHole3m->RegisterYourself();
  tHole3->RegisterYourself();
  tLimit->RegisterYourself();
  tLimit_2->RegisterYourself();
  tLimitm->RegisterYourself();
  tBottom->RegisterYourself();
  tStep_0->RegisterYourself();
  tStep_1->RegisterYourself();

  //Basic shapes for Half_2

  TGeoShape *Framework = new TGeoTubeSeg("Framework", Framework_rmin,Framework_rmax , Framework_z/2.,Framework_Angle,180.-Framework_Angle);
  //This are the elevations at the ends of the framework arc, this number (15) doesn't matter, it just has to be big enough for the cut.
  TGeoShape *Framework_Bottom_0 = new TGeoTubeSeg("Framework_Bottom_0", Framework_rmin,Framework_rmax , Framework_Bottom_z/2.,Framework_Angle,15.+Framework_Angle);
  //Same for the 165 here.
  TGeoShape *Framework_Bottom_1 = new TGeoTubeSeg("Framework_Bottom_1", Framework_rmin,Framework_rmax , Framework_Bottom_z/2.,165. - Framework_Angle,180.-Framework_Angle);
  //This is the elevation at the center of the framework arc. The extra in the z-length is the length of the steps.
  TGeoShape *Framework_Step_0 = new TGeoTubeSeg("Framework_Step_0", Hole_rmax_1, Framework_rmax , (Framework_z+.6)/2.+0.00001,90.-Step_Angle,90.+Step_Angle);
  TGeoShape *Framework_Step_1 = new TGeoTubeSeg("Framework_Step_1", Hole_rmax_1, Hole_rmin_2 , (Framework_z+.15)/2.+0.00001,90.-Step_Angle,90.+Step_Angle);
  //This three are the holes in the framework arc
  TGeoShape *Hole_Framework_0 = new TGeoTubeSeg("Hole_Framework_0", Hole_rmin_0,Hole_rmax_0 , Hole_z/2.,angle_0_min,angle_0_max);
  TGeoShape *Hole_Framework_1 = new TGeoTubeSeg("Hole_Framework_1", Hole_rmin_0,Hole_rmax_1 , Hole_z/2.,angle_1_min,angle_1_max);
  TGeoShape *Hole_Framework_2 = new TGeoTubeSeg("Hole_Framework_2", Hole_rmin_0,Hole_rmax_0 , Hole_z/2.,angle_2_min,angle_2_max);

  //This is the hole in the upper middle of the framework arc (the .005 is to make a good cut)
  TGeoShape *Hole_Framework_3 = new TGeoTubeSeg("Hole_Framework_3", Hole_rmin_2,Framework_rmax+.005 , Hole_z/2.,angle_3_min,angle_3_max);
  //The upper central hole is not just an arc, its limits are straigth lines.
  TGeoShape *Hole3_Limit= new TGeoBBox("Hole3_Limit", 2./2., 5./2., (Hole_z+0.0001)/2.);
  //This are the cuts for the bottom parts of the framework.
  TGeoShape *Framework_Limit_0= new TGeoBBox("Framework_Limit_0", 60./2., 13./2., Hole_z/2.+1.);
  TGeoShape *Framework_Limit_1= new TGeoBBox("Framework_Limit_1", 51.2/2., 14.6/2., Hole_z/2.+1.);

  //Composite shapes for Half_2
  //The first term is the framework arc and the bottom limits
  //The second term is the three other holes in the arc.
  //The third term is just the upper center hole. I also include here the limits to shape the ends of the framework arc.
  //This can be done with smaller pieces, but more of them. I don't know which is better.
  auto * Half_2_Shape_0 = new TGeoCompositeShape("Half_2_Shape_0","Hole_Framework_3-(Hole3_Limit:tHole3m+Hole3_Limit:tHole3)");
  auto * Half_2_Shape_1 = new TGeoCompositeShape("Half_2_Shape_1","(Framework+Framework_Bottom_0:tBottom+Framework_Bottom_1:tBottom) - (Hole_Framework_0+Hole_Framework_1+Hole_Framework_2)-(Half_2_Shape_0+Framework_Limit_1:tLimit_2+Framework_Limit_0:tLimit+Framework_Limit_0:tLimitm+Framework_Limit_0+Framework_Limit_1) ");
  //Add square behind the step in the middle
  auto * Half_2_Shape_2 = new TGeoCompositeShape("Half_2_Shape_2","Half_2_Shape_1+Framework_Step_0:tStep_0+Framework_Step_1:tStep_1");

  auto * Half_2_Volume = new TGeoVolume("Half_2_Volume",Half_2_Shape_2,kMedAlu);

  Half_2->AddNode(Half_2_Volume,1,new TGeoTranslation(0., 0., 0. ));

  //Shell
  //This piece is not coded exactly, 

  auto *Half_3 = new TGeoVolumeAssembly("Half_3");

  //Shell radii
  Float_t Shell_rmax = 60.6+.7;
  Float_t Shell_rmin = 37.5+.7;

  //Rotations and translations
  auto *tShell_0 = new TGeoTranslation("tShell_0",0.,0.,3.1+(25.15+1.)/2.);
  auto *tShell_1 = new TGeoTranslation("tShell_1",0.,0.,-1.6-(25.15+1.)/2.);
  auto *tShellHole = new TGeoTranslation("tShellHole",0.,0.,2./2.+(25.15+1.)/2.);
  auto *tShellHole_0 = new TGeoTranslation("tShellHole_0",0.,-6.9,-26.1/2.-6.2/2.-.1);
  auto *tShellHole_1 = new TGeoTranslation("tShellHole_1",0.,0.,-26.1/2.-6.2/2.-.1);
  auto *tShell_Cut = new TGeoTranslation("tShell_Cut",0.,25./2.,0.);
  auto *tShell_Cut_1 = new TGeoTranslation("tShell_Cut_1",-23.,0.,-8.);
  auto *tShell_Cut_1_inv = new TGeoTranslation("tShell_Cut_1_inv",23.,0.,-8.);
  auto *Rz = new TGeoRotation("Rz",50.,0.,0.);
  auto *Rz_inv = new TGeoRotation("Rz_inv",-50.,0.,0.);
  auto *RShell_Cut = new TGeoRotation("RShell_Cut",90.,90.-24.,-7.5);
  auto *RShell_Cut_inv = new TGeoRotation("RShell_Cut_inv",90.,90.+24.,-7.5);

  auto *cShell_Cut = new TGeoCombiTrans(*tShell_Cut_1, *RShell_Cut);
  auto *cShell_Cut_inv = new TGeoCombiTrans(*tShell_Cut_1_inv, *RShell_Cut_inv);

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

  //Basic shapes for Half_3
  TGeoShape *Shell_0 = new TGeoTubeSeg("Shell_0", Shell_rmax/2.- .1,Shell_rmax/2. , 6.2/2.,12.,168.);
  TGeoShape *Shell_1 = new TGeoTubeSeg("Shell_1", Shell_rmin/2.- .1,Shell_rmin/2. , 3.2/2.,0.,180.);
  TGeoShape *Shell_2 = new TGeoConeSeg("Shell_2",(25.15+1.0)/2.,Shell_rmin/2.-.1,Shell_rmin/2.,Shell_rmax/2.-.1,Shell_rmax/2.,0.,180.);
  TGeoShape *Shell_3 = new TGeoTube("Shell_3", 0.,Shell_rmin/2.+.1 , .1/2.);
  TGeoShape * ShellHole_0 = new TGeoTrd1("ShellHole_0",17.5/4.,42.5/4.,80./2.,(25.15+1.)/2.);
  TGeoShape * ShellHole_1 = new TGeoBBox("ShellHole_1",42.5/4.,80./2.,2./2.+0.00001);
  TGeoShape * ShellHole_2 = new TGeoBBox("ShellHole_2",58.9/4.,(Shell_rmin-2.25)/2.,.4/2.+0.00001);
  TGeoShape * ShellHole_3 = new TGeoBBox("ShellHole_3",80./4.,(Shell_rmin-11.6)/2.,.4/2.+0.00001);

  //For the extra cut, not sure if this is the shape (apprx. distances)
  TGeoShape *Shell_Cut_0 = new TGeoTube("Shell_Cut_0", 0., 3.5, 5./2.);
  TGeoShape *Shell_Cut_1 = new TGeoBBox("Shell_Cut_1", 7./2., 25./2., 5./2.);

  //Composite shapes for Half_3
  auto * Half_3_Shape_0 = new TGeoCompositeShape("Half_3_Shape_0","Shell_Cut_0+Shell_Cut_1:tShell_Cut");
  auto * Half_3_Shape_1 = new TGeoCompositeShape("Half_3_Shape_1","Shell_2-Half_3_Shape_0:cShell_Cut-Half_3_Shape_0:cShell_Cut_inv");
  auto * Half_3_Shape_2 = new TGeoCompositeShape("Half_3_Shape_2","ShellHole_0+ShellHole_1:tShellHole");
  auto * Half_3_Shape_3 = new TGeoCompositeShape("Half_3_Shape_3","Shell_3:tShellHole_1 - (ShellHole_2:tShellHole_1+ShellHole_3:tShellHole_0)");
  auto * Half_3_Shape_4 = new TGeoCompositeShape("Half_3_Shape_4","(Shell_0:tShell_0 + Half_3_Shape_1+ Shell_1:tShell_1) - (Half_3_Shape_2 + Half_3_Shape_2:Rz + Half_3_Shape_2:Rz_inv)+Half_3_Shape_3");

  auto * Half_3_Volume = new TGeoVolume("Half_3_Volume",Half_3_Shape_4,kMedAlu);
  //Position of the piece relative to the origin which for this code is the center of the the Framework piece (See Half_2)
  Half_3->AddNode(Half_3_Volume,1,new TGeoTranslation(0., 0., -19. ));

  //Half_4
  //Front Framework

  //The part is the arc, the two legs of the sides, and 4 cuts.

  auto *Half_4 = new TGeoVolumeAssembly("Half_4");

  //Front dimensions
  Float_t Front_rmin=19.7;
  Float_t Front_rmax=21.5;
  Float_t Front_z=.6;
  Float_t Front_Angle = 38.612;

  //Legs dimensions
  Float_t Leg_x =2.;
  Float_t Leg_y =6.917;
  Float_t Leg_z =.6;
  Float_t Distance_Leg_x = 14.8;
  Float_t Distance_Leg_y = 6.5;

  //Lateral legs cut dimensions
  Float_t Leg_Lateral_Cut_x =1.5;
  Float_t Leg_Lateral_Cut_y =1.15;

  //Translations
  //The position of the legs
  auto *tLeg_Right = new TGeoTranslation("tLeg_Right",Distance_Leg_x+Leg_x/2.,Distance_Leg_y+Leg_y/2.,0.);
  auto *tLeg_Left = new TGeoTranslation("tLeg_Left",-Distance_Leg_x-Leg_x/2.,Distance_Leg_y+Leg_y/2.,0.);
  //Distances to the center of the piece to make the cut.
  auto *tLeg_Cut_Right = new TGeoTranslation("tLeg_Cut_Right",-Distance_Leg_x-(Leg_x-Leg_Lateral_Cut_x)-Leg_Lateral_Cut_x/2.,Distance_Leg_y+Leg_Lateral_Cut_y/2.,0.);
  auto *tLeg_Cut_Left = new TGeoTranslation("tLeg_Cut_Left",Distance_Leg_x+(Leg_x-Leg_Lateral_Cut_x)+Leg_Lateral_Cut_x/2.,Distance_Leg_y+Leg_Lateral_Cut_y/2.,0.);
  //y-length to the upper cut, plus half the y-length of the leg_Upper_Cut
  auto *tLeg_Cut_Upper = new TGeoTranslation("tLeg_Cut_Upper",0.,21.4+.25,0.);

  tLeg_Right->RegisterYourself();
  tLeg_Left->RegisterYourself();
  tLeg_Cut_Right->RegisterYourself();
  tLeg_Cut_Left->RegisterYourself();
  tLeg_Cut_Upper->RegisterYourself();

  //Basic shapes of Half_4

  TGeoShape *Front = new TGeoTubeSeg("Front",Front_rmin , Front_rmax, Front_z/2.,Front_Angle,180.-Front_Angle);
  TGeoShape *leg = new TGeoBBox("leg",Leg_x/2.,Leg_y/2.,Leg_z/2.);
  //The lateral cut is used twice, so there are 4 cuts. The z dimension is just a bit bigger so it cuts completly the piece.
  //To avoid another translation, the y length is made so that it reaches the piece and cuts the needed.
  TGeoShape *leg_Central_Cut = new TGeoBBox("leg_Central_Cut",1.8/2.,40.4/2.,Leg_z/2.+0.001);
  TGeoShape *leg_Lateral_Cut = new TGeoBBox("leg_Lateral_Cut",(1.5+0.0001)/2.,(1.15+0.0001)/2.,Leg_z/2.+0.001);
  //This shape only has to be bigger than the section to cut.
  TGeoShape *leg_Upper_Cut = new TGeoBBox("leg_Upper_Cut",4./2.,.5/2.,Leg_z/2.+0.001);

  //The front piece + the 2 legs, and then making the 4 cuts
  auto * Half_4_Shape_0 = new TGeoCompositeShape("Half_4_Shape_0","(Front+leg:tLeg_Right+leg:tLeg_Left)-(leg_Central_Cut+leg_Lateral_Cut:tLeg_Cut_Right+leg_Lateral_Cut:tLeg_Cut_Left+leg_Upper_Cut:tLeg_Cut_Upper)");

  auto * Half_4_Volume = new TGeoVolume("Half_4_Volume",Half_4_Shape_0,kMedAlu);
  //Position of the piece relative to the origin which for this code is the center of the the Framework piece (See Half_2)
  Half_4->AddNode(Half_4_Volume,1,new TGeoTranslation(0., 0., -25. ));

  //Half_5

  //Support PCB
  auto *Half_5 = new TGeoVolumeAssembly("Half_5");

  //Dimensions

  Float_t PCB_Angle = 24.3838;

  //Float_t PCB_Central_Projection = 25.15;  // overlap issue, fm
  Float_t PCB_Central_Projection = 24.00;

  Float_t PCB_Central_x = .8;
  Float_t PCB_Central_y = .5;
  Float_t PCB_Central_z = PCB_Central_Projection/cos(PCB_Angle*TMath::Pi()/180.);

  Float_t PCB_Right_x = .8;
  Float_t PCB_Right_y = .5;
  Float_t PCB_Right_z = 3.2;

  Float_t PCB_Left_x = .8;
  Float_t PCB_Left_y = .5;
  Float_t PCB_Left_z = 6.2;

  //Translations, rotations and combinations

  auto * RxPCB  = new TGeoRotation("RxPCB" ,   0.,  -PCB_Angle,   0.) ;
  //Distance of one support to the other one. (this refers to the whole piece)
  auto *tPCB = new TGeoTranslation("tPCB",.6,0.,0.);
  auto *tPCB_inv = new TGeoTranslation("tPCB_inv",-.6,0.,0.);
  //Distance of the Left and Right piece from the center piece. (this refers to parts of the piece)
  auto *tPCB_Right = new TGeoTranslation("tPCB_Right",0.,-(12.4-(PCB_Right_y+PCB_Left_y))/2.,-(PCB_Central_Projection+PCB_Right_z)/2.);
  auto *tPCB_Left = new TGeoTranslation("tPCB_Left",0.,(12.4-(PCB_Right_y+PCB_Left_y))/2.,(PCB_Central_Projection+PCB_Left_z)/2.);
  //Distance of the cut
  auto *tPCB_Cut = new TGeoTranslation("tPCB_Cut",0.,-(12.4/2.-3.6+2./2.),-(PCB_Central_Projection/2.-(10.1-PCB_Right_z)-2./2.));

  RxPCB->RegisterYourself();
  tPCB->RegisterYourself();
  tPCB_inv->RegisterYourself();
  tPCB_Right->RegisterYourself();
  tPCB_Left->RegisterYourself();
  tPCB_Cut->RegisterYourself();

  TGeoShape *Central_PCB = new TGeoBBox("Central_PCB",PCB_Central_x/2.,PCB_Central_y/2.,PCB_Central_z/2.);
  TGeoShape *Right_PCB = new TGeoBBox("Right_PCB",PCB_Right_x/2.,PCB_Right_y/2.,PCB_Right_z/2.);
  TGeoShape *Left_PCB = new TGeoBBox("Left_PCB",PCB_Left_x/2.,PCB_Left_y/2.,PCB_Left_z/2.);

  TGeoShape *Cut_PCB = new TGeoBBox("Cut_PCB",4./2.,2./2.,2./2.);

  auto * Half_5_Shape_0 = new TGeoCompositeShape("Half_5_Shape_0","Central_PCB:RxPCB+Right_PCB:tPCB_Right+Left_PCB:tPCB_Left");
  auto * Half_5_Shape_1 = new TGeoCompositeShape("Half_5_Shape_1","Half_5_Shape_0:tPCB + Half_5_Shape_0:tPCB_inv");
  auto * Half_5_Shape_2 = new TGeoCompositeShape("Half_5_Shape_2","Half_5_Shape_1 - Cut_PCB:tPCB_Cut");

  auto * Half_5_Volume = new TGeoVolume("Half_5_Volume",Half_5_Shape_2,kMedAlu);
  Half_5->AddNode(Half_5_Volume,1,new TGeoTranslation(0., 30.283-6.2,-(28.35/2.+3.2)-.9-.6-.5));

  //The final translation and rotation of the Half Cone to its final position, this are the parameters of the function

  //Position of the radius center of the Framework (See Half_2)
  Float_t tTotal_x;
  Float_t tTotal_y;
  Float_t tTotal_z;

  //Angle of the Half Cone, the z-axis is pointing in the beam axis, and the y-axis in the uppward direction. (It is in Euler angles)
  Float_t rTotal_x;
  Float_t rTotal_y;
  Float_t rTotal_z;

  if(half==0){
    rTotal_x = 90.;
    rTotal_y = 180.;
    rTotal_z = 90.;
    tTotal_x = 0.;
    //tTotal_y = -0.1;  // to be defined
    tTotal_y = -0.5;  // to avoid overlap with disks, fm, vertical position of the total
    tTotal_z = -80.0;  // position still to be defined
    
  }
  if(half==1){
    rTotal_x = 90.;
    rTotal_y = 180.;
    rTotal_z = -90.;
    tTotal_x = 0.;
    //tTotal_y = 0.1;    // to be defined
    tTotal_y = 0.5;    // to avoid overlap with disks, fm, vertical position of the total
    tTotal_z = -80.0;  // position still to be defined

  }

  auto *tTotal = new TGeoTranslation("tTotal",tTotal_x,tTotal_y,tTotal_z);
  auto *rTotal = new TGeoRotation("rTotal",rTotal_x,rTotal_y,rTotal_z);
  auto *cTotal = new TGeoCombiTrans(*tTotal, *rTotal);  

  // overlap problem
  HalfConeVolume->AddNode(Half_0, 0,cTotal); // barres intérieures horizontales fm
  HalfConeVolume->AddNode(Half_1, 0,cTotal); // barres intérieures horizontales fm
  HalfConeVolume->AddNode(Half_2, 0,cTotal);
  HalfConeVolume->AddNode(Half_3, 0,cTotal);
  HalfConeVolume->AddNode(Half_4, 0,cTotal); // support milieu perpendiculaire
  HalfConeVolume->AddNode(Half_5, 0,cTotal); // barre médiane
  
  return HalfConeVolume;

}
