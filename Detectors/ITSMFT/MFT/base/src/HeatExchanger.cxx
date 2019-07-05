// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HeatExchanger.cxx
/// \brief Class building the MFT heat exchanger
/// \author P. Demongodin, and Raphael Tieulent <raphael.tieulent@cern.ch>

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
#include "MFTBase/HeatExchanger.h"
#include "MFTBase/Geometry.h"

using namespace o2::mft;

ClassImp(o2::mft::HeatExchanger);

//_____________________________________________________________________________
HeatExchanger::HeatExchanger()
  : TNamed(),
    mHalfDisk(nullptr),
    mHalfDiskRotation(nullptr),
    mHalfDiskTransformation(nullptr),
    mRWater(0.),
    mDRPipe(0.),
    mHeatExchangerThickness(0.),
    mCarbonThickness(0.),
    mHalfDiskGap(0.),
    mRohacellThickness(0.),
    mNPart(),
    mRMin(),
    mZPlan(),
    mSupportXDimensions(),
    mSupportYDimensions(),
    mLWater0(),
    mXPosition0(),
    mAngle0(),
    mRadius0(),
    mLpartial0(),
    mLWater1(),
    mXPosition1(),
    mAngle1(),
    mRadius1(),
    mLpartial1(),
    mLWater2(),
    mXPosition2(),
    mAngle2(),
    mRadius2(),
    mLpartial2(),
    mLWater3(),
    mXPosition3(),
    mAngle3(),
    mRadius3(),
    mLpartial3(),
    mRadius3fourth(),
    mAngle3fourth(),
    mBeta3fourth(),
    mLWater4(),
    mXPosition4(),
    mAngle4(),
    mRadius4(),
    mLpartial4(),
    mAngle4fifth(),
    mRadius4fifth()
{

  mRWater = 0.1 / 2.;
  mDRPipe = 0.005;
  // mHeatExchangerThickness = 1.398; // to get a 13.4 mm thickness for the rohacell... but the water pipes are not
  // inside it, then its density must be increased
  mHeatExchangerThickness =
    //1.4 + 2 * Geometry::sRohacell;  // sRohacell is used to link the rohacell thickness and the ladder positionning
    1.398;                          // sRohacell is used to link the rohacell thickness and the ladder positionning
  mCarbonThickness = (0.0290) / 2.; // half thickness of the carbon plate

  initParameters();
}

//_____________________________________________________________________________
HeatExchanger::HeatExchanger(Double_t rWater, Double_t dRPipe, Double_t heatExchangerThickness,
                             Double_t carbonThickness)
  : TNamed(),
    mHalfDisk(nullptr),
    mHalfDiskRotation(nullptr),
    mHalfDiskTransformation(nullptr),
    mRWater(rWater),
    mDRPipe(dRPipe),
    mHeatExchangerThickness(heatExchangerThickness),
    mCarbonThickness(carbonThickness),
    mHalfDiskGap(0.),
    mRohacellThickness(0.),
    mNPart(),
    mRMin(),
    mZPlan(),
    mSupportXDimensions(),
    mSupportYDimensions(),
    mXPosition0(),
    mAngle0(),
    mRadius0(),
    mLpartial0(),
    mLWater1(),
    mXPosition1(),
    mAngle1(),
    mRadius1(),
    mLpartial1(),
    mLWater2(),
    mXPosition2(),
    mAngle2(),
    mRadius2(),
    mLpartial2(),
    mLWater3(),
    mXPosition3(),
    mAngle3(),
    mRadius3(),
    mLpartial3(),
    mRadius3fourth(),
    mAngle3fourth(),
    mBeta3fourth(),
    mLWater4(),
    mXPosition4(),
    mAngle4(),
    mRadius4(),
    mLpartial4(),
    mAngle4fifth(),
    mRadius4fifth()
{

  initParameters();
}

//_____________________________________________________________________________
TGeoVolumeAssembly* HeatExchanger::create(Int_t half, Int_t disk)
{

  Info("Create", Form("Creating HeatExchanger_%d_%d", disk, half), 0, 0);

  mHalfDisk = new TGeoVolumeAssembly(Form("HeatExchanger_%d_%d", disk, half));
  switch (disk) {
    case 0:
      createHalfDisk0(half);
      break;
    case 1:
      createHalfDisk1(half);
      break;
    case 2:
      createHalfDisk2(half);
      break;
    case 3:
      createHalfDisk3(half);
      break;
    case 4:
      createHalfDisk4(half);
      break;
  }

  Info("Create", Form("... done HeatExchanger_%d_%d", disk, half), 0, 0);

  return mHalfDisk;
}

//_____________________________________________________________________________
void HeatExchanger::createManifold(Int_t disk)
{

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Manifold1
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  Double_t cornerRadiusTop[5];
  cornerRadiusTop[0] = 0.2;
  cornerRadiusTop[1] = 0.2;
  cornerRadiusTop[2] = 0.2;
  cornerRadiusTop[3] = 0.2;
  cornerRadiusTop[4] = 0.2;

  Double_t thicknessTop[5];
  thicknessTop[0] = 0.2;
  thicknessTop[1] = 0.2;
  thicknessTop[2] = 0.2;
  thicknessTop[3] = 0.2;
  thicknessTop[4] = 0.2;

  Double_t thicknessMiddle[5];
  thicknessMiddle[0] = 0.2;
  thicknessMiddle[1] = 0.2;
  thicknessMiddle[2] = 0.2;
  thicknessMiddle[3] = 0.2;
  thicknessMiddle[4] = 0.2;

  Double_t thicknessBottom[5];
  thicknessBottom[0] = 0.6;
  thicknessBottom[1] = 0.6;
  thicknessBottom[2] = 0.6;
  thicknessBottom[3] = 0.6;
  thicknessBottom[4] = 0.6;

  Double_t widthTop1[5];
  widthTop1[0] = 1.8;
  widthTop1[1] = 1.8;
  widthTop1[2] = 1.8;
  widthTop1[3] = 1.8;
  widthTop1[4] = 1.8;

  Double_t widthTop2;
  widthTop2 = widthTop1[disk] - 2 * cornerRadiusTop[disk];

  Double_t widthMiddle1[5];
  widthMiddle1[0] = 2.2;
  widthMiddle1[1] = 2.2;
  widthMiddle1[2] = 2.2;
  widthMiddle1[3] = 2.2;
  widthMiddle1[4] = 2.2;

  Double_t widthBottom1[5];
  widthBottom1[0] = 1.32;
  widthBottom1[1] = 1.32;
  widthBottom1[2] = 1.32;
  widthBottom1[3] = 1.32;
  widthBottom1[4] = 1.32;

  Double_t lengthTop1[5];
  lengthTop1[0] = 5.91;
  lengthTop1[1] = 5.91;
  lengthTop1[2] = 5.91;
  //lengthTop1[3] = 8.40;
  //lengthTop1[4] = 8.40;
  lengthTop1[3] = 5.91;
  lengthTop1[4] = 5.91;

  Double_t lengthTop2;
  lengthTop2 = cornerRadiusTop[disk];

  Double_t lengthMiddle1[5];
  /*
  lengthMiddle1[0] = 6.71;
  lengthMiddle1[1] = 6.71;
  lengthMiddle1[2] = 6.71;
  //lengthMiddle1[3] = 9.20;
  //lengthMiddle1[4] = 9.20;
  lengthMiddle1[3] = 6.71;
  lengthMiddle1[4] = 6.71;
  */
  lengthMiddle1[0] = mSupportYDimensions[0][0];
  lengthMiddle1[1] = mSupportYDimensions[1][0];
  lengthMiddle1[2] = mSupportYDimensions[2][0];
  lengthMiddle1[3] = mSupportYDimensions[3][0];
  lengthMiddle1[4] = mSupportYDimensions[4][0];

  Double_t lengthBottom1;
  lengthBottom1 = lengthMiddle1[disk];

  auto* dummy = new TGeoBBox("dummy", 0, 0, 0);

  auto* Top1 = new TGeoBBox(Form("Top1MF%d", disk), lengthTop1[disk] / 2, widthTop1[disk] / 2, thicknessTop[disk] / 2);
  auto* Top2 = new TGeoBBox(Form("Top2MF%d", disk), lengthTop2 / 2, widthTop2 / 2, thicknessTop[disk] / 2);
  auto* Top3 = new TGeoTube(Form("Top3MF%d", disk), 0, cornerRadiusTop[disk], thicknessTop[disk] / 2);
  auto* Middle1 = new TGeoBBox(Form("Middle1MF%d", disk), lengthMiddle1[disk] / 2, widthMiddle1[disk] / 2, thicknessMiddle[disk] / 2);
  auto* Bottom1 = new TGeoBBox(Form("Bottom1MF%d", disk), lengthBottom1 / 2, widthBottom1[disk] / 2, thicknessBottom[disk] / 2);

  TGeoTranslation* tTop[7];
  tTop[0] = new TGeoTranslation(Form("tTop1MF%d", disk), 0., 0., thicknessMiddle[disk] / 2 + thicknessTop[disk] / 2);
  tTop[1] = new TGeoTranslation(Form("tTop2MF%d", disk), lengthTop1[disk] / 2 + lengthTop2 / 2, 0., thicknessMiddle[disk] / 2 + thicknessTop[disk] / 2);
  tTop[2] = new TGeoTranslation(Form("tTop3MF%d", disk), lengthTop1[disk] / 2, widthTop1[disk] / 2 - cornerRadiusTop[disk], thicknessMiddle[disk] / 2 + thicknessTop[disk] / 2);
  tTop[3] = new TGeoTranslation(Form("tTop4MF%d", disk), lengthTop1[disk] / 2, -(widthTop1[disk] / 2 - cornerRadiusTop[disk]), thicknessMiddle[disk] / 2 + thicknessTop[disk] / 2);
  tTop[4] = new TGeoTranslation(Form("tTop5MF%d", disk), -(lengthTop1[disk] / 2 + lengthTop2 / 2), 0., thicknessMiddle[disk] / 2 + thicknessTop[disk] / 2);
  tTop[5] = new TGeoTranslation(Form("tTop6MF%d", disk), -lengthTop1[disk] / 2, widthTop1[disk] / 2 - cornerRadiusTop[disk], thicknessMiddle[disk] / 2 + thicknessTop[disk] / 2);
  tTop[6] = new TGeoTranslation(Form("tTop7MF%d", disk), -lengthTop1[disk] / 2, -(widthTop1[disk] / 2 - cornerRadiusTop[disk]), thicknessMiddle[disk] / 2 + thicknessTop[disk] / 2);

  for (Int_t i = 0; i < 7; ++i)
    tTop[i]->RegisterYourself();

  TGeoTranslation* tMiddle1 = new TGeoTranslation(Form("tMiddle1MF%d", disk), 0, 0, 0);
  TGeoTranslation* tBottom1 = new TGeoTranslation(Form("tBottom1MF%d", disk), 0, 0, -(thicknessMiddle[disk] / 2 + thicknessBottom[disk] / 2));
  tMiddle1->RegisterYourself();
  tBottom1->RegisterYourself();

  Double_t radiusPipeHole1[5];
  radiusPipeHole1[0] = 0.192 / 2;
  radiusPipeHole1[1] = 0.192 / 2;
  radiusPipeHole1[2] = 0.192 / 2;
  radiusPipeHole1[3] = 0.192 / 2;
  radiusPipeHole1[4] = 0.192 / 2;

  Double_t radiusPipeHole2[5];
  radiusPipeHole2[0] = 0.300 / 2;
  radiusPipeHole2[1] = 0.300 / 2;
  radiusPipeHole2[2] = 0.300 / 2;
  radiusPipeHole2[3] = 0.300 / 2;
  radiusPipeHole2[4] = 0.300 / 2;

  Double_t lengthPipeHole1[5];
  lengthPipeHole1[0] = 0.95;
  lengthPipeHole1[1] = 0.95;
  lengthPipeHole1[2] = 0.95;
  lengthPipeHole1[3] = 0.95;
  lengthPipeHole1[4] = 0.95;

  Double_t lengthPipeHole2 = (thicknessTop[disk] + thicknessMiddle[disk] + thicknessBottom[disk]) - lengthPipeHole1[disk];

  TGeoTube* Pipe1 = new TGeoTube("Pipe1", 0, radiusPipeHole1[disk], lengthPipeHole1[disk] / 2 + Geometry::sEpsilon);
  TGeoCone* Pipe2 = new TGeoCone("Pipe2", lengthPipeHole2 / 2 + Geometry::sEpsilon, 0, radiusPipeHole1[disk], 0, radiusPipeHole2[disk]);

  TGeoTranslation* tPipePart1 = new TGeoTranslation(Form("tPipePart1MF%d", disk), 0, 0, 0);
  TGeoTranslation* tPipePart2 = new TGeoTranslation(Form("tPipePart2MF%d", disk), 0, 0, lengthPipeHole1[disk] / 2 + lengthPipeHole2 / 2);
  tPipePart1->RegisterYourself();
  tPipePart2->RegisterYourself();

  TGeoCompositeShape* shapePipe = new TGeoCompositeShape(Form("shapePipeMF%d", disk), Form("Pipe1:tPipePart1MF%d + Pipe2:tPipePart2MF%d", disk, disk));

  Int_t nPipeRow[5];
  nPipeRow[0] = 3;
  nPipeRow[1] = 3;
  nPipeRow[2] = 3;
  nPipeRow[3] = 4;
  nPipeRow[4] = 5;

  TGeoTranslation* tPipe;

  // The lengthBulge is a parameter for manifold2. This is required to tune the water pipe position.
  Double_t lengthBulge[5];
  lengthBulge[0] = 0.395;
  lengthBulge[1] = 0.395;
  lengthBulge[2] = 0.395;
  //lengthBulge[3] = 0.400;
  //lengthBulge[4] = 0.400;
  lengthBulge[3] = 0.395;
  lengthBulge[4] = 0.395;

  Double_t offsetPipeRow[5][5] = {};
  offsetPipeRow[0][0] = mXPosition0[0] - lengthBulge[0] / 2;
  offsetPipeRow[0][1] = mXPosition0[1] - lengthBulge[0] / 2;
  offsetPipeRow[0][2] = mXPosition0[2] - lengthBulge[0] / 2;

  offsetPipeRow[1][0] = mXPosition1[0] - lengthBulge[1] / 2;
  offsetPipeRow[1][1] = mXPosition1[1] - lengthBulge[1] / 2;
  offsetPipeRow[1][2] = mXPosition1[2] - lengthBulge[1] / 2;

  offsetPipeRow[2][0] = mXPosition2[0] - lengthBulge[2] / 2;
  offsetPipeRow[2][1] = mXPosition2[1] - lengthBulge[2] / 2;
  offsetPipeRow[2][2] = mXPosition2[2] - lengthBulge[2] / 2;

  offsetPipeRow[3][0] = mXPosition3[0] - lengthBulge[3] / 2;
  offsetPipeRow[3][1] = mXPosition3[1] - lengthBulge[3] / 2;
  offsetPipeRow[3][2] = mXPosition3[2] - lengthBulge[3] / 2;
  offsetPipeRow[3][3] = mXPosition3[3] - lengthBulge[3] / 2;

  offsetPipeRow[4][0] = mXPosition4[0] - lengthBulge[4] / 2;
  offsetPipeRow[4][1] = mXPosition4[1] - lengthBulge[4] / 2;
  offsetPipeRow[4][2] = mXPosition4[2] - lengthBulge[4] / 2;
  offsetPipeRow[4][3] = mXPosition4[3] - lengthBulge[4] / 2;
  offsetPipeRow[4][4] = mXPosition4[4] - lengthBulge[4] / 2;

  //for(Int_t ip=0; ip<nPipeRow[disk]; ++ip) printf("Disk%d: %f \n",disk,offsetPipeRow[disk][ip]);

  Double_t deltaz = mHeatExchangerThickness - Geometry::sKaptonOnCarbonThickness * 4 - 2 * mCarbonThickness;

  Double_t lengthTwoPipeCol[5];
  lengthTwoPipeCol[0] = mZPlan[disk] + deltaz / 2. - mCarbonThickness - mRWater - mDRPipe - 2 * Geometry::sGlueRohacellCarbonThickness - 2 * Geometry::sKaptonOnCarbonThickness;
  lengthTwoPipeCol[1] = mZPlan[disk] + deltaz / 2. - mCarbonThickness - mRWater - mDRPipe - 2 * Geometry::sGlueRohacellCarbonThickness - 2 * Geometry::sKaptonOnCarbonThickness;
  lengthTwoPipeCol[2] = mZPlan[disk] + deltaz / 2. - mCarbonThickness - mRWater - mDRPipe - 2 * Geometry::sGlueRohacellCarbonThickness - 2 * Geometry::sKaptonOnCarbonThickness;
  lengthTwoPipeCol[3] = mZPlan[disk] + deltaz / 2. - mCarbonThickness - mRWater - mDRPipe - 2 * Geometry::sGlueRohacellCarbonThickness - 2 * Geometry::sKaptonOnCarbonThickness;
  lengthTwoPipeCol[4] = mZPlan[disk] + deltaz / 2. - mCarbonThickness - mRWater - mDRPipe - 2 * Geometry::sGlueRohacellCarbonThickness - 2 * Geometry::sKaptonOnCarbonThickness;

  TString namePipe = "";

  for (Int_t iPipeRow = 0; iPipeRow < nPipeRow[disk]; iPipeRow++) {
    tPipe = new TGeoTranslation(Form("tPipe%dMF%d", iPipeRow + 1, disk), lengthMiddle1[disk] / 2 - offsetPipeRow[disk][iPipeRow], 0, 0);
    tPipe->RegisterYourself();

    if (iPipeRow == nPipeRow[disk] - 1)
      namePipe += Form("shapePipeMF%d:tPipe%dMF%d", disk, iPipeRow + 1, disk);
    else
      namePipe += Form("shapePipeMF%d:tPipe%dMF%d +", disk, iPipeRow + 1, disk);
  }

  TGeoCompositeShape* posiPipeOneCol = new TGeoCompositeShape(Form("posiPipeOneColMF%d", disk), namePipe);

  tPipe = new TGeoTranslation(Form("tPipeLeftMF%d", disk), 0, -lengthTwoPipeCol[disk], 0);
  tPipe->RegisterYourself();
  tPipe = new TGeoTranslation(Form("tPipeRightMF%d", disk), 0, lengthTwoPipeCol[disk], 0);
  tPipe->RegisterYourself();

  TGeoCompositeShape* shapePipes = new TGeoCompositeShape(Form("shapePipesMF%d", disk), Form("posiPipeOneColMF%d:tPipeLeftMF%d + posiPipeOneColMF%d:tPipeRightMF%d", disk, disk, disk, disk));
  TGeoCompositeShape* shapeBase = new TGeoCompositeShape(Form("shapeBaseMF%d", disk), Form("Top1MF%d:tTop1MF%d + Top2MF%d:tTop2MF%d + Top3MF%d:tTop3MF%d + Top3MF%d:tTop4MF%d + Top2MF%d:tTop5MF%d + Top3MF%d:tTop6MF%d + Top3MF%d:tTop7MF%d + Middle1MF%d:tMiddle1MF%d + Bottom1MF%d:tBottom1MF%d", disk, disk, disk, disk, disk, disk, disk, disk, disk, disk, disk, disk, disk, disk, disk, disk, disk, disk));

  TGeoTranslation* tBase = new TGeoTranslation(Form("tBaseMF%d", disk), 0, 0, -((thicknessTop[disk] + thicknessMiddle[disk] + thicknessBottom[disk]) / 2 - (thicknessBottom[disk] + thicknessMiddle[disk] / 2)));
  tBase->RegisterYourself();
  TGeoTranslation* tPipes = new TGeoTranslation(Form("tPiesMF%d", disk), 0, 0, -((lengthPipeHole1[disk] + lengthPipeHole2) / 2 - lengthPipeHole1[disk] / 2));
  tPipes->RegisterYourself();

  TGeoCompositeShape* shapeManifold1 = new TGeoCompositeShape(Form("shapeManifold1MF%d", disk), Form("shapeBaseMF%d:tBaseMF%d - shapePipesMF%d:tPiesMF%d", disk, disk, disk, disk));

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Manifold2
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  Double_t lengthBody;
  lengthBody = lengthMiddle1[disk];
  Double_t widthBody;
  widthBody = widthMiddle1[disk];
  Double_t thicknessBody[5];
  thicknessBody[0] = 0.8;
  thicknessBody[1] = 0.8;
  thicknessBody[2] = 0.8;
  thicknessBody[3] = 0.8;
  thicknessBody[4] = 0.8;

  Double_t cornerRadiusBodyBathtub1[5];
  cornerRadiusBodyBathtub1[0] = cornerRadiusTop[disk];
  cornerRadiusBodyBathtub1[1] = cornerRadiusTop[disk];
  cornerRadiusBodyBathtub1[2] = cornerRadiusTop[disk];
  cornerRadiusBodyBathtub1[3] = cornerRadiusTop[disk];
  cornerRadiusBodyBathtub1[4] = cornerRadiusTop[disk];

  Double_t lengthBodyBathtub1;
  lengthBodyBathtub1 = lengthTop1[disk];
  Double_t widthBodyBathtub1;
  widthBodyBathtub1 = widthTop1[disk];
  Double_t thicknessBodyBathtub1[5];
  thicknessBodyBathtub1[0] = 0.4;
  thicknessBodyBathtub1[1] = 0.4;
  thicknessBodyBathtub1[2] = 0.4;
  thicknessBodyBathtub1[3] = 0.4;
  thicknessBodyBathtub1[4] = 0.4;

  Double_t lengthBodyBathtub2;
  lengthBodyBathtub2 = cornerRadiusBodyBathtub1[disk];
  Double_t widthBodyBathtub2;
  widthBodyBathtub2 = widthTop1[disk] - 2 * cornerRadiusBodyBathtub1[disk];
  Double_t thicknessBodyBathtub2[5];
  thicknessBodyBathtub2[0] = 0.4;
  thicknessBodyBathtub2[1] = 0.4;
  thicknessBodyBathtub2[2] = 0.4;
  thicknessBodyBathtub2[3] = 0.4;
  thicknessBodyBathtub2[4] = 0.4;

  auto* coverBody1 = new TGeoBBox(Form("coverBody1MF%d", disk), lengthBody / 2, widthBody / 2, thicknessBody[disk] / 2);

  auto* coverBodyBathtub1 = new TGeoBBox(Form("coverBodyBathtub1MF%d", disk), lengthBodyBathtub1 / 2 + Geometry::sEpsilon, widthBodyBathtub1 / 2 + Geometry::sEpsilon,
                                         thicknessBodyBathtub1[disk] / 2 + Geometry::sEpsilon);
  auto* coverBodyBathtub2 = new TGeoBBox(Form("coverBodyBathtub2MF%d", disk), lengthBodyBathtub2 / 2 + Geometry::sEpsilon, widthBodyBathtub2 / 2 + Geometry::sEpsilon,
                                         thicknessBodyBathtub2[disk] / 2 + Geometry::sEpsilon);
  auto* coverBodyBathtub3 = new TGeoTube(Form("coverBodyBathtub3MF%d", disk), 0, cornerRadiusBodyBathtub1[disk] + Geometry::sEpsilon,
                                         thicknessBodyBathtub2[disk] / 2 + Geometry::sEpsilon);

  TGeoTranslation* tcoverBodyBathtub[7];
  tcoverBodyBathtub[0] = new TGeoTranslation(Form("tcoverBodyBathtub1MF%d", disk), 0., 0., 0);

  tcoverBodyBathtub[1] = new TGeoTranslation(Form("tcoverBodyBathtub2MF%d", disk), lengthBodyBathtub1 / 2 + lengthBodyBathtub2 / 2, 0., 0.);
  tcoverBodyBathtub[2] = new TGeoTranslation(Form("tcoverBodyBathtub3MF%d", disk), lengthBodyBathtub1 / 2, widthBodyBathtub1 / 2 - cornerRadiusBodyBathtub1[disk], 0.);
  tcoverBodyBathtub[3] = new TGeoTranslation(Form("tcoverBodyBathtub4MF%d", disk), lengthBodyBathtub1 / 2, -(widthBodyBathtub1 / 2 - cornerRadiusBodyBathtub1[disk]), 0.);

  tcoverBodyBathtub[4] = new TGeoTranslation(Form("tcoverBodyBathtub5MF%d", disk), -(lengthBodyBathtub1 / 2 + lengthBodyBathtub2 / 2), 0., 0.);
  tcoverBodyBathtub[5] = new TGeoTranslation(Form("tcoverBodyBathtub6MF%d", disk), -(lengthBodyBathtub1 / 2), widthBodyBathtub1 / 2 - cornerRadiusBodyBathtub1[disk], 0.);
  tcoverBodyBathtub[6] = new TGeoTranslation(Form("tcoverBodyBathtub7MF%d", disk), -(lengthBodyBathtub1 / 2), -(widthBodyBathtub1 / 2 - cornerRadiusBodyBathtub1[disk]), 0.);

  for (Int_t i = 0; i < 7; ++i)
    tcoverBodyBathtub[i]->RegisterYourself();

  TGeoCompositeShape* shapeCoverBathtub = new TGeoCompositeShape(Form("shapeCoverBathtubMF%d", disk), Form("coverBodyBathtub1MF%d + coverBodyBathtub2MF%d:tcoverBodyBathtub2MF%d + coverBodyBathtub3MF%d:tcoverBodyBathtub3MF%d + coverBodyBathtub3MF%d:tcoverBodyBathtub4MF%d + coverBodyBathtub2MF%d:tcoverBodyBathtub5MF%d + coverBodyBathtub3MF%d:tcoverBodyBathtub6MF%d + coverBodyBathtub3MF%d:tcoverBodyBathtub7MF%d", disk, disk, disk, disk, disk, disk, disk, disk, disk, disk, disk, disk, disk));

  TGeoTranslation* tcoverBathtub = new TGeoTranslation(Form("tcoverBathtubMF%d", disk), 0, 0, thicknessBody[disk] / 2 - thicknessBodyBathtub1[disk] / 2);
  tcoverBathtub->RegisterYourself();

  Double_t cornerRadiusStep1[5];
  cornerRadiusStep1[0] = 0.2;
  cornerRadiusStep1[1] = 0.2;
  cornerRadiusStep1[2] = 0.2;
  cornerRadiusStep1[3] = 0.2;
  cornerRadiusStep1[4] = 0.2;

  Double_t lengthStep1[5];
  lengthStep1[0] = 5.61;
  lengthStep1[1] = 5.61;
  lengthStep1[2] = 5.61;
  //lengthStep1[3] = 7.90;
  //lengthStep1[4] = 7.90;
  lengthStep1[3] = 5.61;
  lengthStep1[4] = 5.61;
  Double_t widthStep1[5];
  widthStep1[0] = 1.40;
  widthStep1[1] = 1.40;
  widthStep1[2] = 1.40;
  widthStep1[3] = 1.40;
  widthStep1[4] = 1.40;
  Double_t thicknessStep1[5];
  thicknessStep1[0] = 0.2;
  thicknessStep1[1] = 0.2;
  thicknessStep1[2] = 0.2;
  thicknessStep1[3] = 0.2;
  thicknessStep1[4] = 0.2;

  Double_t lengthStep2;
  lengthStep2 = cornerRadiusStep1[disk];
  Double_t widthStep2;
  widthStep2 = widthStep1[disk] - cornerRadiusStep1[disk];
  Double_t thicknessStep2;
  thicknessStep2 = thicknessStep1[disk];

  Double_t angleStep3 = 45. / 180. * TMath::Pi();
  Double_t lengthStep3[5];
  lengthStep3[0] = 0.3 / TMath::Cos(angleStep3);
  lengthStep3[1] = 0.3 / TMath::Cos(angleStep3);
  lengthStep3[2] = 0.3 / TMath::Cos(angleStep3);
  lengthStep3[3] = 0.3 / TMath::Cos(angleStep3);
  lengthStep3[4] = 0.3 / TMath::Cos(angleStep3);
  Double_t widthStep3;
  widthStep3 = 0.3 / TMath::Cos(angleStep3);
  Double_t thicknessStep3;
  thicknessStep3 = thicknessStep1[disk];

  auto* coverBodyStep1 = new TGeoBBox(Form("coverBodyStep1MF%d", disk), (lengthStep1[disk] - cornerRadiusStep1[disk]) / 2 + Geometry::sEpsilon, widthStep1[disk] / 2 + Geometry::sEpsilon,
                                      thicknessStep1[disk] / 2 + Geometry::sEpsilon);
  auto* coverBodyStep2 = new TGeoBBox(Form("coverBodyStep2MF%d", disk), lengthStep2 / 2 + Geometry::sEpsilon, widthStep2 / 2 + Geometry::sEpsilon,
                                      thicknessStep2 / 2 + Geometry::sEpsilon);
  auto* coverBodyStep3 = new TGeoTube(Form("coverBodyStep3MF%d", disk), 0, cornerRadiusStep1[disk] + Geometry::sEpsilon,
                                      thicknessStep1[disk] / 2 + Geometry::sEpsilon);
  auto* coverBodyStep4 = new TGeoBBox(Form("coverBodyStep4MF%d", disk), lengthStep3[disk] / 2 + Geometry::sEpsilon, widthStep3 / 2 + Geometry::sEpsilon,
                                      thicknessStep3 / 2 + Geometry::sEpsilon);

  TGeoTranslation* tcoverBodyStep[4];
  tcoverBodyStep[0] = new TGeoTranslation(Form("tcoverBodyStep1MF%d", disk), 0., 0., 0);
  tcoverBodyStep[1] = new TGeoTranslation(Form("tcoverBodyStep2MF%d", disk), (lengthStep1[disk] - cornerRadiusStep1[disk]) / 2 + lengthStep2 / 2, cornerRadiusStep1[disk] / 2, 0.);
  tcoverBodyStep[2] = new TGeoTranslation(Form("tcoverBodyStep3MF%d", disk), (lengthStep1[disk] - cornerRadiusStep1[disk]) / 2, -(widthStep1[disk] / 2 - cornerRadiusStep1[disk]), 0.);
  tcoverBodyStep[3] = new TGeoTranslation(Form("tcoverBodyStep4MF%d", disk), -(lengthStep1[disk] - lengthStep2) / 2, -widthStep1[disk] / 2, 0.);

  TGeoRotation* rcoverBodyStep = new TGeoRotation(Form("rcoverBodyStep4MF%d", disk), 45, 0, 0);
  rcoverBodyStep->RegisterYourself();

  TGeoCompositeShape* coverBodyStep4rotated = new TGeoCompositeShape(Form("coverBodyStep4rotatedMF%d", disk), Form("dummy + coverBodyStep4MF%d:rcoverBodyStep4MF%d", disk, disk));

  for (Int_t i = 0; i < 4; ++i)
    tcoverBodyStep[i]->RegisterYourself();

  TGeoCompositeShape* shapeStep = new TGeoCompositeShape(Form("shapeStepMF%d", disk), Form("coverBodyStep1MF%d:tcoverBodyStep1MF%d + coverBodyStep2MF%d:tcoverBodyStep2MF%d + coverBodyStep3MF%d:tcoverBodyStep3MF%d + coverBodyStep4rotatedMF%d:tcoverBodyStep4MF%d", disk, disk, disk, disk, disk, disk, disk, disk));

  TGeoTranslation* tcoverStep = new TGeoTranslation(Form("tcoverStepMF%d", disk), -(lengthMiddle1[disk] / 2 - (lengthStep1[disk] / 2 - lengthStep2 / 2)),
                                                    (widthBody / 2 - widthStep1[disk] / 2), -(thicknessBody[disk] / 2 - thicknessStep1[disk] / 2));
  tcoverStep->RegisterYourself();

  Double_t widthBulge[5];
  widthBulge[0] = 0.50;
  widthBulge[1] = 0.50;
  widthBulge[2] = 0.50;
  widthBulge[3] = 0.50;
  widthBulge[4] = 0.50;
  Double_t thicknessBulge;
  thicknessBulge = thicknessBody[disk];
  Double_t lengthBulgeSub[5];
  lengthBulgeSub[0] = 0.2;
  lengthBulgeSub[1] = 0.2;
  lengthBulgeSub[2] = 0.2;
  lengthBulgeSub[3] = 0.2;
  lengthBulgeSub[4] = 0.2;
  Double_t widthBulgeSub;
  widthBulgeSub = lengthBulgeSub[disk];
  Double_t thicknessBulgeSub[5];
  thicknessBulgeSub[0] = 0.5;
  thicknessBulgeSub[1] = 0.5;
  thicknessBulgeSub[2] = 0.5;
  thicknessBulgeSub[3] = 0.5;
  thicknessBulgeSub[4] = 0.5;

  auto* coverBodyBulge = new TGeoBBox(Form("coverBodyBulgeMF%d", disk), lengthBulge[disk] / 2, widthBulge[disk] / 2, thicknessBulge / 2);
  auto* coverBodyBulgeSub = new TGeoBBox(Form("coverBodyBulgeSubMF%d", disk), lengthBulgeSub[disk] / 2, widthBulgeSub / 2, thicknessBulgeSub[disk] / 2 + Geometry::sEpsilon);

  TGeoRotation* rcoverBodyBulgeSub = new TGeoRotation(Form("rcoverBodyBulgeSubMF%d", disk), 0, 90, 45);
  rcoverBodyBulgeSub->RegisterYourself();

  TGeoCompositeShape* coverBodyBulgeSubrotated = new TGeoCompositeShape(Form("coverBodyBulgeSubrotatedMF%d", disk), Form("dummy + coverBodyBulgeSubMF%d:rcoverBodyBulgeSubMF%d", disk, disk));

  TGeoTranslation* tcoverBodyBulgeSub = new TGeoTranslation(Form("tcoverBodyBulgeSubMF%d", disk), -lengthBulge[disk] / 2, 0, 0);
  tcoverBodyBulgeSub->RegisterYourself();

  TGeoCompositeShape* shapeBulge = new TGeoCompositeShape(Form("shapeBulgeMF%d", disk), Form("coverBodyBulgeMF%d - coverBodyBulgeSubrotatedMF%d:tcoverBodyBulgeSubMF%d", disk, disk, disk));

  TGeoTranslation* tcoverBulge = new TGeoTranslation(Form("tcoverBulgeMF%d", disk), -(lengthMiddle1[disk] / 2 + lengthBulge[disk] / 2), -(widthBody / 2 - widthBulge[disk] / 2), 0);
  tcoverBulge->RegisterYourself();

  Double_t holeRadius[5];
  holeRadius[0] = 0.25; //0.207;    overlap issue of thread, increase up to 0.25, fm
  holeRadius[1] = 0.25; //0.207;
  holeRadius[2] = 0.25; //0.207;
  holeRadius[3] = 0.25; //0.207;
  holeRadius[4] = 0.25; //0.207;
  Double_t holeOffset[5];
  holeOffset[0] = 0.65;
  holeOffset[1] = 0.65;
  holeOffset[2] = 0.65;
  holeOffset[3] = 0.65;
  holeOffset[4] = 0.65;

  auto* coverBodyHole = new TGeoTube(Form("coverBodyHoleMF%d", disk), 0, holeRadius[disk], thicknessBody[disk] / 2 + Geometry::sEpsilon);
  TGeoTranslation* tcoverBodyHole = new TGeoTranslation(Form("tcoverBodyHoleMF%d", disk), lengthBody / 2 - holeOffset[disk], 0, 0);
  tcoverBodyHole->RegisterYourself();

  TGeoCompositeShape* shapeManifold2 = new TGeoCompositeShape(Form("shapeManifold2MF%d", disk), Form("coverBody1MF%d - coverBodyHoleMF%d:tcoverBodyHoleMF%d - shapeCoverBathtubMF%d:tcoverBathtubMF%d - shapeStepMF%d:tcoverStepMF%d + shapeBulgeMF%d:tcoverBulgeMF%d", disk, disk, disk, disk, disk, disk, disk, disk, disk));

  TGeoRotation* rshapeManifold2 = new TGeoRotation(Form("rshapeManifold2MF%d", disk), 0, 180, 0);
  rshapeManifold2->RegisterYourself();

  TGeoCompositeShape* shapeManifold2rotated = new TGeoCompositeShape(Form("shapeManifold2rotatedMF%d", disk), Form("dummy + shapeManifold2MF%d:rshapeManifold2MF%d", disk, disk));

  TGeoTranslation* tshapeManifold1 = new TGeoTranslation(Form("tshapeManifold1MF%d", disk), 0, 0,
                                                         -((thicknessBody[disk] + thicknessMiddle[disk] + thicknessBottom[disk]) / 2 - (thicknessTop[disk] + thicknessMiddle[disk] + thicknessBottom[disk]) / 2));
  tshapeManifold1->RegisterYourself();
  TGeoTranslation* tshapeManifold2 = new TGeoTranslation(Form("tshapeManifold2MF%d", disk), 0, 0, (thicknessBody[disk] + thicknessMiddle[disk] + thicknessBottom[disk]) / 2 - thicknessBody[disk] / 2);
  tshapeManifold2->RegisterYourself();

  TGeoCompositeShape* shapeManifold = new TGeoCompositeShape("shapeManifold", Form("shapeManifold1MF%d:tshapeManifold1MF%d + shapeManifold2rotatedMF%d:tshapeManifold2MF%d", disk, disk, disk, disk));

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Manifold3
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  Double_t innerRadiusPlug1 = 0.600 / 2;
  Double_t outerRadiusPlug1 = 0.99 / 2;
  Double_t thicknessPlug1 = 0.105;

  auto* plug1 = new TGeoTube(Form("plug1MF%d", disk), innerRadiusPlug1, outerRadiusPlug1, thicknessPlug1 / 2);

  Double_t innerRadiusPlug2 = innerRadiusPlug1;
  Double_t outerMinRadiusPlug2 = 0.94 / 2;
  Double_t outerMaxRadiusPlug2 = outerRadiusPlug1;
  Double_t thicknessPlug2 = 0.025;

  auto* plug2 = new TGeoCone(Form("plug2MF%d", disk), thicknessPlug2 / 2, innerRadiusPlug2, outerMaxRadiusPlug2, innerRadiusPlug2, outerMinRadiusPlug2);

  Double_t innerRadiusPlug3 = innerRadiusPlug1;
  Double_t outerRadiusPlug3 = 0.720 / 2;
  Double_t thicknessPlug3 = 0.086;

  auto* plug3 = new TGeoTube(Form("plug3MF%d", disk), innerRadiusPlug3, outerRadiusPlug3, thicknessPlug3 / 2);

  Double_t innerRadiusPlug4 = innerRadiusPlug1;
  Double_t outerRadiusPlug4 = 1.100 / 2;
  Double_t thicknessPlug4 = 0.534;

  auto* plug4 = new TGeoTube(Form("plug4MF%d", disk), innerRadiusPlug4, outerRadiusPlug4, thicknessPlug4 / 2);

  Double_t innerRadiusPlug5 = innerRadiusPlug1;
  Double_t outerRadiusPlug5 = 1.270 / 2;
  Double_t thicknessPlug5 = 0.700;

  auto* plug5main = new TGeoTube(Form("plug5mainMF%d", disk), innerRadiusPlug5, outerRadiusPlug5, thicknessPlug5 / 2);

  const Int_t nSidePlug5 = 6;
  Double_t anglesubPlug5 = 360. / nSidePlug5 / 2. * TMath::Pi() / 180.;
  Double_t lengthPlug5sub = outerRadiusPlug5 * TMath::Cos(anglesubPlug5) * 2;
  Double_t widthPlug5sub = outerRadiusPlug5 * (1 - TMath::Cos(anglesubPlug5));
  Double_t thicknessPlug5sub = thicknessPlug5 + Geometry::sEpsilon;

  auto* plug5sub = new TGeoBBox(Form("plug5subMF%d", disk), lengthPlug5sub / 2, widthPlug5sub / 2, thicknessPlug5sub / 2);

  TGeoTranslation* tPlug5sub = new TGeoTranslation(Form("tPlug5subMF%d", disk), 0, outerRadiusPlug5 * TMath::Cos(anglesubPlug5) + outerRadiusPlug5 * (1 - TMath::Cos(anglesubPlug5)) / 2, 0.);
  tPlug5sub->RegisterYourself();

  TGeoCompositeShape* disPlug5sub = new TGeoCompositeShape(Form("disPlug5subMF%d", disk), Form("dummy + plug5subMF%d:tPlug5subMF%d", disk, disk));

  TGeoRotation* rPlug5sub[nSidePlug5];

  TString namePlug5 = Form("plug5mainMF%d", disk);

  for (Int_t index = 0; index < nSidePlug5; ++index) {
    rPlug5sub[index] = new TGeoRotation(Form("rPlug5sub%dMF%d", index, disk), index * 60, 0, 0);
    rPlug5sub[index]->RegisterYourself();
    namePlug5 += Form(" - disPlug5subMF%d:rPlug5sub%dMF%d", disk, index, disk);
  }

  TGeoCompositeShape* plug5 = new TGeoCompositeShape(Form("plug5MF%d", disk), namePlug5);

  Double_t innerRadiusPlug6 = 0;
  Double_t outerRadiusPlug6 = 0.780 / 2;
  Double_t thicknessPlug6 = 0.150;

  auto* plug6 = new TGeoTube(Form("plug6MF%d", disk), innerRadiusPlug6, outerRadiusPlug6, thicknessPlug6 / 2);

  Double_t innerRadiusPlug7 = innerRadiusPlug6;
  Double_t outerMinRadiusPlug7 = 0.520 / 2;
  Double_t outerMaxRadiusPlug7 = 0.638 / 2;
  Double_t thicknessPlug7 = 0.050;

  auto* plug7 = new TGeoCone(Form("plug7MF%d", disk), thicknessPlug7 / 2, innerRadiusPlug7, outerMaxRadiusPlug7, innerRadiusPlug7, outerMinRadiusPlug7);

  Double_t innerRadiusPlug8 = innerRadiusPlug6;
  Double_t outerRadiusPlug8 = 0.413 / 2;
  Double_t thicknessPlug8 = 0.042;

  auto* plug8 = new TGeoTube(Form("plug8MF%d", disk), innerRadiusPlug8, outerRadiusPlug8, thicknessPlug8 / 2);

  Double_t innerRadiusPlug9 = innerRadiusPlug6;
  Double_t outerMinRadiusPlug9 = outerRadiusPlug8;
  Double_t outerMaxRadiusPlug9 = 0.500 / 2;
  Double_t thicknessPlug9 = 0.040;

  auto* plug9 = new TGeoCone(Form("plug9MF%d", disk), thicknessPlug9 / 2, innerRadiusPlug9, outerMinRadiusPlug9, innerRadiusPlug9, outerMaxRadiusPlug9);

  Double_t innerRadiusPlug10 = innerRadiusPlug6;
  Double_t outerRadiusPlug10 = outerMaxRadiusPlug9;
  Double_t thicknessPlug10 = 0.125;

  auto* plug10 = new TGeoTube(Form("plug10MF%d", disk), innerRadiusPlug10, outerRadiusPlug10, thicknessPlug10 / 2);

  Double_t innerRadiusPlug11 = innerRadiusPlug6;
  Double_t outerMinRadiusPlug11 = outerRadiusPlug8;
  Double_t outerMaxRadiusPlug11 = 0.500 / 2;
  Double_t thicknessPlug11 = 0.043;

  auto* plug11 = new TGeoCone(Form("plug11MF%d", disk), thicknessPlug11 / 2, innerRadiusPlug11, outerMaxRadiusPlug11, innerRadiusPlug11, outerMinRadiusPlug11);

  Double_t innerRadiusPlug12 = 0;
  Double_t outerRadiusPlug12 = 0.289 / 2;
  Double_t thicknessPlug12 = thicknessPlug6 + thicknessPlug7 + thicknessPlug8 + thicknessPlug9 + thicknessPlug10 + thicknessPlug11;

  auto* plug12main = new TGeoTube(Form("plug12mainMF%d", disk), innerRadiusPlug12, outerRadiusPlug12, thicknessPlug12 / 2 + Geometry::sEpsilon);

  const Int_t nSidePlug12 = 6;
  Double_t anglesubPlug12 = 360. / nSidePlug12 / 2. * TMath::Pi() / 180.;
  Double_t lengthPlug12sub = outerRadiusPlug12 * TMath::Cos(anglesubPlug12) * 2;
  Double_t widthPlug12sub = outerRadiusPlug12 * (1 - TMath::Cos(anglesubPlug12));
  Double_t thicknessPlug12sub = thicknessPlug12;

  auto* plug12sub = new TGeoBBox(Form("plug12subMF%d", disk), lengthPlug12sub / 2, widthPlug12sub / 2, thicknessPlug12sub / 2 + Geometry::sEpsilon);

  TGeoTranslation* tPlug12sub = new TGeoTranslation(Form("tPlug12subMF%d", disk), 0, outerRadiusPlug12 * TMath::Cos(anglesubPlug12) + outerRadiusPlug12 * (1 - TMath::Cos(anglesubPlug12)) / 2, 0.);
  tPlug12sub->RegisterYourself();

  TGeoCompositeShape* disPlug12sub = new TGeoCompositeShape(Form("disPlug12subMF%d", disk), Form("dummy + plug12subMF%d:tPlug12subMF%d", disk, disk));

  TGeoRotation* rPlug12sub[nSidePlug12];

  TString namePlug12 = Form("plug12mainMF%d", disk);

  for (Int_t index = 0; index < nSidePlug12; ++index) {
    rPlug12sub[index] = new TGeoRotation(Form("rPlug12sub%dMF%d", index, disk), index * 60, 0, 0);
    rPlug12sub[index]->RegisterYourself();
    namePlug12 += Form(" - disPlug12subMF%d:rPlug12sub%dMF%d", disk, index, disk);
  }

  TGeoCompositeShape* plug12 = new TGeoCompositeShape(Form("plug12MF%d", disk), namePlug12);

  Double_t refposPlug = (thicknessPlug1 + thicknessPlug2 + thicknessPlug3 + thicknessPlug4 + thicknessPlug5 + thicknessPlug6 + thicknessPlug7 + thicknessPlug8 + thicknessPlug9 + thicknessPlug10 + thicknessPlug11) / 2;

  TGeoTranslation* tPlug[12];
  tPlug[0] = new TGeoTranslation(Form("tPlug1MF%d", disk), 0, 0, -refposPlug + thicknessPlug1 / 2);
  tPlug[1] = new TGeoTranslation(Form("tPlug2MF%d", disk), 0, 0, -refposPlug + thicknessPlug1 + thicknessPlug2 / 2);
  tPlug[2] = new TGeoTranslation(Form("tPlug3MF%d", disk), 0, 0, -refposPlug + thicknessPlug1 + thicknessPlug2 + thicknessPlug3 / 2);
  tPlug[3] = new TGeoTranslation(Form("tPlug4MF%d", disk), 0, 0, -refposPlug + thicknessPlug1 + thicknessPlug2 + thicknessPlug3 + thicknessPlug4 / 2);
  tPlug[4] = new TGeoTranslation(Form("tPlug5MF%d", disk), 0, 0, -refposPlug + thicknessPlug1 + thicknessPlug2 + thicknessPlug3 + thicknessPlug4 + thicknessPlug5 / 2);
  tPlug[5] = new TGeoTranslation(Form("tPlug6MF%d", disk), 0, 0, -refposPlug + thicknessPlug1 + thicknessPlug2 + thicknessPlug3 + thicknessPlug4 + thicknessPlug5 + thicknessPlug6 / 2);
  tPlug[6] = new TGeoTranslation(Form("tPlug7MF%d", disk), 0, 0, -refposPlug + thicknessPlug1 + thicknessPlug2 + thicknessPlug3 + thicknessPlug4 + thicknessPlug5 + thicknessPlug6 + thicknessPlug7 / 2);
  tPlug[7] = new TGeoTranslation(Form("tPlug8MF%d", disk), 0, 0, -refposPlug + thicknessPlug1 + thicknessPlug2 + thicknessPlug3 + thicknessPlug4 + thicknessPlug5 + thicknessPlug6 + thicknessPlug7 + thicknessPlug8 / 2);
  tPlug[8] = new TGeoTranslation(Form("tPlug9MF%d", disk), 0, 0, -refposPlug + thicknessPlug1 + thicknessPlug2 + thicknessPlug3 + thicknessPlug4 + thicknessPlug5 + thicknessPlug6 + thicknessPlug7 + thicknessPlug8 + thicknessPlug9 / 2);
  tPlug[9] = new TGeoTranslation(Form("tPlug10MF%d", disk), 0, 0, -refposPlug + thicknessPlug1 + thicknessPlug2 + thicknessPlug3 + thicknessPlug4 + thicknessPlug5 + thicknessPlug6 + thicknessPlug7 + thicknessPlug8 + thicknessPlug9 + thicknessPlug10 / 2);
  tPlug[10] = new TGeoTranslation(Form("tPlug11MF%d", disk), 0, 0, -refposPlug + thicknessPlug1 + thicknessPlug2 + thicknessPlug3 + thicknessPlug4 + thicknessPlug5 + thicknessPlug6 + thicknessPlug7 + thicknessPlug8 + thicknessPlug9 + thicknessPlug10 + thicknessPlug11 / 2);
  tPlug[11] = new TGeoTranslation(Form("tPlug12MF%d", disk), 0, 0, -refposPlug + thicknessPlug1 + thicknessPlug2 + thicknessPlug3 + thicknessPlug4 + thicknessPlug5 + thicknessPlug12 / 2);

  TString namePlug = "";
  for (Int_t ipart = 0; ipart < 12; ++ipart) {
    tPlug[ipart]->RegisterYourself();
    if (ipart == 0)
      namePlug += Form("plug1MF%d:tPlug1MF%d", disk, disk);
    else if (ipart == 11)
      namePlug += Form(" - plug%dMF%d:tPlug%dMF%d", ipart + 1, disk, ipart + 1, disk);
    else
      namePlug += Form(" + plug%dMF%d:tPlug%dMF%d", ipart + 1, disk, ipart + 1, disk);
  }

  TGeoCompositeShape* shapePlug = new TGeoCompositeShape(Form("shapePlugMF%d", disk), namePlug);

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Water
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  Double_t watherThickness[5];
  watherThickness[0] = thicknessBodyBathtub2[0] - thicknessTop[0];
  watherThickness[1] = thicknessBodyBathtub2[1] - thicknessTop[1];
  watherThickness[2] = thicknessBodyBathtub2[2] - thicknessTop[2];
  watherThickness[3] = thicknessBodyBathtub2[3] - thicknessTop[3];
  watherThickness[4] = thicknessBodyBathtub2[4] - thicknessTop[4];

  auto* water1 = new TGeoBBox(Form("water1MF%d", disk), lengthBodyBathtub1 / 2, widthBodyBathtub1 / 2, watherThickness[disk] / 2);
  auto* water2 = new TGeoBBox(Form("water2MF%d", disk), lengthBodyBathtub2 / 2, widthBodyBathtub2 / 2, watherThickness[disk] / 2);
  auto* water3 = new TGeoTube(Form("water3MF%d", disk), 0, cornerRadiusBodyBathtub1[disk], watherThickness[disk] / 2);

  TGeoTranslation* twater[7];
  twater[0] = new TGeoTranslation(Form("twater1MF%d", disk), 0., 0., 0);

  twater[1] = new TGeoTranslation(Form("twater2MF%d", disk), lengthBodyBathtub1 / 2 + lengthBodyBathtub2 / 2, 0., 0.);
  twater[2] = new TGeoTranslation(Form("twater3MF%d", disk), lengthBodyBathtub1 / 2, widthBodyBathtub1 / 2 - cornerRadiusBodyBathtub1[disk], 0.);
  twater[3] = new TGeoTranslation(Form("twater4MF%d", disk), lengthBodyBathtub1 / 2, -(widthBodyBathtub1 / 2 - cornerRadiusBodyBathtub1[disk]), 0.);

  twater[4] = new TGeoTranslation(Form("twater5MF%d", disk), -(lengthBodyBathtub1 / 2 + lengthBodyBathtub2 / 2), 0., 0.);
  twater[5] = new TGeoTranslation(Form("twater6MF%d", disk), -(lengthBodyBathtub1 / 2), widthBodyBathtub1 / 2 - cornerRadiusBodyBathtub1[disk], 0.);
  twater[6] = new TGeoTranslation(Form("twater7MF%d", disk), -(lengthBodyBathtub1 / 2), -(widthBodyBathtub1 / 2 - cornerRadiusBodyBathtub1[disk]), 0.);

  for (Int_t i = 0; i < 7; ++i)
    twater[i]->RegisterYourself();

  TGeoCompositeShape* shapeWater = new TGeoCompositeShape(Form("shapeCoverBathtubMF%d", disk),
                                                          Form("water1MF%d + water2MF%d:twater2MF%d + water3MF%d:twater3MF%d + water3MF%d:twater4MF%d + water2MF%d:twater5MF%d +"
                                                               "water3MF%d:twater6MF%d + water3MF%d:twater7MF%d",
                                                               disk, disk, disk, disk, disk, disk, disk, disk, disk, disk, disk, disk, disk));

  TGeoCombiTrans* transformation1 = nullptr;
  TGeoCombiTrans* transformation2 = nullptr;
  TGeoCombiTrans* transformationplug1 = nullptr;
  TGeoCombiTrans* transformationplug2 = nullptr;
  TGeoCombiTrans* transformationwater1 = nullptr;
  TGeoCombiTrans* transformationwater2 = nullptr;
  TGeoRotation* rotation = nullptr;

  TGeoMedium* kMedPeek = gGeoManager->GetMedium("MFT_PEEK$");
  TGeoMedium* kMed_plug = gGeoManager->GetMedium("MFT_Alu$");
  TGeoMedium* kMed_Water = gGeoManager->GetMedium("MFT_Water$");

  Double_t thicknessTotMF = (thicknessBody[disk] + thicknessMiddle[disk] + thicknessBottom[disk]);

  Double_t deltay = 0.2;     // shift respect to the median plan of the MFT
  Double_t mfX = 2.2;        // width
  Double_t mfY = 6.8 - 0.1;  // height, decrease to avoid overlap with support, to be solved, fm
  Double_t mfZ = 1.7 - 0.85; // thickness, decrease to avoid overlap with support, to be solved, fm
  Double_t fShift = 0;
  if (disk == 3 || disk == 4)
    fShift = 0.015; // to avoid overlap with the 2 curved water pipes on the 2 upstream chambers

  auto* MF01 = new TGeoVolume(Form("MF%d1", disk), shapeManifold, kMedPeek);
  auto* MFplug01 = new TGeoVolume(Form("MFplug%d1", disk), shapePlug, kMed_plug);
  auto* MFwater01 = new TGeoVolume(Form("MFwater%d1", disk), shapeWater, kMed_Water);
  TGeoRotation* rotation1 = new TGeoRotation(Form("rotation1MF%d", disk), 90, 90., 180.);

  transformation1 =
    //new TGeoCombiTrans(mSupportXDimensions[disk][0] / 2 + mfZ / 2 + fShift, mfY / 2 + deltay, mZPlan[disk], rotation1);
    new TGeoCombiTrans(mSupportXDimensions[disk][0] / 2 + 0.1 + thicknessTotMF / 2,
                       mHalfDiskGap + mSupportYDimensions[disk][0] / 2, mZPlan[disk], rotation1);
  mHalfDisk->AddNode(MF01, 1, transformation1);

  TGeoRotation* rotationplug1 = new TGeoRotation(Form("rotationplug1MF%d", disk), -90, 90., 180.);
  transformationplug1 =
    //new TGeoCombiTrans(mSupportXDimensions[disk][0] / 2 + mfZ / 2 + fShift + thicknessTotMF/2 + refposPlug - (thicknessPlug8 + thicknessPlug9 + thicknessPlug10 + thicknessPlug11),mfY / 2 + deltay - ((lengthBody)/2 - holeOffset[3]), mZPlan[disk], rotationplug1);
    new TGeoCombiTrans(mSupportXDimensions[disk][0] / 2 + 0.1 + thicknessTotMF / 2 + thicknessTotMF / 2 + refposPlug - (thicknessPlug8 + thicknessPlug9 + thicknessPlug10 + thicknessPlug11),
                       mHalfDiskGap + mSupportYDimensions[disk][0] / 2 - ((lengthBody) / 2 - holeOffset[3]), mZPlan[disk], rotationplug1);
  mHalfDisk->AddNode(MFplug01, 1, transformationplug1);

  transformationwater1 =
    //new TGeoCombiTrans(mSupportXDimensions[disk][0] / 2 + mfZ / 2 + fShift + (thicknessTotMF/2 - watherThickness[disk]/2) - (thicknessBody[disk] - thicknessTop[disk] - watherThickness[disk]),mfY / 2 + deltay, mZPlan[disk], rotation1);
    new TGeoCombiTrans(mSupportXDimensions[disk][0] / 2 + 0.1 + thicknessTotMF / 2 + (thicknessTotMF / 2 - watherThickness[disk] / 2) - (thicknessBody[disk] - thicknessTop[disk] - watherThickness[disk]),
                       mHalfDiskGap + mSupportYDimensions[disk][0] / 2, mZPlan[disk], rotation1);
  mHalfDisk->AddNode(MFwater01, 1, transformationwater1);

  auto* MF02 = new TGeoVolume(Form("MF%d2", disk), shapeManifold, kMedPeek);
  auto* MFplug02 = new TGeoVolume(Form("MFplug%d2", disk), shapePlug, kMed_plug);
  auto* MFwater02 = new TGeoVolume(Form("MFwater%d2", disk), shapeWater, kMed_Water);
  TGeoRotation* rotation2 = new TGeoRotation(Form("rotation2MF%d", disk), 90, 90., 0.);

  transformation2 =
    //new TGeoCombiTrans(mSupportXDimensions[disk][0] / 2 + mfZ / 2 + fShift, -mfY / 2 - deltay, mZPlan[disk], rotation2);
    new TGeoCombiTrans(mSupportXDimensions[disk][0] / 2 + 0.1 + thicknessTotMF / 2,
                       -mHalfDiskGap - mSupportYDimensions[disk][0] / 2, mZPlan[disk], rotation2);
  mHalfDisk->AddNode(MF02, 1, transformation2);

  TGeoRotation* rotationplug2 = new TGeoRotation(Form("rotationplug1MF%d", disk), -90, 90., 0.);
  transformationplug2 =
    //new TGeoCombiTrans(mSupportXDimensions[disk][0] / 2 + mfZ / 2 + fShift + thicknessTotMF/2 + refposPlug - (thicknessPlug8 + thicknessPlug9 + thicknessPlug10 + thicknessPlug11),-mfY / 2 - deltay + ((lengthBody)/2 - holeOffset[3]), mZPlan[disk], rotationplug2);
    new TGeoCombiTrans(mSupportXDimensions[disk][0] / 2 + 0.1 + thicknessTotMF / 2 + thicknessTotMF / 2 + refposPlug - (thicknessPlug8 + thicknessPlug9 + thicknessPlug10 + thicknessPlug11),
                       -mHalfDiskGap - mSupportYDimensions[disk][0] / 2 + ((lengthBody) / 2 - holeOffset[3]), mZPlan[disk], rotationplug2);
  mHalfDisk->AddNode(MFplug02, 1, transformationplug2);

  transformationwater2 =
    //new TGeoCombiTrans(mSupportXDimensions[disk][0] / 2 + mfZ / 2 + fShift +  ((thicknessTotMF/2 - watherThickness[disk]/2) - (thicknessBody[disk] - thicknessTop[disk] - watherThickness[disk])),-mfY / 2 - deltay, mZPlan[disk], rotation2);
    new TGeoCombiTrans(mSupportXDimensions[disk][0] / 2 + 0.1 + thicknessTotMF / 2 + ((thicknessTotMF / 2 - watherThickness[disk] / 2) - (thicknessBody[disk] - thicknessTop[disk] - watherThickness[disk])),
                       -mHalfDiskGap - mSupportYDimensions[disk][0] / 2, mZPlan[disk], rotation2);
  mHalfDisk->AddNode(MFwater02, 1, transformationwater2);
}

//_____________________________________________________________________________
void HeatExchanger::createHalfDisk0(Int_t half)
{

  Int_t disk = 0;

  if (half == Top)
    printf("Creating MFT heat exchanger for disk0 top\n");
  else if (half == Bottom)
    printf("Creating MFT heat exchanger for disk0 bottom\n");
  else
    printf("No valid option for MFT heat exchanger on disk0\n");

  // mCarbon   = gGeoManager->GetMedium("MFT_Carbon$");
  mCarbon = gGeoManager->GetMedium("MFT_CarbonFiber$");

  mWater = gGeoManager->GetMedium("MFT_Water$");
  mRohacell = gGeoManager->GetMedium("MFT_Rohacell");
  mPipe = gGeoManager->GetMedium("MFT_Polyimide");

  auto* cooling = new TGeoVolumeAssembly(Form("cooling_D0_H%d", half));

  //Float_t lMiddle = mSupportXDimensions[disk][0] - 2. * mLWater; // length of central part

  TGeoTranslation* translation = nullptr;
  TGeoRotation* rotation = nullptr;
  TGeoCombiTrans* transformation = nullptr;

  TGeoCombiTrans* transformation1 = nullptr;
  TGeoCombiTrans* transformation2 = nullptr;

  // **************************************** Water part ****************************************
  // ********************** Four parameters mLwater0, mRadius0, mAngle0, mLpartial0 *************
  Double_t ivolume = 0; // offset chamber 0
  Double_t mRadiusCentralTore[4];
  Double_t xPos0[4];
  Double_t yPos0[4];

  for (Int_t itube = 0; itube < 3; itube++) {
    TGeoVolume* waterTube1 = gGeoManager->MakeTube(Form("waterTube1%d_D0_H%d", itube, half), mWater, 0., mRWater, mLWater0[itube] / 2.);
    translation = new TGeoTranslation(mXPosition0[itube], 0., mSupportXDimensions[0][0] / 2. + mMoreLength - mLWater0[itube] / 2.);
    cooling->AddNode(waterTube1, ivolume++, translation);

    TGeoVolume* waterTorus1 = gGeoManager->MakeTorus(Form("waterTorus1%d_D0_H%d", itube, half), mWater, mRadius0[itube], 0., mRWater, 0., mAngle0[itube]);
    rotation = new TGeoRotation("rotation", 180., -90., 0.);
    transformation = new TGeoCombiTrans(mRadius0[itube] + mXPosition0[itube], 0., mSupportXDimensions[0][0] / 2. + mMoreLength - mLWater0[itube], rotation);
    cooling->AddNode(waterTorus1, ivolume++, transformation);

    TGeoVolume* waterTube2 = gGeoManager->MakeTube(Form("waterTube2%d_D0_H%d", itube, half), mWater, 0., mRWater, mLpartial0[itube] / 2.);
    rotation = new TGeoRotation("rotation", 90., 180 - mAngle0[itube], 0.);
    xPos0[itube] = mLWater0[itube] + mRadius0[itube] * TMath::Sin(mAngle0[itube] * TMath::DegToRad()) + mLpartial0[itube] / 2 * TMath::Cos(mAngle0[itube] * TMath::DegToRad());
    yPos0[itube] = mXPosition0[itube] + mRadius0[itube] * (1 - TMath::Cos(mAngle0[itube] * TMath::DegToRad())) + mLpartial0[itube] / 2 * TMath::Sin(mAngle0[itube] * TMath::DegToRad());
    transformation = new TGeoCombiTrans(yPos0[itube], 0., mSupportXDimensions[0][0] / 2. + mMoreLength - xPos0[itube], rotation);
    cooling->AddNode(waterTube2, ivolume++, transformation);

    mRadiusCentralTore[itube] = (mSupportXDimensions[0][0] / 2. + mMoreLength - xPos0[itube] - mLpartial0[itube] / 2 * TMath::Cos(mAngle0[itube] * TMath::DegToRad())) / TMath::Sin(mAngle0[itube] * TMath::DegToRad());
    TGeoVolume* waterTorusCentral = gGeoManager->MakeTorus(Form("waterTorusCentral%d_D0_H%d", itube, half), mWater, mRadiusCentralTore[itube], 0., mRWater, -mAngle0[itube], 2. * mAngle0[itube]);
    rotation = new TGeoRotation("rotation", 0., 90., 0.);
    transformation = new TGeoCombiTrans(yPos0[itube] + mLpartial0[itube] / 2 * TMath::Sin(mAngle0[itube] * TMath::DegToRad()) - mRadiusCentralTore[itube] * TMath::Cos(mAngle0[itube] * TMath::DegToRad()), 0., 0., rotation);
    cooling->AddNode(waterTorusCentral, ivolume++, transformation);

    TGeoVolume* waterTube3 = gGeoManager->MakeTube(Form("waterTube3%d_D0_H%d", 2, half), mWater, 0., mRWater, mLpartial0[itube] / 2.);
    rotation = new TGeoRotation("rotation", -90., 0 - mAngle0[itube], 0.);
    transformation = new TGeoCombiTrans(yPos0[itube], 0., -(mSupportXDimensions[0][0] / 2. + mMoreLength - xPos0[itube]), rotation);
    cooling->AddNode(waterTube3, ivolume++, transformation);

    TGeoVolume* waterTorus2 = gGeoManager->MakeTorus(Form("waterTorus2%d_D0_H%d", itube, half), mWater, mRadius0[itube], 0., mRWater, 0., mAngle0[itube]);
    rotation = new TGeoRotation("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(mRadius0[itube] + mXPosition0[itube], 0., -(mSupportXDimensions[0][0] / 2. + mMoreLength - mLWater0[itube]), rotation);
    cooling->AddNode(waterTorus2, ivolume++, transformation);

    TGeoVolume* waterTube4 = gGeoManager->MakeTube(Form("waterTube4%d_D0_H%d", itube, half), mWater, 0., mRWater, mLWater0[itube] / 2.);
    translation = new TGeoTranslation(mXPosition0[itube], 0., -(mSupportXDimensions[0][0] / 2. + mMoreLength - mLWater0[itube] / 2.));
    cooling->AddNode(waterTube4, ivolume++, translation);
  }

  // **************************************************** Tube part ************************************************
  // ****************************** Four parameters mLwater0, mRadius0, mAngle0, mLpartial0 ************************
  for (Int_t itube = 0; itube < 3; itube++) {

    TGeoVolume* pipeTube1 = gGeoManager->MakeTube(Form("pipeTube1%d_D0_H%d", itube, half), mPipe, mRWater, mRWater + mDRPipe, mLWater0[itube] / 2.);
    translation = new TGeoTranslation(mXPosition0[itube], 0., mSupportXDimensions[0][0] / 2. + mMoreLength - mLWater0[itube] / 2.);
    cooling->AddNode(pipeTube1, ivolume++, translation);

    TGeoVolume* pipeTorus1 = gGeoManager->MakeTorus(Form("pipeTorus1%d_D0_H%d", itube, half), mPipe, mRadius0[itube], mRWater, mRWater + mDRPipe, 0., mAngle0[itube]);
    rotation = new TGeoRotation("rotation", 180., -90., 0.);
    transformation = new TGeoCombiTrans(mRadius0[itube] + mXPosition0[itube], 0., mSupportXDimensions[0][0] / 2. + mMoreLength - mLWater0[itube], rotation);
    cooling->AddNode(pipeTorus1, ivolume++, transformation);

    TGeoVolume* pipeTube2 = gGeoManager->MakeTube(Form("pipeTube2%d_D0_H%d", itube, half), mPipe, mRWater, mRWater + mDRPipe, mLpartial0[itube] / 2.);
    rotation = new TGeoRotation("rotation", 90., 180 - mAngle0[itube], 0.);
    xPos0[itube] = mLWater0[itube] + mRadius0[itube] * TMath::Sin(mAngle0[itube] * TMath::DegToRad()) + mLpartial0[itube] / 2 * TMath::Cos(mAngle0[itube] * TMath::DegToRad());
    yPos0[itube] = mXPosition0[itube] + mRadius0[itube] * (1 - TMath::Cos(mAngle0[itube] * TMath::DegToRad())) + mLpartial0[itube] / 2 * TMath::Sin(mAngle0[itube] * TMath::DegToRad());
    transformation = new TGeoCombiTrans(yPos0[itube], 0., mSupportXDimensions[0][0] / 2. + mMoreLength - xPos0[itube], rotation);
    cooling->AddNode(pipeTube2, ivolume++, transformation);

    mRadiusCentralTore[itube] = (mSupportXDimensions[0][0] / 2. + mMoreLength - xPos0[itube] - mLpartial0[itube] / 2 * TMath::Cos(mAngle0[itube] * TMath::DegToRad())) / TMath::Sin(mAngle0[itube] * TMath::DegToRad());
    TGeoVolume* pipeTorusCentral = gGeoManager->MakeTorus(Form("pipeTorusCentral%d_D0_H%d", itube, half), mPipe, mRadiusCentralTore[itube], mRWater,
                                                          mRWater + mDRPipe, -mAngle0[itube], 2. * mAngle0[itube]);
    rotation = new TGeoRotation("rotation", 0., 90., 0.);
    transformation = new TGeoCombiTrans(yPos0[itube] + mLpartial0[itube] / 2 * TMath::Sin(mAngle0[itube] * TMath::DegToRad()) - mRadiusCentralTore[itube] * TMath::Cos(mAngle0[itube] * TMath::DegToRad()), 0., 0., rotation);
    cooling->AddNode(pipeTorusCentral, ivolume++, transformation);

    TGeoVolume* pipeTube3 = gGeoManager->MakeTube(Form("pipeTube3%d_D0_H%d", 2, half), mPipe, mRWater, mRWater + mDRPipe, mLpartial0[itube] / 2.);
    rotation = new TGeoRotation("rotation", -90., 0 - mAngle0[itube], 0.);
    transformation = new TGeoCombiTrans(yPos0[itube], 0., -(mSupportXDimensions[0][0] / 2. + mMoreLength - xPos0[itube]), rotation);
    cooling->AddNode(pipeTube3, ivolume++, transformation);

    TGeoVolume* pipeTorus2 = gGeoManager->MakeTorus(Form("pipeTorus2%d_D0_H%d", itube, half), mPipe, mRadius0[itube], mRWater, mRWater + mDRPipe, 0., mAngle0[itube]);
    rotation = new TGeoRotation("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(mRadius0[itube] + mXPosition0[itube], 0., -(mSupportXDimensions[0][0] / 2. + mMoreLength - mLWater0[itube]), rotation);
    cooling->AddNode(pipeTorus2, ivolume++, transformation);

    TGeoVolume* pipeTube4 = gGeoManager->MakeTube(Form("pipeTube4%d_D0_H%d", itube, half), mPipe, mRWater, mRWater + mDRPipe, mLWater0[itube] / 2.);
    translation = new TGeoTranslation(mXPosition0[itube], 0., -(mSupportXDimensions[0][0] / 2. + mMoreLength - mLWater0[itube] / 2.));
    cooling->AddNode(pipeTube4, ivolume++, translation);
  }
  // ***********************************************************************************************

  //Double_t deltaz = mHeatExchangerThickness - mCarbonThickness * 2; // distance between pair of carbon plates
  Double_t deltaz = mHeatExchangerThickness - Geometry::sKaptonOnCarbonThickness * 4 - Geometry::sKaptonGlueThickness * 4 - 2 * mCarbonThickness;

  //  if (half == Top) {
  //    rotation = new TGeoRotation ("rotation", 90., 90., 0.);
  //    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] + deltaz/2. - mCarbonThickness - mRWater - mDRPipe,
  //    rotation);
  //    mHalfDisk->AddNode(cooling, 1, transformation);
  //    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] - deltaz/2. + mCarbonThickness + mRWater + mDRPipe,
  //    rotation);
  //    mHalfDisk->AddNode(cooling, 2, transformation);
  //  }
  //  else if (half == Bottom) {
  rotation = new TGeoRotation("rotation", -90., 90., 0.);
  transformation =
    //new TGeoCombiTrans(0., 0., mZPlan[disk] + deltaz / 2. - mCarbonThickness - mRWater - mDRPipe, rotation);
    new TGeoCombiTrans(0., 0., mZPlan[disk] + deltaz / 2. - mCarbonThickness - mRWater - mDRPipe - 2 * Geometry::sGlueRohacellCarbonThickness, rotation);
  mHalfDisk->AddNode(cooling, 3, transformation);
  transformation =
    //new TGeoCombiTrans(0., 0., mZPlan[disk] - deltaz / 2. + mCarbonThickness + mRWater + mDRPipe, rotation);
    new TGeoCombiTrans(0., 0., mZPlan[disk] - deltaz / 2. + mCarbonThickness + mRWater + mDRPipe + 2 * Geometry::sGlueRohacellCarbonThickness, rotation);
  mHalfDisk->AddNode(cooling, 4, transformation);
  //  }

  // **************************************** Carbon Plates ****************************************

  auto* carbonPlate = new TGeoVolumeAssembly(Form("carbonPlate_D0_H%d", half));

  auto* carbonBase0 = new TGeoBBox(Form("carbonBase0_D0_H%d", half), (mSupportXDimensions[disk][0] / 2. + mMoreLength),
                                   (mSupportYDimensions[disk][0]) / 2., mCarbonThickness);
  auto* t01 = new TGeoTranslation("t01", 0., (mSupportYDimensions[disk][0]) / 2. + mHalfDiskGap, 0.);
  t01->RegisterYourself();

  auto* holeCarbon0 =
    new TGeoTubeSeg(Form("holeCarbon0_D0_H%d", half), 0., mRMin[disk], mCarbonThickness + 0.000001, 0, 180.);
  auto* t02 = new TGeoTranslation("t02", 0., -mHalfDiskGap, 0.);
  t02->RegisterYourself();

  /// TGeoCompositeShape *cs0 = new TGeoCompositeShape(Form("cs0_D0_H%d",half),
  /// Form("(carbonBase0_D0_H%d:t01)-(holeCarbon0_D0_H%d:t02)",half,half));
  auto* carbonhole0 = new TGeoSubtraction(carbonBase0, holeCarbon0, t01, t02);
  auto* ch0 = new TGeoCompositeShape(Form("Carbon0_D0_H%d", half), carbonhole0);
  auto* carbonBaseWithHole0 = new TGeoVolume(Form("carbonBaseWithHole_D0_H%d", half), ch0, mCarbon);

  carbonBaseWithHole0->SetLineColor(kGray + 3);
  rotation = new TGeoRotation("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  carbonPlate->AddNode(carbonBaseWithHole0, 0, new TGeoTranslation(0., 0., mZPlan[disk]));

  Double_t ty = mSupportYDimensions[disk][0];

  for (Int_t ipart = 1; ipart < mNPart[disk]; ipart++) {
    ty += mSupportYDimensions[disk][ipart] / 2.;
    TGeoVolume* partCarbon =
      gGeoManager->MakeBox(Form("partCarbon_D0_H%d_%d", half, ipart), mCarbon, mSupportXDimensions[disk][ipart] / 2.,
                           mSupportYDimensions[disk][ipart] / 2., mCarbonThickness);
    partCarbon->SetLineColor(kGray + 3);
    auto* t = new TGeoTranslation("t", 0, ty + mHalfDiskGap, mZPlan[disk]);
    carbonPlate->AddNode(partCarbon, ipart, t);
    ty += mSupportYDimensions[disk][ipart] / 2.;
  }

  //  if (half == Top) {
  //    rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  //    transformation = new TGeoCombiTrans(0., 0., deltaz/2., rotation);
  //    mHalfDisk->AddNode(carbonPlate, 1, transformation);
  //    transformation = new TGeoCombiTrans(0., 0., -deltaz/2., rotation);
  //    mHalfDisk->AddNode(carbonPlate, 2, transformation);
  //  }
  //  else if (half == Bottom) {
  rotation = new TGeoRotation("rotation", 180., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., deltaz / 2., rotation);
  mHalfDisk->AddNode(carbonPlate, 3, transformation);
  transformation = new TGeoCombiTrans(0., 0., -deltaz / 2., rotation);
  mHalfDisk->AddNode(carbonPlate, 4, transformation);
  //  }

  // **************************************** Glue Bwtween Carbon Plate and Rohacell Plate ****************************************

  TGeoMedium* mGlueRohacellCarbon = gGeoManager->GetMedium("MFT_Epoxy$");

  auto* glueRohacellCarbon = new TGeoVolumeAssembly(Form("glueRohacellCarbon_D0_H%d", half));

  auto* glueRohacellCarbonBase0 = new TGeoBBox(Form("glueRohacellCarbonBase0_D0_H%d", half), (mSupportXDimensions[disk][0]) / 2.,
                                               (mSupportYDimensions[disk][0]) / 2., Geometry::sGlueRohacellCarbonThickness);

  auto* translation_gluRC01 = new TGeoTranslation("translation_gluRC01", 0., (mSupportYDimensions[disk][0]) / 2. + mHalfDiskGap, 0.);
  translation_gluRC01->RegisterYourself();
  auto* translation_gluRC02 = new TGeoTranslation("translation_gluRC02", 0., -mHalfDiskGap, 0.);
  translation_gluRC02->RegisterYourself();

  auto* holeglueRohacellCarbon0 =
    new TGeoTubeSeg(Form("holeglueRohacellCarbon0_D0_H%d", half), 0., mRMin[disk], Geometry::sGlueRohacellCarbonThickness + 0.000001, 0, 180.);

  auto* glueRohacellCarbonhole0 = new TGeoSubtraction(glueRohacellCarbonBase0, holeglueRohacellCarbon0, translation_gluRC01, translation_gluRC02);
  auto* gRC0 = new TGeoCompositeShape(Form("glueRohacellCarbon0_D0_H%d", half), glueRohacellCarbonhole0);
  auto* glueRohacellCarbonBaseWithHole0 = new TGeoVolume(Form("glueRohacellCarbonWithHole_D0_H%d", half), gRC0, mGlueRohacellCarbon);

  glueRohacellCarbonBaseWithHole0->SetLineColor(kGreen);
  rotation = new TGeoRotation("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  glueRohacellCarbon->AddNode(glueRohacellCarbonBaseWithHole0, 0, new TGeoTranslation(0., 0., mZPlan[disk]));

  Double_t tyGRC = mSupportYDimensions[disk][0];

  for (Int_t ipart = 1; ipart < mNPart[disk]; ipart++) {
    tyGRC += mSupportYDimensions[disk][ipart] / 2.;
    TGeoVolume* partGlueRohacellCarbon =
      gGeoManager->MakeBox(Form("partGlueRohacellCarbon_D0_H%d_%d", half, ipart), mGlueRohacellCarbon, mSupportXDimensions[disk][ipart] / 2.,
                           mSupportYDimensions[disk][ipart] / 2., Geometry::sGlueRohacellCarbonThickness);
    partGlueRohacellCarbon->SetLineColor(kGreen);
    auto* t = new TGeoTranslation("t", 0, tyGRC + mHalfDiskGap, mZPlan[disk]);
    glueRohacellCarbon->AddNode(partGlueRohacellCarbon, ipart, t);
    tyGRC += mSupportYDimensions[disk][ipart] / 2.;
  }

  rotation = new TGeoRotation("rotation", 180., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., deltaz / 2. - mCarbonThickness - Geometry::sGlueRohacellCarbonThickness, rotation);
  mHalfDisk->AddNode(glueRohacellCarbon, 3, transformation);
  transformation = new TGeoCombiTrans(0., 0., -(deltaz / 2. - mCarbonThickness - Geometry::sGlueRohacellCarbonThickness), rotation);
  mHalfDisk->AddNode(glueRohacellCarbon, 4, transformation);

  // **************************************** Kapton on Carbon Plate ****************************************

  TGeoMedium* mKaptonOnCarbon = gGeoManager->GetMedium("MFT_Kapton$");

  auto* kaptonOnCarbon = new TGeoVolumeAssembly(Form("kaptonOnCarbon_D0_H%d", half));
  auto* kaptonOnCarbonBase0 = new TGeoBBox(Form("kaptonOnCarbonBase0_D0_H%d", half), (mSupportXDimensions[disk][0] / 2. + mMoreLength),
                                           (mSupportYDimensions[disk][0]) / 2., Geometry::sKaptonOnCarbonThickness);

  auto* translation_KC01 = new TGeoTranslation("translation_KC01", 0., (mSupportYDimensions[disk][0]) / 2. + mHalfDiskGap, 0.);
  translation_KC01->RegisterYourself();
  auto* translation_KC02 = new TGeoTranslation("translation_KC02", 0., -mHalfDiskGap, 0.);
  translation_KC02->RegisterYourself();

  auto* holekaptonOnCarbon0 =
    new TGeoTubeSeg(Form("holekaptonOnCarbon0_D0_H%d", half), 0., mRMin[disk], Geometry::sKaptonOnCarbonThickness + 0.000001, 0, 180.);

  auto* kaptonOnCarbonhole0 = new TGeoSubtraction(kaptonOnCarbonBase0, holekaptonOnCarbon0, translation_KC01, translation_KC02);
  auto* KC0 = new TGeoCompositeShape(Form("kaptonOnCarbon_D0_H%d", half), kaptonOnCarbonhole0);
  auto* kaptonOnCarbonBaseWithHole0 = new TGeoVolume(Form("kaptonOnCarbonWithHole_D0_H%d", half), KC0, mKaptonOnCarbon);

  kaptonOnCarbonBaseWithHole0->SetLineColor(kMagenta);
  rotation = new TGeoRotation("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  kaptonOnCarbon->AddNode(kaptonOnCarbonBaseWithHole0, 0, new TGeoTranslation(0., 0., mZPlan[disk]));

  Double_t tyKC = mSupportYDimensions[disk][0];

  for (Int_t ipart = 1; ipart < mNPart[disk]; ipart++) {
    tyKC += mSupportYDimensions[disk][ipart] / 2.;
    TGeoVolume* partkaptonOnCarbonBase =
      gGeoManager->MakeBox(Form("partkaptonOnCarbon_D0_H%d_%d", half, ipart), mKaptonOnCarbon, mSupportXDimensions[disk][ipart] / 2.,
                           mSupportYDimensions[disk][ipart] / 2., Geometry::sKaptonOnCarbonThickness);
    partkaptonOnCarbonBase->SetLineColor(kMagenta);
    auto* t = new TGeoTranslation("t", 0, tyKC + mHalfDiskGap, mZPlan[disk]);
    kaptonOnCarbon->AddNode(partkaptonOnCarbonBase, ipart, t);
    tyKC += mSupportYDimensions[disk][ipart] / 2.;
  }

  rotation = new TGeoRotation("rotation", 180., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., deltaz / 2 + Geometry::sKaptonOnCarbonThickness + mCarbonThickness + Geometry::sKaptonGlueThickness * 2, rotation);
  mHalfDisk->AddNode(kaptonOnCarbon, 3, transformation);
  transformation = new TGeoCombiTrans(0., 0., -(deltaz / 2 + Geometry::sKaptonOnCarbonThickness + mCarbonThickness + Geometry::sKaptonGlueThickness * 2), rotation);
  mHalfDisk->AddNode(kaptonOnCarbon, 4, transformation);

  // **************************************** Kapton glue on the carbon plate ****************************************

  TGeoMedium* mGlueKaptonCarbon = gGeoManager->GetMedium("MFT_Epoxy$");

  auto* glueKaptonCarbon = new TGeoVolumeAssembly(Form("glueKaptonCarbon_D0_H%d", half));

  auto* glueKaptonCarbonBase0 = new TGeoBBox(Form("glueKaptonCarbonBase0_D0_H%d", half), (mSupportXDimensions[disk][0] / 2. + mMoreLength),
                                             (mSupportYDimensions[disk][0]) / 2., Geometry::sKaptonGlueThickness);

  auto* translation_gluKC01 = new TGeoTranslation("translation_gluKC01", 0., (mSupportYDimensions[disk][0]) / 2. + mHalfDiskGap, 0.);
  translation_gluKC01->RegisterYourself();
  auto* translation_gluKC02 = new TGeoTranslation("translation_gluKC02", 0., -mHalfDiskGap, 0.);
  translation_gluKC02->RegisterYourself();

  auto* holeglueKaptonCarbon0 =
    new TGeoTubeSeg(Form("holeglueKaptonCarbon0_D0_H%d", half), 0., mRMin[disk], Geometry::sKaptonGlueThickness + 0.000001, 0, 180.);

  auto* glueKaptonCarbonhole0 = new TGeoSubtraction(glueKaptonCarbonBase0, holeglueKaptonCarbon0, translation_gluKC01, translation_gluKC02);
  auto* gKC0 = new TGeoCompositeShape(Form("glueKaptonCarbon0_D0_H%d", half), glueKaptonCarbonhole0);
  auto* glueKaptonCarbonBaseWithHole0 = new TGeoVolume(Form("glueKaptonCarbonWithHole_D0_H%d", half), gKC0, mGlueKaptonCarbon);

  glueKaptonCarbonBaseWithHole0->SetLineColor(kGreen);
  rotation = new TGeoRotation("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  glueKaptonCarbon->AddNode(glueKaptonCarbonBaseWithHole0, 0, new TGeoTranslation(0., 0., mZPlan[disk]));

  Double_t tyGKC = mSupportYDimensions[disk][0];

  for (Int_t ipart = 1; ipart < mNPart[disk]; ipart++) {
    tyGKC += mSupportYDimensions[disk][ipart] / 2.;
    TGeoVolume* partGlueKaptonCarbon =
      gGeoManager->MakeBox(Form("partGlueKaptonCarbon_D0_H%d_%d", half, ipart), mGlueKaptonCarbon, mSupportXDimensions[disk][ipart] / 2.,
                           mSupportYDimensions[disk][ipart] / 2., Geometry::sKaptonGlueThickness);
    partGlueKaptonCarbon->SetLineColor(kGreen);
    auto* t = new TGeoTranslation("t", 0, tyGKC + mHalfDiskGap, mZPlan[disk]);
    glueKaptonCarbon->AddNode(partGlueKaptonCarbon, ipart, t);
    tyGKC += mSupportYDimensions[disk][ipart] / 2.;
  }

  rotation = new TGeoRotation("rotation", 180., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., deltaz / 2. + mCarbonThickness + Geometry::sKaptonGlueThickness, rotation);
  mHalfDisk->AddNode(glueKaptonCarbon, 3, transformation);
  transformation = new TGeoCombiTrans(0., 0., -(deltaz / 2. + mCarbonThickness + Geometry::sKaptonGlueThickness), rotation);
  mHalfDisk->AddNode(glueKaptonCarbon, 4, transformation);

  // **************************************** Rohacell Plate ****************************************

  auto* rohacellPlate = new TGeoVolumeAssembly(Form("rohacellPlate_D0_H%d", half));

  auto* rohacellBase0 = new TGeoBBox(Form("rohacellBase0_D0_H%d", half), (mSupportXDimensions[disk][0]) / 2., (mSupportYDimensions[disk][0]) / 2.,
                                     mRohacellThickness);
  //rohacellBase0->SetName("rB");
  // TGeoTranslation *t3 = new TGeoTranslation ("t3",0., (fSupportYDimensions[disk][0])/2. + fHalfDiskGap , 0.);
  // t3 -> RegisterYourself();

  auto* holeRohacell0 = new TGeoTubeSeg(Form("holeRohacell0_D0_H%d", half), 0., mRMin[disk], mRohacellThickness + 0.000001, 0, 180.);
  // TGeoTranslation *t4= new TGeoTranslation ("t4", 0., - fHalfDiskGap , 0.);
  // t4-> RegisterYourself();
  //auto* cs0 = new TGeoCompositeShape("cs0", Form("(rohacellBase0_D0_H%d:t01)-(holeRohacell0_D0_H%d:t02)",half,half));

  // **************************************** GROOVES *************************************************
  Double_t diameter = 0.21; // groove diameter
  Double_t epsilon = 0.06;  // outside shift of the goove
  Int_t iCount = 0;
  Double_t mPosition[4];

  TGeoCombiTrans* transfo[7][3];
  TGeoTube* grooveTube[7][3];
  TGeoTorus* grooveTorus[7][3];
  TGeoSubtraction* rohacellBaseGroove[300];
  TGeoCompositeShape* rohacellGroove[300];

  for (Int_t igroove = 0; igroove < 3; igroove++) {
    grooveTube[0][igroove] = new TGeoTube("linear", 0., diameter, mLWater0[igroove] / 2.);
    grooveTorus[1][igroove] = new TGeoTorus("SideTorus", mRadius0[igroove], 0., diameter, 0., mAngle0[igroove]);
    grooveTube[2][igroove] = new TGeoTube("tiltedLinear", 0., diameter, mLpartial0[igroove] / 2.);
    grooveTorus[3][igroove] = new TGeoTorus("centralTorus", mRadiusCentralTore[igroove], 0., diameter, -mAngle0[igroove], 2. * mAngle0[igroove]);
    grooveTube[4][igroove] = new TGeoTube("tiltedLinear", 0., diameter, mLpartial0[igroove] / 2.);
    grooveTorus[5][igroove] = new TGeoTorus("SideTorus", mRadius0[igroove], 0., diameter, 0., mAngle0[igroove]);
    grooveTube[6][igroove] = new TGeoTube("linear", 0., diameter, mLWater0[igroove] / 2.);
  }

  // Rotation matrix
  TGeoRotation* rotationLinear = new TGeoRotation("rotation", -90., 90., 0.);
  TGeoRotation* rotationSideTorusL = new TGeoRotation("rotationSideTorusLeft", -90., 0., 0.);
  TGeoRotation* rotationSideTorusR = new TGeoRotation("rotationSideTorusRight", 90., 180., 180.);
  TGeoRotation* rotationCentralTorus = new TGeoRotation("rotationCentralTorus", 90., 0., 0.);
  TGeoRotation* rotationTiltedLinearR;
  TGeoRotation* rotationTiltedLinearL;

  // Creating grooves
  for (Int_t iface = 1; iface > -2; iface -= 2) {     // front and rear
    for (Int_t igroove = 0; igroove < 3; igroove++) { // 3 grooves
      mPosition[igroove] = mXPosition0[igroove] - mSupportYDimensions[disk][0] / 2. - mHalfDiskGap;
      for (Int_t ip = 0; ip < 7; ip++) { // each groove is made of 7 parts

        switch (ip) {
          case 0: // Linear
            transfo[ip][igroove] = new TGeoCombiTrans(mSupportXDimensions[disk][0] / 2. + mMoreLength - mLWater0[igroove] / 2., mPosition[igroove],
                                                      iface * (mRohacellThickness + epsilon), rotationLinear);
            if (igroove == 0 && iface == 1) {
              rohacellBaseGroove[0] = new TGeoSubtraction(rohacellBase0, grooveTube[ip][igroove], nullptr, transfo[ip][igroove]);
              rohacellGroove[0] = new TGeoCompositeShape(Form("rohacell0Groove%d_G%d_F%d_H%d", ip, igroove, iface, half), rohacellBaseGroove[0]);
            };
            break;
          case 1: // side torus
            transfo[ip][igroove] = new TGeoCombiTrans(mSupportXDimensions[disk][0] / 2. + mMoreLength - mLWater0[igroove], mRadius0[igroove] + mXPosition0[igroove] - mSupportYDimensions[disk][0] / 2. - mHalfDiskGap, iface * (mRohacellThickness + epsilon), rotationSideTorusR);
            break;
          case 2: // Linear tilted
            rotationTiltedLinearR = new TGeoRotation("rotationTiltedLinearRight", 90. - mAngle0[igroove], 90., 0.);
            transfo[ip][igroove] = new TGeoCombiTrans(mSupportXDimensions[disk][0] / 2. + mMoreLength - xPos0[igroove], yPos0[igroove] - mSupportYDimensions[disk][0] / 2. - mHalfDiskGap, iface * (mRohacellThickness + epsilon), rotationTiltedLinearR);
            break;
          case 3: // Central Torus
            transfo[ip][igroove] = new TGeoCombiTrans(0., yPos0[igroove] + mLpartial0[igroove] / 2 * TMath::Sin(mAngle0[igroove] * TMath::DegToRad()) - mRadiusCentralTore[igroove] * TMath::Cos(mAngle0[igroove] * TMath::DegToRad()) - mSupportYDimensions[disk][0] / 2. - mHalfDiskGap,
                                                      iface * (mRohacellThickness + epsilon), rotationCentralTorus);
            break;
          case 4: // Linear tilted
            rotationTiltedLinearL = new TGeoRotation("rotationTiltedLinearLeft", 90. + mAngle0[igroove], 90., 0.);
            transfo[ip][igroove] = new TGeoCombiTrans(-(mSupportXDimensions[disk][0] / 2. + mMoreLength - xPos0[igroove]), yPos0[igroove] - mSupportYDimensions[disk][0] / 2. - mHalfDiskGap,
                                                      iface * (mRohacellThickness + epsilon), rotationTiltedLinearL);
            break;
          case 5: // side torus
            transfo[ip][igroove] = new TGeoCombiTrans(-(mSupportXDimensions[disk][0] / 2. + mMoreLength - mLWater0[igroove]), mRadius0[igroove] + mPosition[igroove],
                                                      iface * (mRohacellThickness + epsilon), rotationSideTorusL);
            break;
          case 6: // Linear
            transfo[ip][igroove] = new TGeoCombiTrans(-(mSupportXDimensions[disk][0] / 2. + mMoreLength - mLWater0[igroove] / 2.), mPosition[igroove],
                                                      iface * (mRohacellThickness + epsilon), rotationLinear);
            break;
        }

        if (!(ip == 0 && igroove == 0 && iface == 1)) {
          if (ip & 1) {
            rohacellBaseGroove[iCount] = new TGeoSubtraction(rohacellGroove[iCount - 1], grooveTorus[ip][igroove], nullptr, transfo[ip][igroove]);
          } else {
            rohacellBaseGroove[iCount] = new TGeoSubtraction(rohacellGroove[iCount - 1], grooveTube[ip][igroove], nullptr, transfo[ip][igroove]);
          }
          rohacellGroove[iCount] = new TGeoCompositeShape(Form("rohacell0Groove%d_G%d_F%d_H%d", iCount, igroove, iface, half), rohacellBaseGroove[iCount]);
        }
        iCount++;
      }
    }
  }
  // **************************************************************************************************

  // Passage du beam pipe
  //auto* rohacellBase = new TGeoSubtraction(rohacellBase0, holeRohacell0, t01, t02);
  auto* rohacellBase = new TGeoSubtraction(rohacellGroove[iCount - 1], holeRohacell0, t01, t02);
  auto* rh0 = new TGeoCompositeShape(Form("rohacellTore%d_D0_H%d", 0, half), rohacellBase);

  //auto* rohacellBaseWithHole = new TGeoVolume(Form("rohacellBaseWithHole_D0_H%d", half), rh0test, mRohacell);
  auto* rohacellBaseWithHole = new TGeoVolume(Form("rohacellBaseWithHole_D0_H%d", half), rh0, mRohacell);
  //auto* rohacellBaseWithHole = new TGeoVolume(Form("rohacellBaseWithHole_D0_H%d", half), cs0, mRohacell);

  TGeoVolume* partRohacell;
  rohacellBaseWithHole->SetLineColor(kGray);
  rotation = new TGeoRotation("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  rohacellPlate->AddNode(rohacellBaseWithHole, 0, new TGeoTranslation(0., 0., mZPlan[disk]));

  ty = mSupportYDimensions[disk][0];

  for (Int_t ipart = 1; ipart < mNPart[disk]; ipart++) {
    ty += mSupportYDimensions[disk][ipart] / 2.;
    /*
      partRohacell = gGeoManager->MakeBox(Form("partRohacelli_D0_H%d_%d", half, ipart), mRohacell,
      mSupportXDimensions[disk][ipart] / 2.,
      mSupportYDimensions[disk][ipart] / 2., mRohacellThickness);
      partRohacell->SetLineColor(kGray);
    */
    auto* t = new TGeoTranslation("t", 0, ty + mHalfDiskGap, mZPlan[disk]);

    //===========================================================================================================
    //===========================================================================================================
    auto* partRohacell0 =
      new TGeoBBox(Form("rohacellBase0_D0_H%d_%d", half, ipart), mSupportXDimensions[disk][ipart] / 2.,
                   mSupportYDimensions[disk][ipart] / 2., mRohacellThickness);

    // ************************ Creating grooves for the other parts of the rohacell plate **********************
    Double_t mShift;
    for (Int_t iface = 1; iface > -2; iface -= 2) {     // front and rear
      for (Int_t igroove = 0; igroove < 3; igroove++) { // 3 grooves
        if (ipart == 1) {
          mPosition[ipart] = mXPosition0[igroove] - mSupportYDimensions[disk][ipart] / 2. - mHalfDiskGap - mSupportYDimensions[disk][ipart - 1];
          mShift = -mSupportYDimensions[disk][ipart - 1];
        };
        if (ipart == 2) {
          mPosition[ipart] = mXPosition0[igroove] - mSupportYDimensions[disk][ipart] / 2. - mHalfDiskGap - mSupportYDimensions[disk][ipart - 1] - mSupportYDimensions[disk][ipart - 2];
          mShift = -mSupportYDimensions[disk][ipart - 1] - mSupportYDimensions[disk][ipart - 2];
        };
        if (ipart == 3) {
          mPosition[ipart] = mXPosition0[igroove] - mSupportYDimensions[disk][ipart] / 2. - mHalfDiskGap - mSupportYDimensions[disk][ipart - 1] - mSupportYDimensions[disk][ipart - 2] - mSupportYDimensions[disk][ipart - 3];
          mShift = -mSupportYDimensions[disk][ipart - 1] - mSupportYDimensions[disk][ipart - 2] - mSupportYDimensions[disk][ipart - 3];
        };

        for (Int_t ip = 0; ip < 7; ip++) { // each groove is made of 7 parts

          switch (ip) {
            case 0: // Linear
              transfo[ip][igroove] = new TGeoCombiTrans(mSupportXDimensions[0][0] / 2. - mLWater0[igroove] / 2., mPosition[ipart], iface * (mRohacellThickness + epsilon), rotationLinear);
              if (igroove == 0 && iface == 1) {
                rohacellBaseGroove[iCount] = new TGeoSubtraction(partRohacell0, grooveTube[ip][igroove], nullptr, transfo[ip][igroove]);
                rohacellGroove[iCount] = new TGeoCompositeShape(Form("rohacell0Groove%d_G%d_F%d_H%d", ip, igroove, iface, half), rohacellBaseGroove[iCount]);
              };
              break;
            case 1: // side torus
              transfo[ip][igroove] = new TGeoCombiTrans(mSupportXDimensions[0][0] / 2. + mMoreLength - mLWater0[igroove], mPosition[ipart] + mRadius0[igroove],
                                                        iface * (mRohacellThickness + epsilon), rotationSideTorusR);
              break;
            case 2: // Linear tilted
              rotationTiltedLinearR = new TGeoRotation("rotationTiltedLinearRight", 90. - mAngle0[igroove], 90., 0.);
              transfo[ip][igroove] = new TGeoCombiTrans(mSupportXDimensions[0][0] / 2. + mMoreLength - xPos0[igroove], yPos0[igroove] + mShift - mHalfDiskGap - mSupportYDimensions[disk][ipart] / 2., iface * (mRohacellThickness + epsilon), rotationTiltedLinearR);
              break;
            case 3: // Central Torus
              transfo[ip][igroove] = new TGeoCombiTrans(0., mPosition[ipart] + yPos0[igroove] + mLpartial0[igroove] / 2 * TMath::Sin(mAngle0[igroove] * TMath::DegToRad()) - mRadiusCentralTore[igroove] * TMath::Cos(mAngle0[igroove] * TMath::DegToRad()) - mXPosition0[igroove],
                                                        iface * (mRohacellThickness + epsilon), rotationCentralTorus);
              break;
            case 4: // Linear tilted
              rotationTiltedLinearL = new TGeoRotation("rotationTiltedLinearLeft", 90. + mAngle0[igroove], 90., 0.);
              transfo[ip][igroove] = new TGeoCombiTrans(-(mSupportXDimensions[0][0] / 2. + mMoreLength - xPos0[igroove]), yPos0[igroove] + mPosition[ipart] - mXPosition0[igroove],
                                                        iface * (mRohacellThickness + epsilon), rotationTiltedLinearL);
              break;
            case 5: // side torus
              transfo[ip][igroove] = new TGeoCombiTrans(-(mSupportXDimensions[0][0] / 2. + mMoreLength - mLWater0[igroove]), mRadius0[igroove] + mPosition[ipart],
                                                        iface * (mRohacellThickness + epsilon), rotationSideTorusL);
              break;
            case 6: // Linear
              transfo[ip][igroove] = new TGeoCombiTrans(-(mSupportXDimensions[0][0] / 2. + mMoreLength - mLWater0[igroove] / 2.), mPosition[ipart],
                                                        iface * (mRohacellThickness + epsilon), rotationLinear);
              break;
          }
          if (!(ip == 0 && igroove == 0 && iface == 1)) {
            if (ip & 1) {
              rohacellBaseGroove[iCount] = new TGeoSubtraction(rohacellGroove[iCount - 1], grooveTorus[ip][igroove], nullptr, transfo[ip][igroove]);
            } else {
              rohacellBaseGroove[iCount] = new TGeoSubtraction(rohacellGroove[iCount - 1], grooveTube[ip][igroove], nullptr, transfo[ip][igroove]);
            }

            rohacellGroove[iCount] = new TGeoCompositeShape(Form("rohacell0Groove%d_G%d_F%d_H%d", iCount, igroove, iface, half), rohacellBaseGroove[iCount]);
          }
          iCount++;
        }
      }
    }

    partRohacell = new TGeoVolume(Form("partRohacelli_D0_H%d_%d", half, ipart), rohacellGroove[iCount - 1], mRohacell);
    //partRohacell = new TGeoVolume(Form("partRohacelli_D0_H%d_%d", half, ipart), partRohacell0, mRohacell);
    //===========================================================================================================
    //===========================================================================================================

    rohacellPlate->AddNode(partRohacell, ipart, t);

    ty += mSupportYDimensions[disk][ipart] / 2.;
  }

  //  if (half == Top) {
  //    rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  //    transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  //    mHalfDisk->AddNode(rohacellPlate, 1, transformation);
  //  }
  //  if (half == Bottom) {
  rotation = new TGeoRotation("rotation", 180., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  mHalfDisk->AddNode(rohacellPlate, 1, transformation);
  //  }

  createManifold(disk);
}

//_____________________________________________________________________________
void HeatExchanger::createHalfDisk1(Int_t half)
{

  Int_t disk = 1;

  if (half == Top)
    printf("Creating MFT heat exchanger for disk1 top\n");
  else if (half == Bottom)
    printf("Creating MFT heat exchanger for disk1 bottom\n");
  else
    printf("No valid option for MFT heat exchanger on disk1\n");

  // mCarbon   = gGeoManager->GetMedium("MFT_Carbon$");
  mCarbon = gGeoManager->GetMedium("MFT_CarbonFiber$");

  mWater = gGeoManager->GetMedium("MFT_Water$");
  mRohacell = gGeoManager->GetMedium("MFT_Rohacell");
  mPipe = gGeoManager->GetMedium("MFT_Polyimide");

  auto* cooling = new TGeoVolumeAssembly(Form("cooling_D1_H%d", half));

  //Float_t lMiddle = mSupportXDimensions[disk][0] - 2. * mLWater; // length of central part

  TGeoTranslation* translation = nullptr;
  TGeoRotation* rotation = nullptr;
  TGeoCombiTrans* transformation = nullptr;

  // **************************************** Water part ****************************************
  // ********************** Four parameters mLwater1, mRadius1, mAngle1, mLpartial1 *************
  Double_t ivolume = 100; // offset chamber 1
  Double_t mRadiusCentralTore[4];
  Double_t xPos1[4];
  Double_t yPos1[4];
  for (Int_t itube = 0; itube < 3; itube++) {
    TGeoVolume* waterTube1 = gGeoManager->MakeTube(Form("waterTube1%d_D1_H%d", itube, half), mWater, 0., mRWater, mLWater1[itube] / 2.);
    translation = new TGeoTranslation(mXPosition1[itube], 0., mSupportXDimensions[1][0] / 2. + mMoreLength - mLWater1[itube] / 2.);
    cooling->AddNode(waterTube1, ivolume++, translation);

    TGeoVolume* waterTorus1 = gGeoManager->MakeTorus(Form("waterTorus1%d_D1_H%d", itube, half), mWater, mRadius1[itube], 0., mRWater, 0., mAngle1[itube]);
    rotation = new TGeoRotation("rotation", 180., -90., 0.);
    transformation = new TGeoCombiTrans(mRadius1[itube] + mXPosition1[itube], 0., mSupportXDimensions[1][0] / 2. + mMoreLength - mLWater1[itube], rotation);
    cooling->AddNode(waterTorus1, ivolume++, transformation);

    TGeoVolume* waterTube2 = gGeoManager->MakeTube(Form("waterTube2%d_D1_H%d", itube, half), mWater, 0., mRWater, mLpartial1[itube] / 2.);
    rotation = new TGeoRotation("rotation", 90., 180 - mAngle1[itube], 0.);
    xPos1[itube] = mLWater1[itube] + mRadius1[itube] * TMath::Sin(mAngle1[itube] * TMath::DegToRad()) + mLpartial1[itube] / 2 * TMath::Cos(mAngle1[itube] * TMath::DegToRad());
    yPos1[itube] = mXPosition1[itube] + mRadius1[itube] * (1 - TMath::Cos(mAngle1[itube] * TMath::DegToRad())) + mLpartial1[itube] / 2 * TMath::Sin(mAngle1[itube] * TMath::DegToRad());
    transformation = new TGeoCombiTrans(yPos1[itube], 0., mSupportXDimensions[1][0] / 2. + mMoreLength - xPos1[itube], rotation);
    cooling->AddNode(waterTube2, ivolume++, transformation);

    mRadiusCentralTore[itube] = (mSupportXDimensions[1][0] / 2. + mMoreLength - xPos1[itube] - mLpartial1[itube] / 2 * TMath::Cos(mAngle1[itube] * TMath::DegToRad())) / TMath::Sin(mAngle1[itube] * TMath::DegToRad());
    TGeoVolume* waterTorusCentral = gGeoManager->MakeTorus(Form("waterTorusCentral%d_D1_H%d", itube, half), mWater, mRadiusCentralTore[itube], 0., mRWater,
                                                           -mAngle1[itube], 2. * mAngle1[itube]);
    rotation = new TGeoRotation("rotation", 0., 90., 0.);
    transformation = new TGeoCombiTrans(yPos1[itube] + mLpartial1[itube] / 2 * TMath::Sin(mAngle1[itube] * TMath::DegToRad()) - mRadiusCentralTore[itube] * TMath::Cos(mAngle1[itube] * TMath::DegToRad()), 0., 0., rotation);
    cooling->AddNode(waterTorusCentral, ivolume++, transformation);

    TGeoVolume* waterTube3 = gGeoManager->MakeTube(Form("waterTube3%d_D1_H%d", 2, half), mWater, 0., mRWater, mLpartial1[itube] / 2.);
    rotation = new TGeoRotation("rotation", -90., 0 - mAngle1[itube], 0.);
    transformation = new TGeoCombiTrans(yPos1[itube], 0., -(mSupportXDimensions[1][0] / 2. + mMoreLength - xPos1[itube]), rotation);
    cooling->AddNode(waterTube3, ivolume++, transformation);

    TGeoVolume* waterTorus2 = gGeoManager->MakeTorus(Form("waterTorus2%d_D1_H%d", itube, half), mWater, mRadius1[itube], 0., mRWater, 0., mAngle1[itube]);
    rotation = new TGeoRotation("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(mRadius1[itube] + mXPosition1[itube], 0., -(mSupportXDimensions[1][0] / 2. + mMoreLength - mLWater1[itube]), rotation);
    cooling->AddNode(waterTorus2, ivolume++, transformation);

    TGeoVolume* waterTube4 = gGeoManager->MakeTube(Form("waterTube4%d_D1_H%d", itube, half), mWater, 0., mRWater, mLWater1[itube] / 2.);
    translation = new TGeoTranslation(mXPosition1[itube], 0., -(mSupportXDimensions[1][0] / 2. + mMoreLength - mLWater1[itube] / 2.));
    cooling->AddNode(waterTube4, ivolume++, translation);
  }

  // **************************************************** Tube part ************************************************
  // ****************************** Four parameters mLwater1, mRadius1, mAngle1, mLpartial1 ************************
  for (Int_t itube = 0; itube < 3; itube++) {

    TGeoVolume* pipeTube1 = gGeoManager->MakeTube(Form("pipeTube1%d_D1_H%d", itube, half), mPipe, mRWater, mRWater + mDRPipe, mLWater1[itube] / 2.);
    translation = new TGeoTranslation(mXPosition1[itube], 0., mSupportXDimensions[1][0] / 2. + mMoreLength - mLWater1[itube] / 2.);
    cooling->AddNode(pipeTube1, ivolume++, translation);

    TGeoVolume* pipeTorus1 = gGeoManager->MakeTorus(Form("pipeTorus1%d_D1_H%d", itube, half), mPipe, mRadius1[itube], mRWater, mRWater + mDRPipe, 0., mAngle1[itube]);
    rotation = new TGeoRotation("rotation", 180., -90., 0.);
    transformation = new TGeoCombiTrans(mRadius1[itube] + mXPosition1[itube], 0., mSupportXDimensions[1][0] / 2. + mMoreLength - mLWater1[itube], rotation);
    cooling->AddNode(pipeTorus1, ivolume++, transformation);

    TGeoVolume* pipeTube2 = gGeoManager->MakeTube(Form("pipeTube2%d_D1_H%d", itube, half), mPipe, mRWater, mRWater + mDRPipe, mLpartial1[itube] / 2.);
    rotation = new TGeoRotation("rotation", 90., 180 - mAngle1[itube], 0.);
    xPos1[itube] = mLWater1[itube] + mRadius1[itube] * TMath::Sin(mAngle1[itube] * TMath::DegToRad()) + mLpartial1[itube] / 2 * TMath::Cos(mAngle1[itube] * TMath::DegToRad());
    yPos1[itube] = mXPosition1[itube] + mRadius1[itube] * (1 - TMath::Cos(mAngle1[itube] * TMath::DegToRad())) + mLpartial1[itube] / 2 * TMath::Sin(mAngle1[itube] * TMath::DegToRad());
    transformation = new TGeoCombiTrans(yPos1[itube], 0., mSupportXDimensions[1][0] / 2. + mMoreLength - xPos1[itube], rotation);
    cooling->AddNode(pipeTube2, ivolume++, transformation);

    mRadiusCentralTore[itube] = (mSupportXDimensions[1][0] / 2. + mMoreLength - xPos1[itube] - mLpartial1[itube] / 2 * TMath::Cos(mAngle1[itube] * TMath::DegToRad())) / TMath::Sin(mAngle1[itube] * TMath::DegToRad());
    TGeoVolume* pipeTorusCentral = gGeoManager->MakeTorus(Form("pipeTorusCentral%d_D1_H%d", itube, half), mPipe, mRadiusCentralTore[itube], mRWater, mRWater + mDRPipe,
                                                          -mAngle1[itube], 2. * mAngle1[itube]);
    rotation = new TGeoRotation("rotation", 0., 90., 0.);
    transformation = new TGeoCombiTrans(yPos1[itube] + mLpartial1[itube] / 2 * TMath::Sin(mAngle1[itube] * TMath::DegToRad()) - mRadiusCentralTore[itube] * TMath::Cos(mAngle1[itube] * TMath::DegToRad()), 0., 0., rotation);
    cooling->AddNode(pipeTorusCentral, ivolume++, transformation);

    TGeoVolume* pipeTube3 = gGeoManager->MakeTube(Form("pipeTube3%d_D1_H%d", 2, half), mPipe, mRWater, mRWater + mDRPipe, mLpartial1[itube] / 2.);
    rotation = new TGeoRotation("rotation", -90., 0 - mAngle1[itube], 0.);
    transformation = new TGeoCombiTrans(yPos1[itube], 0., -(mSupportXDimensions[1][0] / 2. + mMoreLength - xPos1[itube]), rotation);
    cooling->AddNode(pipeTube3, ivolume++, transformation);

    TGeoVolume* pipeTorus2 = gGeoManager->MakeTorus(Form("pipeTorus2%d_D1_H%d", itube, half), mPipe, mRadius1[itube], mRWater, mRWater + mDRPipe, 0., mAngle1[itube]);
    rotation = new TGeoRotation("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(mRadius1[itube] + mXPosition1[itube], 0., -(mSupportXDimensions[1][0] / 2. + mMoreLength - mLWater1[itube]), rotation);
    cooling->AddNode(pipeTorus2, ivolume++, transformation);

    TGeoVolume* pipeTube4 = gGeoManager->MakeTube(Form("pipeTube4%d_D1_H%d", itube, half), mPipe, mRWater, mRWater + mDRPipe, mLWater1[itube] / 2.);
    translation = new TGeoTranslation(mXPosition1[itube], 0., -(mSupportXDimensions[1][0] / 2. + mMoreLength - mLWater1[itube] / 2.));
    cooling->AddNode(pipeTube4, ivolume++, translation);
  }
  // ***********************************************************************************************

  //Double_t deltaz = mHeatExchangerThickness - mCarbonThickness * 2; // distance between pair of carbon plates
  //Double_t deltaz = mHeatExchangerThickness - Geometry::sKaptonOnCarbonThickness * 4 - 2 * mCarbonThickness;
  Double_t deltaz = mHeatExchangerThickness - Geometry::sKaptonOnCarbonThickness * 4 - Geometry::sKaptonGlueThickness * 4 - 2 * mCarbonThickness;

  //  if (half == Top) {
  //    rotation = new TGeoRotation ("rotation", 90., 90., 0.);
  //    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] + deltaz/2. - mCarbonThickness - mRWater - mDRPipe,
  //    rotation);
  //    mHalfDisk->AddNode(cooling, 1, transformation);
  //    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] - deltaz/2. + mCarbonThickness + mRWater + mDRPipe,
  //    rotation);
  //    mHalfDisk->AddNode(cooling, 2, transformation);
  //  }
  //  else if (half == Bottom) {
  rotation = new TGeoRotation("rotation", -90., 90., 0.);
  transformation =
    //new TGeoCombiTrans(0., 0., mZPlan[disk] + deltaz / 2. - mCarbonThickness - mRWater - mDRPipe, rotation);
    new TGeoCombiTrans(0., 0., mZPlan[disk] + deltaz / 2. - mCarbonThickness - mRWater - mDRPipe - 2 * Geometry::sGlueRohacellCarbonThickness, rotation);
  mHalfDisk->AddNode(cooling, 0, transformation);
  transformation =
    //new TGeoCombiTrans(0., 0., mZPlan[disk] - deltaz / 2. + mCarbonThickness + mRWater + mDRPipe, rotation);
    new TGeoCombiTrans(0., 0., mZPlan[disk] - deltaz / 2. + mCarbonThickness + mRWater + mDRPipe + 2 * Geometry::sGlueRohacellCarbonThickness, rotation);
  mHalfDisk->AddNode(cooling, 1, transformation);
  //  }

  // **************************************** Carbon Plates ****************************************

  auto* carbonPlate = new TGeoVolumeAssembly(Form("carbonPlate_D1_H%d", half));

  auto* carbonBase1 = new TGeoBBox(Form("carbonBase1_D1_H%d", half), (mSupportXDimensions[disk][0]) / 2. + mMoreLength,
                                   (mSupportYDimensions[disk][0]) / 2., mCarbonThickness);
  auto* t11 = new TGeoTranslation("t11", 0., (mSupportYDimensions[disk][0]) / 2. + mHalfDiskGap, 0.);
  t11->RegisterYourself();

  auto* holeCarbon1 =
    new TGeoTubeSeg(Form("holeCarbon1_D1_H%d", half), 0., mRMin[disk], mCarbonThickness + 0.000001, 0, 180.);
  auto* t12 = new TGeoTranslation("t12", 0., -mHalfDiskGap, 0.);
  t12->RegisterYourself();

  // TGeoCompositeShape *cs1 = new TGeoCompositeShape(Form("Carbon1_D1_H%d",half),
  // Form("(carbonBase1_D1_H%d:t11)-(holeCarbon1_D1_H%d:t12)",half,half));
  auto* carbonhole1 = new TGeoSubtraction(carbonBase1, holeCarbon1, t11, t12);
  auto* ch1 = new TGeoCompositeShape(Form("Carbon1_D1_H%d", half), carbonhole1);
  auto* carbonBaseWithHole1 = new TGeoVolume(Form("carbonBaseWithHole_D1_H%d", half), ch1, mCarbon);

  carbonBaseWithHole1->SetLineColor(kGray + 3);
  rotation = new TGeoRotation("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  carbonPlate->AddNode(carbonBaseWithHole1, 0, new TGeoTranslation(0., 0., mZPlan[disk]));

  Double_t ty = mSupportYDimensions[disk][0];

  for (Int_t ipart = 1; ipart < mNPart[disk]; ipart++) {
    ty += mSupportYDimensions[disk][ipart] / 2.;
    TGeoVolume* partCarbon =
      gGeoManager->MakeBox(Form("partCarbon_D1_H%d_%d", half, ipart), mCarbon, mSupportXDimensions[disk][ipart] / 2.,
                           mSupportYDimensions[disk][ipart] / 2., mCarbonThickness);
    partCarbon->SetLineColor(kGray + 3);
    auto* t = new TGeoTranslation("t", 0, ty + mHalfDiskGap, mZPlan[disk]);
    carbonPlate->AddNode(partCarbon, ipart, t);
    ty += mSupportYDimensions[disk][ipart] / 2.;
  }

  //  if (half == Top) {
  //    rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  //    transformation = new TGeoCombiTrans(0., 0., deltaz/2., rotation);
  //    mHalfDisk->AddNode(carbonPlate, 1, transformation);
  //    transformation = new TGeoCombiTrans(0., 0., -deltaz/2., rotation);
  //    mHalfDisk->AddNode(carbonPlate, 2, transformation);
  //  }
  //  else if (half == Bottom) {
  rotation = new TGeoRotation("rotation", 180., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., deltaz / 2., rotation);
  mHalfDisk->AddNode(carbonPlate, 0, transformation);
  transformation = new TGeoCombiTrans(0., 0., -deltaz / 2., rotation);
  mHalfDisk->AddNode(carbonPlate, 1, transformation);
  //  }

  // **************************************** Glue Bwtween Carbon Plate and Rohacell Plate ****************************************

  TGeoMedium* mGlueRohacellCarbon = gGeoManager->GetMedium("MFT_Epoxy$");

  auto* glueRohacellCarbon = new TGeoVolumeAssembly(Form("glueRohacellCarbon_D0_H%d", half));

  auto* glueRohacellCarbonBase0 = new TGeoBBox(Form("glueRohacellCarbonBase0_D0_H%d", half), (mSupportXDimensions[disk][0]) / 2.,
                                               (mSupportYDimensions[disk][0]) / 2., Geometry::sGlueRohacellCarbonThickness);

  auto* translation_gluRC01 = new TGeoTranslation("translation_gluRC01", 0., (mSupportYDimensions[disk][0]) / 2. + mHalfDiskGap, 0.);
  translation_gluRC01->RegisterYourself();
  auto* translation_gluRC02 = new TGeoTranslation("translation_gluRC02", 0., -mHalfDiskGap, 0.);
  translation_gluRC02->RegisterYourself();

  auto* holeglueRohacellCarbon0 =
    new TGeoTubeSeg(Form("holeglueRohacellCarbon0_D0_H%d", half), 0., mRMin[disk], Geometry::sGlueRohacellCarbonThickness + 0.000001, 0, 180.);

  auto* glueRohacellCarbonhole0 = new TGeoSubtraction(glueRohacellCarbonBase0, holeglueRohacellCarbon0, translation_gluRC01, translation_gluRC02);
  auto* gRC0 = new TGeoCompositeShape(Form("glueRohacellCarbon0_D0_H%d", half), glueRohacellCarbonhole0);
  auto* glueRohacellCarbonBaseWithHole0 = new TGeoVolume(Form("glueRohacellCarbonWithHole_D0_H%d", half), gRC0, mGlueRohacellCarbon);

  glueRohacellCarbonBaseWithHole0->SetLineColor(kGreen);
  rotation = new TGeoRotation("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  glueRohacellCarbon->AddNode(glueRohacellCarbonBaseWithHole0, 0, new TGeoTranslation(0., 0., mZPlan[disk]));

  Double_t tyGRC = mSupportYDimensions[disk][0];

  for (Int_t ipart = 1; ipart < mNPart[disk]; ipart++) {
    tyGRC += mSupportYDimensions[disk][ipart] / 2.;
    TGeoVolume* partGlueRohacellCarbon =
      gGeoManager->MakeBox(Form("partGlueRohacellCarbon_D0_H%d_%d", half, ipart), mGlueRohacellCarbon, mSupportXDimensions[disk][ipart] / 2.,
                           mSupportYDimensions[disk][ipart] / 2., Geometry::sGlueRohacellCarbonThickness);
    partGlueRohacellCarbon->SetLineColor(kGreen);
    auto* t = new TGeoTranslation("t", 0, tyGRC + mHalfDiskGap, mZPlan[disk]);
    glueRohacellCarbon->AddNode(partGlueRohacellCarbon, ipart, t);
    tyGRC += mSupportYDimensions[disk][ipart] / 2.;
  }

  rotation = new TGeoRotation("rotation", 180., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., deltaz / 2. - mCarbonThickness - Geometry::sGlueRohacellCarbonThickness, rotation);
  mHalfDisk->AddNode(glueRohacellCarbon, 3, transformation);
  transformation = new TGeoCombiTrans(0., 0., -(deltaz / 2. - mCarbonThickness - Geometry::sGlueRohacellCarbonThickness), rotation);
  mHalfDisk->AddNode(glueRohacellCarbon, 4, transformation);

  // **************************************** Kapton on Carbon Plate ****************************************

  TGeoMedium* mKaptonOnCarbon = gGeoManager->GetMedium("MFT_Kapton$");

  auto* kaptonOnCarbon = new TGeoVolumeAssembly(Form("kaptonOnCarbon_D0_H%d", half));
  auto* kaptonOnCarbonBase0 = new TGeoBBox(Form("kaptonOnCarbonBase0_D0_H%d", half), (mSupportXDimensions[disk][0]) / 2. + mMoreLength,
                                           (mSupportYDimensions[disk][0]) / 2., Geometry::sKaptonOnCarbonThickness);

  auto* translation_KC01 = new TGeoTranslation("translation_KC01", 0., (mSupportYDimensions[disk][0]) / 2. + mHalfDiskGap, 0.);
  translation_KC01->RegisterYourself();
  auto* translation_KC02 = new TGeoTranslation("translation_KC02", 0., -mHalfDiskGap, 0.);
  translation_KC02->RegisterYourself();

  auto* holekaptonOnCarbon0 =
    new TGeoTubeSeg(Form("holekaptonOnCarbon0_D0_H%d", half), 0., mRMin[disk], Geometry::sKaptonOnCarbonThickness + 0.000001, 0, 180.);

  auto* kaptonOnCarbonhole0 = new TGeoSubtraction(kaptonOnCarbonBase0, holekaptonOnCarbon0, translation_KC01, translation_KC02);
  auto* KC0 = new TGeoCompositeShape(Form("kaptonOnCarbon_D0_H%d", half), kaptonOnCarbonhole0);
  auto* kaptonOnCarbonBaseWithHole0 = new TGeoVolume(Form("kaptonOnCarbonWithHole_D0_H%d", half), KC0, mKaptonOnCarbon);

  kaptonOnCarbonBaseWithHole0->SetLineColor(kMagenta);
  rotation = new TGeoRotation("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  kaptonOnCarbon->AddNode(kaptonOnCarbonBaseWithHole0, 0, new TGeoTranslation(0., 0., mZPlan[disk]));

  Double_t tyKC = mSupportYDimensions[disk][0];

  for (Int_t ipart = 1; ipart < mNPart[disk]; ipart++) {
    tyKC += mSupportYDimensions[disk][ipart] / 2.;
    TGeoVolume* partkaptonOnCarbonBase =
      gGeoManager->MakeBox(Form("partkaptonOnCarbon_D0_H%d_%d", half, ipart), mKaptonOnCarbon, mSupportXDimensions[disk][ipart] / 2.,
                           mSupportYDimensions[disk][ipart] / 2., Geometry::sKaptonOnCarbonThickness);
    partkaptonOnCarbonBase->SetLineColor(kMagenta);
    auto* t = new TGeoTranslation("t", 0, tyKC + mHalfDiskGap, mZPlan[disk]);
    kaptonOnCarbon->AddNode(partkaptonOnCarbonBase, ipart, t);
    tyKC += mSupportYDimensions[disk][ipart] / 2.;
  }

  rotation = new TGeoRotation("rotation", 180., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., deltaz / 2 + Geometry::sKaptonOnCarbonThickness + mCarbonThickness + Geometry::sKaptonGlueThickness * 2, rotation);
  mHalfDisk->AddNode(kaptonOnCarbon, 3, transformation);
  transformation = new TGeoCombiTrans(0., 0., -(deltaz / 2 + Geometry::sKaptonOnCarbonThickness + mCarbonThickness + Geometry::sKaptonGlueThickness * 2), rotation);
  mHalfDisk->AddNode(kaptonOnCarbon, 4, transformation);

  // **************************************** Kapton glue on the carbon plate ****************************************

  TGeoMedium* mGlueKaptonCarbon = gGeoManager->GetMedium("MFT_Epoxy$");

  auto* glueKaptonCarbon = new TGeoVolumeAssembly(Form("glueKaptonCarbon_D0_H%d", half));

  auto* glueKaptonCarbonBase0 = new TGeoBBox(Form("glueKaptonCarbonBase0_D0_H%d", half), (mSupportXDimensions[disk][0]) / 2.,
                                             (mSupportYDimensions[disk][0]) / 2., Geometry::sKaptonGlueThickness);

  auto* translation_gluKC01 = new TGeoTranslation("translation_gluKC01", 0., (mSupportYDimensions[disk][0]) / 2. + mHalfDiskGap, 0.);
  translation_gluKC01->RegisterYourself();
  auto* translation_gluKC02 = new TGeoTranslation("translation_gluKC02", 0., -mHalfDiskGap, 0.);
  translation_gluKC02->RegisterYourself();

  auto* holeglueKaptonCarbon0 =
    new TGeoTubeSeg(Form("holeglueKaptonCarbon0_D0_H%d", half), 0., mRMin[disk], Geometry::sKaptonGlueThickness + 0.000001, 0, 180.);

  auto* glueKaptonCarbonhole0 = new TGeoSubtraction(glueKaptonCarbonBase0, holeglueKaptonCarbon0, translation_gluKC01, translation_gluKC02);
  auto* gKC0 = new TGeoCompositeShape(Form("glueKaptonCarbon0_D0_H%d", half), glueKaptonCarbonhole0);
  auto* glueKaptonCarbonBaseWithHole0 = new TGeoVolume(Form("glueKaptonCarbonWithHole_D0_H%d", half), gKC0, mGlueKaptonCarbon);

  glueKaptonCarbonBaseWithHole0->SetLineColor(kGreen);
  rotation = new TGeoRotation("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  glueKaptonCarbon->AddNode(glueKaptonCarbonBaseWithHole0, 0, new TGeoTranslation(0., 0., mZPlan[disk]));

  Double_t tyGKC = mSupportYDimensions[disk][0];

  for (Int_t ipart = 1; ipart < mNPart[disk]; ipart++) {
    tyGKC += mSupportYDimensions[disk][ipart] / 2.;
    TGeoVolume* partGlueKaptonCarbon =
      gGeoManager->MakeBox(Form("partGlueKaptonCarbon_D0_H%d_%d", half, ipart), mGlueKaptonCarbon, mSupportXDimensions[disk][ipart] / 2.,
                           mSupportYDimensions[disk][ipart] / 2., Geometry::sKaptonGlueThickness);
    partGlueKaptonCarbon->SetLineColor(kGreen);
    auto* t = new TGeoTranslation("t", 0, tyGKC + mHalfDiskGap, mZPlan[disk]);
    glueKaptonCarbon->AddNode(partGlueKaptonCarbon, ipart, t);
    tyGKC += mSupportYDimensions[disk][ipart] / 2.;
  }

  rotation = new TGeoRotation("rotation", 180., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., deltaz / 2. + mCarbonThickness + Geometry::sKaptonGlueThickness, rotation);
  mHalfDisk->AddNode(glueKaptonCarbon, 3, transformation);
  transformation = new TGeoCombiTrans(0., 0., -(deltaz / 2. + mCarbonThickness + Geometry::sKaptonGlueThickness), rotation);
  mHalfDisk->AddNode(glueKaptonCarbon, 4, transformation);

  // **************************************** Rohacell Plate ****************************************

  auto* rohacellPlate = new TGeoVolumeAssembly(Form("rohacellPlate_D1_H%d", half));

  auto* rohacellBase1 = new TGeoBBox("rohacellBase1", (mSupportXDimensions[disk][0]) / 2.,
                                     (mSupportYDimensions[disk][0]) / 2., mRohacellThickness);
  // TGeoTranslation *t3 = new TGeoTranslation ("t3",0., (fSupportYDimensions[disk][0])/2. + fHalfDiskGap , 0.);
  // t3 -> RegisterYourself();

  auto* holeRohacell1 = new TGeoTubeSeg("holeRohacell1", 0., mRMin[disk], mRohacellThickness + 0.000001, 0, 180.);
  // TGeoTranslation *t4= new TGeoTranslation ("t4", 0., - fHalfDiskGap , 0.);
  // t4-> RegisterYourself();

  //////cs1 = new TGeoCompositeShape(Form("rohacell_D1_H%d",half), "(rohacellBase1:t11)-(holeRohacell1:t12)");

  // **************************************** GROOVES *************************************************
  Double_t diameter = 0.21; // groove diameter
  Double_t epsilon = 0.06;  // outside shift of the goove
  Int_t iCount = 0;
  Double_t mPosition[4];

  TGeoCombiTrans* transfo[7][3];
  TGeoTube* grooveTube[7][3];
  TGeoTorus* grooveTorus[7][3];
  TGeoSubtraction* rohacellBaseGroove[300];
  TGeoCompositeShape* rohacellGroove[300];

  for (Int_t igroove = 0; igroove < 3; igroove++) {
    grooveTube[0][igroove] = new TGeoTube("linear", 0., diameter, mLWater1[igroove] / 2.);
    grooveTorus[1][igroove] = new TGeoTorus("SideTorus", mRadius1[igroove], 0., diameter, 0., mAngle1[igroove]);
    grooveTube[2][igroove] = new TGeoTube("tiltedLinear", 0., diameter, mLpartial1[igroove] / 2.);
    grooveTorus[3][igroove] = new TGeoTorus("centralTorus", mRadiusCentralTore[igroove], 0., diameter, -mAngle1[igroove], 2. * mAngle1[igroove]);
    grooveTube[4][igroove] = new TGeoTube("tiltedLinear", 0., diameter, mLpartial1[igroove] / 2.);
    grooveTorus[5][igroove] = new TGeoTorus("SideTorus", mRadius1[igroove], 0., diameter, 0., mAngle1[igroove]);
    grooveTube[6][igroove] = new TGeoTube("linear", 0., diameter, mLWater1[igroove] / 2.);
  }

  // Rotation matrix
  TGeoRotation* rotationLinear = new TGeoRotation("rotation", -90., 90., 0.);
  TGeoRotation* rotationSideTorusL = new TGeoRotation("rotationSideTorusLeft", -90., 0., 0.);
  TGeoRotation* rotationSideTorusR = new TGeoRotation("rotationSideTorusRight", 90., 180., 180.);
  TGeoRotation* rotationCentralTorus = new TGeoRotation("rotationCentralTorus", 90., 0., 0.);
  TGeoRotation* rotationTiltedLinearR;
  TGeoRotation* rotationTiltedLinearL;

  // Creating grooves
  for (Int_t iface = 1; iface > -2; iface -= 2) {     // front and rear
    for (Int_t igroove = 0; igroove < 3; igroove++) { // 3 grooves
      mPosition[igroove] = mXPosition1[igroove] - mSupportYDimensions[disk][0] / 2. - mHalfDiskGap;
      for (Int_t ip = 0; ip < 7; ip++) { // each groove is made of 7 parts

        switch (ip) {
          case 0: // Linear
            transfo[ip][igroove] = new TGeoCombiTrans(mSupportXDimensions[1][0] / 2. + mMoreLength - mLWater1[igroove] / 2., mPosition[igroove],
                                                      iface * (mRohacellThickness + epsilon), rotationLinear);
            if (igroove == 0 && iface == 1) {
              rohacellBaseGroove[0] = new TGeoSubtraction(rohacellBase1, grooveTube[ip][igroove], nullptr, transfo[ip][igroove]);
              rohacellGroove[0] = new TGeoCompositeShape(Form("rohacell1Groove%d_G%d_F%d_H%d", ip, igroove, iface, half), rohacellBaseGroove[0]);
            };
            break;
          case 1: // side torus
            transfo[ip][igroove] = new TGeoCombiTrans(mSupportXDimensions[1][0] / 2. + mMoreLength - mLWater1[igroove], mRadius1[igroove] + mXPosition1[igroove] - mSupportYDimensions[disk][0] / 2. - mHalfDiskGap, iface * (mRohacellThickness + epsilon), rotationSideTorusR);
            break;
          case 2: // Linear tilted
            rotationTiltedLinearR = new TGeoRotation("rotationTiltedLinearRight", 90. - mAngle1[igroove], 90., 0.);
            transfo[ip][igroove] = new TGeoCombiTrans(mSupportXDimensions[1][0] / 2. + mMoreLength - xPos1[igroove], yPos1[igroove] - mSupportYDimensions[disk][0] / 2. - mHalfDiskGap,
                                                      iface * (mRohacellThickness + epsilon), rotationTiltedLinearR);
            break;
          case 3: // Central Torus
            transfo[ip][igroove] = new TGeoCombiTrans(0., yPos1[igroove] + mLpartial1[igroove] / 2 * TMath::Sin(mAngle1[igroove] * TMath::DegToRad()) - mRadiusCentralTore[igroove] * TMath::Cos(mAngle1[igroove] * TMath::DegToRad()) - mSupportYDimensions[disk][0] / 2. - mHalfDiskGap,
                                                      iface * (mRohacellThickness + epsilon), rotationCentralTorus);
            break;
          case 4: // Linear tilted
            rotationTiltedLinearL = new TGeoRotation("rotationTiltedLinearLeft", 90. + mAngle1[igroove], 90., 0.);
            transfo[ip][igroove] = new TGeoCombiTrans(-(mSupportXDimensions[1][0] / 2. + mMoreLength - xPos1[igroove]), yPos1[igroove] - mSupportYDimensions[disk][0] / 2. - mHalfDiskGap,
                                                      iface * (mRohacellThickness + epsilon), rotationTiltedLinearL);
            break;
          case 5: // side torus
            transfo[ip][igroove] = new TGeoCombiTrans(-(mSupportXDimensions[1][0] / 2. + mMoreLength - mLWater1[igroove]), mRadius1[igroove] + mPosition[igroove],
                                                      iface * (mRohacellThickness + epsilon), rotationSideTorusL);
            break;
          case 6: // Linear
            transfo[ip][igroove] = new TGeoCombiTrans(-(mSupportXDimensions[1][0] / 2. + mMoreLength - mLWater1[igroove] / 2.), mPosition[igroove],
                                                      iface * (mRohacellThickness + epsilon), rotationLinear);
            break;
        }

        if (!(ip == 0 && igroove == 0 && iface == 1)) {
          if (ip & 1) {
            rohacellBaseGroove[iCount] = new TGeoSubtraction(rohacellGroove[iCount - 1], grooveTorus[ip][igroove], nullptr, transfo[ip][igroove]);
          } else {
            rohacellBaseGroove[iCount] = new TGeoSubtraction(rohacellGroove[iCount - 1], grooveTube[ip][igroove], nullptr, transfo[ip][igroove]);
          }
          rohacellGroove[iCount] = new TGeoCompositeShape(Form("rohacell1Groove%d_G%d_F%d_H%d", iCount, igroove, iface, half), rohacellBaseGroove[iCount]);
        }
        iCount++;
      }
    }
  }

  // **************************************************************************************************

  //auto* rohacellhole1 = new TGeoSubtraction(rohacellBase1, holeRohacell1, t11, t12);
  auto* rohacellBase = new TGeoSubtraction(rohacellGroove[iCount - 1], holeRohacell1, t11, t12);
  auto* rh1 = new TGeoCompositeShape(Form("rohacellBase1_D1_H%d", half), rohacellBase);
  //auto* rh1 = new TGeoCompositeShape(Form("rohacellBase1_D1_H%d", half), rohacellhole1);
  auto* rohacellBaseWithHole = new TGeoVolume(Form("rohacellBaseWithHole_D1_H%d", half), rh1, mRohacell);

  TGeoVolume* partRohacell;
  rohacellBaseWithHole->SetLineColor(kGray);
  rotation = new TGeoRotation("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  rohacellPlate->AddNode(rohacellBaseWithHole, 0, new TGeoTranslation(0., 0., mZPlan[disk]));

  ty = mSupportYDimensions[disk][0];

  for (Int_t ipart = 1; ipart < mNPart[disk]; ipart++) {
    ty += mSupportYDimensions[disk][ipart] / 2.;
    /*
    TGeoVolume* partRohacell = gGeoManager->MakeBox(Form("partRohacelli_D1_H%d_%d", half, ipart), mRohacell,
                                                    mSupportXDimensions[disk][ipart] / 2.,
                                                    mSupportYDimensions[disk][ipart] / 2., mRohacellThickness);
    partRohacell->SetLineColor(kGray);
    */
    auto* t = new TGeoTranslation("t", 0, ty + mHalfDiskGap, mZPlan[disk]);

    //===========================================================================================================
    //===========================================================================================================
    auto* partRohacell0 =
      new TGeoBBox(Form("rohacellBase0_D1_H%d_%d", half, ipart), mSupportXDimensions[disk][ipart] / 2.,
                   mSupportYDimensions[disk][ipart] / 2., mRohacellThickness);

    // ***************** Creating grooves for the other parts of the rohacell plate **********************
    Double_t mShift;
    for (Int_t iface = 1; iface > -2; iface -= 2) {     // front and rear
      for (Int_t igroove = 0; igroove < 3; igroove++) { // 3 grooves
        if (ipart == 1) {
          mPosition[ipart] = mXPosition1[igroove] - mSupportYDimensions[disk][ipart] / 2. - mHalfDiskGap - mSupportYDimensions[disk][ipart - 1];
          mShift = -mSupportYDimensions[disk][ipart - 1];
        };
        if (ipart == 2) {
          mPosition[ipart] = mXPosition1[igroove] - mSupportYDimensions[disk][ipart] / 2. - mHalfDiskGap - mSupportYDimensions[disk][ipart - 1] - mSupportYDimensions[disk][ipart - 2];
          mShift = -mSupportYDimensions[disk][ipart - 1] - mSupportYDimensions[disk][ipart - 2];
        };
        if (ipart == 3) {
          mPosition[ipart] = mXPosition1[igroove] - mSupportYDimensions[disk][ipart] / 2. - mHalfDiskGap - mSupportYDimensions[disk][ipart - 1] - mSupportYDimensions[disk][ipart - 2] - mSupportYDimensions[disk][ipart - 3];
          mShift = -mSupportYDimensions[disk][ipart - 1] - mSupportYDimensions[disk][ipart - 2] - mSupportYDimensions[disk][ipart - 3];
        };

        for (Int_t ip = 0; ip < 7; ip++) { // each groove is made of 7 parts

          switch (ip) {
            case 0: // Linear
              transfo[ip][igroove] = new TGeoCombiTrans(mSupportXDimensions[disk][0] / 2. + mMoreLength - mLWater1[igroove] / 2., mPosition[ipart],
                                                        iface * (mRohacellThickness + epsilon), rotationLinear);
              if (igroove == 0 && iface == 1) {
                rohacellBaseGroove[iCount] = new TGeoSubtraction(partRohacell0, grooveTube[ip][igroove], nullptr, transfo[ip][igroove]);
                rohacellGroove[iCount] = new TGeoCompositeShape(Form("rohacell1Groove%d_G%d_F%d_H%d", ip, igroove, iface, half), rohacellBaseGroove[iCount]);
              };
              break;
            case 1: // side torus
              transfo[ip][igroove] = new TGeoCombiTrans(mSupportXDimensions[disk][0] / 2. + mMoreLength - mLWater1[igroove], mPosition[ipart] + mRadius1[igroove],
                                                        iface * (mRohacellThickness + epsilon), rotationSideTorusR);
              break;
            case 2: // Linear tilted
              rotationTiltedLinearR = new TGeoRotation("rotationTiltedLinearRight", 90. - mAngle1[igroove], 90., 0.);
              transfo[ip][igroove] = new TGeoCombiTrans(mSupportXDimensions[disk][0] / 2. + mMoreLength - xPos1[igroove], yPos1[igroove] + mShift - mHalfDiskGap - mSupportYDimensions[disk][ipart] / 2., iface * (mRohacellThickness + epsilon), rotationTiltedLinearR);
              break;
            case 3: // Central Torus
              transfo[ip][igroove] = new TGeoCombiTrans(0., mPosition[ipart] + yPos1[igroove] + mLpartial1[igroove] / 2 * TMath::Sin(mAngle1[igroove] * TMath::DegToRad()) - mRadiusCentralTore[igroove] * TMath::Cos(mAngle1[igroove] * TMath::DegToRad()) - mXPosition1[igroove],
                                                        iface * (mRohacellThickness + epsilon), rotationCentralTorus);
              break;
            case 4: // Linear tilted
              rotationTiltedLinearL = new TGeoRotation("rotationTiltedLinearLeft", 90. + mAngle1[igroove], 90., 0.);
              transfo[ip][igroove] = new TGeoCombiTrans(-(mSupportXDimensions[disk][0] / 2. + mMoreLength - xPos1[igroove]), yPos1[igroove] + mPosition[ipart] - mXPosition1[igroove],
                                                        iface * (mRohacellThickness + epsilon), rotationTiltedLinearL);
              break;
            case 5: // side torus
              transfo[ip][igroove] = new TGeoCombiTrans(-(mSupportXDimensions[disk][0] / 2. + mMoreLength - mLWater1[igroove]), mRadius1[igroove] + mPosition[ipart],
                                                        iface * (mRohacellThickness + epsilon), rotationSideTorusL);
              break;
            case 6: // Linear
              transfo[ip][igroove] = new TGeoCombiTrans(-(mSupportXDimensions[disk][0] / 2. + mMoreLength - mLWater1[igroove] / 2.), mPosition[ipart],
                                                        iface * (mRohacellThickness + epsilon), rotationLinear);
              break;
          }
          if (!(ip == 0 && igroove == 0 && iface == 1)) {
            if (ip & 1) {
              rohacellBaseGroove[iCount] =
                new TGeoSubtraction(rohacellGroove[iCount - 1], grooveTorus[ip][igroove], nullptr, transfo[ip][igroove]);
            } else {
              rohacellBaseGroove[iCount] =
                new TGeoSubtraction(rohacellGroove[iCount - 1], grooveTube[ip][igroove], nullptr, transfo[ip][igroove]);
            }

            rohacellGroove[iCount] = new TGeoCompositeShape(Form("rohacell1Groove%d_G%d_F%d_H%d", iCount, igroove, iface, half), rohacellBaseGroove[iCount]);
          }
          iCount++;
        }
      }
    }

    // **************************************************************************************************

    partRohacell = new TGeoVolume(Form("partRohacelli_D1_H%d_%d", half, ipart), rohacellGroove[iCount - 1], mRohacell);
    //partRohacell = new TGeoVolume(Form("partRohacelli_D1_H%d_%d", half, ipart), partRohacell0, mRohacell);
    //===========================================================================================================
    //===========================================================================================================

    rohacellPlate->AddNode(partRohacell, ipart, t);
    ty += mSupportYDimensions[disk][ipart] / 2.;
  }

  //  if (half == Top) {
  //    rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  //    transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  //    mHalfDisk->AddNode(rohacellPlate, 1, transformation);
  //  }
  //  if (half == Bottom) {
  rotation = new TGeoRotation("rotation", 180., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  mHalfDisk->AddNode(rohacellPlate, 2, transformation);
  //  }

  createManifold(disk);
}

//_____________________________________________________________________________
void HeatExchanger::createHalfDisk2(Int_t half)
{

  Int_t disk = 2;

  if (half == Top)
    printf("Creating MFT heat exchanger for disk2 top\n");
  else if (half == Bottom)
    printf("Creating MFT heat exchanger for disk2 bottom\n");
  else
    printf("No valid option for MFT heat exchanger on disk2\n");

  // mCarbon   = gGeoManager->GetMedium("MFT_Carbon$");
  mCarbon = gGeoManager->GetMedium("MFT_CarbonFiber$");

  mWater = gGeoManager->GetMedium("MFT_Water$");
  mRohacell = gGeoManager->GetMedium("MFT_Rohacell");
  mPipe = gGeoManager->GetMedium("MFT_Polyimide");

  auto* cooling = new TGeoVolumeAssembly(Form("cooling_D2_H%d", half));

  //Float_t lMiddle = mSupportXDimensions[disk][0] - 2. * mLWater; // length of central part

  TGeoTranslation* translation = nullptr;
  TGeoRotation* rotation = nullptr;
  TGeoCombiTrans* transformation = nullptr;

  // **************************************** Water part ****************************************

  // ********************** Four parameters mLwater2, mRadius2, mAngle2, mLpartial2 *************
  Double_t ivolume = 200; // offset chamber 2
  Double_t mRadiusCentralTore[4];
  Double_t xPos2[4];
  Double_t yPos2[4];
  for (Int_t itube = 0; itube < 3; itube++) {
    TGeoVolume* waterTube1 = gGeoManager->MakeTube(Form("waterTube1%d_D2_H%d", itube, half), mWater, 0., mRWater, mLWater2[itube] / 2.);
    translation = new TGeoTranslation(mXPosition2[itube], 0., mSupportXDimensions[2][0] / 2. + mMoreLength - mLWater2[itube] / 2.);
    cooling->AddNode(waterTube1, ivolume++, translation);

    TGeoVolume* waterTorus1 = gGeoManager->MakeTorus(Form("waterTorus1%d_D2_H%d", itube, half), mWater, mRadius2[itube], 0., mRWater, 0., mAngle2[itube]);
    rotation = new TGeoRotation("rotation", 180., -90., 0.);
    transformation = new TGeoCombiTrans(mRadius2[itube] + mXPosition2[itube], 0., mSupportXDimensions[2][0] / 2. + mMoreLength - mLWater2[itube], rotation);
    cooling->AddNode(waterTorus1, ivolume++, transformation);

    TGeoVolume* waterTube2 = gGeoManager->MakeTube(Form("waterTube2%d_D2_H%d", itube, half), mWater, 0., mRWater, mLpartial2[itube] / 2.);
    rotation = new TGeoRotation("rotation", 90., 180 - mAngle2[itube], 0.);
    xPos2[itube] = mLWater2[itube] + mRadius2[itube] * TMath::Sin(mAngle2[itube] * TMath::DegToRad()) + mLpartial2[itube] / 2 * TMath::Cos(mAngle2[itube] * TMath::DegToRad());
    yPos2[itube] = mXPosition2[itube] + mRadius2[itube] * (1 - TMath::Cos(mAngle2[itube] * TMath::DegToRad())) + mLpartial2[itube] / 2 * TMath::Sin(mAngle2[itube] * TMath::DegToRad());
    transformation = new TGeoCombiTrans(yPos2[itube], 0., mSupportXDimensions[2][0] / 2. + mMoreLength - xPos2[itube], rotation);
    cooling->AddNode(waterTube2, ivolume++, transformation);

    mRadiusCentralTore[itube] = (mSupportXDimensions[2][0] / 2. + mMoreLength - xPos2[itube] - mLpartial2[itube] / 2 * TMath::Cos(mAngle2[itube] * TMath::DegToRad())) / TMath::Sin(mAngle2[itube] * TMath::DegToRad());
    TGeoVolume* waterTorusCentral = gGeoManager->MakeTorus(Form("waterTorusCentral%d_D2_H%d", itube, half), mWater, mRadiusCentralTore[itube], 0., mRWater, -mAngle2[itube], 2. * mAngle2[itube]);
    rotation = new TGeoRotation("rotation", 0., 90., 0.);
    transformation = new TGeoCombiTrans(yPos2[itube] + mLpartial2[itube] / 2 * TMath::Sin(mAngle2[itube] * TMath::DegToRad()) - mRadiusCentralTore[itube] * TMath::Cos(mAngle2[itube] * TMath::DegToRad()), 0., 0., rotation);
    cooling->AddNode(waterTorusCentral, ivolume++, transformation);

    TGeoVolume* waterTube3 = gGeoManager->MakeTube(Form("waterTube3%d_D2_H%d", 2, half), mWater, 0., mRWater, mLpartial2[itube] / 2.);
    rotation = new TGeoRotation("rotation", -90., 0 - mAngle2[itube], 0.);
    transformation = new TGeoCombiTrans(yPos2[itube], 0., -(mSupportXDimensions[2][0] / 2. + mMoreLength - xPos2[itube]), rotation);
    cooling->AddNode(waterTube3, ivolume++, transformation);

    TGeoVolume* waterTorus2 = gGeoManager->MakeTorus(Form("waterTorus2%d_D2_H%d", itube, half), mWater, mRadius2[itube], 0., mRWater, 0., mAngle2[itube]);
    rotation = new TGeoRotation("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(mRadius2[itube] + mXPosition2[itube], 0., -(mSupportXDimensions[2][0] / 2. + mMoreLength - mLWater2[itube]), rotation);
    cooling->AddNode(waterTorus2, ivolume++, transformation);

    TGeoVolume* waterTube4 = gGeoManager->MakeTube(Form("waterTube4%d_D2_H%d", itube, half), mWater, 0., mRWater, mLWater2[itube] / 2.);
    translation = new TGeoTranslation(mXPosition2[itube], 0., -(mSupportXDimensions[2][0] / 2. + mMoreLength - mLWater2[itube] / 2.));
    cooling->AddNode(waterTube4, ivolume++, translation);
  }

  // **************************************************** Tube part ************************************************
  // ****************************** Four parameters mLwater2, mRadius2, mAngle2, mLpartial2 ************************
  for (Int_t itube = 0; itube < 3; itube++) {

    TGeoVolume* pipeTube1 = gGeoManager->MakeTube(Form("pipeTube1%d_D2_H%d", itube, half), mPipe, mRWater, mRWater + mDRPipe, mLWater2[itube] / 2.);
    translation = new TGeoTranslation(mXPosition2[itube], 0., mSupportXDimensions[2][0] / 2. + mMoreLength - mLWater2[itube] / 2.);
    cooling->AddNode(pipeTube1, ivolume++, translation);

    TGeoVolume* pipeTorus1 = gGeoManager->MakeTorus(Form("pipeTorus1%d_D2_H%d", itube, half), mPipe, mRadius2[itube], mRWater, mRWater + mDRPipe, 0., mAngle2[itube]);
    rotation = new TGeoRotation("rotation", 180., -90., 0.);
    transformation = new TGeoCombiTrans(mRadius2[itube] + mXPosition2[itube], 0., mSupportXDimensions[2][0] / 2. + mMoreLength - mLWater2[itube], rotation);
    cooling->AddNode(pipeTorus1, ivolume++, transformation);

    TGeoVolume* pipeTube2 = gGeoManager->MakeTube(Form("pipeTube2%d_D2_H%d", itube, half), mPipe, mRWater, mRWater + mDRPipe, mLpartial2[itube] / 2.);
    rotation = new TGeoRotation("rotation", 90., 180 - mAngle2[itube], 0.);
    xPos2[itube] = mLWater2[itube] + mRadius2[itube] * TMath::Sin(mAngle2[itube] * TMath::DegToRad()) + mLpartial2[itube] / 2 * TMath::Cos(mAngle2[itube] * TMath::DegToRad());
    yPos2[itube] = mXPosition2[itube] + mRadius2[itube] * (1 - TMath::Cos(mAngle2[itube] * TMath::DegToRad())) + mLpartial2[itube] / 2 * TMath::Sin(mAngle2[itube] * TMath::DegToRad());
    transformation = new TGeoCombiTrans(yPos2[itube], 0., mSupportXDimensions[2][0] / 2. + mMoreLength - xPos2[itube], rotation);
    cooling->AddNode(pipeTube2, ivolume++, transformation);

    mRadiusCentralTore[itube] = (mSupportXDimensions[2][0] / 2. + mMoreLength - xPos2[itube] - mLpartial2[itube] / 2 * TMath::Cos(mAngle2[itube] * TMath::DegToRad())) / TMath::Sin(mAngle2[itube] * TMath::DegToRad());
    TGeoVolume* pipeTorusCentral = gGeoManager->MakeTorus(Form("pipeTorusCentral%d_D2_H%d", itube, half), mPipe, mRadiusCentralTore[itube], mRWater, mRWater + mDRPipe,
                                                          -mAngle2[itube], 2. * mAngle2[itube]);
    rotation = new TGeoRotation("rotation", 0., 90., 0.);
    transformation = new TGeoCombiTrans(yPos2[itube] + mLpartial2[itube] / 2 * TMath::Sin(mAngle2[itube] * TMath::DegToRad()) - mRadiusCentralTore[itube] * TMath::Cos(mAngle2[itube] * TMath::DegToRad()), 0., 0., rotation);
    cooling->AddNode(pipeTorusCentral, ivolume++, transformation);

    TGeoVolume* pipeTube3 = gGeoManager->MakeTube(Form("pipeTube3%d_D2_H%d", 2, half), mPipe, mRWater, mRWater + mDRPipe, mLpartial2[itube] / 2.);
    rotation = new TGeoRotation("rotation", -90., 0 - mAngle2[itube], 0.);
    transformation = new TGeoCombiTrans(yPos2[itube], 0., -(mSupportXDimensions[2][0] / 2. + mMoreLength - xPos2[itube]), rotation);
    cooling->AddNode(pipeTube3, ivolume++, transformation);

    TGeoVolume* pipeTorus2 = gGeoManager->MakeTorus(Form("pipeTorus2%d_D2_H%d", itube, half), mPipe, mRadius2[itube], mRWater, mRWater + mDRPipe, 0., mAngle2[itube]);
    rotation = new TGeoRotation("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(mRadius2[itube] + mXPosition2[itube], 0., -(mSupportXDimensions[2][0] / 2. + mMoreLength - mLWater2[itube]), rotation);
    cooling->AddNode(pipeTorus2, ivolume++, transformation);

    TGeoVolume* pipeTube4 = gGeoManager->MakeTube(Form("pipeTube4%d_D2_H%d", itube, half), mPipe, mRWater, mRWater + mDRPipe, mLWater2[itube] / 2.);
    translation = new TGeoTranslation(mXPosition2[itube], 0., -(mSupportXDimensions[2][0] / 2. + mMoreLength - mLWater2[itube] / 2.));
    cooling->AddNode(pipeTube4, ivolume++, translation);
  }
  // ***********************************************************************************************

  //Double_t deltaz = mHeatExchangerThickness - mCarbonThickness * 2; // distance between pair of carbon plates
  Double_t deltaz = mHeatExchangerThickness - Geometry::sKaptonOnCarbonThickness * 4 - Geometry::sKaptonGlueThickness * 4 - 2 * mCarbonThickness;

  //  if (half == Top) {
  //    rotation = new TGeoRotation ("rotation", 90., 90., 0.);
  //    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] + deltaz/2. - mCarbonThickness - mRWater - mDRPipe,
  //    rotation);
  //    mHalfDisk->AddNode(cooling, 1, transformation);
  //    transformation = new TGeoCombiTrans(0., 0., mZPlan[disk] - deltaz/2. + mCarbonThickness + mRWater + mDRPipe,
  //    rotation);
  //    mHalfDisk->AddNode(cooling, 2, transformation);
  //  }
  //  else if (half == Bottom) {
  rotation = new TGeoRotation("rotation", -90., 90., 0.);
  transformation =
    //new TGeoCombiTrans(0., 0., mZPlan[disk] + deltaz / 2. - mCarbonThickness - mRWater - mDRPipe, rotation);
    new TGeoCombiTrans(0., 0., mZPlan[disk] + deltaz / 2. - mCarbonThickness - mRWater - mDRPipe - 2 * Geometry::sGlueRohacellCarbonThickness, rotation);
  mHalfDisk->AddNode(cooling, 3, transformation);
  transformation =
    //new TGeoCombiTrans(0., 0., mZPlan[disk] - deltaz / 2. + mCarbonThickness + mRWater + mDRPipe, rotation);
    new TGeoCombiTrans(0., 0., mZPlan[disk] - deltaz / 2. + mCarbonThickness + mRWater + mDRPipe + 2 * Geometry::sGlueRohacellCarbonThickness, rotation);
  mHalfDisk->AddNode(cooling, 4, transformation);
  //  }

  // **************************************** Carbon Plates ****************************************

  auto* carbonPlate = new TGeoVolumeAssembly(Form("carbonPlate_D2_H%d", half));

  auto* carbonBase2 = new TGeoBBox(Form("carbonBase2_D2_H%d", half), (mSupportXDimensions[disk][0]) / 2. + mMoreLength,
                                   (mSupportYDimensions[disk][0]) / 2., mCarbonThickness);
  auto* t21 = new TGeoTranslation("t21", 0., (mSupportYDimensions[disk][0]) / 2. + mHalfDiskGap, 0.);
  t21->RegisterYourself();

  auto* holeCarbon2 =
    new TGeoTubeSeg(Form("holeCarbon2_D2_H%d", half), 0., mRMin[disk], mCarbonThickness + 0.000001, 0, 180.);
  auto* t22 = new TGeoTranslation("t22", 0., -mHalfDiskGap, 0.);
  t22->RegisterYourself();

  // TGeoCompositeShape *cs2 = new
  // TGeoCompositeShape(Form("carbon2_D2_H%d",half),Form("(carbonBase2_D2_H%d:t21)-(holeCarbon2_D2_H%d:t22)",half,half));
  auto* carbonhole2 = new TGeoSubtraction(carbonBase2, holeCarbon2, t21, t22);
  auto* cs2 = new TGeoCompositeShape(Form("Carbon2_D2_H%d", half), carbonhole2);
  auto* carbonBaseWithHole2 = new TGeoVolume(Form("carbonBaseWithHole_D2_H%d", half), cs2, mCarbon);

  carbonBaseWithHole2->SetLineColor(kGray + 3);
  rotation = new TGeoRotation("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  carbonPlate->AddNode(carbonBaseWithHole2, 0, new TGeoTranslation(0., 0., mZPlan[disk]));

  Double_t ty = mSupportYDimensions[disk][0];

  for (Int_t ipart = 1; ipart < mNPart[disk]; ipart++) {
    ty += mSupportYDimensions[disk][ipart] / 2.;
    TGeoVolume* partCarbon =
      gGeoManager->MakeBox(Form("partCarbon_D2_H%d_%d", half, ipart), mCarbon, mSupportXDimensions[disk][ipart] / 2.,
                           mSupportYDimensions[disk][ipart] / 2., mCarbonThickness);
    partCarbon->SetLineColor(kGray + 3);
    auto* t = new TGeoTranslation("t", 0, ty + mHalfDiskGap, mZPlan[disk]);
    carbonPlate->AddNode(partCarbon, ipart, t);
    ty += mSupportYDimensions[disk][ipart] / 2.;
  }

  //  if (half == Top) {
  //    rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  //    transformation = new TGeoCombiTrans(0., 0., deltaz/2., rotation);
  //    mHalfDisk->AddNode(carbonPlate, 1, transformation);
  //    transformation = new TGeoCombiTrans(0., 0., -deltaz/2., rotation);
  //    mHalfDisk->AddNode(carbonPlate, 2, transformation);
  //  }
  //  else if (half == Bottom) {
  rotation = new TGeoRotation("rotation", 180., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., deltaz / 2., rotation);
  mHalfDisk->AddNode(carbonPlate, 3, transformation);
  transformation = new TGeoCombiTrans(0., 0., -deltaz / 2., rotation);
  mHalfDisk->AddNode(carbonPlate, 4, transformation);
  //  }

  // **************************************** Glue Bwtween Carbon Plate and Rohacell Plate ****************************************

  TGeoMedium* mGlueRohacellCarbon = gGeoManager->GetMedium("MFT_Epoxy$");

  auto* glueRohacellCarbon = new TGeoVolumeAssembly(Form("glueRohacellCarbon_D0_H%d", half));

  auto* glueRohacellCarbonBase0 = new TGeoBBox(Form("glueRohacellCarbonBase0_D0_H%d", half), (mSupportXDimensions[disk][0]) / 2.,
                                               (mSupportYDimensions[disk][0]) / 2., Geometry::sGlueRohacellCarbonThickness);

  auto* translation_gluRC01 = new TGeoTranslation("translation_gluRC01", 0., (mSupportYDimensions[disk][0]) / 2. + mHalfDiskGap, 0.);
  translation_gluRC01->RegisterYourself();
  auto* translation_gluRC02 = new TGeoTranslation("translation_gluRC02", 0., -mHalfDiskGap, 0.);
  translation_gluRC02->RegisterYourself();

  auto* holeglueRohacellCarbon0 =
    new TGeoTubeSeg(Form("holeglueRohacellCarbon0_D0_H%d", half), 0., mRMin[disk], Geometry::sGlueRohacellCarbonThickness + 0.000001, 0, 180.);

  auto* glueRohacellCarbonhole0 = new TGeoSubtraction(glueRohacellCarbonBase0, holeglueRohacellCarbon0, translation_gluRC01, translation_gluRC02);
  auto* gRC0 = new TGeoCompositeShape(Form("glueRohacellCarbon0_D0_H%d", half), glueRohacellCarbonhole0);
  auto* glueRohacellCarbonBaseWithHole0 = new TGeoVolume(Form("glueRohacellCarbonWithHole_D0_H%d", half), gRC0, mGlueRohacellCarbon);

  glueRohacellCarbonBaseWithHole0->SetLineColor(kGreen);
  rotation = new TGeoRotation("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  glueRohacellCarbon->AddNode(glueRohacellCarbonBaseWithHole0, 0, new TGeoTranslation(0., 0., mZPlan[disk]));

  Double_t tyGRC = mSupportYDimensions[disk][0];

  for (Int_t ipart = 1; ipart < mNPart[disk]; ipart++) {
    tyGRC += mSupportYDimensions[disk][ipart] / 2.;
    TGeoVolume* partGlueRohacellCarbon =
      gGeoManager->MakeBox(Form("partGlueRohacellCarbon_D0_H%d_%d", half, ipart), mGlueRohacellCarbon, mSupportXDimensions[disk][ipart] / 2.,
                           mSupportYDimensions[disk][ipart] / 2., Geometry::sGlueRohacellCarbonThickness);
    partGlueRohacellCarbon->SetLineColor(kGreen);
    auto* t = new TGeoTranslation("t", 0, tyGRC + mHalfDiskGap, mZPlan[disk]);
    glueRohacellCarbon->AddNode(partGlueRohacellCarbon, ipart, t);
    tyGRC += mSupportYDimensions[disk][ipart] / 2.;
  }

  rotation = new TGeoRotation("rotation", 180., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., deltaz / 2. - mCarbonThickness - Geometry::sGlueRohacellCarbonThickness, rotation);
  mHalfDisk->AddNode(glueRohacellCarbon, 3, transformation);
  transformation = new TGeoCombiTrans(0., 0., -(deltaz / 2. - mCarbonThickness - Geometry::sGlueRohacellCarbonThickness), rotation);
  mHalfDisk->AddNode(glueRohacellCarbon, 4, transformation);

  // **************************************** Kapton on Carbon Plate ****************************************

  TGeoMedium* mKaptonOnCarbon = gGeoManager->GetMedium("MFT_Kapton$");

  auto* kaptonOnCarbon = new TGeoVolumeAssembly(Form("kaptonOnCarbon_D0_H%d", half));
  auto* kaptonOnCarbonBase0 = new TGeoBBox(Form("kaptonOnCarbonBase0_D0_H%d", half), (mSupportXDimensions[disk][0]) / 2. + mMoreLength,
                                           (mSupportYDimensions[disk][0]) / 2., Geometry::sKaptonOnCarbonThickness);

  auto* translation_KC01 = new TGeoTranslation("translation_KC01", 0., (mSupportYDimensions[disk][0]) / 2. + mHalfDiskGap, 0.);
  translation_KC01->RegisterYourself();
  auto* translation_KC02 = new TGeoTranslation("translation_KC02", 0., -mHalfDiskGap, 0.);
  translation_KC02->RegisterYourself();

  auto* holekaptonOnCarbon0 =
    new TGeoTubeSeg(Form("holekaptonOnCarbon0_D0_H%d", half), 0., mRMin[disk], Geometry::sKaptonOnCarbonThickness + 0.000001, 0, 180.);

  auto* kaptonOnCarbonhole0 = new TGeoSubtraction(kaptonOnCarbonBase0, holekaptonOnCarbon0, translation_KC01, translation_KC02);
  auto* KC0 = new TGeoCompositeShape(Form("kaptonOnCarbon_D0_H%d", half), kaptonOnCarbonhole0);
  auto* kaptonOnCarbonBaseWithHole0 = new TGeoVolume(Form("kaptonOnCarbonWithHole_D0_H%d", half), KC0, mKaptonOnCarbon);

  kaptonOnCarbonBaseWithHole0->SetLineColor(kMagenta);
  rotation = new TGeoRotation("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  kaptonOnCarbon->AddNode(kaptonOnCarbonBaseWithHole0, 0, new TGeoTranslation(0., 0., mZPlan[disk]));

  Double_t tyKC = mSupportYDimensions[disk][0];

  for (Int_t ipart = 1; ipart < mNPart[disk]; ipart++) {
    tyKC += mSupportYDimensions[disk][ipart] / 2.;
    TGeoVolume* partkaptonOnCarbonBase =
      gGeoManager->MakeBox(Form("partkaptonOnCarbon_D0_H%d_%d", half, ipart), mKaptonOnCarbon, mSupportXDimensions[disk][ipart] / 2.,
                           mSupportYDimensions[disk][ipart] / 2., Geometry::sKaptonOnCarbonThickness);
    partkaptonOnCarbonBase->SetLineColor(kMagenta);
    auto* t = new TGeoTranslation("t", 0, tyKC + mHalfDiskGap, mZPlan[disk]);
    kaptonOnCarbon->AddNode(partkaptonOnCarbonBase, ipart, t);
    tyKC += mSupportYDimensions[disk][ipart] / 2.;
  }

  rotation = new TGeoRotation("rotation", 180., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., deltaz / 2 + Geometry::sKaptonOnCarbonThickness + mCarbonThickness + Geometry::sKaptonGlueThickness * 2, rotation);
  mHalfDisk->AddNode(kaptonOnCarbon, 3, transformation);
  transformation = new TGeoCombiTrans(0., 0., -(deltaz / 2 + Geometry::sKaptonOnCarbonThickness + mCarbonThickness + Geometry::sKaptonGlueThickness * 2), rotation);
  mHalfDisk->AddNode(kaptonOnCarbon, 4, transformation);
  // **************************************** Kapton glue on the carbon plate ****************************************

  TGeoMedium* mGlueKaptonCarbon = gGeoManager->GetMedium("MFT_Epoxy$");

  auto* glueKaptonCarbon = new TGeoVolumeAssembly(Form("glueKaptonCarbon_D0_H%d", half));

  auto* glueKaptonCarbonBase0 = new TGeoBBox(Form("glueKaptonCarbonBase0_D0_H%d", half), (mSupportXDimensions[disk][0]) / 2.,
                                             (mSupportYDimensions[disk][0]) / 2., Geometry::sKaptonGlueThickness);

  auto* translation_gluKC01 = new TGeoTranslation("translation_gluKC01", 0., (mSupportYDimensions[disk][0]) / 2. + mHalfDiskGap, 0.);
  translation_gluKC01->RegisterYourself();
  auto* translation_gluKC02 = new TGeoTranslation("translation_gluKC02", 0., -mHalfDiskGap, 0.);
  translation_gluKC02->RegisterYourself();

  auto* holeglueKaptonCarbon0 =
    new TGeoTubeSeg(Form("holeglueKaptonCarbon0_D0_H%d", half), 0., mRMin[disk], Geometry::sKaptonGlueThickness + 0.000001, 0, 180.);

  auto* glueKaptonCarbonhole0 = new TGeoSubtraction(glueKaptonCarbonBase0, holeglueKaptonCarbon0, translation_gluKC01, translation_gluKC02);
  auto* gKC0 = new TGeoCompositeShape(Form("glueKaptonCarbon0_D0_H%d", half), glueKaptonCarbonhole0);
  auto* glueKaptonCarbonBaseWithHole0 = new TGeoVolume(Form("glueKaptonCarbonWithHole_D0_H%d", half), gKC0, mGlueKaptonCarbon);

  glueKaptonCarbonBaseWithHole0->SetLineColor(kGreen);
  rotation = new TGeoRotation("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  glueKaptonCarbon->AddNode(glueKaptonCarbonBaseWithHole0, 0, new TGeoTranslation(0., 0., mZPlan[disk]));

  Double_t tyGKC = mSupportYDimensions[disk][0];

  for (Int_t ipart = 1; ipart < mNPart[disk]; ipart++) {
    tyGKC += mSupportYDimensions[disk][ipart] / 2.;
    TGeoVolume* partGlueKaptonCarbon =
      gGeoManager->MakeBox(Form("partGlueKaptonCarbon_D0_H%d_%d", half, ipart), mGlueKaptonCarbon, mSupportXDimensions[disk][ipart] / 2.,
                           mSupportYDimensions[disk][ipart] / 2., Geometry::sKaptonGlueThickness);
    partGlueKaptonCarbon->SetLineColor(kGreen);
    auto* t = new TGeoTranslation("t", 0, tyGKC + mHalfDiskGap, mZPlan[disk]);
    glueKaptonCarbon->AddNode(partGlueKaptonCarbon, ipart, t);
    tyGKC += mSupportYDimensions[disk][ipart] / 2.;
  }

  rotation = new TGeoRotation("rotation", 180., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., deltaz / 2. + mCarbonThickness + Geometry::sKaptonGlueThickness, rotation);
  mHalfDisk->AddNode(glueKaptonCarbon, 3, transformation);
  transformation = new TGeoCombiTrans(0., 0., -(deltaz / 2. + mCarbonThickness + Geometry::sKaptonGlueThickness), rotation);
  mHalfDisk->AddNode(glueKaptonCarbon, 4, transformation);

  // **************************************** Rohacell Plate ****************************************

  auto* rohacellPlate = new TGeoVolumeAssembly(Form("rohacellPlate_D2_H%d", half));

  auto* rohacellBase2 = new TGeoBBox(Form("rohacellBase2_D2_H%d", half), (mSupportXDimensions[disk][0]) / 2.,
                                     (mSupportYDimensions[disk][0]) / 2., mRohacellThickness);
  // TGeoTranslation *t3 = new TGeoTranslation ("t3",0., (fSupportYDimensions[disk][0])/2. + fHalfDiskGap , 0.);
  // t3 -> RegisterYourself();

  auto* holeRohacell2 =
    new TGeoTubeSeg(Form("holeRohacell2_D2_H%d", half), 0., mRMin[disk], mRohacellThickness + 0.000001, 0, 180.);
  // TGeoTranslation *t4= new TGeoTranslation ("t4", 0., - fHalfDiskGap , 0.);
  // t4-> RegisterYourself();

  /// cs2 = new TGeoCompositeShape(Form("rohacell_D2_H%d",half),
  /// Form("(rohacellBase2_D2_H%d:t21)-(holeRohacell2_D2_H%d:t22)",half,half));

  // **************************************** GROOVES *************************************************
  Double_t diameter = 0.21; // groove diameter
  Double_t epsilon = 0.06;  // outside shift of the goove
  Int_t iCount = 0;
  Double_t mPosition[4];

  TGeoCombiTrans* transfo[7][3];
  TGeoTube* grooveTube[7][3];
  TGeoTorus* grooveTorus[7][3];
  TGeoSubtraction* rohacellBaseGroove[300];
  TGeoCompositeShape* rohacellGroove[300];

  for (Int_t igroove = 0; igroove < 3; igroove++) {
    grooveTube[0][igroove] = new TGeoTube("linear", 0., diameter, mLWater2[igroove] / 2.);
    grooveTorus[1][igroove] = new TGeoTorus("SideTorus", mRadius2[igroove], 0., diameter, 0., mAngle2[igroove]);
    grooveTube[2][igroove] = new TGeoTube("tiltedLinear", 0., diameter, mLpartial2[igroove] / 2.);
    grooveTorus[3][igroove] = new TGeoTorus("centralTorus", mRadiusCentralTore[igroove], 0., diameter, -mAngle2[igroove], 2. * mAngle2[igroove]);
    grooveTube[4][igroove] = new TGeoTube("tiltedLinear", 0., diameter, mLpartial2[igroove] / 2.);
    grooveTorus[5][igroove] = new TGeoTorus("SideTorus", mRadius2[igroove], 0., diameter, 0., mAngle2[igroove]);
    grooveTube[6][igroove] = new TGeoTube("linear", 0., diameter, mLWater2[igroove] / 2.);
  }

  // Rotation matrix
  TGeoRotation* rotationLinear = new TGeoRotation("rotation", -90., 90., 0.);
  TGeoRotation* rotationSideTorusL = new TGeoRotation("rotationSideTorusLeft", -90., 0., 0.);
  TGeoRotation* rotationSideTorusR = new TGeoRotation("rotationSideTorusRight", 90., 180., 180.);
  TGeoRotation* rotationCentralTorus = new TGeoRotation("rotationCentralTorus", 90., 0., 0.);
  TGeoRotation* rotationTiltedLinearR;
  TGeoRotation* rotationTiltedLinearL;

  // Creating grooves
  for (Int_t iface = 1; iface > -2; iface -= 2) {     // front and rear
    for (Int_t igroove = 0; igroove < 3; igroove++) { // 3 grooves
      mPosition[igroove] = mXPosition2[igroove] - mSupportYDimensions[disk][0] / 2. - mHalfDiskGap;
      for (Int_t ip = 0; ip < 7; ip++) { // each groove is made of 7 parts

        switch (ip) {
          case 0: // Linear
            transfo[ip][igroove] = new TGeoCombiTrans(mSupportXDimensions[disk][0] / 2. + mMoreLength - mLWater2[igroove] / 2., mPosition[igroove],
                                                      iface * (mRohacellThickness + epsilon), rotationLinear);
            if (igroove == 0 && iface == 1) {
              rohacellBaseGroove[0] = new TGeoSubtraction(rohacellBase2, grooveTube[ip][igroove], nullptr, transfo[ip][igroove]);
              rohacellGroove[0] = new TGeoCompositeShape(Form("rohacell2Groove%d_G%d_F%d_H%d", ip, igroove, iface, half), rohacellBaseGroove[0]);
            };
            break;
          case 1: // side torus
            transfo[ip][igroove] = new TGeoCombiTrans(mSupportXDimensions[disk][0] / 2. + mMoreLength - mLWater2[igroove], mRadius2[igroove] + mXPosition2[igroove] - mSupportYDimensions[disk][0] / 2. - mHalfDiskGap, iface * (mRohacellThickness + epsilon), rotationSideTorusR);
            break;
          case 2: // Linear tilted
            rotationTiltedLinearR = new TGeoRotation("rotationTiltedLinearRight", 90. - mAngle2[igroove], 90., 0.);
            transfo[ip][igroove] = new TGeoCombiTrans(mSupportXDimensions[disk][0] / 2. + mMoreLength - xPos2[igroove], yPos2[igroove] - mSupportYDimensions[disk][0] / 2. - mHalfDiskGap,
                                                      iface * (mRohacellThickness + epsilon), rotationTiltedLinearR);
            break;
          case 3: // Central Torus
            transfo[ip][igroove] = new TGeoCombiTrans(0., yPos2[igroove] + mLpartial2[igroove] / 2 * TMath::Sin(mAngle2[igroove] * TMath::DegToRad()) - mRadiusCentralTore[igroove] * TMath::Cos(mAngle2[igroove] * TMath::DegToRad()) - mSupportYDimensions[disk][0] / 2. - mHalfDiskGap,
                                                      iface * (mRohacellThickness + epsilon), rotationCentralTorus);
            break;
          case 4: // Linear tilted
            rotationTiltedLinearL = new TGeoRotation("rotationTiltedLinearLeft", 90. + mAngle2[igroove], 90., 0.);
            transfo[ip][igroove] = new TGeoCombiTrans(-(mSupportXDimensions[disk][0] / 2. + mMoreLength - xPos2[igroove]), yPos2[igroove] - mSupportYDimensions[disk][0] / 2. - mHalfDiskGap,
                                                      iface * (mRohacellThickness + epsilon), rotationTiltedLinearL);
            break;
          case 5: // side torus
            transfo[ip][igroove] = new TGeoCombiTrans(-(mSupportXDimensions[disk][0] / 2. + mMoreLength - mLWater2[igroove]), mRadius2[igroove] + mPosition[igroove],
                                                      iface * (mRohacellThickness + epsilon), rotationSideTorusL);
            break;
          case 6: // Linear
            transfo[ip][igroove] = new TGeoCombiTrans(-(mSupportXDimensions[disk][0] / 2. + mMoreLength - mLWater2[igroove] / 2.), mPosition[igroove],
                                                      iface * (mRohacellThickness + epsilon), rotationLinear);
            break;
        }

        if (!(ip == 0 && igroove == 0 && iface == 1)) {
          if (ip & 1) {
            rohacellBaseGroove[iCount] = new TGeoSubtraction(rohacellGroove[iCount - 1], grooveTorus[ip][igroove], nullptr, transfo[ip][igroove]);
          } else {
            rohacellBaseGroove[iCount] = new TGeoSubtraction(rohacellGroove[iCount - 1], grooveTube[ip][igroove], nullptr, transfo[ip][igroove]);
          }
          rohacellGroove[iCount] = new TGeoCompositeShape(Form("rohacell2Groove%d_G%d_F%d_H%d", iCount, igroove, iface, half), rohacellBaseGroove[iCount]);
        }
        iCount++;
      }
    }
  }

  // **************************************************************************************************

  //auto* rohacellhole2 = new TGeoSubtraction(rohacellBase2, holeRohacell2, t21, t22);
  auto* rohacellBase = new TGeoSubtraction(rohacellGroove[iCount - 1], holeRohacell2, t21, t22);
  auto* rh2 = new TGeoCompositeShape(Form("rohacellBase2_D2_H%d", half), rohacellBase);
  //auto* rh2 = new TGeoCompositeShape(Form("rohacellBase2_D2_H%d", half), rohacellhole2);
  auto* rohacellBaseWithHole = new TGeoVolume(Form("rohacellBaseWithHole_D2_H%d", half), rh2, mRohacell);

  TGeoVolume* partRohacell;
  rohacellBaseWithHole->SetLineColor(kGray);
  rotation = new TGeoRotation("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  rohacellPlate->AddNode(rohacellBaseWithHole, 0, new TGeoTranslation(0., 0., mZPlan[disk]));

  ty = mSupportYDimensions[disk][0];

  for (Int_t ipart = 1; ipart < mNPart[disk]; ipart++) {
    ty += mSupportYDimensions[disk][ipart] / 2.;
    /*
    TGeoVolume* partRohacell = gGeoManager->MakeBox(Form("partRohacelli_D2_H%d_%d", half, ipart), mRohacell,
                                                    mSupportXDimensions[disk][ipart] / 2.,
                                                    mSupportYDimensions[disk][ipart] / 2., mRohacellThickness);
    partRohacell->SetLineColor(kGray);
    */
    auto* t = new TGeoTranslation("t", 0, ty + mHalfDiskGap, mZPlan[disk]);

    //===========================================================================================================
    //===========================================================================================================
    auto* partRohacell0 =
      new TGeoBBox(Form("rohacellBase0_D2_H%d_%d", half, ipart), mSupportXDimensions[disk][ipart] / 2.,
                   mSupportYDimensions[disk][ipart] / 2., mRohacellThickness);

    // ***************** Creating grooves for the other parts of the rohacell plate **********************
    Double_t mShift;
    for (Int_t iface = 1; iface > -2; iface -= 2) {     // front and rear
      for (Int_t igroove = 0; igroove < 3; igroove++) { // 3 grooves
        if (ipart == 1) {
          mPosition[ipart] = mXPosition2[igroove] - mSupportYDimensions[disk][ipart] / 2. - mHalfDiskGap - mSupportYDimensions[disk][ipart - 1];
          mShift = -mSupportYDimensions[disk][ipart - 1];
        };
        if (ipart == 2) {
          mPosition[ipart] = mXPosition2[igroove] - mSupportYDimensions[disk][ipart] / 2. - mHalfDiskGap - mSupportYDimensions[disk][ipart - 1] - mSupportYDimensions[disk][ipart - 2];
          mShift = -mSupportYDimensions[disk][ipart - 1] - mSupportYDimensions[disk][ipart - 2];
        };
        if (ipart == 3) {
          mPosition[ipart] = mXPosition2[igroove] - mSupportYDimensions[disk][ipart] / 2. - mHalfDiskGap - mSupportYDimensions[disk][ipart - 1] - mSupportYDimensions[disk][ipart - 2] - mSupportYDimensions[disk][ipart - 3];
          mShift = -mSupportYDimensions[disk][ipart - 1] - mSupportYDimensions[disk][ipart - 2] - mSupportYDimensions[disk][ipart - 3];
        };

        for (Int_t ip = 0; ip < 7; ip++) { // each groove is made of 7 parts

          switch (ip) {
            case 0: // Linear
              transfo[ip][igroove] = new TGeoCombiTrans(mSupportXDimensions[disk][0] / 2. + mMoreLength - mLWater2[igroove] / 2., mPosition[ipart],
                                                        iface * (mRohacellThickness + epsilon), rotationLinear);
              if (igroove == 0 && iface == 1) {
                rohacellBaseGroove[iCount] = new TGeoSubtraction(partRohacell0, grooveTube[ip][igroove], nullptr, transfo[ip][igroove]);
                rohacellGroove[iCount] = new TGeoCompositeShape(Form("rohacell2Groove%d_G%d_F%d_H%d", ip, igroove, iface, half), rohacellBaseGroove[iCount]);
              };
              break;
            case 1: // side torus
              transfo[ip][igroove] = new TGeoCombiTrans(mSupportXDimensions[2][0] / 2. + mMoreLength - mLWater2[igroove], mPosition[ipart] + mRadius2[igroove],
                                                        iface * (mRohacellThickness + epsilon), rotationSideTorusR);
              break;
            case 2: // Linear tilted
              rotationTiltedLinearR = new TGeoRotation("rotationTiltedLinearRight", 90. - mAngle2[igroove], 90., 0.);
              transfo[ip][igroove] = new TGeoCombiTrans(mSupportXDimensions[disk][0] / 2. + mMoreLength - xPos2[igroove], yPos2[igroove] + mShift - mHalfDiskGap - mSupportYDimensions[disk][ipart] / 2., iface * (mRohacellThickness + epsilon), rotationTiltedLinearR);
              break;
            case 3: // Central Torus
              transfo[ip][igroove] = new TGeoCombiTrans(0., mPosition[ipart] + yPos2[igroove] + mLpartial2[igroove] / 2 * TMath::Sin(mAngle2[igroove] * TMath::DegToRad()) - mRadiusCentralTore[igroove] * TMath::Cos(mAngle2[igroove] * TMath::DegToRad()) - mXPosition2[igroove],
                                                        iface * (mRohacellThickness + epsilon), rotationCentralTorus);
              break;
            case 4: // Linear tilted
              rotationTiltedLinearL = new TGeoRotation("rotationTiltedLinearLeft", 90. + mAngle2[igroove], 90., 0.);
              transfo[ip][igroove] = new TGeoCombiTrans(-(mSupportXDimensions[disk][0] / 2. + mMoreLength - xPos2[igroove]), yPos2[igroove] + mPosition[ipart] - mXPosition2[igroove],
                                                        iface * (mRohacellThickness + epsilon), rotationTiltedLinearL);
              break;
            case 5: // side torus
              transfo[ip][igroove] = new TGeoCombiTrans(-(mSupportXDimensions[disk][0] / 2. + mMoreLength - mLWater2[igroove]), mRadius2[igroove] + mPosition[ipart],
                                                        iface * (mRohacellThickness + epsilon), rotationSideTorusL);
              break;
            case 6: // Linear
              transfo[ip][igroove] = new TGeoCombiTrans(-(mSupportXDimensions[disk][0] / 2. + mMoreLength - mLWater2[igroove] / 2.), mPosition[ipart],
                                                        iface * (mRohacellThickness + epsilon), rotationLinear);
              break;
          }
          if (!(ip == 0 && igroove == 0 && iface == 1)) {
            if (ip & 1) {
              rohacellBaseGroove[iCount] =
                new TGeoSubtraction(rohacellGroove[iCount - 1], grooveTorus[ip][igroove], nullptr, transfo[ip][igroove]);
            } else {
              rohacellBaseGroove[iCount] =
                new TGeoSubtraction(rohacellGroove[iCount - 1], grooveTube[ip][igroove], nullptr, transfo[ip][igroove]);
            }

            rohacellGroove[iCount] = new TGeoCompositeShape(Form("rohacell2Groove%d_G%d_F%d_H%d", iCount, igroove, iface, half), rohacellBaseGroove[iCount]);
          }
          iCount++;
        }
      }
    }
    // **************************************************************************************************

    partRohacell = new TGeoVolume(Form("partRohacelli_D2_H%d_%d", half, ipart), rohacellGroove[iCount - 1], mRohacell);
    //partRohacell = new TGeoVolume(Form("partRohacelli_D2_H%d_%d", half, ipart), partRohacell0, mRohacell);
    //===========================================================================================================
    //===========================================================================================================

    rohacellPlate->AddNode(partRohacell, ipart, t);
    ty += mSupportYDimensions[disk][ipart] / 2.;
  }

  //  if (half == Top) {
  //    rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  //    transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  //    mHalfDisk->AddNode(rohacellPlate, 1, transformation);
  //  }
  //  if (half == Bottom) {
  rotation = new TGeoRotation("rotation", 180., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  mHalfDisk->AddNode(rohacellPlate, 2, transformation);
  //  }

  createManifold(disk);
}

//_____________________________________________________________________________
void HeatExchanger::createHalfDisk3(Int_t half)
{

  Int_t disk = 3;

  if (half == Top)
    printf("Creating MFT heat exchanger for disk3 top\n");
  else if (half == Bottom)
    printf("Creating MFT heat exchanger for disk3 bottom\n");
  else
    printf("No valid option for MFT heat exchanger on disk3\n");

  // mCarbon   = gGeoManager->GetMedium("MFT_Carbon$");
  mCarbon = gGeoManager->GetMedium("MFT_CarbonFiber$");

  mWater = gGeoManager->GetMedium("MFT_Water$");
  mRohacell = gGeoManager->GetMedium("MFT_Rohacell");
  mPipe = gGeoManager->GetMedium("MFT_Polyimide");

  auto* cooling = new TGeoVolumeAssembly(Form("cooling_D3_H%d", half));
  //Double_t deltaz = mHeatExchangerThickness - mCarbonThickness * 2; // distance between pair of carbon plans
  Double_t deltaz = mHeatExchangerThickness - Geometry::sKaptonOnCarbonThickness * 4 - Geometry::sKaptonGlueThickness * 4 - 2 * mCarbonThickness;

  TGeoTranslation* translation = nullptr;
  TGeoRotation* rotation = nullptr;
  TGeoCombiTrans* transformation = nullptr;

  // **************************************** Water part ****************************************

  // ********************** Four parameters mLwater3, mRadius3, mAngle3, mLpartial3 *************
  Double_t ivolume = 300; // offset chamber 3
  Double_t mRadiusCentralTore[4];
  Double_t xPos3[4];
  Double_t yPos3[4];
  for (Int_t itube = 0; itube < 4; itube++) {
    TGeoVolume* waterTube1 = gGeoManager->MakeTube(Form("waterTube1%d_D3_H%d", itube, half), mWater, 0., mRWater, mLWater3[itube] / 2.);
    translation = new TGeoTranslation(mXPosition3[itube], 0., mSupportXDimensions[3][0] / 2. + mMoreLength - mLWater3[itube] / 2.);
    cooling->AddNode(waterTube1, ivolume++, translation);

    TGeoVolume* waterTorus1 = gGeoManager->MakeTorus(Form("waterTorus1%d_D3_H%d", itube, half), mWater, mRadius3[itube], 0., mRWater, 0., mAngle3[itube]);
    rotation = new TGeoRotation("rotation", 180., -90., 0.);
    transformation = new TGeoCombiTrans(mRadius3[itube] + mXPosition3[itube], 0., mSupportXDimensions[3][0] / 2. + mMoreLength - mLWater3[itube], rotation);
    cooling->AddNode(waterTorus1, ivolume++, transformation);

    TGeoVolume* waterTube2 = gGeoManager->MakeTube(Form("waterTube2%d_D3_H%d", itube, half), mWater, 0., mRWater, mLpartial3[itube] / 2.);
    rotation = new TGeoRotation("rotation", 90., 180 - mAngle3[itube], 0.);
    xPos3[itube] = mLWater3[itube] + mRadius3[itube] * TMath::Sin(mAngle3[itube] * TMath::DegToRad()) + mLpartial3[itube] / 2 * TMath::Cos(mAngle3[itube] * TMath::DegToRad());
    yPos3[itube] = mXPosition3[itube] + mRadius3[itube] * (1 - TMath::Cos(mAngle3[itube] * TMath::DegToRad())) + mLpartial3[itube] / 2 * TMath::Sin(mAngle3[itube] * TMath::DegToRad());
    transformation = new TGeoCombiTrans(yPos3[itube], 0., mSupportXDimensions[3][0] / 2. + mMoreLength - xPos3[itube], rotation);
    cooling->AddNode(waterTube2, ivolume++, transformation);

    mRadiusCentralTore[itube] = (mSupportXDimensions[3][0] / 2. + mMoreLength - xPos3[itube] - mLpartial3[itube] / 2 * TMath::Cos(mAngle3[itube] * TMath::DegToRad())) / TMath::Sin(mAngle3[itube] * TMath::DegToRad());
    TGeoVolume* waterTorusCentral = gGeoManager->MakeTorus(Form("waterTorusCentral%d_D3_H%d", itube, half), mWater, mRadiusCentralTore[itube], 0., mRWater,
                                                           -mAngle3[itube], 2. * mAngle3[itube]);
    rotation = new TGeoRotation("rotation", 0., 90., 0.);
    transformation = new TGeoCombiTrans(yPos3[itube] + mLpartial3[itube] / 2 * TMath::Sin(mAngle3[itube] * TMath::DegToRad()) - mRadiusCentralTore[itube] * TMath::Cos(mAngle3[itube] * TMath::DegToRad()), 0., 0., rotation);
    cooling->AddNode(waterTorusCentral, ivolume++, transformation);

    TGeoVolume* waterTube3 = gGeoManager->MakeTube(Form("waterTube3%d_D3_H%d", 2, half), mWater, 0., mRWater, mLpartial3[itube] / 2.);
    rotation = new TGeoRotation("rotation", -90., 0 - mAngle3[itube], 0.);
    transformation = new TGeoCombiTrans(yPos3[itube], 0., -(mSupportXDimensions[3][0] / 2. + mMoreLength - xPos3[itube]), rotation);
    cooling->AddNode(waterTube3, ivolume++, transformation);

    TGeoVolume* waterTorus2 = gGeoManager->MakeTorus(Form("waterTorus2%d_D3_H%d", itube, half), mWater, mRadius3[itube], 0., mRWater, 0., mAngle3[itube]);
    rotation = new TGeoRotation("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(mRadius3[itube] + mXPosition3[itube], 0., -(mSupportXDimensions[3][0] / 2. + mMoreLength - mLWater3[itube]), rotation);
    cooling->AddNode(waterTorus2, ivolume++, transformation);

    TGeoVolume* waterTube4 = gGeoManager->MakeTube(Form("waterTube4%d_D3_H%d", itube, half), mWater, 0., mRWater, mLWater3[itube] / 2.);
    translation = new TGeoTranslation(mXPosition3[itube], 0., -(mSupportXDimensions[3][0] / 2. + mMoreLength - mLWater3[itube] / 2.));
    cooling->AddNode(waterTube4, ivolume++, translation);
  }

  // **************************************************** Tube part ************************************************
  // ****************************** Four parameters mLwater3, mRadius3, mAngle3, mLpartial3 ************************
  for (Int_t itube = 0; itube < 4; itube++) {

    TGeoVolume* pipeTube1 = gGeoManager->MakeTube(Form("pipeTube1%d_D3_H%d", itube, half), mPipe, mRWater, mRWater + mDRPipe, mLWater3[itube] / 2.);
    translation = new TGeoTranslation(mXPosition3[itube], 0., mSupportXDimensions[3][0] / 2. + mMoreLength - mLWater3[itube] / 2.);
    cooling->AddNode(pipeTube1, ivolume++, translation);

    TGeoVolume* pipeTorus1 = gGeoManager->MakeTorus(Form("pipeTorus1%d_D3_H%d", itube, half), mPipe, mRadius3[itube], mRWater, mRWater + mDRPipe, 0., mAngle3[itube]);
    rotation = new TGeoRotation("rotation", 180., -90., 0.);
    transformation = new TGeoCombiTrans(mRadius3[itube] + mXPosition3[itube], 0., mSupportXDimensions[3][0] / 2. + mMoreLength - mLWater3[itube], rotation);
    cooling->AddNode(pipeTorus1, ivolume++, transformation);

    TGeoVolume* pipeTube2 = gGeoManager->MakeTube(Form("pipeTube2%d_D3_H%d", itube, half), mPipe, mRWater, mRWater + mDRPipe, mLpartial3[itube] / 2.);
    rotation = new TGeoRotation("rotation", 90., 180 - mAngle3[itube], 0.);
    xPos3[itube] = mLWater3[itube] + mRadius3[itube] * TMath::Sin(mAngle3[itube] * TMath::DegToRad()) + mLpartial3[itube] / 2 * TMath::Cos(mAngle3[itube] * TMath::DegToRad());
    yPos3[itube] = mXPosition3[itube] + mRadius3[itube] * (1 - TMath::Cos(mAngle3[itube] * TMath::DegToRad())) + mLpartial3[itube] / 2 * TMath::Sin(mAngle3[itube] * TMath::DegToRad());
    transformation = new TGeoCombiTrans(yPos3[itube], 0., mSupportXDimensions[3][0] / 2. + mMoreLength - xPos3[itube], rotation);
    cooling->AddNode(pipeTube2, ivolume++, transformation);

    mRadiusCentralTore[itube] = (mSupportXDimensions[3][0] / 2. + mMoreLength - xPos3[itube] - mLpartial3[itube] / 2 * TMath::Cos(mAngle3[itube] * TMath::DegToRad())) / TMath::Sin(mAngle3[itube] * TMath::DegToRad());
    TGeoVolume* pipeTorusCentral = gGeoManager->MakeTorus(Form("pipeTorusCentral%d_D3_H%d", itube, half), mPipe, mRadiusCentralTore[itube], mRWater, mRWater + mDRPipe,
                                                          -mAngle3[itube], 2. * mAngle3[itube]);
    rotation = new TGeoRotation("rotation", 0., 90., 0.);
    transformation = new TGeoCombiTrans(yPos3[itube] + mLpartial3[itube] / 2 * TMath::Sin(mAngle3[itube] * TMath::DegToRad()) - mRadiusCentralTore[itube] * TMath::Cos(mAngle3[itube] * TMath::DegToRad()), 0., 0., rotation);
    cooling->AddNode(pipeTorusCentral, ivolume++, transformation);

    TGeoVolume* pipeTube3 = gGeoManager->MakeTube(Form("pipeTube3%d_D3_H%d", 2, half), mPipe, mRWater, mRWater + mDRPipe, mLpartial3[itube] / 2.);
    rotation = new TGeoRotation("rotation", -90., 0 - mAngle3[itube], 0.);
    transformation = new TGeoCombiTrans(yPos3[itube], 0., -(mSupportXDimensions[3][0] / 2. + mMoreLength - xPos3[itube]), rotation);
    cooling->AddNode(pipeTube3, ivolume++, transformation);

    TGeoVolume* pipeTorus2 = gGeoManager->MakeTorus(Form("pipeTorus2%d_D3_H%d", itube, half), mPipe, mRadius3[itube], mRWater, mRWater + mDRPipe, 0., mAngle3[itube]);
    rotation = new TGeoRotation("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(mRadius3[itube] + mXPosition3[itube], 0., -(mSupportXDimensions[3][0] / 2. + mMoreLength - mLWater3[itube]), rotation);
    cooling->AddNode(pipeTorus2, ivolume++, transformation);

    TGeoVolume* pipeTube4 = gGeoManager->MakeTube(Form("pipeTube4%d_D3_H%d", itube, half), mPipe, mRWater, mRWater + mDRPipe, mLWater3[itube] / 2.);
    translation = new TGeoTranslation(mXPosition3[itube], 0., -(mSupportXDimensions[3][0] / 2. + mMoreLength - mLWater3[itube] / 2.));
    cooling->AddNode(pipeTube4, ivolume++, translation);
  }
  // ***********************************************************************************************
  rotation = new TGeoRotation("rotation", -90., 90., 0.);
  transformation =
    //new TGeoCombiTrans(0., 0., mZPlan[disk] + deltaz / 2. - mCarbonThickness - mRWater - mDRPipe, rotation);
    new TGeoCombiTrans(0., 0., mZPlan[disk] + deltaz / 2. - mCarbonThickness - mRWater - mDRPipe - 2 * Geometry::sGlueRohacellCarbonThickness, rotation);
  mHalfDisk->AddNode(cooling, 3, transformation);
  transformation =
    //new TGeoCombiTrans(0., 0., mZPlan[disk] - deltaz / 2. + mCarbonThickness + mRWater + mDRPipe, rotation);
    new TGeoCombiTrans(0., 0., mZPlan[disk] - deltaz / 2. + mCarbonThickness + mRWater + mDRPipe + 2 * Geometry::sGlueRohacellCarbonThickness, rotation);
  mHalfDisk->AddNode(cooling, 4, transformation);

  // **************************************** Carbon Plates ****************************************

  auto* carbonPlate = new TGeoVolumeAssembly(Form("carbonPlate_D3_H%d", half));

  auto* carbonBase3 = new TGeoBBox(Form("carbonBase3_D3_H%d", half), (mSupportXDimensions[disk][0]) / 2. + mMoreLength,
                                   (mSupportYDimensions[disk][0]) / 2., mCarbonThickness);
  auto* t31 = new TGeoTranslation("t31", 0., (mSupportYDimensions[disk][0]) / 2. + mHalfDiskGap, 0.);
  t31->RegisterYourself();

  auto* holeCarbon3 =
    new TGeoTubeSeg(Form("holeCarbon3_D3_H%d", half), 0., mRMin[disk], mCarbonThickness + 0.000001, 0, 180.);
  auto* t32 = new TGeoTranslation("t32", 0., -mHalfDiskGap, 0.);
  t32->RegisterYourself();

  /// TGeoCompositeShape *cs3 = new
  /// TGeoCompositeShape(Form("Carbon3_D3_H%d",half),Form("(carbonBase3_D3_H%d:t31)-(holeCarbon3_D3_H%d:t32)",half,half)
  /// );
  auto* carbonhole3 = new TGeoSubtraction(carbonBase3, holeCarbon3, t31, t32);
  auto* cs3 = new TGeoCompositeShape(Form("Carbon3_D3_H%d", half), carbonhole3);
  auto* carbonBaseWithHole3 = new TGeoVolume(Form("carbonBaseWithHole_D3_H%d", half), cs3, mCarbon);

  carbonBaseWithHole3->SetLineColor(kGray + 3);
  rotation = new TGeoRotation("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  carbonPlate->AddNode(carbonBaseWithHole3, 0, new TGeoTranslation(0., 0., mZPlan[disk]));

  Double_t ty = mSupportYDimensions[disk][0];

  for (Int_t ipart = 1; ipart < mNPart[disk]; ipart++) {
    ty += mSupportYDimensions[disk][ipart] / 2.;
    TGeoVolume* partCarbon =
      gGeoManager->MakeBox(Form("partCarbon_D3_H%d_%d", half, ipart), mCarbon, mSupportXDimensions[disk][ipart] / 2.,
                           mSupportYDimensions[disk][ipart] / 2., mCarbonThickness);
    partCarbon->SetLineColor(kGray + 3);
    auto* t = new TGeoTranslation("t", 0, ty + mHalfDiskGap, mZPlan[disk]);
    carbonPlate->AddNode(partCarbon, ipart, t);
    ty += mSupportYDimensions[disk][ipart] / 2.;
  }

  //  if (half == Top) {
  //    rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  //    transformation = new TGeoCombiTrans(0., 0., deltaz/2., rotation);
  //    mHalfDisk->AddNode(carbonPlate, 1, transformation);
  //    transformation = new TGeoCombiTrans(0., 0., -deltaz/2., rotation);
  //    mHalfDisk->AddNode(carbonPlate, 2, transformation);
  //  }
  //  else if (half == Bottom) {
  rotation = new TGeoRotation("rotation", 180., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., deltaz / 2., rotation);
  mHalfDisk->AddNode(carbonPlate, 3, transformation);
  transformation = new TGeoCombiTrans(0., 0., -deltaz / 2., rotation);
  mHalfDisk->AddNode(carbonPlate, 4, transformation);
  //  }

  // **************************************** Glue Bwtween Carbon Plate and Rohacell Plate ****************************************

  TGeoMedium* mGlueRohacellCarbon = gGeoManager->GetMedium("MFT_Epoxy$");

  auto* glueRohacellCarbon = new TGeoVolumeAssembly(Form("glueRohacellCarbon_D0_H%d", half));

  auto* glueRohacellCarbonBase0 = new TGeoBBox(Form("glueRohacellCarbonBase0_D0_H%d", half), (mSupportXDimensions[disk][0]) / 2.,
                                               (mSupportYDimensions[disk][0]) / 2., Geometry::sGlueRohacellCarbonThickness);

  auto* translation_gluRC01 = new TGeoTranslation("translation_gluRC01", 0., (mSupportYDimensions[disk][0]) / 2. + mHalfDiskGap, 0.);
  translation_gluRC01->RegisterYourself();
  auto* translation_gluRC02 = new TGeoTranslation("translation_gluRC02", 0., -mHalfDiskGap, 0.);
  translation_gluRC02->RegisterYourself();

  auto* holeglueRohacellCarbon0 =
    new TGeoTubeSeg(Form("holeglueRohacellCarbon0_D0_H%d", half), 0., mRMin[disk], Geometry::sGlueRohacellCarbonThickness + 0.000001, 0, 180.);

  auto* glueRohacellCarbonhole0 = new TGeoSubtraction(glueRohacellCarbonBase0, holeglueRohacellCarbon0, translation_gluRC01, translation_gluRC02);
  auto* gRC0 = new TGeoCompositeShape(Form("glueRohacellCarbon0_D0_H%d", half), glueRohacellCarbonhole0);
  auto* glueRohacellCarbonBaseWithHole0 = new TGeoVolume(Form("glueRohacellCarbonWithHole_D0_H%d", half), gRC0, mGlueRohacellCarbon);

  glueRohacellCarbonBaseWithHole0->SetLineColor(kGreen);
  rotation = new TGeoRotation("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  glueRohacellCarbon->AddNode(glueRohacellCarbonBaseWithHole0, 0, new TGeoTranslation(0., 0., mZPlan[disk]));

  Double_t tyGRC = mSupportYDimensions[disk][0];

  for (Int_t ipart = 1; ipart < mNPart[disk]; ipart++) {
    tyGRC += mSupportYDimensions[disk][ipart] / 2.;
    TGeoVolume* partGlueRohacellCarbon =
      gGeoManager->MakeBox(Form("partGlueRohacellCarbon_D0_H%d_%d", half, ipart), mGlueRohacellCarbon, mSupportXDimensions[disk][ipart] / 2.,
                           mSupportYDimensions[disk][ipart] / 2., Geometry::sGlueRohacellCarbonThickness);
    partGlueRohacellCarbon->SetLineColor(kGreen);
    auto* t = new TGeoTranslation("t", 0, tyGRC + mHalfDiskGap, mZPlan[disk]);
    glueRohacellCarbon->AddNode(partGlueRohacellCarbon, ipart, t);
    tyGRC += mSupportYDimensions[disk][ipart] / 2.;
  }

  rotation = new TGeoRotation("rotation", 180., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., deltaz / 2. - mCarbonThickness - Geometry::sGlueRohacellCarbonThickness, rotation);
  mHalfDisk->AddNode(glueRohacellCarbon, 3, transformation);
  transformation = new TGeoCombiTrans(0., 0., -(deltaz / 2. - mCarbonThickness - Geometry::sGlueRohacellCarbonThickness), rotation);
  mHalfDisk->AddNode(glueRohacellCarbon, 4, transformation);

  // **************************************** Kapton on Carbon Plate ****************************************

  TGeoMedium* mKaptonOnCarbon = gGeoManager->GetMedium("MFT_Kapton$");

  auto* kaptonOnCarbon = new TGeoVolumeAssembly(Form("kaptonOnCarbon_D0_H%d", half));
  auto* kaptonOnCarbonBase0 = new TGeoBBox(Form("kaptonOnCarbonBase0_D0_H%d", half), (mSupportXDimensions[disk][0]) / 2. + mMoreLength,
                                           (mSupportYDimensions[disk][0]) / 2., Geometry::sKaptonOnCarbonThickness);

  auto* translation_KC01 = new TGeoTranslation("translation_KC01", 0., (mSupportYDimensions[disk][0]) / 2. + mHalfDiskGap, 0.);
  translation_KC01->RegisterYourself();
  auto* translation_KC02 = new TGeoTranslation("translation_KC02", 0., -mHalfDiskGap, 0.);
  translation_KC02->RegisterYourself();

  auto* holekaptonOnCarbon0 =
    new TGeoTubeSeg(Form("holekaptonOnCarbon0_D0_H%d", half), 0., mRMin[disk], Geometry::sKaptonOnCarbonThickness + 0.000001, 0, 180.);

  auto* kaptonOnCarbonhole0 = new TGeoSubtraction(kaptonOnCarbonBase0, holekaptonOnCarbon0, translation_KC01, translation_KC02);
  auto* KC0 = new TGeoCompositeShape(Form("kaptonOnCarbon_D0_H%d", half), kaptonOnCarbonhole0);
  auto* kaptonOnCarbonBaseWithHole0 = new TGeoVolume(Form("kaptonOnCarbonWithHole_D0_H%d", half), KC0, mKaptonOnCarbon);

  kaptonOnCarbonBaseWithHole0->SetLineColor(kMagenta);
  rotation = new TGeoRotation("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  kaptonOnCarbon->AddNode(kaptonOnCarbonBaseWithHole0, 0, new TGeoTranslation(0., 0., mZPlan[disk]));

  Double_t tyKC = mSupportYDimensions[disk][0];

  for (Int_t ipart = 1; ipart < mNPart[disk]; ipart++) {
    tyKC += mSupportYDimensions[disk][ipart] / 2.;
    TGeoVolume* partkaptonOnCarbonBase =
      gGeoManager->MakeBox(Form("partkaptonOnCarbon_D0_H%d_%d", half, ipart), mKaptonOnCarbon, mSupportXDimensions[disk][ipart] / 2.,
                           mSupportYDimensions[disk][ipart] / 2., Geometry::sKaptonOnCarbonThickness);
    partkaptonOnCarbonBase->SetLineColor(kMagenta);
    auto* t = new TGeoTranslation("t", 0, tyKC + mHalfDiskGap, mZPlan[disk]);
    kaptonOnCarbon->AddNode(partkaptonOnCarbonBase, ipart, t);
    tyKC += mSupportYDimensions[disk][ipart] / 2.;
  }

  rotation = new TGeoRotation("rotation", 180., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., deltaz / 2 + Geometry::sKaptonOnCarbonThickness + mCarbonThickness + Geometry::sKaptonGlueThickness * 2, rotation);
  mHalfDisk->AddNode(kaptonOnCarbon, 3, transformation);
  transformation = new TGeoCombiTrans(0., 0., -(deltaz / 2 + Geometry::sKaptonOnCarbonThickness + mCarbonThickness + Geometry::sKaptonGlueThickness * 2), rotation);
  mHalfDisk->AddNode(kaptonOnCarbon, 4, transformation);

  // **************************************** Kapton glue on the carbon plate ****************************************

  TGeoMedium* mGlueKaptonCarbon = gGeoManager->GetMedium("MFT_Epoxy$");

  auto* glueKaptonCarbon = new TGeoVolumeAssembly(Form("glueKaptonCarbon_D0_H%d", half));

  auto* glueKaptonCarbonBase0 = new TGeoBBox(Form("glueKaptonCarbonBase0_D0_H%d", half), (mSupportXDimensions[disk][0]) / 2.,
                                             (mSupportYDimensions[disk][0]) / 2., Geometry::sKaptonGlueThickness);

  auto* translation_gluKC01 = new TGeoTranslation("translation_gluKC01", 0., (mSupportYDimensions[disk][0]) / 2. + mHalfDiskGap, 0.);
  translation_gluKC01->RegisterYourself();
  auto* translation_gluKC02 = new TGeoTranslation("translation_gluKC02", 0., -mHalfDiskGap, 0.);
  translation_gluKC02->RegisterYourself();

  auto* holeglueKaptonCarbon0 =
    new TGeoTubeSeg(Form("holeglueKaptonCarbon0_D0_H%d", half), 0., mRMin[disk], Geometry::sKaptonGlueThickness + 0.000001, 0, 180.);

  auto* glueKaptonCarbonhole0 = new TGeoSubtraction(glueKaptonCarbonBase0, holeglueKaptonCarbon0, translation_gluKC01, translation_gluKC02);
  auto* gKC0 = new TGeoCompositeShape(Form("glueKaptonCarbon0_D0_H%d", half), glueKaptonCarbonhole0);
  auto* glueKaptonCarbonBaseWithHole0 = new TGeoVolume(Form("glueKaptonCarbonWithHole_D0_H%d", half), gKC0, mGlueKaptonCarbon);

  glueKaptonCarbonBaseWithHole0->SetLineColor(kGreen);
  rotation = new TGeoRotation("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  glueKaptonCarbon->AddNode(glueKaptonCarbonBaseWithHole0, 0, new TGeoTranslation(0., 0., mZPlan[disk]));

  Double_t tyGKC = mSupportYDimensions[disk][0];

  for (Int_t ipart = 1; ipart < mNPart[disk]; ipart++) {
    tyGKC += mSupportYDimensions[disk][ipart] / 2.;
    TGeoVolume* partGlueKaptonCarbon =
      gGeoManager->MakeBox(Form("partGlueKaptonCarbon_D0_H%d_%d", half, ipart), mGlueKaptonCarbon, mSupportXDimensions[disk][ipart] / 2.,
                           mSupportYDimensions[disk][ipart] / 2., Geometry::sKaptonGlueThickness);
    partGlueKaptonCarbon->SetLineColor(kGreen);
    auto* t = new TGeoTranslation("t", 0, tyGKC + mHalfDiskGap, mZPlan[disk]);
    glueKaptonCarbon->AddNode(partGlueKaptonCarbon, ipart, t);
    tyGKC += mSupportYDimensions[disk][ipart] / 2.;
  }

  rotation = new TGeoRotation("rotation", 180., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., deltaz / 2. + mCarbonThickness + Geometry::sKaptonGlueThickness, rotation);
  mHalfDisk->AddNode(glueKaptonCarbon, 3, transformation);
  transformation = new TGeoCombiTrans(0., 0., -(deltaz / 2. + mCarbonThickness + Geometry::sKaptonGlueThickness), rotation);
  mHalfDisk->AddNode(glueKaptonCarbon, 4, transformation);

  // **************************************** Rohacell Plate ****************************************

  auto* rohacellPlate = new TGeoVolumeAssembly(Form("rohacellPlate_D3_H%d", half));

  auto* rohacellBase3 = new TGeoBBox(Form("rohacellBase3_D3_H%d", half), (mSupportXDimensions[disk][0]) / 2., (mSupportYDimensions[disk][0]) / 2., mRohacellThickness);
  // TGeoTranslation *t3 = new TGeoTranslation ("t3",0., (fSupportYDimensions[disk][0])/2. + fHalfDiskGap , 0.);
  // t3 -> RegisterYourself();

  auto* holeRohacell3 =
    new TGeoTubeSeg(Form("holeRohacell3_D3_H%d", half), 0., mRMin[disk], mRohacellThickness + 0.000001, 0, 180.);
  // TGeoTranslation *t4= new TGeoTranslation ("t4", 0., - fHalfDiskGap , 0.);
  // t4-> RegisterYourself();

  /// cs3 = new TGeoCompositeShape(Form("rohacell_D3_H%d",half),
  /// Form("(rohacellBase3_D3_H%d:t31)-(holeRohacell3_D3_H%d:t32)",half,half));

  //===========================================================================================================
  //===========================================================================================================

  // **************************************** GROOVES *************************************************
  Double_t diameter = 0.21; // groove diameter
  Double_t epsilon = 0.06;  // outside shift of the goove
  Int_t iCount = 0;
  Double_t mPosition[4];

  TGeoCombiTrans* transfo[7][4];
  TGeoTube* grooveTube[7][4];
  TGeoTorus* grooveTorus[7][4];
  TGeoSubtraction* rohacellBaseGroove[300];
  TGeoCompositeShape* rohacellGroove[300];

  for (Int_t igroove = 0; igroove < 4; igroove++) {
    grooveTube[0][igroove] = new TGeoTube("linear", 0., diameter, mLWater3[igroove] / 2.);
    grooveTorus[1][igroove] = new TGeoTorus("SideTorus", mRadius3[igroove], 0., diameter, 0., mAngle3[igroove]);
    grooveTube[2][igroove] = new TGeoTube("tiltedLinear", 0., diameter, mLpartial3[igroove] / 2.);
    grooveTorus[3][igroove] = new TGeoTorus("centralTorus", mRadiusCentralTore[igroove], 0., diameter, -mAngle3[igroove], 2. * mAngle3[igroove]);
    grooveTube[4][igroove] = new TGeoTube("tiltedLinear", 0., diameter, mLpartial3[igroove] / 2.);
    grooveTorus[5][igroove] = new TGeoTorus("SideTorus", mRadius3[igroove], 0., diameter, 0., mAngle3[igroove]);
    grooveTube[6][igroove] = new TGeoTube("linear", 0., diameter, mLWater3[igroove] / 2.);
  }

  // Rotation matrix
  TGeoRotation* rotationLinear = new TGeoRotation("rotation", -90., 90., 0.);
  TGeoRotation* rotationSideTorusL = new TGeoRotation("rotationSideTorusLeft", -90., 0., 0.);
  TGeoRotation* rotationSideTorusR = new TGeoRotation("rotationSideTorusRight", 90., 180., 180.);
  TGeoRotation* rotationCentralTorus = new TGeoRotation("rotationCentralTorus", 90., 0., 0.);
  TGeoRotation* rotationTiltedLinearR;
  TGeoRotation* rotationTiltedLinearL;

  // Creating grooves
  for (Int_t iface = 1; iface > -2; iface -= 2) {     // front and rear
    for (Int_t igroove = 0; igroove < 4; igroove++) { // 4 grooves
      mPosition[igroove] = mXPosition3[igroove] - mSupportYDimensions[disk][0] / 2. - mHalfDiskGap;
      for (Int_t ip = 0; ip < 7; ip++) { // each groove is made of 7 parts

        switch (ip) {
          case 0: // Linear
            transfo[ip][igroove] = new TGeoCombiTrans(mSupportXDimensions[3][0] / 2. + mMoreLength - mLWater3[igroove] / 2., mPosition[igroove], iface * (mRohacellThickness + epsilon), rotationLinear);
            if (igroove == 0 && iface == 1) {
              rohacellBaseGroove[0] = new TGeoSubtraction(rohacellBase3, grooveTube[ip][igroove], nullptr, transfo[ip][igroove]);
              rohacellGroove[0] = new TGeoCompositeShape(Form("rohacell3Groove%d_G%d_F%d_H%d", ip, igroove, iface, half), rohacellBaseGroove[0]);
            };
            break;
          case 1: // side torus
            transfo[ip][igroove] = new TGeoCombiTrans(mSupportXDimensions[3][0] / 2. + mMoreLength - mLWater3[igroove], mRadius3[igroove] + mXPosition3[igroove] - mSupportYDimensions[disk][0] / 2. - mHalfDiskGap, iface * (mRohacellThickness + epsilon), rotationSideTorusR);
            break;
          case 2: // Linear tilted
            rotationTiltedLinearR = new TGeoRotation("rotationTiltedLinearRight", 90. - mAngle3[igroove], 90., 0.);
            transfo[ip][igroove] = new TGeoCombiTrans(mSupportXDimensions[3][0] / 2. + mMoreLength - xPos3[igroove], yPos3[igroove] - mSupportYDimensions[disk][0] / 2. - mHalfDiskGap,
                                                      iface * (mRohacellThickness + epsilon), rotationTiltedLinearR);
            break;
          case 3: // Central Torus
            transfo[ip][igroove] = new TGeoCombiTrans(0., yPos3[igroove] + mLpartial3[igroove] / 2 * TMath::Sin(mAngle3[igroove] * TMath::DegToRad()) - mRadiusCentralTore[igroove] * TMath::Cos(mAngle3[igroove] * TMath::DegToRad()) - mSupportYDimensions[disk][0] / 2. - mHalfDiskGap,
                                                      iface * (mRohacellThickness + epsilon), rotationCentralTorus);
            break;
          case 4: // Linear tilted
            rotationTiltedLinearL = new TGeoRotation("rotationTiltedLinearLeft", 90. + mAngle3[igroove], 90., 0.);
            transfo[ip][igroove] = new TGeoCombiTrans(-(mSupportXDimensions[3][0] / 2. + mMoreLength - xPos3[igroove]), yPos3[igroove] - mSupportYDimensions[disk][0] / 2. - mHalfDiskGap,
                                                      iface * (mRohacellThickness + epsilon), rotationTiltedLinearL);
            break;
          case 5: // side torus
            transfo[ip][igroove] = new TGeoCombiTrans(-(mSupportXDimensions[3][0] / 2. + mMoreLength - mLWater3[igroove]), mRadius3[igroove] + mPosition[igroove],
                                                      iface * (mRohacellThickness + epsilon), rotationSideTorusL);
            break;
          case 6: // Linear
            transfo[ip][igroove] = new TGeoCombiTrans(-(mSupportXDimensions[3][0] / 2. + mMoreLength - mLWater3[igroove] / 2.), mPosition[igroove],
                                                      iface * (mRohacellThickness + epsilon), rotationLinear);
            break;
        }

        if (!(ip == 0 && igroove == 0 && iface == 1)) {
          if (ip & 1) {
            rohacellBaseGroove[iCount] = new TGeoSubtraction(rohacellGroove[iCount - 1], grooveTorus[ip][igroove], nullptr, transfo[ip][igroove]);
          } else {
            rohacellBaseGroove[iCount] = new TGeoSubtraction(rohacellGroove[iCount - 1], grooveTube[ip][igroove], nullptr, transfo[ip][igroove]);
          }
          rohacellGroove[iCount] = new TGeoCompositeShape(Form("rohacell3Groove%d_G%d_F%d_H%d", iCount, igroove, iface, half), rohacellBaseGroove[iCount]);
        }
        iCount++;
      }
    }
  }
  // **********************************************************************************************************

  auto* rohacellhole3 = new TGeoSubtraction(rohacellGroove[iCount - 1], holeRohacell3, t31, t32);
  //auto* rohacellhole3 = new TGeoSubtraction(rohacellBase3, holeRohacell3, t31, t32);
  auto* rh3 = new TGeoCompositeShape(Form("rohacellBase3_D3_H%d", half), rohacellhole3);
  auto* rohacellBaseWithHole = new TGeoVolume(Form("rohacellBaseWithHole_D3_H%d", half), rh3, mRohacell);

  rohacellBaseWithHole->SetLineColor(kGray);
  rotation = new TGeoRotation("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  rohacellPlate->AddNode(rohacellBaseWithHole, 0, new TGeoTranslation(0., 0., mZPlan[disk]));

  ty = mSupportYDimensions[disk][0];

  TGeoVolume* partRohacell;
  for (Int_t ipart = 1; ipart < mNPart[disk]; ipart++) {
    //for (Int_t ipart = 1; ipart < 4; ipart++) {
    ty += mSupportYDimensions[disk][ipart] / 2.;
    /*
    TGeoVolume* partRohacell = gGeoManager->MakeBox(Form("partRohacelli_D3_H%d_%d", half, ipart), mRohacell,
                                                    mSupportXDimensions[disk][ipart] / 2.,
                                                    mSupportYDimensions[disk][ipart] / 2., mRohacellThickness);
    partRohacell->SetLineColor(kGray);
    */
    auto* t = new TGeoTranslation("t", 0, ty + mHalfDiskGap, mZPlan[disk]);

    //===========================================================================================================
    //===========================================================================================================
    auto* partRohacell0 =
      new TGeoBBox(Form("rohacellBase0_D3_H%d_%d", half, ipart), mSupportXDimensions[disk][ipart] / 2.,
                   mSupportYDimensions[disk][ipart] / 2., mRohacellThickness);
    Double_t mShift;

    // ****************  Creating grooves for the other parts of the rohacell plate **********************
    for (Int_t iface = 1; iface > -2; iface -= 2) {     // front and rear
      for (Int_t igroove = 0; igroove < 4; igroove++) { // 4 grooves
        if (ipart == 1) {
          mPosition[ipart] = mXPosition3[igroove] - mSupportYDimensions[disk][ipart] / 2. - mHalfDiskGap - mSupportYDimensions[disk][ipart - 1];
          mShift = -mSupportYDimensions[disk][ipart - 1];
        };
        if (ipart == 2) {
          mPosition[ipart] = mXPosition3[igroove] - mSupportYDimensions[disk][ipart] / 2. - mHalfDiskGap - mSupportYDimensions[disk][ipart - 1] - mSupportYDimensions[disk][ipart - 2];
          mShift = -mSupportYDimensions[disk][ipart - 1] - mSupportYDimensions[disk][ipart - 2];
        };
        if (ipart == 3) {
          mPosition[ipart] = mXPosition3[igroove] - mSupportYDimensions[disk][ipart] / 2. - mHalfDiskGap - mSupportYDimensions[disk][ipart - 1] - mSupportYDimensions[disk][ipart - 2] - mSupportYDimensions[disk][ipart - 3];
          mShift = -mSupportYDimensions[disk][ipart - 1] - mSupportYDimensions[disk][ipart - 2] - mSupportYDimensions[disk][ipart - 3];
        };

        for (Int_t ip = 0; ip < 7; ip++) { // each groove is made of 7 parts

          switch (ip) {
            case 0: // Linear
              transfo[ip][igroove] = new TGeoCombiTrans(mSupportXDimensions[3][0] / 2. + mMoreLength - mLWater3[igroove] / 2., mPosition[ipart],
                                                        iface * (mRohacellThickness + epsilon), rotationLinear);
              if (igroove == 0 && iface == 1) {
                rohacellBaseGroove[iCount] = new TGeoSubtraction(partRohacell0, grooveTube[ip][igroove], nullptr, transfo[ip][igroove]);
                rohacellGroove[iCount] = new TGeoCompositeShape(Form("rohacell3Groove%d_G%d_F%d_H%d", ip, igroove, iface, half), rohacellBaseGroove[iCount]);
              };
              break;
            case 1: // side torus
              transfo[ip][igroove] = new TGeoCombiTrans(mSupportXDimensions[3][0] / 2. + mMoreLength - mLWater3[igroove], mPosition[ipart] + mRadius3[igroove],
                                                        iface * (mRohacellThickness + epsilon), rotationSideTorusR);
              break;
            case 2: // Linear tilted
              rotationTiltedLinearR = new TGeoRotation("rotationTiltedLinearRight", 90. - mAngle3[igroove], 90., 0.);
              transfo[ip][igroove] = new TGeoCombiTrans(mSupportXDimensions[3][0] / 2. + mMoreLength - xPos3[igroove], yPos3[igroove] + mShift - mHalfDiskGap - mSupportYDimensions[disk][ipart] / 2., iface * (mRohacellThickness + epsilon), rotationTiltedLinearR);
              break;
            case 3: // Central Torus
              transfo[ip][igroove] = new TGeoCombiTrans(0., mPosition[ipart] + yPos3[igroove] + mLpartial3[igroove] / 2 * TMath::Sin(mAngle3[igroove] * TMath::DegToRad()) - mRadiusCentralTore[igroove] * TMath::Cos(mAngle3[igroove] * TMath::DegToRad()) - mXPosition3[igroove],
                                                        iface * (mRohacellThickness + epsilon), rotationCentralTorus);
              break;
            case 4: // Linear tilted
              rotationTiltedLinearL = new TGeoRotation("rotationTiltedLinearLeft", 90. + mAngle3[igroove], 90., 0.);
              transfo[ip][igroove] = new TGeoCombiTrans(-(mSupportXDimensions[3][0] / 2. + mMoreLength - xPos3[igroove]), yPos3[igroove] + mPosition[ipart] - mXPosition3[igroove],
                                                        iface * (mRohacellThickness + epsilon), rotationTiltedLinearL);
              break;
            case 5: // side torus
              transfo[ip][igroove] = new TGeoCombiTrans(-(mSupportXDimensions[3][0] / 2. + mMoreLength - mLWater3[igroove]), mRadius3[igroove] + mPosition[ipart],
                                                        iface * (mRohacellThickness + epsilon), rotationSideTorusL);
              break;
            case 6: // Linear
              transfo[ip][igroove] = new TGeoCombiTrans(-(mSupportXDimensions[3][0] / 2. + mMoreLength - mLWater3[igroove] / 2.), mPosition[ipart],
                                                        iface * (mRohacellThickness + epsilon), rotationLinear);
              break;
          }
          if (!(ip == 0 && igroove == 0 && iface == 1)) {
            if (ip & 1) {
              rohacellBaseGroove[iCount] =
                new TGeoSubtraction(rohacellGroove[iCount - 1], grooveTorus[ip][igroove], nullptr, transfo[ip][igroove]);
            } else {
              rohacellBaseGroove[iCount] =
                new TGeoSubtraction(rohacellGroove[iCount - 1], grooveTube[ip][igroove], nullptr, transfo[ip][igroove]);
            }

            rohacellGroove[iCount] = new TGeoCompositeShape(Form("rohacell3Groove%d_G%d_F%d_H%d", iCount, igroove, iface, half), rohacellBaseGroove[iCount]);
          }
          iCount++;
        }
      }
    }

    partRohacell = new TGeoVolume(Form("partRohacelli_D3_H%d_%d", half, ipart), rohacellGroove[iCount - 1], mRohacell);
    //partRohacell = new TGeoVolume(Form("partRohacelli_D3_H%d_%d", half, ipart), partRohacell0, mRohacell);
    //===========================================================================================================
    //===========================================================================================================

    rohacellPlate->AddNode(partRohacell, ipart, t);
    ty += mSupportYDimensions[disk][ipart] / 2.;
  }

  //  if (half == Top) {
  //    rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  //    transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  //    mHalfDisk->AddNode(rohacellPlate, 1, transformation);
  //  }
  //  if (half == Bottom) {
  rotation = new TGeoRotation("rotation", 180., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  mHalfDisk->AddNode(rohacellPlate, 2, transformation);
  //  }

  createManifold(disk);
}

//_____________________________________________________________________________
void HeatExchanger::createHalfDisk4(Int_t half)
{

  Int_t disk = 4;

  if (half == Top)
    printf("Creating MFT heat exchanger for disk4 top\n");
  else if (half == Bottom)
    printf("Creating MFT heat exchanger for disk4 bottom\n");
  else
    printf("No valid option for MFT heat exchanger on disk4\n");

  // mCarbon   = gGeoManager->GetMedium("MFT_Carbon$");
  mCarbon = gGeoManager->GetMedium("MFT_CarbonFiber$");

  mWater = gGeoManager->GetMedium("MFT_Water$");
  mRohacell = gGeoManager->GetMedium("MFT_Rohacell");
  mPipe = gGeoManager->GetMedium("MFT_Polyimide");

  auto* cooling = new TGeoVolumeAssembly(Form("cooling_D4_H%d", half));
  //Double_t deltaz = mHeatExchangerThickness - mCarbonThickness * 2; // distance between pair of carbon plans
  Double_t deltaz = mHeatExchangerThickness - Geometry::sKaptonOnCarbonThickness * 4 - Geometry::sKaptonGlueThickness * 4 - 2 * mCarbonThickness;

  TGeoTranslation* translation = nullptr;
  TGeoRotation* rotation = nullptr;
  TGeoCombiTrans* transformation = nullptr;

  // **************************************** Water part ******************************************
  // ********************* Four parameters mLwater4, mRadius4, mAngle4, mLpartial4 ****************
  Double_t ivolume = 400; // offset chamber 4
  Double_t mRadiusCentralTore[4];
  Double_t xPos4[4];
  Double_t yPos4[4];
  for (Int_t itube = 0; itube < 4; itube++) {
    TGeoVolume* waterTube1 = gGeoManager->MakeTube(Form("waterTube1%d_D4_H%d", itube, half), mWater, 0., mRWater, mLWater4[itube] / 2.);
    translation = new TGeoTranslation(mXPosition4[itube], 0., mSupportXDimensions[4][0] / 2. + mMoreLength - mLWater4[itube] / 2.);
    cooling->AddNode(waterTube1, ivolume++, translation);

    TGeoVolume* waterTorus1 = gGeoManager->MakeTorus(Form("waterTorus1%d_D4_H%d", itube, half), mWater, mRadius4[itube], 0., mRWater, 0., mAngle4[itube]);
    rotation = new TGeoRotation("rotation", 180., -90., 0.);
    transformation = new TGeoCombiTrans(mRadius4[itube] + mXPosition4[itube], 0., mSupportXDimensions[4][0] / 2. + mMoreLength - mLWater4[itube], rotation);
    //cooling->AddNode(waterTorus1, ivolume++, transformation);

    TGeoVolume* waterTube2 = gGeoManager->MakeTube(Form("waterTube2%d_D4_H%d", itube, half), mWater, 0., mRWater, mLpartial4[itube] / 2.);
    rotation = new TGeoRotation("rotation", 90., 180 - mAngle4[itube], 0.);
    xPos4[itube] = mLWater4[itube] + mRadius4[itube] * TMath::Sin(mAngle4[itube] * TMath::DegToRad()) + mLpartial4[itube] / 2 * TMath::Cos(mAngle4[itube] * TMath::DegToRad());
    yPos4[itube] = mXPosition4[itube] + mRadius4[itube] * (1 - TMath::Cos(mAngle4[itube] * TMath::DegToRad())) + mLpartial4[itube] / 2 * TMath::Sin(mAngle4[itube] * TMath::DegToRad());
    transformation = new TGeoCombiTrans(yPos4[itube], 0., mSupportXDimensions[4][0] / 2. + mMoreLength - xPos4[itube], rotation);
    cooling->AddNode(waterTube2, ivolume++, transformation);

    mRadiusCentralTore[itube] = (mSupportXDimensions[4][0] / 2. + mMoreLength - xPos4[itube] - mLpartial4[itube] / 2 * TMath::Cos(mAngle4[itube] * TMath::DegToRad())) / TMath::Sin(mAngle4[itube] * TMath::DegToRad());
    TGeoVolume* waterTorusCentral = gGeoManager->MakeTorus(Form("waterTorusCentral%d_D4_H%d", itube, half), mWater, mRadiusCentralTore[itube], 0.,
                                                           mRWater, -mAngle4[itube], 2. * mAngle4[itube]);
    rotation = new TGeoRotation("rotation", 0., 90., 0.);
    transformation = new TGeoCombiTrans(yPos4[itube] + mLpartial4[itube] / 2 * TMath::Sin(mAngle4[itube] * TMath::DegToRad()) -
                                          mRadiusCentralTore[itube] * TMath::Cos(mAngle4[itube] * TMath::DegToRad()),
                                        0., 0., rotation);
    cooling->AddNode(waterTorusCentral, ivolume++, transformation);

    TGeoVolume* waterTube3 = gGeoManager->MakeTube(Form("waterTube3%d_D4_H%d", 2, half), mWater, 0., mRWater, mLpartial4[itube] / 2.);
    rotation = new TGeoRotation("rotation", -90., 0 - mAngle4[itube], 0.);
    transformation = new TGeoCombiTrans(yPos4[itube], 0., -(mSupportXDimensions[4][0] / 2. + mMoreLength - xPos4[itube]), rotation);
    cooling->AddNode(waterTube3, ivolume++, transformation);

    TGeoVolume* waterTorus2 = gGeoManager->MakeTorus(Form("waterTorus2%d_D4_H%d", itube, half), mWater, mRadius4[itube], 0., mRWater, 0., mAngle4[itube]);
    rotation = new TGeoRotation("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(mRadius4[itube] + mXPosition4[itube], 0., -(mSupportXDimensions[4][0] / 2. + mMoreLength - mLWater4[itube]), rotation);
    cooling->AddNode(waterTorus2, ivolume++, transformation);

    TGeoVolume* waterTube4 = gGeoManager->MakeTube(Form("waterTube4%d_D4_H%d", itube, half), mWater, 0., mRWater, mLWater4[itube] / 2.);
    translation = new TGeoTranslation(mXPosition4[itube], 0., -(mSupportXDimensions[4][0] / 2. + mMoreLength - mLWater4[itube] / 2.));
    cooling->AddNode(waterTube4, ivolume++, translation);
  }

  // **************************************** Tube part *******************************************
  // ********************* Four parameters mLwater4, mRadius4, mAngle4, mLpartial4 ****************
  for (Int_t itube = 0; itube < 4; itube++) {

    TGeoVolume* pipeTube1 = gGeoManager->MakeTube(Form("pipeTube1%d_D4_H%d", itube, half), mPipe, mRWater, mRWater + mDRPipe, mLWater4[itube] / 2.);
    translation = new TGeoTranslation(mXPosition4[itube], 0., mSupportXDimensions[4][0] / 2. + mMoreLength - mLWater4[itube] / 2.);
    cooling->AddNode(pipeTube1, ivolume++, translation);

    TGeoVolume* pipeTorus1 = gGeoManager->MakeTorus(Form("pipeTorus1%d_D4_H%d", itube, half), mPipe, mRadius4[itube], mRWater, mRWater + mDRPipe, 0., mAngle4[itube]);
    rotation = new TGeoRotation("rotation", 180., -90., 0.);
    transformation = new TGeoCombiTrans(mRadius4[itube] + mXPosition4[itube], 0., mSupportXDimensions[4][0] / 2. + mMoreLength - mLWater4[itube], rotation);
    cooling->AddNode(pipeTorus1, ivolume++, transformation);

    TGeoVolume* pipeTube2 = gGeoManager->MakeTube(Form("pipeTube2%d_D4_H%d", itube, half), mPipe, mRWater, mRWater + mDRPipe, mLpartial4[itube] / 2.);
    rotation = new TGeoRotation("rotation", 90., 180 - mAngle4[itube], 0.);
    xPos4[itube] = mLWater4[itube] + mRadius4[itube] * TMath::Sin(mAngle4[itube] * TMath::DegToRad()) + mLpartial4[itube] / 2 * TMath::Cos(mAngle4[itube] * TMath::DegToRad());
    yPos4[itube] = mXPosition4[itube] + mRadius4[itube] * (1 - TMath::Cos(mAngle4[itube] * TMath::DegToRad())) + mLpartial4[itube] / 2 * TMath::Sin(mAngle4[itube] * TMath::DegToRad());
    transformation = new TGeoCombiTrans(yPos4[itube], 0., mSupportXDimensions[4][0] / 2. + mMoreLength - xPos4[itube], rotation);
    cooling->AddNode(pipeTube2, ivolume++, transformation);

    mRadiusCentralTore[itube] = (mSupportXDimensions[4][0] / 2. + mMoreLength - xPos4[itube] - mLpartial4[itube] / 2 * TMath::Cos(mAngle4[itube] * TMath::DegToRad())) / TMath::Sin(mAngle4[itube] * TMath::DegToRad());
    TGeoVolume* pipeTorusCentral = gGeoManager->MakeTorus(Form("pipeTorusCentral%d_D4_H%d", itube, half), mPipe, mRadiusCentralTore[itube], mRWater, mRWater + mDRPipe,
                                                          -mAngle4[itube], 2. * mAngle4[itube]);
    rotation = new TGeoRotation("rotation", 0., 90., 0.);
    transformation = new TGeoCombiTrans(yPos4[itube] + mLpartial4[itube] / 2 * TMath::Sin(mAngle4[itube] * TMath::DegToRad()) - mRadiusCentralTore[itube] * TMath::Cos(mAngle4[itube] * TMath::DegToRad()), 0., 0., rotation);
    cooling->AddNode(pipeTorusCentral, ivolume++, transformation);

    TGeoVolume* pipeTube3 = gGeoManager->MakeTube(Form("pipeTube3%d_D4_H%d", 2, half), mPipe, mRWater, mRWater + mDRPipe, mLpartial4[itube] / 2.);
    rotation = new TGeoRotation("rotation", -90., 0 - mAngle4[itube], 0.);
    transformation = new TGeoCombiTrans(yPos4[itube], 0., -(mSupportXDimensions[4][0] / 2. + mMoreLength - xPos4[itube]), rotation);
    cooling->AddNode(pipeTube3, ivolume++, transformation);

    TGeoVolume* pipeTorus2 = gGeoManager->MakeTorus(Form("pipeTorus2%d_D4_H%d", itube, half), mPipe, mRadius4[itube], mRWater, mRWater + mDRPipe, 0., mAngle4[itube]);
    rotation = new TGeoRotation("rotation", 180., 90., 0.);
    transformation = new TGeoCombiTrans(mRadius4[itube] + mXPosition4[itube], 0., -(mSupportXDimensions[4][0] / 2. + mMoreLength - mLWater4[itube]), rotation);
    cooling->AddNode(pipeTorus2, ivolume++, transformation);

    TGeoVolume* pipeTube4 = gGeoManager->MakeTube(Form("pipeTube4%d_D4_H%d", itube, half), mPipe, mRWater, mRWater + mDRPipe, mLWater4[itube] / 2.);
    translation = new TGeoTranslation(mXPosition4[itube], 0., -(mSupportXDimensions[4][0] / 2. + mMoreLength - mLWater4[itube] / 2.));
    cooling->AddNode(pipeTube4, ivolume++, translation);
  }
  //------------------------------------------------------------------
  //------------------------------------------------------------------

  rotation = new TGeoRotation("rotation", -90., 90., 0.);
  transformation =
    //new TGeoCombiTrans(0., 0., mZPlan[disk] + deltaz / 2. - mCarbonThickness - mRWater - mDRPipe, rotation);
    new TGeoCombiTrans(0., 0., mZPlan[disk] + deltaz / 2. - mCarbonThickness - mRWater - mDRPipe - 2 * Geometry::sGlueRohacellCarbonThickness, rotation);
  mHalfDisk->AddNode(cooling, 3, transformation);
  transformation =
    //new TGeoCombiTrans(0., 0., mZPlan[disk] - deltaz / 2. + mCarbonThickness + mRWater + mDRPipe, rotation);
    new TGeoCombiTrans(0., 0., mZPlan[disk] - deltaz / 2. + mCarbonThickness + mRWater + mDRPipe + 2 * Geometry::sGlueRohacellCarbonThickness, rotation);
  mHalfDisk->AddNode(cooling, 4, transformation);

  // **************************************** Carbon Plates ****************************************

  auto* carbonPlate = new TGeoVolumeAssembly(Form("carbonPlate_D4_H%d", half));

  auto* carbonBase4 = new TGeoBBox(Form("carbonBase4_D4_H%d", half), (mSupportXDimensions[disk][0]) / 2. + mMoreLength,
                                   (mSupportYDimensions[disk][0]) / 2., mCarbonThickness);
  auto* t41 = new TGeoTranslation("t41", 0., (mSupportYDimensions[disk][0]) / 2. + mHalfDiskGap, 0.);
  t41->RegisterYourself();

  auto* holeCarbon4 =
    new TGeoTubeSeg(Form("holeCarbon4_D4_H%d", half), 0., mRMin[disk], mCarbonThickness + 0.000001, 0, 180.);
  auto* t42 = new TGeoTranslation("t42", 0., -mHalfDiskGap, 0.);
  t42->RegisterYourself();

  /// TGeoCompositeShape *cs4 = new
  /// TGeoCompositeShape(Form("Carbon4_D4_H%d",half),Form("(carbonBase4_D4_H%d:t41)-(holeCarbon4_D4_H%d:t42)",half,half));
  auto* carbonhole4 = new TGeoSubtraction(carbonBase4, holeCarbon4, t41, t42);
  auto* cs4 = new TGeoCompositeShape(Form("Carbon4_D4_H%d", half), carbonhole4);
  auto* carbonBaseWithHole4 = new TGeoVolume(Form("carbonBaseWithHole_D4_H%d", half), cs4, mCarbon);

  carbonBaseWithHole4->SetLineColor(kGray + 3);
  rotation = new TGeoRotation("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  carbonPlate->AddNode(carbonBaseWithHole4, 0, new TGeoTranslation(0., 0., mZPlan[disk]));

  Double_t ty = mSupportYDimensions[disk][0];

  for (Int_t ipart = 1; ipart < mNPart[disk]; ipart++) {
    ty += mSupportYDimensions[disk][ipart] / 2.;
    TGeoVolume* partCarbon =
      gGeoManager->MakeBox(Form("partCarbon_D4_H%d_%d", half, ipart), mCarbon, mSupportXDimensions[disk][ipart] / 2.,
                           mSupportYDimensions[disk][ipart] / 2., mCarbonThickness);
    partCarbon->SetLineColor(kGray + 3);
    auto* t = new TGeoTranslation("t", 0, ty + mHalfDiskGap, mZPlan[disk]);
    carbonPlate->AddNode(partCarbon, ipart, t);
    ty += mSupportYDimensions[disk][ipart] / 2.;
  }

  //  if (half == Top) {
  //    rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  //    transformation = new TGeoCombiTrans(0., 0., deltaz/2., rotation);
  //    mHalfDisk->AddNode(carbonPlate, 1, transformation);
  //    transformation = new TGeoCombiTrans(0., 0., -deltaz/2., rotation);
  //    mHalfDisk->AddNode(carbonPlate, 2, transformation);
  //  }
  //  else if (half == Bottom) {
  rotation = new TGeoRotation("rotation", 180., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., deltaz / 2., rotation);
  mHalfDisk->AddNode(carbonPlate, 3, transformation);
  transformation = new TGeoCombiTrans(0., 0., -deltaz / 2., rotation);
  mHalfDisk->AddNode(carbonPlate, 4, transformation);
  //  }

  // **************************************** Glue Bwtween Carbon Plate and Rohacell Plate ****************************************

  TGeoMedium* mGlueRohacellCarbon = gGeoManager->GetMedium("MFT_Epoxy$");

  auto* glueRohacellCarbon = new TGeoVolumeAssembly(Form("glueRohacellCarbon_D0_H%d", half));

  auto* glueRohacellCarbonBase0 = new TGeoBBox(Form("glueRohacellCarbonBase0_D0_H%d", half), (mSupportXDimensions[disk][0]) / 2.,
                                               (mSupportYDimensions[disk][0]) / 2., Geometry::sGlueRohacellCarbonThickness);

  auto* translation_gluRC01 = new TGeoTranslation("translation_gluRC01", 0., (mSupportYDimensions[disk][0]) / 2. + mHalfDiskGap, 0.);
  translation_gluRC01->RegisterYourself();
  auto* translation_gluRC02 = new TGeoTranslation("translation_gluRC02", 0., -mHalfDiskGap, 0.);
  translation_gluRC02->RegisterYourself();

  auto* holeglueRohacellCarbon0 =
    new TGeoTubeSeg(Form("holeglueRohacellCarbon0_D0_H%d", half), 0., mRMin[disk], Geometry::sGlueRohacellCarbonThickness + 0.000001, 0, 180.);

  auto* glueRohacellCarbonhole0 = new TGeoSubtraction(glueRohacellCarbonBase0, holeglueRohacellCarbon0, translation_gluRC01, translation_gluRC02);
  auto* gRC0 = new TGeoCompositeShape(Form("glueRohacellCarbon0_D0_H%d", half), glueRohacellCarbonhole0);
  auto* glueRohacellCarbonBaseWithHole0 = new TGeoVolume(Form("glueRohacellCarbonWithHole_D0_H%d", half), gRC0, mGlueRohacellCarbon);

  glueRohacellCarbonBaseWithHole0->SetLineColor(kGreen);
  rotation = new TGeoRotation("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  glueRohacellCarbon->AddNode(glueRohacellCarbonBaseWithHole0, 0, new TGeoTranslation(0., 0., mZPlan[disk]));

  Double_t tyGRC = mSupportYDimensions[disk][0];

  for (Int_t ipart = 1; ipart < mNPart[disk]; ipart++) {
    tyGRC += mSupportYDimensions[disk][ipart] / 2.;
    TGeoVolume* partGlueRohacellCarbon =
      gGeoManager->MakeBox(Form("partGlueRohacellCarbon_D0_H%d_%d", half, ipart), mGlueRohacellCarbon, mSupportXDimensions[disk][ipart] / 2.,
                           mSupportYDimensions[disk][ipart] / 2., Geometry::sGlueRohacellCarbonThickness);
    partGlueRohacellCarbon->SetLineColor(kGreen);
    auto* t = new TGeoTranslation("t", 0, tyGRC + mHalfDiskGap, mZPlan[disk]);
    glueRohacellCarbon->AddNode(partGlueRohacellCarbon, ipart, t);
    tyGRC += mSupportYDimensions[disk][ipart] / 2.;
  }

  rotation = new TGeoRotation("rotation", 180., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., deltaz / 2. - mCarbonThickness - Geometry::sGlueRohacellCarbonThickness, rotation);
  mHalfDisk->AddNode(glueRohacellCarbon, 3, transformation);
  transformation = new TGeoCombiTrans(0., 0., -(deltaz / 2. - mCarbonThickness - Geometry::sGlueRohacellCarbonThickness), rotation);
  mHalfDisk->AddNode(glueRohacellCarbon, 4, transformation);

  // **************************************** Kapton on Carbon Plate ****************************************

  TGeoMedium* mKaptonOnCarbon = gGeoManager->GetMedium("MFT_Kapton$");

  auto* kaptonOnCarbon = new TGeoVolumeAssembly(Form("kaptonOnCarbon_D0_H%d", half));
  auto* kaptonOnCarbonBase0 = new TGeoBBox(Form("kaptonOnCarbonBase0_D0_H%d", half), (mSupportXDimensions[disk][0]) / 2. + mMoreLength,
                                           (mSupportYDimensions[disk][0]) / 2., Geometry::sKaptonOnCarbonThickness);

  auto* translation_KC01 = new TGeoTranslation("translation_KC01", 0., (mSupportYDimensions[disk][0]) / 2. + mHalfDiskGap, 0.);
  translation_KC01->RegisterYourself();
  auto* translation_KC02 = new TGeoTranslation("translation_KC02", 0., -mHalfDiskGap, 0.);
  translation_KC02->RegisterYourself();

  auto* holekaptonOnCarbon0 =
    new TGeoTubeSeg(Form("holekaptonOnCarbon0_D0_H%d", half), 0., mRMin[disk], Geometry::sKaptonOnCarbonThickness + 0.000001, 0, 180.);

  auto* kaptonOnCarbonhole0 = new TGeoSubtraction(kaptonOnCarbonBase0, holekaptonOnCarbon0, translation_KC01, translation_KC02);
  auto* KC0 = new TGeoCompositeShape(Form("kaptonOnCarbon_D0_H%d", half), kaptonOnCarbonhole0);
  auto* kaptonOnCarbonBaseWithHole0 = new TGeoVolume(Form("kaptonOnCarbonWithHole_D0_H%d", half), KC0, mKaptonOnCarbon);

  kaptonOnCarbonBaseWithHole0->SetLineColor(kMagenta);
  rotation = new TGeoRotation("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  kaptonOnCarbon->AddNode(kaptonOnCarbonBaseWithHole0, 0, new TGeoTranslation(0., 0., mZPlan[disk]));

  Double_t tyKC = mSupportYDimensions[disk][0];

  for (Int_t ipart = 1; ipart < mNPart[disk]; ipart++) {
    tyKC += mSupportYDimensions[disk][ipart] / 2.;
    TGeoVolume* partkaptonOnCarbonBase =
      gGeoManager->MakeBox(Form("partkaptonOnCarbon_D0_H%d_%d", half, ipart), mKaptonOnCarbon, mSupportXDimensions[disk][ipart] / 2.,
                           mSupportYDimensions[disk][ipart] / 2., Geometry::sKaptonOnCarbonThickness);
    partkaptonOnCarbonBase->SetLineColor(kMagenta);
    auto* t = new TGeoTranslation("t", 0, tyKC + mHalfDiskGap, mZPlan[disk]);
    kaptonOnCarbon->AddNode(partkaptonOnCarbonBase, ipart, t);
    tyKC += mSupportYDimensions[disk][ipart] / 2.;
  }

  rotation = new TGeoRotation("rotation", 180., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., deltaz / 2 + Geometry::sKaptonOnCarbonThickness + mCarbonThickness + Geometry::sKaptonGlueThickness * 2, rotation);
  mHalfDisk->AddNode(kaptonOnCarbon, 3, transformation);
  transformation = new TGeoCombiTrans(0., 0., -(deltaz / 2 + Geometry::sKaptonOnCarbonThickness + mCarbonThickness + Geometry::sKaptonGlueThickness * 2), rotation);
  mHalfDisk->AddNode(kaptonOnCarbon, 4, transformation);

  // **************************************** Kapton glue on the carbon plate ****************************************

  TGeoMedium* mGlueKaptonCarbon = gGeoManager->GetMedium("MFT_Epoxy$");

  auto* glueKaptonCarbon = new TGeoVolumeAssembly(Form("glueKaptonCarbon_D0_H%d", half));

  auto* glueKaptonCarbonBase0 = new TGeoBBox(Form("glueKaptonCarbonBase0_D0_H%d", half), (mSupportXDimensions[disk][0]) / 2., (mSupportYDimensions[disk][0]) / 2., Geometry::sKaptonGlueThickness);

  auto* translation_gluKC01 = new TGeoTranslation("translation_gluKC01", 0., (mSupportYDimensions[disk][0]) / 2. + mHalfDiskGap, 0.);
  translation_gluKC01->RegisterYourself();
  auto* translation_gluKC02 = new TGeoTranslation("translation_gluKC02", 0., -mHalfDiskGap, 0.);
  translation_gluKC02->RegisterYourself();

  auto* holeglueKaptonCarbon0 =
    new TGeoTubeSeg(Form("holeglueKaptonCarbon0_D0_H%d", half), 0., mRMin[disk], Geometry::sKaptonGlueThickness + 0.000001, 0, 180.);

  auto* glueKaptonCarbonhole0 = new TGeoSubtraction(glueKaptonCarbonBase0, holeglueKaptonCarbon0, translation_gluKC01, translation_gluKC02);
  auto* gKC0 = new TGeoCompositeShape(Form("glueKaptonCarbon0_D0_H%d", half), glueKaptonCarbonhole0);
  auto* glueKaptonCarbonBaseWithHole0 = new TGeoVolume(Form("glueKaptonCarbonWithHole_D0_H%d", half), gKC0, mGlueKaptonCarbon);

  glueKaptonCarbonBaseWithHole0->SetLineColor(kGreen);
  rotation = new TGeoRotation("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  glueKaptonCarbon->AddNode(glueKaptonCarbonBaseWithHole0, 0, new TGeoTranslation(0., 0., mZPlan[disk]));

  Double_t tyGKC = mSupportYDimensions[disk][0];

  for (Int_t ipart = 1; ipart < mNPart[disk]; ipart++) {
    tyGKC += mSupportYDimensions[disk][ipart] / 2.;
    TGeoVolume* partGlueKaptonCarbon =
      gGeoManager->MakeBox(Form("partGlueKaptonCarbon_D0_H%d_%d", half, ipart), mGlueKaptonCarbon, mSupportXDimensions[disk][ipart] / 2.,
                           mSupportYDimensions[disk][ipart] / 2., Geometry::sKaptonGlueThickness);
    partGlueKaptonCarbon->SetLineColor(kGreen);
    auto* t = new TGeoTranslation("t", 0, tyGKC + mHalfDiskGap, mZPlan[disk]);
    glueKaptonCarbon->AddNode(partGlueKaptonCarbon, ipart, t);
    tyGKC += mSupportYDimensions[disk][ipart] / 2.;
  }

  rotation = new TGeoRotation("rotation", 180., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., deltaz / 2. + mCarbonThickness + Geometry::sKaptonGlueThickness, rotation);
  mHalfDisk->AddNode(glueKaptonCarbon, 3, transformation);
  transformation = new TGeoCombiTrans(0., 0., -(deltaz / 2. + mCarbonThickness + Geometry::sKaptonGlueThickness), rotation);
  mHalfDisk->AddNode(glueKaptonCarbon, 4, transformation);

  // **************************************** Rohacell Plate ****************************************

  auto* rohacellPlate = new TGeoVolumeAssembly(Form("rohacellPlate_D4_H%d", half));

  auto* rohacellBase4 = new TGeoBBox(Form("rohacellBase4_D4_H%d", half), (mSupportXDimensions[disk][0]) / 2.,
                                     (mSupportYDimensions[disk][0]) / 2., mRohacellThickness);
  // TGeoTranslation *t3 = new TGeoTranslation ("t3",0., (fSupportYDimensions[disk][0])/2. + fHalfDiskGap , 0.);
  // t3 -> RegisterYourself();

  auto* holeRohacell4 =
    new TGeoTubeSeg(Form("holeRohacell4_D4_H%d", half), 0., mRMin[disk], mRohacellThickness + 0.000001, 0, 180.);
  // TGeoTranslation *t4= new TGeoTranslation ("t4", 0., - fHalfDiskGap , 0.);
  // t4-> RegisterYourself();

  /// cs4 = new TGeoCompositeShape(Form("rohacell_D4_H%d",half),
  /// Form("(rohacellBase4_D4_H%d:t41)-(holeRohacell4_D4_H%d:t42)",half,half));

  // *************************************** Grooves *************************************************
  Double_t diameter = 0.21; // groove diameter
  Double_t epsilon = 0.06;  // outside shift of the goove
  Int_t iCount = 0;
  Double_t mPosition[5];
  TGeoCombiTrans* transfo[7][4];
  TGeoTube* grooveTube[8][4];
  TGeoTorus* grooveTorus[8][4];
  TGeoSubtraction* rohacellBaseGroove[300];
  TGeoCompositeShape* rohacellGroove[300];
  TGeoRotation* rotationTorus5[8];

  for (Int_t igroove = 0; igroove < 4; igroove++) {
    grooveTube[0][igroove] = new TGeoTube("linear", 0., diameter, mLWater4[igroove] / 2.);
    grooveTorus[1][igroove] = new TGeoTorus("SideTorus", mRadius4[igroove], 0., diameter, 0., mAngle4[igroove]);
    grooveTube[2][igroove] = new TGeoTube("tiltedLinear", 0., diameter, mLpartial4[igroove] / 2.);
    grooveTorus[3][igroove] = new TGeoTorus("centralTorus", mRadiusCentralTore[igroove], 0., diameter, -mAngle4[igroove], 2. * mAngle4[igroove]);
    grooveTube[4][igroove] = new TGeoTube("tiltedLinear", 0., diameter, mLpartial4[igroove] / 2.);
    grooveTorus[5][igroove] = new TGeoTorus("SideTorus", mRadius4[igroove], 0., diameter, 0., mAngle4[igroove]);
    grooveTube[6][igroove] = new TGeoTube("linear", 0., diameter, mLWater4[igroove] / 2.);
  }

  // Rotation matrix
  TGeoRotation* rotationLinear = new TGeoRotation("rotation", 90., 90., 0.);
  TGeoRotation* rotationSideTorusL = new TGeoRotation("rotationSideTorusLeft", -90., 0., 0.);
  TGeoRotation* rotationSideTorusR = new TGeoRotation("rotationSideTorusRight", 90., 180., 180.);
  TGeoRotation* rotationCentralTorus = new TGeoRotation("rotationCentralTorus", 90., 0., 0.);
  TGeoRotation* rotationTiltedLinearR;
  TGeoRotation* rotationTiltedLinearL;

  // Creating grooves
  for (Int_t iface = 1; iface > -2; iface -= 2) {     // front and rear
    for (Int_t igroove = 0; igroove < 4; igroove++) { // 4 grooves
      mPosition[igroove] = mXPosition4[igroove] - mSupportYDimensions[disk][0] / 2. - mHalfDiskGap;
      for (Int_t ip = 0; ip < 7; ip++) { // each groove is made of 7 parts

        switch (ip) {
          case 0: // Linear
            transfo[ip][igroove] = new TGeoCombiTrans(mSupportXDimensions[4][0] / 2. + mMoreLength - mLWater4[igroove] / 2., mPosition[igroove],
                                                      iface * (mRohacellThickness + epsilon), rotationLinear);
            if (igroove == 0 && iface == 1) {
              rohacellBaseGroove[0] = new TGeoSubtraction(rohacellBase4, grooveTube[ip][igroove], nullptr, transfo[ip][igroove]);
              rohacellGroove[0] = new TGeoCompositeShape(Form("rohacell4Groove%d_G%d_F%d_H%d", ip, igroove, iface, half), rohacellBaseGroove[0]);
            };
            break;
          case 1: // side torus
            transfo[ip][igroove] = new TGeoCombiTrans(mSupportXDimensions[4][0] / 2. + mMoreLength - mLWater4[igroove], mRadius4[igroove] + mXPosition4[igroove] - mSupportYDimensions[disk][0] / 2. - mHalfDiskGap, iface * (mRohacellThickness + epsilon), rotationSideTorusR);
            break;
          case 2: // Linear tilted
            rotationTiltedLinearR = new TGeoRotation("rotationTiltedLinearRight", 90. - mAngle4[igroove], 90., 0.);
            transfo[ip][igroove] = new TGeoCombiTrans(mSupportXDimensions[4][0] / 2. + mMoreLength - xPos4[igroove], yPos4[igroove] - mSupportYDimensions[disk][0] / 2. - mHalfDiskGap,
                                                      iface * (mRohacellThickness + epsilon), rotationTiltedLinearR);
            break;
          case 3: // Central Torus
            transfo[ip][igroove] = new TGeoCombiTrans(0., yPos4[igroove] + mLpartial4[igroove] / 2 * TMath::Sin(mAngle4[igroove] * TMath::DegToRad()) - mRadiusCentralTore[igroove] * TMath::Cos(mAngle4[igroove] * TMath::DegToRad()) - mSupportYDimensions[disk][0] / 2. - mHalfDiskGap,
                                                      iface * (mRohacellThickness + epsilon), rotationCentralTorus);
            break;
          case 4: // Linear tilted
            rotationTiltedLinearL = new TGeoRotation("rotationTiltedLinearLeft", 90. + mAngle4[igroove], 90., 0.);
            transfo[ip][igroove] = new TGeoCombiTrans(-(mSupportXDimensions[4][0] / 2. + mMoreLength - xPos4[igroove]), yPos4[igroove] - mSupportYDimensions[disk][0] / 2. - mHalfDiskGap,
                                                      iface * (mRohacellThickness + epsilon), rotationTiltedLinearL);
            break;
          case 5: // side torus
            transfo[ip][igroove] = new TGeoCombiTrans(-(mSupportXDimensions[4][0] / 2. + mMoreLength - mLWater4[igroove]), mRadius4[igroove] + mPosition[igroove],
                                                      iface * (mRohacellThickness + epsilon), rotationSideTorusL);
            break;
          case 6: // Linear
            transfo[ip][igroove] = new TGeoCombiTrans(-(mSupportXDimensions[4][0] / 2. + mMoreLength - mLWater4[igroove] / 2.), mPosition[igroove],
                                                      iface * (mRohacellThickness + epsilon), rotationLinear);
            break;
        }

        if (!(ip == 0 && igroove == 0 && iface == 1)) {
          if (ip & 1) {
            rohacellBaseGroove[iCount] =
              new TGeoSubtraction(rohacellGroove[iCount - 1], grooveTorus[ip][igroove], nullptr, transfo[ip][igroove]);
          } else {
            rohacellBaseGroove[iCount] =
              new TGeoSubtraction(rohacellGroove[iCount - 1], grooveTube[ip][igroove], nullptr, transfo[ip][igroove]);
          }
          rohacellGroove[iCount] = new TGeoCompositeShape(Form("rohacell4Groove%d_G%d_F%d_H%d", iCount, igroove, iface, half), rohacellBaseGroove[iCount]);
        }
        iCount++;
      }
    }
  }
  // **********************************************************************************************************

  //auto* rohacellhole4 = new TGeoSubtraction(rohacellBase4, holeRohacell4, t41, t42);
  auto* rohacellhole4 = new TGeoSubtraction(rohacellGroove[iCount - 1], holeRohacell4, t41, t42);
  auto* rh4 = new TGeoCompositeShape(Form("rohacellBase4_D4_H%d", half), rohacellhole4);
  auto* rohacellBaseWithHole = new TGeoVolume(Form("rohacellBaseWithHole_D4_H%d", half), rh4, mRohacell);

  rohacellBaseWithHole->SetLineColor(kGray);
  rotation = new TGeoRotation("rotation", 0., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  rohacellPlate->AddNode(rohacellBaseWithHole, 0, new TGeoTranslation(0., 0., mZPlan[disk]));

  ty = mSupportYDimensions[disk][0];

  TGeoVolume* partRohacell;

  for (Int_t ipart = 1; ipart < mNPart[disk]; ipart++) {
    ty += mSupportYDimensions[disk][ipart] / 2.;
    /*
    TGeoVolume* partRohacell = gGeoManager->MakeBox(Form("partRohacelli_D4_H%d_%d", half, ipart), mRohacell,
                                                    mSupportXDimensions[disk][ipart] / 2.,
                                                    mSupportYDimensions[disk][ipart] / 2., mRohacellThickness);
    partRohacell->SetLineColor(kGray);
    */
    auto* t = new TGeoTranslation("t", 0, ty + mHalfDiskGap, mZPlan[disk]);

    //===========================================================================================================
    //===========================================================================================================
    auto* partRohacell0 =
      new TGeoBBox(Form("rohacellBase0_D4_H%d_%d", half, ipart), mSupportXDimensions[disk][ipart] / 2.,
                   mSupportYDimensions[disk][ipart] / 2., mRohacellThickness);
    Double_t mShift;

    // ****************  Creating grooves for the other parts of the rohacell plate **********************
    for (Int_t iface = 1; iface > -2; iface -= 2) {     // front and rear
      for (Int_t igroove = 0; igroove < 4; igroove++) { // 4 grooves
        if (ipart == 1) {
          mPosition[ipart] = mXPosition4[igroove] - mSupportYDimensions[disk][ipart] / 2. - mHalfDiskGap - mSupportYDimensions[disk][ipart - 1];
          mShift = -mSupportYDimensions[disk][ipart - 1];
        };
        if (ipart == 2) {
          mPosition[ipart] = mXPosition4[igroove] - mSupportYDimensions[disk][ipart] / 2. - mHalfDiskGap - mSupportYDimensions[disk][ipart - 1] - mSupportYDimensions[disk][ipart - 2];
          mShift = -mSupportYDimensions[disk][ipart - 1] - mSupportYDimensions[disk][ipart - 2];
        };
        if (ipart == 3) {
          mPosition[ipart] = mXPosition4[igroove] - mSupportYDimensions[disk][ipart] / 2. - mHalfDiskGap - mSupportYDimensions[disk][ipart - 1] - mSupportYDimensions[disk][ipart - 2] - mSupportYDimensions[disk][ipart - 3];
          mShift = -mSupportYDimensions[disk][ipart - 1] - mSupportYDimensions[disk][ipart - 2] - mSupportYDimensions[disk][ipart - 3];
        };

        for (Int_t ip = 0; ip < 7; ip++) { // each groove is made of 7 parts

          switch (ip) {
            case 0: // Linear
              transfo[ip][igroove] = new TGeoCombiTrans(mSupportXDimensions[4][0] / 2. + mMoreLength - mLWater4[igroove] / 2., mPosition[ipart],
                                                        iface * (mRohacellThickness + epsilon), rotationLinear);
              if (igroove == 0 && iface == 1) {
                rohacellBaseGroove[iCount] = new TGeoSubtraction(partRohacell0, grooveTube[ip][igroove], nullptr, transfo[ip][igroove]);
                rohacellGroove[iCount] = new TGeoCompositeShape(Form("rohacell4Groove%d_G%d_F%d_H%d", ip, igroove, iface, half), rohacellBaseGroove[iCount]);
              };
              break;
            case 1: // side torus
              transfo[ip][igroove] = new TGeoCombiTrans(mSupportXDimensions[4][0] / 2. + mMoreLength - mLWater4[igroove], mPosition[ipart] + mRadius4[igroove],
                                                        iface * (mRohacellThickness + epsilon), rotationSideTorusR);
              break;
            case 2: // Linear tilted
              rotationTiltedLinearR = new TGeoRotation("rotationTiltedLinearRight", 90. - mAngle4[igroove], 90., 0.);
              transfo[ip][igroove] = new TGeoCombiTrans(mSupportXDimensions[4][0] / 2. + mMoreLength - xPos4[igroove], yPos4[igroove] + mShift - mHalfDiskGap - mSupportYDimensions[disk][ipart] / 2., iface * (mRohacellThickness + epsilon), rotationTiltedLinearR);
              break;
            case 3: // Central Torus
              transfo[ip][igroove] = new TGeoCombiTrans(0., mPosition[ipart] + yPos4[igroove] + mLpartial4[igroove] / 2 * TMath::Sin(mAngle4[igroove] * TMath::DegToRad()) - mRadiusCentralTore[igroove] * TMath::Cos(mAngle4[igroove] * TMath::DegToRad()) - mXPosition4[igroove],
                                                        iface * (mRohacellThickness + epsilon), rotationCentralTorus);
              break;
            case 4: // Linear tilted
              rotationTiltedLinearL = new TGeoRotation("rotationTiltedLinearLeft", 90. + mAngle4[igroove], 90., 0.);
              transfo[ip][igroove] = new TGeoCombiTrans(-(mSupportXDimensions[4][0] / 2. + mMoreLength - xPos4[igroove]), yPos4[igroove] + mPosition[ipart] - mXPosition4[igroove],
                                                        iface * (mRohacellThickness + epsilon), rotationTiltedLinearL);
              break;
            case 5: // side torus
              transfo[ip][igroove] = new TGeoCombiTrans(-(mSupportXDimensions[4][0] / 2. + mMoreLength - mLWater4[igroove]), mRadius4[igroove] + mPosition[ipart],
                                                        iface * (mRohacellThickness + epsilon), rotationSideTorusL);
              break;
            case 6: // Linear
              transfo[ip][igroove] = new TGeoCombiTrans(-(mSupportXDimensions[4][0] / 2. + mMoreLength - mLWater4[igroove] / 2.), mPosition[ipart],
                                                        iface * (mRohacellThickness + epsilon), rotationLinear);
              break;
          }
          if (!(ip == 0 && igroove == 0 && iface == 1)) {
            if (ip & 1) {
              rohacellBaseGroove[iCount] =
                new TGeoSubtraction(rohacellGroove[iCount - 1], grooveTorus[ip][igroove], nullptr, transfo[ip][igroove]);
            } else {
              rohacellBaseGroove[iCount] =
                new TGeoSubtraction(rohacellGroove[iCount - 1], grooveTube[ip][igroove], nullptr, transfo[ip][igroove]);
            }

            rohacellGroove[iCount] = new TGeoCompositeShape(Form("rohacell4Groove%d_G%d_F%d_H%d", iCount, igroove, iface, half), rohacellBaseGroove[iCount]);
          }
          iCount++;
        }
      }
    }

    partRohacell = new TGeoVolume(Form("partRohacelli_D4_H%d_%d", half, ipart), rohacellGroove[iCount - 1], mRohacell);
    //partRohacell = new TGeoVolume(Form("partRohacelli_D4_H%d_%d", half, ipart), partRohacell0, mRohacell);
    //===========================================================================================================
    //===========================================================================================================

    rohacellPlate->AddNode(partRohacell, ipart, t);

    ty += mSupportYDimensions[disk][ipart] / 2.;
  }

  //  if (half == Top) {
  //    rotation = new TGeoRotation ("rotation", 0., 0., 0.);
  //    transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  //    mHalfDisk->AddNode(rohacellPlate, 1, transformation);
  //  }
  //  if (half == Bottom) {
  rotation = new TGeoRotation("rotation", 180., 0., 0.);
  transformation = new TGeoCombiTrans(0., 0., 0., rotation);
  mHalfDisk->AddNode(rohacellPlate, 2, transformation);
  //  }

  createManifold(disk);
}

//_____________________________________________________________________________
void HeatExchanger::initParameters()
{

  mHalfDiskRotation = new TGeoRotation**[constants::DisksNumber];
  mHalfDiskTransformation = new TGeoCombiTrans**[constants::DisksNumber];
  for (Int_t idisk = 0; idisk < constants::DisksNumber; idisk++) {
    mHalfDiskRotation[idisk] = new TGeoRotation*[constants::HalvesNumber];
    mHalfDiskTransformation[idisk] = new TGeoCombiTrans*[constants::HalvesNumber];
    for (Int_t ihalf = 0; ihalf < constants::HalvesNumber; ihalf++) {
      mHalfDiskRotation[idisk][ihalf] = new TGeoRotation(Form("rotation%d%d", idisk, ihalf), 0., 0., 0.);
      mHalfDiskTransformation[idisk][ihalf] =
        new TGeoCombiTrans(Form("transformation%d%d", idisk, ihalf), 0., 0., 0., mHalfDiskRotation[idisk][ihalf]);
    }
  }

  //mRohacellThickness = mHeatExchangerThickness / 2. - 2. * mCarbonThickness - 2 * (mRWater + mDRPipe);
  //mRohacellThickness = mHeatExchangerThickness / 2. - 2. * mCarbonThickness;
  mRohacellThickness = mHeatExchangerThickness / 2. - 2. * mCarbonThickness - 2 * Geometry::sGlueRohacellCarbonThickness - 2 * Geometry::sKaptonOnCarbonThickness - 2 * Geometry::sKaptonGlueThickness;

  mHalfDiskGap = 0.2;

  mNPart[0] = 3;
  mNPart[1] = 3;
  mNPart[2] = 3;
  mNPart[3] = 4; //5
  mNPart[4] = 4;

  mRMin[0] = 2.35;
  mRMin[1] = 2.35;
  mRMin[2] = 2.35;
  mRMin[3] = 3.35;
  mRMin[4] = 3.75;

  // mZPlan[0] = 46;
  // mZPlan[1] = 49.3;
  // mZPlan[2] = 53.1;
  // mZPlan[3] = 68.7;
  // mZPlan[4] = 76.8;

  mZPlan[0] = 0;
  mZPlan[1] = 0;
  mZPlan[2] = 0;
  mZPlan[3] = 0;
  mZPlan[4] = 0;

  mSupportXDimensions = new Double_t*[constants::DisksNumber];
  mSupportYDimensions = new Double_t*[constants::DisksNumber];

  for (Int_t i = 0; i < constants::DisksNumber; i++) {
    mSupportXDimensions[i] = new double[mNPart[i]];
    mSupportYDimensions[i] = new double[mNPart[i]];
  }

  mMoreLength = 0.7; // additional length of water pipes outside the rohacell plate, should be 7 mm

  // disk width
  // disks 0, 1
  mSupportXDimensions[0][0] = mSupportXDimensions[1][0] = 21.8;
  mSupportXDimensions[0][1] = mSupportXDimensions[1][1] = 15.7;
  mSupportXDimensions[0][2] = mSupportXDimensions[1][2] = 5.6;

  // disk 2
  mSupportXDimensions[2][0] = 22.59; // 23
  mSupportXDimensions[2][1] = 19.2;  // 14
  mSupportXDimensions[2][2] = 12.4;  // 5.1

  // disk 3
  mSupportXDimensions[3][0] = 28.4; //28.4
  mSupportXDimensions[3][1] = 19.2; //21.9;
  mSupportXDimensions[3][2] = 9.0;  //18.5;
  mSupportXDimensions[3][3] = 7.3;  //8.3;

  // disk 4
  mSupportXDimensions[4][0] = 29.39; //28.4;
  mSupportXDimensions[4][1] = 22.60; //24.204;
  mSupportXDimensions[4][2] = 15.80; //21.9;
  mSupportXDimensions[4][3] = 5.60;  //15.1;

  // disk height
  // disks 0, 1
  mSupportYDimensions[0][0] = mSupportYDimensions[1][0] = 6.7;
  mSupportYDimensions[0][1] = mSupportYDimensions[1][1] = 2.5;
  mSupportYDimensions[0][2] = mSupportYDimensions[1][2] = 2.4;

  // disk 2
  mSupportYDimensions[2][0] = 6.66; // 6.7
  mSupportYDimensions[2][1] = 2.5;  // 2.6
  mSupportYDimensions[2][2] = 2.99; // 2.0

  // disk 3
  mSupportYDimensions[3][0] = 9.2; //6.61;
  mSupportYDimensions[3][1] = 3.0; //3.01;
  mSupportYDimensions[3][2] = 2.3; //3.01;
  mSupportYDimensions[3][3] = 1.2; //1.8;

  // disk 4
  mSupportYDimensions[4][0] = 9.20; //6.61;
  mSupportYDimensions[4][1] = 3.00; //3.01;
  mSupportYDimensions[4][2] = 3.00; //3.01;
  mSupportYDimensions[4][3] = 0.90; //2.42;

  // Parameters for disks 0, 1
  //mLWater0[0] = mLWater1[0] =5.98;
  mLWater0[0] = mLWater1[0] = 5.98 + mMoreLength;
  mLWater0[1] = mLWater1[1] = 5.91 + mMoreLength;
  mLWater0[2] = mLWater1[2] = 2.93 + mMoreLength;

  mXPosition0[0] = mXPosition1[0] = 1.7;
  mXPosition0[1] = mXPosition1[1] = 4.61;
  mXPosition0[2] = mXPosition1[2] = 6.31;

  mAngle0[0] = mAngle1[0] = 31.;
  mAngle0[1] = mAngle1[1] = 29.;
  mAngle0[2] = mAngle1[2] = 28.5;

  mRadius0[0] = mRadius1[0] = 4.0;
  mRadius0[1] = mRadius1[1] = 4.0;
  mRadius0[2] = mRadius1[2] = 4.0;

  mLpartial0[0] = mLpartial1[0] = 0.916;
  mLpartial0[1] = mLpartial1[1] = 0.73;
  mLpartial0[2] = mLpartial1[2] = 4.85;

  // Parameters for disk 2
  mLWater2[0] = 5.29 + mMoreLength;
  mLWater2[1] = 5.29 + mMoreLength;
  mLWater2[2] = 1.31 + mMoreLength;

  mXPosition2[0] = 1.7;
  mXPosition2[1] = 4.61;
  mXPosition2[2] = 6.41;

  mAngle2[0] = 28.;
  mAngle2[1] = 29.;
  mAngle2[2] = 27.;

  mRadius2[0] = 4.0;
  mRadius2[1] = 4.0;
  mRadius2[2] = 4.0;

  mLpartial2[0] = 2.60;
  mLpartial2[1] = 2.38;
  mLpartial2[2] = 7.16;

  // Parameters for disk 3
  mLWater3[0] = 6.35 + mMoreLength; //8.032;
  mLWater3[1] = 5.6 + mMoreLength;  //8.032;
  mLWater3[2] = 2.88 + mMoreLength; //8.2;
  mLWater3[3] = 2.4 + mMoreLength;  //8.2;

  mXPosition3[0] = 1.4;
  mXPosition3[1] = 3.9;
  mXPosition3[2] = 5.9;
  mXPosition3[3] = 7.9;

  mAngle3[0] = 34.; //41.3;
  mAngle3[1] = 30.; //41.3;
  mAngle3[2] = 28.; //28;
  mAngle3[3] = 32.; //28;

  mRadius3[0] = 4.; //4.3
  mRadius3[1] = 4.; //4.3
  mRadius3[2] = 4.; //7.4;
  mRadius3[3] = 4.; //7.4;

  mAngleThirdPipe3 = 15.;
  mLpartial3[0] = 4.0;  //2.3;
  mLpartial3[1] = 5.26; //2.3;
  mLpartial3[2] = 8.53; // fm
  mLpartial3[3] = 8.90; // fm

  mRadius3fourth[0] = 9.6;
  mRadius3fourth[1] = 2.9;
  mRadius3fourth[2] = 2.9;
  mRadius3fourth[3] = 0.;

  mAngle3fourth[0] = 40.8;
  mAngle3fourth[1] = 50.;
  mAngle3fourth[2] = 60.;
  mAngle3fourth[3] = 8 + mAngle3fourth[0] - mAngle3fourth[1] + mAngle3fourth[2];

  // Parameters for disk 4

  mLWater4[0] = 6.20 + mMoreLength; //5.911;
  mLWater4[1] = 4.67 + mMoreLength; //3.697;
  mLWater4[2] = 2.70 + mMoreLength; //3.038;
  mLWater4[3] = 1.20 + mMoreLength; //3.038;

  mXPosition4[0] = 1.3; //1.7;
  mXPosition4[1] = 3.8; //3.492;
  mXPosition4[2] = 5.8; //4.61;
  mXPosition4[3] = 7.8; //5.5;
  mXPosition4[4] = 5.8;

  mAngle4[0] = 31.; //35.5;
  mAngle4[1] = 28.; //30.;
  mAngle4[2] = 28.; //54.;
  mAngle4[3] = 29.; //53.;
  mAngle4[4] = 40;
  mAngle4[5] = (mAngle4[3] - mAngle4[4]);

  mRadius4[0] = 4.0; //6.6;
  mRadius4[1] = 4.0; //7.2;
  mRadius4[2] = 4.0; //4.6;
  mRadius4[3] = 4.0; //6.2;
  mRadius4[4] = 4.0; //6.;

  mLpartial4[0] = 5.095;  //2.5;
  mLpartial4[1] = 7.026;  //3.6;
  mLpartial4[2] = 9.327;  //2.5;
  mLpartial4[3] = 11.006; //3.6;

  mAngle4fifth[0] = 64.;
  mAngle4fifth[1] = 30.;
  mAngle4fifth[2] = 27.;
  mAngle4fifth[3] = mAngle4fifth[0] - mAngle4fifth[1] + mAngle4fifth[2];

  mRadius4fifth[0] = 2.7;
  mRadius4fifth[1] = 5.;
  mRadius4fifth[2] = 5.1;
  mRadius4fifth[3] = 4.3;
}
