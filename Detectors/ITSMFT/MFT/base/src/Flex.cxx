// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Flex.cxx
/// \brief Flex class for ALICE MFT upgrade
/// \author Franck Manso <franck.manso@cern.ch>

#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoTrd2.h"
#include "TGeoMatrix.h"
#include "TGeoBBox.h"
#include "TGeoTube.h"
#include "TGeoCompositeShape.h"
#include "TGeoBoolNode.h"
#include "TMath.h"

#include "FairLogger.h"

#include "MFTBase/LadderSegmentation.h"
#include "MFTBase/ChipSegmentation.h"
#include "MFTBase/Flex.h"
#include "MFTBase/Ladder.h"
#include "MFTBase/Geometry.h"
#include "ITSMFTBase/SegmentationAlpide.h"

using namespace o2::mft;
using namespace o2::itsmft;

ClassImp(o2::mft::Flex);

//_____________________________________________________________________________
Flex::Flex() : mFlexOrigin(), mLadderSeg(nullptr)
{
  // Constructor
}

//_____________________________________________________________________________
Flex::~Flex() = default;

//_____________________________________________________________________________
Flex::Flex(LadderSegmentation* ladder) : mFlexOrigin(), mLadderSeg(ladder)
{
  // Constructor
}

//_____________________________________________________________________________
TGeoVolumeAssembly* Flex::makeFlex(Int_t nbsensors, Double_t length)
{

  // Informations from the technical report mft_flex_proto_5chip_v08_laz50p.docx on MFT twiki and private communications

  // For the naming
  Geometry* mftGeom = Geometry::instance();
  Int_t idHalfMFT = mftGeom->getHalfID(mLadderSeg->GetUniqueID());
  Int_t idHalfDisk = mftGeom->getDiskID(mLadderSeg->GetUniqueID());
  Int_t idLadder = mftGeom->getLadderID(mLadderSeg->GetUniqueID());

  // First a global pointer for the flex
  TGeoMedium* kMedAir = gGeoManager->GetMedium("MFT_Air$");
  auto* flex = new TGeoVolumeAssembly(Form("flex_%d_%d_%d", idHalfMFT, idHalfDisk, idLadder));

  // Defining one single layer for the strips and the AVDD and DVDD
  TGeoVolume* lines = makeLines(nbsensors, length - Geometry::sClearance, Geometry::sFlexHeight - Geometry::sClearance,
                                Geometry::sAluThickness);

  // AGND and DGND layers
  TGeoVolume* agnd_dgnd = makeAGNDandDGND(length - Geometry::sClearance, Geometry::sFlexHeight - Geometry::sClearance,
                                          Geometry::sAluThickness);

  // The others layers
  TGeoVolume* kaptonlayer = makeKapton(length, Geometry::sFlexHeight, Geometry::sKaptonThickness);
  TGeoVolume* varnishlayerIn = makeVarnish(length, Geometry::sFlexHeight, Geometry::sVarnishThickness, 0);
  TGeoVolume* varnishlayerOut = makeVarnish(length, Geometry::sFlexHeight, Geometry::sVarnishThickness, 1);

  // Final flex building
  Double_t zvarnishIn = Geometry::sKaptonThickness / 2 + Geometry::sAluThickness + Geometry::sVarnishThickness / 2 -
                        Geometry::sGlueThickness;
  Double_t zgnd = Geometry::sKaptonThickness / 2 + Geometry::sAluThickness / 2 - Geometry::sGlueThickness;
  Double_t zkaptonlayer = -Geometry::sGlueThickness;
  Double_t zlines = -Geometry::sKaptonThickness / 2 - Geometry::sAluThickness / 2 - Geometry::sGlueThickness;
  Double_t zvarnishOut = -Geometry::sKaptonThickness / 2 - Geometry::sAluThickness - Geometry::sVarnishThickness / 2 -
                         Geometry::sGlueThickness;

  //-----------------------------------------------------------------------------------------
  //-------------------------- Adding all layers of the FPC ----------------------------------
  //-----------------------------------------------------------------------------------------

  flex->AddNode(varnishlayerIn, 1, new TGeoTranslation(0., 0., zvarnishIn)); // inside, in front of the cold plate
  flex->AddNode(agnd_dgnd, 1, new TGeoTranslation(0., 0., zgnd));
  flex->AddNode(kaptonlayer, 1, new TGeoTranslation(0., 0., zkaptonlayer));
  flex->AddNode(lines, 1, new TGeoTranslation(0., 0., zlines));
  flex->AddNode(varnishlayerOut, 1, new TGeoTranslation(0., 0., zvarnishOut)); // outside

  makeElectricComponents(flex, nbsensors, length, zvarnishOut);

  return flex;
}

//_____________________________________________________________________________
void Flex::makeElectricComponents(TGeoVolumeAssembly* flex, Int_t nbsensors, Double_t length, Double_t zvarnish)
{

  // Making and adding all the electric components
  TGeoVolumeAssembly* electric[200];

  // 2 components on the connector side
  Int_t total;

  auto* rotation = new TGeoRotation("rotation", 90., 0., 0.);
  auto* rotationpi = new TGeoRotation("rotationpi", 180., 0., 0.);
  auto* transformation0 =
    new TGeoCombiTrans(length / 2 - 0.1, Geometry::sFlexHeight / 2 - 0.2,
                       zvarnish - Geometry::sVarnishThickness / 2 - Geometry::sCapacitorDz / 2, rotation);
  auto* transformation1 =
    new TGeoCombiTrans(length / 2 - 0.1, Geometry::sFlexHeight / 2 - 0.6,
                       zvarnish - Geometry::sVarnishThickness / 2 - Geometry::sCapacitorDz / 2, rotation);

  for (Int_t id = 0; id < 2; id++) {
    electric[id] = makeElectricComponent(Geometry::sCapacitorDy, Geometry::sCapacitorDx, Geometry::sCapacitorDz, id);
  }
  flex->AddNode(electric[0], 1, transformation0);
  flex->AddNode(electric[1], 2, transformation1);
  total = 2;

  // 2 lines of electric components along the FPC in the middle (4 per sensor)
  for (Int_t id = 0; id < 4 * nbsensors; id++) {
    electric[id + total] =
      makeElectricComponent(Geometry::sCapacitorDy, Geometry::sCapacitorDx, Geometry::sCapacitorDz, id + total);
  }
  for (Int_t id = 0; id < 2 * nbsensors; id++) {
    flex->AddNode(electric[id + total], id + 1000,
                  new TGeoTranslation(-length / 2 + (id + 0.5) * SegmentationAlpide::SensorSizeCols / 2,
                                      Geometry::sFlexHeight / 2 - 0.35,
                                      zvarnish - Geometry::sVarnishThickness / 2 - Geometry::sCapacitorDz / 2));
    flex->AddNode(electric[id + total + 2 * nbsensors], id + 2000,
                  new TGeoTranslation(-length / 2 + (id + 0.5) * SegmentationAlpide::SensorSizeCols / 2, 0.,
                                      zvarnish - Geometry::sVarnishThickness / 2 - Geometry::sCapacitorDz / 2));
  }
  total = total + 4 * nbsensors;

  // ------- 3 components on the FPC side --------
  for (Int_t id = 0; id < 3; id++) {
    electric[id + total] =
      makeElectricComponent(Geometry::sCapacitorDy, Geometry::sCapacitorDx, Geometry::sCapacitorDz, id + total);
  }
  for (Int_t id = 0; id < 3; id++) {
    flex->AddNode(electric[id + total], id + 3000,
                  new TGeoTranslation(-length / 2 + SegmentationAlpide::SensorSizeCols + (id + 1) * 0.3 - 0.6,
                                      -Geometry::sFlexHeight / 2 + 0.2,
                                      zvarnish - Geometry::sVarnishThickness / 2 - Geometry::sCapacitorDz / 2));
  }
  total = total + 3;

  /*
  // The connector of the FPC
  for(Int_t id=0; id < 74; id++)electric[id+total] = makeElectricComponent(Geometry::sConnectorLength,
  Geometry::sConnectorWidth,
                                                                            Geometry::sConnectorThickness, id+total);
  for(Int_t id=0; id < 37; id++){
    flex->AddNode(electric[id+total], id+100, new TGeoTranslation(length/2+0.15-Geometry::sConnectorOffset,
  id*0.04-Geometry::sFlexHeight/2 + 0.1,
                                                                  zvarnish-Geometry::sVarnishThickness/2-Geometry::sCapacitorDz/2));
    flex->AddNode(electric[id+total+37], id+200, new TGeoTranslation(length/2-0.15-Geometry::sConnectorOffset,
  id*0.04-Geometry::sFlexHeight/2 + 0.1,
                                                                     zvarnish - Geometry::sVarnishThickness/2 -
  Geometry::sCapacitorDz/2));
  }
  total=total+74;
  */

  //-------------------------- New Connector ----------------------
  TGeoMedium* kMedAlu = gGeoManager->GetMedium("MFT_Alu$");
  TGeoMedium* kMedPeek = gGeoManager->GetMedium("MFT_PEEK$");

  auto* connect = new TGeoBBox("connect", Geometry::sConnectorLength / 2, Geometry::sConnectorWidth / 2,
                               Geometry::sConnectorHeight / 2);
  auto* remov =
    new TGeoBBox("remov", Geometry::sConnectorLength / 2, Geometry::sConnectorWidth / 2 + Geometry::sEpsilon,
                 Geometry::sConnectorHeight / 2 + Geometry::sEpsilon);

  auto* t1 = new TGeoTranslation("t1", Geometry::sConnectorThickness, 0., -0.01);
  auto* connecto = new TGeoSubtraction(connect, remov, nullptr, t1);
  auto* connector = new TGeoCompositeShape("connector", connecto);
  auto* connectord = new TGeoVolume("connectord", connector, kMedAlu);
  connectord->SetVisibility(kTRUE);
  connectord->SetLineColor(kRed);
  connectord->SetLineWidth(1);
  connectord->SetFillColor(connectord->GetLineColor());
  connectord->SetFillStyle(4000); // 0% transparent

  Double_t interspace = 0.1; // interspace inside the 2 ranges of connector pads
  Double_t step = 0.04;      // interspace between each pad inside the connector
  for (Int_t id = 0; id < 37; id++) {
    flex->AddNode(
      connectord, id + total,
      new TGeoTranslation(length / 2 + interspace / 2 + Geometry::sConnectorLength / 2 - Geometry::sConnectorOffset,
                          id * step - Geometry::sFlexHeight / 2 + 0.1,
                          zvarnish - Geometry::sVarnishThickness / 2 - Geometry::sConnectorHeight / 2));
    auto* transformationpi =
      new TGeoCombiTrans(length / 2 - interspace / 2 - Geometry::sConnectorLength / 2 - Geometry::sConnectorOffset,
                         id * step - Geometry::sFlexHeight / 2 + 0.1,
                         zvarnish - Geometry::sVarnishThickness / 2 - Geometry::sConnectorHeight / 2, rotationpi);
    flex->AddNode(connectord, id + total + 37, transformationpi);
  }

  Double_t boxthickness = 0.05;
  auto* boxconnect = new TGeoBBox("boxconnect", (2 * Geometry::sConnectorThickness + interspace + boxthickness) / 2,
                                  Geometry::sFlexHeight / 2 - 0.04, Geometry::sConnectorHeight / 2);
  auto* boxremov = new TGeoBBox("boxremov", (2 * Geometry::sConnectorThickness + interspace) / 2,
                                (Geometry::sFlexHeight - 0.1 - step) / 2, Geometry::sConnectorHeight / 2 + 0.001);
  auto* boxconnecto = new TGeoSubtraction(boxconnect, boxremov, nullptr, nullptr);
  auto* boxconnector = new TGeoCompositeShape("boxconnector", boxconnecto);
  auto* boxconnectord = new TGeoVolume("boxconnectord", boxconnector, kMedPeek);
  flex->AddNode(boxconnectord, 1,
                new TGeoTranslation(length / 2 - Geometry::sConnectorOffset, -step / 2,
                                    zvarnish - Geometry::sVarnishThickness / 2 - Geometry::sConnectorHeight / 2 -
                                      Geometry::sConnectorThickness));
}

//_____________________________________________________________________________
TGeoVolumeAssembly* Flex::makeElectricComponent(Double_t dx, Double_t dy, Double_t dz, Int_t id)
{

  Geometry* mftGeom = Geometry::instance();
  Int_t idHalfMFT = mftGeom->getHalfID(mLadderSeg->GetUniqueID());
  Int_t idHalfDisk = mftGeom->getDiskID(mLadderSeg->GetUniqueID());
  Int_t idLadder = mftGeom->getLadderID(mLadderSeg->GetUniqueID());
  //------------------------------------------------------
  TGeoMedium* kmedX7R = gGeoManager->GetMedium("MFT_X7Rcapacitors$");
  TGeoMedium* kmedX7Rw = gGeoManager->GetMedium("MFT_X7Rweld$");

  auto* X7R0402 = new TGeoVolumeAssembly(Form("X7R_%d_%d_%d_%d", idHalfMFT, idHalfDisk, idLadder, id));

  auto* capacit = new TGeoBBox("capacitor", dx / 2, dy / 2, dz / 2);
  auto* weld = new TGeoBBox("weld", (dx / 4) / 2, dy / 2, (dz / 2) / 2);
  auto* capacitor =
    new TGeoVolume(Form("capacitor_%d_%d_%d_%d", idHalfMFT, idHalfDisk, idLadder, id), capacit, kmedX7R);
  auto* welding0 = new TGeoVolume(Form("welding0_%d_%d_%d_%d", idHalfMFT, idHalfDisk, idLadder, id), weld, kmedX7Rw);
  auto* welding1 = new TGeoVolume(Form("welding1_%d_%d_%d_%d", idHalfMFT, idHalfDisk, idLadder, id), weld, kmedX7Rw);
  capacitor->SetVisibility(kTRUE);
  capacitor->SetLineColor(kRed);
  capacitor->SetLineWidth(1);
  capacitor->SetFillColor(capacitor->GetLineColor());
  capacitor->SetFillStyle(4000); // 0% transparent

  welding0->SetVisibility(kTRUE);
  welding0->SetLineColor(kGray);
  welding0->SetLineWidth(1);
  welding0->SetFillColor(welding0->GetLineColor());
  welding0->SetFillStyle(4000); // 0% transparent

  welding1->SetVisibility(kTRUE);
  welding1->SetLineColor(kGray);
  welding1->SetLineWidth(1);
  welding1->SetFillColor(welding1->GetLineColor());
  welding1->SetFillStyle(4000); // 0% transparent

  X7R0402->AddNode(capacitor, 1, new TGeoTranslation(0., 0., 0.));
  X7R0402->AddNode(welding0, 1, new TGeoTranslation(dx / 2 + (dx / 4) / 2, 0., (dz / 2) / 2));
  X7R0402->AddNode(welding1, 1, new TGeoTranslation(-dx / 2 - (dx / 4) / 2, 0., (dz / 2) / 2));

  X7R0402->SetVisibility(kTRUE);

  return X7R0402;

  //------------------------------------------------------

  /*
  // the medium has to be changed, see ITS capacitors...
  TGeoMedium *kMedCopper = gGeoManager->GetMedium("MFT_Cu$");

  Geometry * mftGeom = Geometry::instance();
  Int_t idHalfMFT = mftGeom->getHalfID(mLadderSeg->GetUniqueID());
  Int_t idHalfDisk = mftGeom->getDiskID(mLadderSeg->GetUniqueID());
  Int_t idLadder = mftGeom->getLadderID(mLadderSeg->GetUniqueID());

  TGeoVolume* electriccomponent = new TGeoVolume(Form("electric_%d_%d_%d_%d",idHalfMFT,idHalfDisk,idLadder,id), new
  TGeoBBox("BOX", dy/2, dx/2, dz/2), kMedCopper);
  electriccomponent->SetVisibility(1);
  electriccomponent->SetLineColor(kRed);
  return electriccomponent;
  */
}

//_____________________________________________________________________________
TGeoVolume* Flex::makeLines(Int_t nbsensors, Double_t length, Double_t widthflex, Double_t thickness)
{

  // One line is built by removing 3 lines of aluminium in the TGeoBBox *layer_def layer. Then one line is made by the 2
  // remaining aluminium strips.

  // the initial layer of aluminium
  auto* layer_def = new TGeoBBox("layer_def", length / 2, widthflex / 2, thickness / 2);

  // Two holes for fixing and positionning of the FPC on the cold plate
  auto* hole1 = new TGeoTube("hole1", 0., Geometry::sRadiusHole1, thickness / 2 + Geometry::sEpsilon);
  auto* hole2 = new TGeoTube("hole2", 0., Geometry::sRadiusHole2, thickness / 2 + Geometry::sEpsilon);

  auto* t1 = new TGeoTranslation("t1", length / 2 - Geometry::sHoleShift1, 0., 0.);
  auto* layerholesub1 = new TGeoSubtraction(layer_def, hole1, nullptr, t1);
  auto* layerhole1 = new TGeoCompositeShape("layerhole1", layerholesub1);

  auto* t2 = new TGeoTranslation("t2", length / 2 - Geometry::sHoleShift2, 0., 0.);
  auto* layerholesub2 = new TGeoSubtraction(layerhole1, hole2, nullptr, t2);
  auto* layer = new TGeoCompositeShape("layerhole2", layerholesub2);

  TGeoBBox* line[25];
  TGeoTranslation *t[6], *ts[15], *tvdd, *tl[2];
  TGeoSubtraction* layerl[25];
  TGeoCompositeShape* layern[25];
  Int_t istart, istop;
  Int_t kTotalLinesNb = 0;
  Int_t kTotalLinesNb1, kTotalLinesNb2;
  Double_t length_line;

  // ----------- two lines along the FPC digital side --------------
  t[0] = new TGeoTranslation("t0", SegmentationAlpide::SensorSizeCols / 2 - Geometry::sConnectorOffset / 2,
                             -widthflex / 2 + 2 * Geometry::sLineWidth, 0.);
  line[0] = new TGeoBBox("line0", length / 2 - Geometry::sConnectorOffset / 2 - SegmentationAlpide::SensorSizeCols / 2,
                         Geometry::sLineWidth / 2, thickness / 2 + Geometry::sEpsilon);
  layerl[0] = new TGeoSubtraction(layer, line[0], nullptr, t[0]);
  layern[0] = new TGeoCompositeShape(Form("layer%d", 0), layerl[0]);

  istart = 1;
  istop = 6;
  for (int iline = istart; iline < istop; iline++) {
    t[iline] =
      new TGeoTranslation(Form("t%d", iline), SegmentationAlpide::SensorSizeCols / 2 - Geometry::sConnectorOffset / 2,
                          -widthflex / 2 + 2 * (iline + 1) * Geometry::sLineWidth, 0.);
    line[iline] = new TGeoBBox(Form("line%d", iline),
                               length / 2 - Geometry::sConnectorOffset / 2 - SegmentationAlpide::SensorSizeCols / 2,
                               Geometry::sLineWidth / 2, thickness / 2 + Geometry::sEpsilon);
    layerl[iline] = new TGeoSubtraction(layern[iline - 1], line[iline], nullptr, t[iline]);
    layern[iline] = new TGeoCompositeShape(Form("layer%d", iline), layerl[iline]);
    kTotalLinesNb++;
  }

  // ---------  lines for the sensors, one line/sensor -------------
  istart = kTotalLinesNb + 1;
  istop = 6 + 3 * nbsensors;
  for (int iline = istart; iline < istop; iline++) {
    length_line = length - Geometry::sConnectorOffset -
                  TMath::Nint((iline - 6) / 3) * SegmentationAlpide::SensorSizeCols -
                  SegmentationAlpide::SensorSizeCols / 2;
    ts[iline] = new TGeoTranslation(Form("t%d", iline), length / 2 - length_line / 2 - Geometry::sConnectorOffset,
                                    -2 * (iline - 6) * Geometry::sLineWidth + 0.5 - widthflex / 2, 0.);
    line[iline] = new TGeoBBox(Form("line%d", iline), length_line / 2, Geometry::sLineWidth / 2,
                               thickness / 2 + Geometry::sEpsilon);
    layerl[iline] = new TGeoSubtraction(layern[iline - 1], line[iline], nullptr, ts[iline]);
    layern[iline] = new TGeoCompositeShape(Form("layer%d", iline), layerl[iline]);
    kTotalLinesNb++;
  }

  // ---------  an interspace to separate AVDD and DVDD -------------
  kTotalLinesNb++;
  tvdd = new TGeoTranslation("tvdd", 0., widthflex / 2 - Geometry::sShiftDDGNDline, 0.);
  line[kTotalLinesNb] = new TGeoBBox(Form("line%d", kTotalLinesNb), length / 2, 2 * Geometry::sLineWidth / 2,
                                     thickness / 2 + Geometry::sEpsilon);
  layerl[kTotalLinesNb] = new TGeoSubtraction(layern[kTotalLinesNb - 1], line[kTotalLinesNb], nullptr, tvdd);
  layern[kTotalLinesNb] = new TGeoCompositeShape(Form("layer%d", kTotalLinesNb), layerl[kTotalLinesNb]);
  kTotalLinesNb++;

  // ---------  one line along the FPC analog side -------------
  istart = kTotalLinesNb;
  istop = kTotalLinesNb + 2;
  for (int iline = istart; iline < istop; iline++) {
    length_line = length - Geometry::sConnectorOffset;
    tl[iline - istart] =
      new TGeoTranslation(Form("tl%d", iline), length / 2 - length_line / 2 - Geometry::sConnectorOffset,
                          widthflex / 2 - Geometry::sShiftline - 2. * (iline - istart) * Geometry::sLineWidth, 0.);
    line[iline] = new TGeoBBox(Form("line%d", iline), length_line / 2, Geometry::sLineWidth / 2,
                               thickness / 2 + Geometry::sEpsilon);
    layerl[iline] = new TGeoSubtraction(layern[iline - 1], line[iline], nullptr, tl[iline - istart]);
    layern[iline] = new TGeoCompositeShape(Form("layer%d", iline), layerl[iline]);
    kTotalLinesNb++;
  }

  Geometry* mftGeom = Geometry::instance();
  Int_t idHalfMFT = mftGeom->getHalfID(mLadderSeg->GetUniqueID());
  Int_t idHalfDisk = mftGeom->getDiskID(mLadderSeg->GetUniqueID());
  Int_t idLadder = mftGeom->getLadderID(mLadderSeg->GetUniqueID());

  TGeoMedium* kMedAlu = gGeoManager->GetMedium("MFT_Alu$");

  auto* lineslayer =
    new TGeoVolume(Form("lineslayer_%d_%d_%d", idHalfMFT, idHalfDisk, idLadder), layern[kTotalLinesNb - 1], kMedAlu);
  lineslayer->SetVisibility(true);
  lineslayer->SetLineColor(kBlue);

  return lineslayer;
}

//_____________________________________________________________________________
TGeoVolume* Flex::makeAGNDandDGND(Double_t length, Double_t widthflex, Double_t thickness)
{

  // AGND and DGND layers
  auto* layer = new TGeoBBox("layer", length / 2, widthflex / 2, thickness / 2);
  auto* hole1 = new TGeoTube("hole1", 0., Geometry::sRadiusHole1, thickness / 2 + Geometry::sEpsilon);
  auto* hole2 = new TGeoTube("hole2", 0., Geometry::sRadiusHole2, thickness / 2 + Geometry::sEpsilon);

  auto* t1 = new TGeoTranslation("t1", length / 2 - Geometry::sHoleShift1, 0., 0.);
  auto* layerholesub1 = new TGeoSubtraction(layer, hole1, nullptr, t1);
  auto* layerhole1 = new TGeoCompositeShape("layerhole1", layerholesub1);

  auto* t2 = new TGeoTranslation("t2", length / 2 - Geometry::sHoleShift2, 0., 0.);
  auto* layerholesub2 = new TGeoSubtraction(layerhole1, hole2, nullptr, t2);
  auto* layerhole2 = new TGeoCompositeShape("layerhole2", layerholesub2);

  //--------------
  TGeoBBox* line[3];
  TGeoTranslation* t[3];
  TGeoCompositeShape* layern[3];
  TGeoSubtraction* layerl[3];
  Double_t length_line;
  length_line = length - Geometry::sConnectorOffset;

  // First, the two lines along the FPC side
  t[0] = new TGeoTranslation("t0", length / 2 - length_line / 2 - Geometry::sConnectorOffset,
                             widthflex / 2 - Geometry::sShiftline, 0.);
  line[0] = new TGeoBBox("line0", length / 2 - Geometry::sConnectorOffset / 2, Geometry::sLineWidth / 2,
                         thickness / 2 + Geometry::sEpsilon);
  layerl[0] = new TGeoSubtraction(layerhole2, line[0], nullptr, t[0]);
  layern[0] = new TGeoCompositeShape(Form("layer%d", 0), layerl[0]);

  t[1] = new TGeoTranslation("t1", length / 2 - length_line / 2 - Geometry::sConnectorOffset,
                             widthflex / 2 - Geometry::sShiftline - 2 * Geometry::sLineWidth, 0.);
  line[1] = new TGeoBBox("line1", length / 2 - Geometry::sConnectorOffset / 2, Geometry::sLineWidth / 2,
                         thickness / 2 + Geometry::sEpsilon);
  layerl[1] = new TGeoSubtraction(layern[0], line[1], nullptr, t[1]);
  layern[1] = new TGeoCompositeShape(Form("layer%d", 1), layerl[1]);

  // Now the interspace to separate the AGND et DGND --> same interspace compare the AVDD et DVDD
  t[2] = new TGeoTranslation("t2", length / 2 - length_line / 2, widthflex / 2 - Geometry::sShiftDDGNDline, 0.);
  line[2] = new TGeoBBox("line2", length / 2 - Geometry::sConnectorOffset / 2, Geometry::sLineWidth,
                         thickness / 2 + Geometry::sEpsilon);
  layerl[2] = new TGeoSubtraction(layern[1], line[2], nullptr, t[2]);
  layern[2] = new TGeoCompositeShape(Form("layer%d", 2), layerl[2]);

  //--------------

  Geometry* mftGeom = Geometry::instance();
  Int_t idHalfMFT = mftGeom->getHalfID(mLadderSeg->GetUniqueID());
  Int_t idHalfDisk = mftGeom->getDiskID(mLadderSeg->GetUniqueID());
  Int_t idLadder = mftGeom->getLadderID(mLadderSeg->GetUniqueID());

  TGeoMedium* kMedAlu = gGeoManager->GetMedium("MFT_Alu$");
  auto* alulayer = new TGeoVolume(Form("alulayer_%d_%d_%d", idHalfMFT, idHalfDisk, idLadder), layern[2], kMedAlu);
  alulayer->SetVisibility(true);
  alulayer->SetLineColor(kBlue);

  return alulayer;
}

//_____________________________________________________________________________
TGeoVolume* Flex::makeKapton(Double_t length, Double_t widthflex, Double_t thickness)
{

  auto* layer = new TGeoBBox("layer", length / 2, widthflex / 2, thickness / 2);
  // Two holes for fixing and positionning of the FPC on the cold plate
  auto* hole1 = new TGeoTube("hole1", 0., Geometry::sRadiusHole1, thickness / 2 + Geometry::sEpsilon);
  auto* hole2 = new TGeoTube("hole2", 0., Geometry::sRadiusHole2, thickness / 2 + Geometry::sEpsilon);

  auto* t1 = new TGeoTranslation("t1", length / 2 - Geometry::sHoleShift1, 0., 0.);
  auto* layerholesub1 = new TGeoSubtraction(layer, hole1, nullptr, t1);
  auto* layerhole1 = new TGeoCompositeShape("layerhole1", layerholesub1);

  auto* t2 = new TGeoTranslation("t2", length / 2 - Geometry::sHoleShift2, 0., 0.);
  auto* layerholesub2 = new TGeoSubtraction(layerhole1, hole2, nullptr, t2);
  auto* layerhole2 = new TGeoCompositeShape("layerhole2", layerholesub2);

  Geometry* mftGeom = Geometry::instance();
  Int_t idHalfMFT = mftGeom->getHalfID(mLadderSeg->GetUniqueID());
  Int_t idHalfDisk = mftGeom->getDiskID(mLadderSeg->GetUniqueID());
  Int_t idLadder = mftGeom->getLadderID(mLadderSeg->GetUniqueID());

  TGeoMedium* kMedKapton = gGeoManager->GetMedium("MFT_Kapton$");
  auto* kaptonlayer =
    new TGeoVolume(Form("kaptonlayer_%d_%d_%d", idHalfMFT, idHalfDisk, idLadder), layerhole2, kMedKapton);
  kaptonlayer->SetVisibility(true);
  kaptonlayer->SetLineColor(kYellow);

  return kaptonlayer;
}

//_____________________________________________________________________________
TGeoVolume* Flex::makeVarnish(Double_t length, Double_t widthflex, Double_t thickness, Int_t iflag)
{

  auto* layer = new TGeoBBox("layer", length / 2, widthflex / 2, thickness / 2);
  // Two holes for fixing and positionning of the FPC on the cold plate
  auto* hole1 = new TGeoTube("hole1", 0., Geometry::sRadiusHole1, thickness / 2 + Geometry::sEpsilon);
  auto* hole2 = new TGeoTube("hole2", 0., Geometry::sRadiusHole2, thickness / 2 + Geometry::sEpsilon);

  auto* t1 = new TGeoTranslation("t1", length / 2 - Geometry::sHoleShift1, 0., 0.);
  auto* layerholesub1 = new TGeoSubtraction(layer, hole1, nullptr, t1);
  auto* layerhole1 = new TGeoCompositeShape("layerhole1", layerholesub1);

  auto* t2 = new TGeoTranslation("t2", length / 2 - Geometry::sHoleShift2, 0., 0.);
  auto* layerholesub2 = new TGeoSubtraction(layerhole1, hole2, nullptr, t2);
  auto* layerhole2 = new TGeoCompositeShape("layerhole2", layerholesub2);

  Geometry* mftGeom = Geometry::instance();
  Int_t idHalfMFT = mftGeom->getHalfID(mLadderSeg->GetUniqueID());
  Int_t idHalfDisk = mftGeom->getDiskID(mLadderSeg->GetUniqueID());
  Int_t idLadder = mftGeom->getLadderID(mLadderSeg->GetUniqueID());

  TGeoMedium* kMedVarnish = gGeoManager->GetMedium("MFT_Epoxy$"); // we assume that varnish = epoxy ...
  TGeoMaterial* kMatVarnish = kMedVarnish->GetMaterial();
  // kMatVarnish->Dump();
  auto* varnishlayer =
    new TGeoVolume(Form("varnishlayer_%d_%d_%d_%d", idHalfMFT, idHalfDisk, idLadder, iflag), layerhole2, kMedVarnish);
  varnishlayer->SetVisibility(true);
  varnishlayer->SetLineColor(kGreen - 1);

  return varnishlayer;
}
