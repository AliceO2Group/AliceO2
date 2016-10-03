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

#include "FairLogger.h"

#include "MFTBase/Constants.h"
#include "MFTSimulation/LadderSegmentation.h"
#include "MFTSimulation/ChipSegmentation.h"
#include "MFTSimulation/Flex.h"
#include "MFTSimulation/Chip.h"
#include "MFTSimulation/Ladder.h"
#include "MFTSimulation/Geometry.h"
#include "MFTSimulation/Plane.h"

using namespace AliceO2::MFT;

/// \cond CLASSIMP
ClassImp(AliceO2::MFT::Flex)
/// \endcond

//_____________________________________________________________________________
Flex::Flex():
TNamed(),
fFlexOrigin(),
fLadderSeg(NULL)
{
  // Constructor
}

//_____________________________________________________________________________
Flex::~Flex() 
{

}

//_____________________________________________________________________________
Flex::Flex(LadderSegmentation *ladder):
TNamed(),
fFlexOrigin(),
fLadderSeg(ladder)
{
  // Constructor
}


//_____________________________________________________________________________
TGeoVolumeAssembly* Flex::MakeFlex(Int_t nbsensors, Double_t length)
{
  // Informations from the technical report mft_flex_proto_5chip_v08_laz50p.docx on MFT twiki and private communications

  // For the naming
  Geometry * mftGeom = Geometry::Instance();
  Int_t idHalfMFT = mftGeom->GetHalfID(fLadderSeg->GetUniqueID());
  Int_t idHalfDisk = mftGeom->GetHalfDiskID(fLadderSeg->GetUniqueID());
  Int_t idLadder = mftGeom->GetLadderID(fLadderSeg->GetUniqueID());

  // First a global pointer for the flex
  TGeoMedium *kMedAir = gGeoManager->GetMedium("MFT_Air$");
  TGeoVolumeAssembly*  flex  = new TGeoVolumeAssembly(Form("flex_%d_%d_%d",idHalfMFT,idHalfDisk,idLadder));

  // Defining one single layer for the strips and the AVDD and DVDD
  TGeoVolume* lines = Make_Lines(nbsensors,length-Constants::kClearance, Constants::kFlexHeight - Constants::kClearance, Constants::kAluThickness);

  // AGND and DGND layers
  TGeoVolume* agnd_dgnd = Make_AGND_DGND(length-Constants::kClearance, Constants::kFlexHeight-Constants::kClearance, Constants::kAluThickness);

  // The others layers
  TGeoVolume* kaptonlayer     = Make_Kapton(length, Constants::kFlexHeight, Constants::kKaptonThickness);
  TGeoVolume* varnishlayerIn  = Make_Varnish(length, Constants::kFlexHeight, Constants::kVarnishThickness,0);
  TGeoVolume* varnishlayerOut = Make_Varnish(length, Constants::kFlexHeight, Constants::kVarnishThickness,1);
    
  // Final flex building
  Double_t zvarnishIn = Constants::kKaptonThickness/2+Constants::kAluThickness+Constants::kVarnishThickness/2-Constants::kGlueThickness;
  Double_t zgnd = Constants::kKaptonThickness/2+Constants::kAluThickness/2-Constants::kGlueThickness;
  Double_t zkaptonlayer = -Constants::kGlueThickness;
  Double_t zlines = -Constants::kKaptonThickness/2-Constants::kAluThickness/2-Constants::kGlueThickness;
  Double_t zvarnishOut = -Constants::kKaptonThickness/2-Constants::kAluThickness-Constants::kVarnishThickness/2-Constants::kGlueThickness;

  //-----------------------------------------------------------------------------------------
  //-------------------------- Adding all layers of the FPC ----------------------------------
  //-----------------------------------------------------------------------------------------
  
  flex->AddNode(varnishlayerIn,  1,  new TGeoTranslation(0., 0., zvarnishIn));    // inside, in front of the cold plate
  flex->AddNode(agnd_dgnd,       1,  new TGeoTranslation(0., 0., zgnd));
  flex->AddNode(kaptonlayer,     1,  new TGeoTranslation(0., 0., zkaptonlayer));
  flex->AddNode(lines,           1,  new TGeoTranslation(0., 0., zlines));
  flex->AddNode(varnishlayerOut, 1,  new TGeoTranslation(0., 0., zvarnishOut));   // outside

  Make_ElectricComponents(flex, nbsensors, length, zvarnishOut);
  //-----------------------------------------------------------------------------------------
  //-----------------------------------------------------------------------------------------
  //-----------------------------------------------------------------------------------------

  return flex;
}


//_____________________________________________________________________________
void Flex::Make_ElectricComponents(TGeoVolumeAssembly*  flex, Int_t nbsensors, Double_t length, Double_t zvarnish) 
{

  // Making and adding all the electric components
  TGeoVolumeAssembly *electric[200];

  // 2 components on the connector side
  Int_t total;

  TGeoRotation *rotation = new TGeoRotation ("rotation", 90., 0., 0.);
  TGeoRotation *rotationpi = new TGeoRotation ("rotationpi", 180., 0., 0.);
  TGeoCombiTrans *transformation0 = new TGeoCombiTrans(length/2 - 0.1, Constants::kFlexHeight/2-0.2, zvarnish-Constants::kVarnishThickness/2-Constants::kCapacitorDz/2, rotation);
  TGeoCombiTrans *transformation1 = new TGeoCombiTrans(length/2 - 0.1, Constants::kFlexHeight/2-0.6, zvarnish-Constants::kVarnishThickness/2-Constants::kCapacitorDz/2, rotation);

  for(Int_t id = 0; id < 2; id++) 
    electric[id] = Make_ElectricComponent(Constants::kCapacitorDy, Constants::kCapacitorDx, Constants::kCapacitorDz, id);
  flex->AddNode(electric[0], 1, transformation0);
  flex->AddNode(electric[1], 2, transformation1);
  total=2;

  // 2 lines of electric components along the FPC in the middle (4 per sensor)
  for(Int_t id=0; id < 4*nbsensors; id++)
    electric[id+total] = Make_ElectricComponent(Constants::kCapacitorDy, Constants::kCapacitorDx, Constants::kCapacitorDz, id+total);
  for(Int_t id=0; id < 2*nbsensors; id++) {
    flex->AddNode(electric[id+total], id+1000, new TGeoTranslation(-length/2 + (id+0.5)*Constants::kSensorLength/2, Constants::kFlexHeight/2 - 0.35, zvarnish - Constants::kVarnishThickness/2 - Constants::kCapacitorDz/2));
    flex->AddNode(electric[id+total+2*nbsensors], id+2000, new TGeoTranslation(-length/2 + (id+0.5)*Constants::kSensorLength/2, 0., zvarnish - Constants::kVarnishThickness/2 - Constants::kCapacitorDz/2));
  }
  total=total+4*nbsensors;
  
  // ------- 3 components on the FPC side -------- 
  for(Int_t id=0; id < 3; id++)
    electric[id+total] = Make_ElectricComponent(Constants::kCapacitorDy, Constants::kCapacitorDx, Constants::kCapacitorDz, id+total);
  for(Int_t id=0 ; id < 3; id++) {
    flex->AddNode(electric[id+total], id+3000, new TGeoTranslation(-length/2+Constants::kSensorLength+(id+1)*0.3-0.6, -Constants::kFlexHeight/2 + 0.2, zvarnish-Constants::kVarnishThickness/2 - Constants::kCapacitorDz/2));
  }
  total=total+3;
  
  /*
  // The connector of the FPC
  for(Int_t id=0; id < 74; id++)electric[id+total] = Make_ElectricComponent(Constants::kConnectorLength, Constants::kConnectorWidth, 
									    Constants::kConnectorThickness, id+total);
  for(Int_t id=0; id < 37; id++){
    flex->AddNode(electric[id+total], id+100, new TGeoTranslation(length/2+0.15-Constants::kConnectorOffset, id*0.04-Constants::kFlexHeight/2 + 0.1, 
								  zvarnish-Constants::kVarnishThickness/2-Constants::kCapacitorDz/2));
    flex->AddNode(electric[id+total+37], id+200, new TGeoTranslation(length/2-0.15-Constants::kConnectorOffset, id*0.04-Constants::kFlexHeight/2 + 0.1, 
								     zvarnish - Constants::kVarnishThickness/2 - Constants::kCapacitorDz/2));
  }
  total=total+74;
  */
    
  //-------------------------- New Connector ----------------------
  TGeoMedium *kMedAlu = gGeoManager->GetMedium("MFT_Alu$");
  TGeoMedium *kMedPeek = gGeoManager->GetMedium("MFT_PEEK$");

  TGeoBBox *connect = new TGeoBBox("connect", Constants::kConnectorLength/2, Constants::kConnectorWidth/2, Constants::kConnectorHeight/2);
  TGeoBBox *remov = new TGeoBBox("remov", Constants::kConnectorLength/2, Constants::kConnectorWidth/2+Constants::kEpsilon, Constants::kConnectorHeight/2+Constants::kEpsilon);

  TGeoTranslation    *t1= new TGeoTranslation ("t1", Constants::kConnectorThickness, 0., -0.01);
  TGeoSubtraction    *connecto = new TGeoSubtraction(connect, remov, NULL, t1);
  TGeoCompositeShape *connector = new TGeoCompositeShape("connector", connecto);
  TGeoVolume *connectord = new TGeoVolume("connectord", connector, kMedAlu);
  connectord->SetVisibility(kTRUE);
  connectord->SetLineColor(kRed);
  connectord->SetLineWidth(1);
  connectord->SetFillColor(connectord->GetLineColor());
  connectord->SetFillStyle(4000); // 0% transparent

  Double_t interspace = 0.1; // interspace inside the 2 ranges of connector pads
  Double_t step = 0.04;      // interspace between each pad inside the connector
  for(Int_t id=0; id < 37; id++) {
    flex->AddNode(connectord,id+total,new TGeoTranslation(length/2+interspace/2+Constants::kConnectorLength/2-Constants::kConnectorOffset, id*step-Constants::kFlexHeight/2 + 0.1, zvarnish - Constants::kVarnishThickness/2 - Constants::kConnectorHeight));
    TGeoCombiTrans *transformationpi = new TGeoCombiTrans(length/2-interspace/2-Constants::kConnectorLength/2-Constants::kConnectorOffset, id*step-Constants::kFlexHeight/2 + 0.1, zvarnish - Constants::kVarnishThickness/2 - Constants::kConnectorHeight, rotationpi);
    flex->AddNode(connectord,id+total+37, transformationpi);
  }
  
  Double_t boxthickness = 0.05;
  TGeoBBox *boxconnect = new TGeoBBox("boxconnect", (2*Constants::kConnectorThickness+interspace+boxthickness)/2, Constants::kFlexHeight/2-0.04, Constants::kConnectorHeight/2);
  TGeoBBox *boxremov = new TGeoBBox("boxremov", (2*Constants::kConnectorThickness+interspace)/2, (Constants::kFlexHeight-0.1-step)/2, Constants::kConnectorHeight/2+0.001);
  TGeoSubtraction *boxconnecto = new TGeoSubtraction(boxconnect, boxremov, NULL, NULL);
  TGeoCompositeShape *boxconnector = new TGeoCompositeShape("boxconnector", boxconnecto);
  TGeoVolume *boxconnectord = new TGeoVolume("boxconnectord", boxconnector, kMedPeek);
  flex->AddNode(boxconnectord,1,new TGeoTranslation(length/2-Constants::kConnectorOffset, -step/2, zvarnish-Constants::kVarnishThickness/2-Constants::kConnectorHeight-Constants::kConnectorThickness));
  
  //---------------------------------------------------------------
  
}

//_____________________________________________________________________________
TGeoVolumeAssembly* Flex::Make_ElectricComponent(Double_t dx, Double_t dy,  Double_t dz, Int_t id)
{
  
  Geometry * mftGeom = Geometry::Instance();
  Int_t idHalfMFT = mftGeom->GetHalfID(fLadderSeg->GetUniqueID());
  Int_t idHalfDisk = mftGeom->GetHalfDiskID(fLadderSeg->GetUniqueID());
  Int_t idLadder = mftGeom->GetLadderID(fLadderSeg->GetUniqueID());
  //------------------------------------------------------
  TGeoMedium *kmedX7R  = gGeoManager->GetMedium("MFT_X7Rcapacitors$");
  TGeoMedium *kmedX7Rw = gGeoManager->GetMedium("MFT_X7Rweld$");

  TGeoVolumeAssembly* X7R0402 = new TGeoVolumeAssembly(Form("X7R_%d_%d_%d_%d",idHalfMFT,idHalfDisk,idLadder,id));

  TGeoBBox *capacit = new TGeoBBox("capacitor", dx/2, dy/2, dz/2);
  TGeoBBox *weld = new TGeoBBox("weld", (dx/4)/2, dy/2, (dz/2)/2);
  TGeoVolume*  capacitor = new TGeoVolume(Form("capacitor_%d_%d_%d_%d",idHalfMFT,idHalfDisk,idLadder,id), capacit, kmedX7R);
  TGeoVolume*  welding0 = new TGeoVolume(Form("welding0_%d_%d_%d_%d",idHalfMFT,idHalfDisk,idLadder,id), weld, kmedX7Rw);
  TGeoVolume*  welding1 = new TGeoVolume(Form("welding1_%d_%d_%d_%d",idHalfMFT,idHalfDisk,idLadder,id), weld, kmedX7Rw);
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

  X7R0402->AddNode(capacitor,  1,  new TGeoTranslation(0., 0., 0.)); 
  X7R0402->AddNode(welding0,   1,  new TGeoTranslation( dx/2+(dx/4)/2, 0., (dz/2)/2)); 
  X7R0402->AddNode(welding1,   1,  new TGeoTranslation(-dx/2-(dx/4)/2, 0., (dz/2)/2));
  
  X7R0402->SetVisibility(kTRUE);

  return X7R0402;
  
  //------------------------------------------------------

  /*
  // the medium has to be changed, see ITS capacitors...
  TGeoMedium *kMedCopper = gGeoManager->GetMedium("MFT_Cu$");

  Geometry * mftGeom = Geometry::Instance();
  Int_t idHalfMFT = mftGeom->GetHalfID(fLadderSeg->GetUniqueID());
  Int_t idHalfDisk = mftGeom->GetHalfDiskID(fLadderSeg->GetUniqueID());
  Int_t idLadder = mftGeom->GetLadderID(fLadderSeg->GetUniqueID());
  
  TGeoVolume* electriccomponent = new TGeoVolume(Form("electric_%d_%d_%d_%d",idHalfMFT,idHalfDisk,idLadder,id), new TGeoBBox("BOX", dy/2, dx/2, dz/2), kMedCopper);
  electriccomponent->SetVisibility(1);
  electriccomponent->SetLineColor(kRed);
  return electriccomponent;
  */
}

//_____________________________________________________________________________
TGeoVolume* Flex::Make_Lines(Int_t nbsensors, Double_t length, Double_t widthflex,  Double_t thickness)
{

  // One line is built by removing 3 lines of aluminium in the TGeoBBox *layer_def layer. Then one line is made by the 2 remaining aluminium strips. 

  // the initial layer of aluminium
  TGeoBBox *layer_def = new TGeoBBox("layer_def", length/2, widthflex/2, thickness/2);

  // Two holes for fixing and positionning of the FPC on the cold plate
  TGeoTube *hole1 = new TGeoTube("hole1", 0., Constants::kRadiusHole1, thickness/2 + Constants::kEpsilon);
  TGeoTube *hole2 = new TGeoTube("hole2", 0., Constants::kRadiusHole2, thickness/2 + Constants::kEpsilon);

  TGeoTranslation    *t1= new TGeoTranslation ("t1", length/2 - Constants::kHoleShift1, 0., 0.);
  TGeoSubtraction    *layerholesub1 = new TGeoSubtraction(layer_def, hole1, NULL, t1);
  TGeoCompositeShape *layerhole1 = new TGeoCompositeShape("layerhole1", layerholesub1);

  TGeoTranslation    *t2= new TGeoTranslation ("t2", length/2 - Constants::kHoleShift2, 0., 0.);
  TGeoSubtraction    *layerholesub2 = new TGeoSubtraction(layerhole1, hole2, NULL, t2);
  TGeoCompositeShape *layer = new TGeoCompositeShape("layerhole2", layerholesub2);

  TGeoBBox *line[25];
  TGeoTranslation *t[6],*ts[15],*tvdd, *tl[2];
  TGeoSubtraction *layerl[25];
  TGeoCompositeShape *layern[25];
  Int_t istart, istop;
  Int_t kTotalLinesNb=0;
  Int_t kTotalLinesNb1, kTotalLinesNb2;
  Double_t length_line;

  // ----------- two lines along the FPC digital side --------------
  t[0] = new TGeoTranslation ("t0", Constants::kSensorLength/2-Constants::kConnectorOffset/2, -widthflex/2 + 2*Constants::kLineWidth, 0.);    
  line[0]  = new TGeoBBox("line0",  length/2 - Constants::kConnectorOffset/2 - Constants::kSensorLength/2, Constants::kLineWidth/2, thickness/2 + Constants::kEpsilon);
  layerl[0] = new TGeoSubtraction(layer, line[0], NULL, t[0]);
  layern[0] = new TGeoCompositeShape(Form("layer%d",0), layerl[0]);

  istart = 1; istop = 6;
  for (int iline = istart; iline < istop; iline++){
    t[iline] = new TGeoTranslation (Form("t%d",iline), Constants::kSensorLength/2 - Constants::kConnectorOffset/2, -widthflex/2 + 2*(iline+1)*Constants::kLineWidth, 0.);
    line[iline]  = new TGeoBBox(Form("line%d",iline),  length/2 - Constants::kConnectorOffset/2 - Constants::kSensorLength/2, Constants::kLineWidth/2, thickness/2 + Constants::kEpsilon);
    layerl[iline] = new TGeoSubtraction(layern[iline-1], line[iline], NULL, t[iline]);
    layern[iline] = new TGeoCompositeShape(Form("layer%d",iline), layerl[iline]);
    kTotalLinesNb++;
  }

  // ---------  lines for the sensors, one line/sensor -------------
  istart = kTotalLinesNb+1; istop = 6+3*nbsensors;
  for (int iline = istart; iline < istop; iline++){
    length_line=length - Constants::kConnectorOffset - TMath::Nint((iline-6)/3)*Constants::kSensorLength - Constants::kSensorLength/2;
    ts[iline] = new TGeoTranslation (Form("t%d",iline), length/2-length_line/2-Constants::kConnectorOffset, -2*(iline-6)*Constants::kLineWidth+0.5-widthflex/2, 0.);
    line[iline]  = new TGeoBBox(Form("line%d",iline), length_line/2, Constants::kLineWidth/2, thickness/2 + Constants::kEpsilon);
    layerl[iline] = new TGeoSubtraction(layern[iline-1], line[iline], NULL, ts[iline]);
    layern[iline] = new TGeoCompositeShape(Form("layer%d",iline), layerl[iline]);
    kTotalLinesNb++;
  }

  // ---------  an interspace to separate AVDD and DVDD -------------
  kTotalLinesNb++;
  tvdd = new TGeoTranslation ("tvdd", 0., widthflex/2-Constants::kShiftDDGNDline, 0.);    
  line[kTotalLinesNb]  = new TGeoBBox(Form("line%d",kTotalLinesNb),  length/2, 2*Constants::kLineWidth/2, thickness/2 + Constants::kEpsilon);
  layerl[kTotalLinesNb] = new TGeoSubtraction(layern[kTotalLinesNb-1], line[kTotalLinesNb], NULL, tvdd);
  layern[kTotalLinesNb] = new TGeoCompositeShape(Form("layer%d",kTotalLinesNb), layerl[kTotalLinesNb]);
  kTotalLinesNb++;

  // ---------  one line along the FPC analog side -------------  
  istart = kTotalLinesNb; istop = kTotalLinesNb + 2;
  for (int iline = istart; iline < istop; iline++){
    length_line=length - Constants::kConnectorOffset;
    tl[iline-istart] = new TGeoTranslation (Form("tl%d",iline), length/2-length_line/2-Constants::kConnectorOffset, widthflex/2-Constants::kShiftline-2.*(iline-istart)*Constants::kLineWidth, 0.);
    line[iline]  = new TGeoBBox(Form("line%d",iline), length_line/2, Constants::kLineWidth/2, thickness/2 + Constants::kEpsilon);
    layerl[iline] = new TGeoSubtraction(layern[iline-1], line[iline], NULL, tl[iline-istart]);
    layern[iline] = new TGeoCompositeShape(Form("layer%d",iline), layerl[iline]);
    kTotalLinesNb++;
  }

  Geometry * mftGeom = Geometry::Instance();
  Int_t idHalfMFT = mftGeom->GetHalfID(fLadderSeg->GetUniqueID());
  Int_t idHalfDisk = mftGeom->GetHalfDiskID(fLadderSeg->GetUniqueID());
  Int_t idLadder = mftGeom->GetLadderID(fLadderSeg->GetUniqueID());

  TGeoMedium *kMedAlu = gGeoManager->GetMedium("MFT_Alu$");

  TGeoVolume *lineslayer = new TGeoVolume(Form("lineslayer_%d_%d_%d",idHalfMFT,idHalfDisk,idLadder), layern[kTotalLinesNb-1], kMedAlu);
  lineslayer->SetVisibility(1);
  lineslayer->SetLineColor(kBlue);

  return lineslayer;

}

//_____________________________________________________________________________
TGeoVolume* Flex::Make_AGND_DGND(Double_t length, Double_t widthflex,  Double_t thickness)
{  

  // AGND and DGND layers
  TGeoBBox *layer = new TGeoBBox("layer", length/2, widthflex/2, thickness/2);
  TGeoTube *hole1 = new TGeoTube("hole1", 0., Constants::kRadiusHole1, thickness/2 + Constants::kEpsilon);
  TGeoTube *hole2 = new TGeoTube("hole2", 0., Constants::kRadiusHole2, thickness/2 + Constants::kEpsilon);
  
  TGeoTranslation    *t1= new TGeoTranslation ("t1", length/2-Constants::kHoleShift1, 0., 0.);
  TGeoSubtraction    *layerholesub1 = new TGeoSubtraction(layer, hole1, NULL, t1);
  TGeoCompositeShape *layerhole1 = new TGeoCompositeShape("layerhole1", layerholesub1);

  TGeoTranslation    *t2= new TGeoTranslation ("t2", length/2-Constants::kHoleShift2, 0., 0.);
  TGeoSubtraction    *layerholesub2 = new TGeoSubtraction(layerhole1, hole2, NULL, t2);
  TGeoCompositeShape *layerhole2 = new TGeoCompositeShape("layerhole2", layerholesub2);

  //--------------
  TGeoBBox *line[3];
  TGeoTranslation *t[3];
  TGeoCompositeShape *layern[3];
  TGeoSubtraction *layerl[3];
  Double_t length_line;
  length_line=length - Constants::kConnectorOffset;

  // First, the two lines along the FPC side
  t[0] = new TGeoTranslation("t0", length/2-length_line/2-Constants::kConnectorOffset, widthflex/2 - Constants::kShiftline, 0.);
  line[0]  = new TGeoBBox("line0",  length/2 - Constants::kConnectorOffset/2, Constants::kLineWidth/2, thickness/2 + Constants::kEpsilon);
  layerl[0] = new TGeoSubtraction(layerhole2, line[0], NULL, t[0]);
  layern[0] = new TGeoCompositeShape(Form("layer%d",0), layerl[0]);

  t[1] = new TGeoTranslation("t1", length/2-length_line/2-Constants::kConnectorOffset, widthflex/2 - Constants::kShiftline - 2*Constants::kLineWidth, 0.);
  line[1]  = new TGeoBBox("line1", length/2 - Constants::kConnectorOffset/2, Constants::kLineWidth/2, thickness/2 + Constants::kEpsilon);
  layerl[1] = new TGeoSubtraction(layern[0], line[1], NULL, t[1]);
  layern[1] = new TGeoCompositeShape(Form("layer%d",1), layerl[1]);

  // Now the interspace to separate the AGND et DGND --> same interspace compare the AVDD et DVDD
  t[2] = new TGeoTranslation("t2", length/2-length_line/2, widthflex/2 - Constants::kShiftDDGNDline, 0.);
  line[2]  = new TGeoBBox("line2", length/2 - Constants::kConnectorOffset/2, Constants::kLineWidth, thickness/2 + Constants::kEpsilon);
  layerl[2] = new TGeoSubtraction(layern[1], line[2], NULL, t[2]);
  layern[2] = new TGeoCompositeShape(Form("layer%d",2), layerl[2]);

  //--------------

  Geometry * mftGeom = Geometry::Instance();
  Int_t idHalfMFT = mftGeom->GetHalfID(fLadderSeg->GetUniqueID());
  Int_t idHalfDisk = mftGeom->GetHalfDiskID(fLadderSeg->GetUniqueID());
  Int_t idLadder = mftGeom->GetLadderID(fLadderSeg->GetUniqueID());

  TGeoMedium *kMedAlu = gGeoManager->GetMedium("MFT_Alu$");
  TGeoVolume *alulayer = new TGeoVolume(Form("alulayer_%d_%d_%d",idHalfMFT,idHalfDisk,idLadder), layern[2], kMedAlu);
  alulayer->SetVisibility(1);
  alulayer->SetLineColor(kBlue);

  return alulayer;

}

//_____________________________________________________________________________
TGeoVolume* Flex::Make_Kapton(Double_t length, Double_t widthflex, Double_t thickness)
{

  TGeoBBox *layer = new TGeoBBox("layer", length/2, widthflex/2, thickness/2);
  // Two holes for fixing and positionning of the FPC on the cold plate
  TGeoTube *hole1 = new TGeoTube("hole1", 0., Constants::kRadiusHole1, thickness/2+Constants::kEpsilon);
  TGeoTube *hole2 = new TGeoTube("hole2", 0., Constants::kRadiusHole2, thickness/2+Constants::kEpsilon);
  
  TGeoTranslation    *t1= new TGeoTranslation ("t1", length/2-Constants::kHoleShift1, 0., 0.);
  TGeoSubtraction    *layerholesub1 = new TGeoSubtraction(layer, hole1, NULL, t1);
  TGeoCompositeShape *layerhole1 = new TGeoCompositeShape("layerhole1", layerholesub1);

  TGeoTranslation    *t2= new TGeoTranslation ("t2", length/2-Constants::kHoleShift2, 0., 0.);
  TGeoSubtraction    *layerholesub2 = new TGeoSubtraction(layerhole1, hole2, NULL, t2);
  TGeoCompositeShape *layerhole2 = new TGeoCompositeShape("layerhole2", layerholesub2);

  Geometry * mftGeom = Geometry::Instance();
  Int_t idHalfMFT = mftGeom->GetHalfID(fLadderSeg->GetUniqueID());
  Int_t idHalfDisk = mftGeom->GetHalfDiskID(fLadderSeg->GetUniqueID());
  Int_t idLadder = mftGeom->GetLadderID(fLadderSeg->GetUniqueID());

  TGeoMedium *kMedKapton = gGeoManager->GetMedium("MFT_Kapton$");
  TGeoVolume *kaptonlayer = new TGeoVolume(Form("kaptonlayer_%d_%d_%d",idHalfMFT,idHalfDisk,idLadder), layerhole2, kMedKapton);
  kaptonlayer->SetVisibility(1);
  kaptonlayer->SetLineColor(kYellow);

  return kaptonlayer;

}

//_____________________________________________________________________________
TGeoVolume* Flex::Make_Varnish(Double_t length, Double_t widthflex,  Double_t thickness, Int_t iflag)
{

  TGeoBBox *layer = new TGeoBBox("layer", length/2, widthflex/2, thickness/2);
  // Two holes for fixing and positionning of the FPC on the cold plate
  TGeoTube *hole1 = new TGeoTube("hole1", 0., Constants::kRadiusHole1, thickness/2+Constants::kEpsilon);
  TGeoTube *hole2 = new TGeoTube("hole2", 0., Constants::kRadiusHole2, thickness/2+Constants::kEpsilon);
  
  TGeoTranslation    *t1= new TGeoTranslation ("t1", length/2-Constants::kHoleShift1, 0., 0.);
  TGeoSubtraction    *layerholesub1 = new TGeoSubtraction(layer, hole1, NULL, t1);
  TGeoCompositeShape *layerhole1 = new TGeoCompositeShape("layerhole1", layerholesub1);

  TGeoTranslation    *t2= new TGeoTranslation ("t2", length/2-Constants::kHoleShift2, 0., 0.);
  TGeoSubtraction    *layerholesub2 = new TGeoSubtraction(layerhole1, hole2, NULL, t2);
  TGeoCompositeShape *layerhole2 = new TGeoCompositeShape("layerhole2", layerholesub2);

  Geometry * mftGeom = Geometry::Instance();
  Int_t idHalfMFT = mftGeom->GetHalfID(fLadderSeg->GetUniqueID());
  Int_t idHalfDisk = mftGeom->GetHalfDiskID(fLadderSeg->GetUniqueID());
  Int_t idLadder = mftGeom->GetLadderID(fLadderSeg->GetUniqueID());

  TGeoMedium *kMedVarnish = gGeoManager->GetMedium("MFT_Epoxy$");  // we assume that varnish = epoxy ...
  TGeoVolume *varnishlayer = new TGeoVolume(Form("varnishlayer_%d_%d_%d_%d",idHalfMFT,idHalfDisk,idLadder,iflag), layerhole2, kMedVarnish);
  varnishlayer->SetVisibility(1);
  varnishlayer->SetLineColor(kGreen-1);

  return varnishlayer;

}

