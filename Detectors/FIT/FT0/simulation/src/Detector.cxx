// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TGeoManager.h" // for TGeoManager
#include "TMath.h"
#include "TGraph.h"
#include "TString.h"
#include "TSystem.h"
#include "TVirtualMC.h"
#include "TVector3.h"
#include "TGeoTube.h"
#include "TGeoCompositeShape.h"

#include "FairRootManager.h" // for FairRootManager
#include "FairLogger.h"
#include "FairVolume.h"

#include "FairRootManager.h"
#include "FairVolume.h"

#include <sstream>
#include "FT0Base/Geometry.h"
#include "FT0Simulation/Detector.h"
#include "SimulationDataFormat/Stack.h"

using namespace o2::ft0;
using o2::ft0::Geometry;

ClassImp(Detector);

Detector::Detector(Bool_t Active)
  : o2::base::DetImpl<Detector>("FT0", Active), mIdSens1(0), mPMTeff(nullptr), mHits(o2::utils::createSimVector<o2::ft0::HitType>())

{
  // Gegeo  = GetGeometry() ;

  //  TString gn(geo->GetName());
}

Detector::Detector(const Detector& rhs)
  : o2::base::DetImpl<Detector>(rhs), mIdSens1(rhs.mIdSens1), mPMTeff(rhs.mPMTeff), mHits(o2::utils::createSimVector<o2::ft0::HitType>())
{
}

Detector::~Detector()
{
  o2::utils::freeSimVector(mHits);
}

void Detector::InitializeO2Detector()
{
  // FIXME: we need to register the sensitive volumes with FairRoot
  TGeoVolume* v = gGeoManager->GetVolume("0REG");
  if (v == nullptr) {
    LOG(WARN) << "@@@@ Sensitive volume 0REG not found!!!!!!!!";
  } else {

    AddSensitiveVolume(v);
  }
}

void Detector::ConstructGeometry()
{
  LOG(DEBUG) << "Creating FT0 geometry\n";
  CreateMaterials();

  Float_t zdetA = 333;
  Float_t zdetC = 82;

  Int_t idrotm[999];
  Double_t x, y, z;
  Float_t pstartC[3] = {20., 20, 5};
  Float_t pstartA[3] = {20, 20, 5};
  Float_t pinstart[3] = {2.95, 2.95, 4.34};
  Float_t pmcp[3] = {2.949, 2.949, 1.}; // MCP

  int nCellsA = Geometry::NCellsA;
  int nCellsC = Geometry::NCellsC;

  Geometry geometry;
  TVector3 centerMCP = geometry.centerMCP(2);
  Matrix(idrotm[901], 90., 0., 90., 90., 180., 0.);

  // C side Concave Geometry

  Double_t crad = 82.; // define concave c-side radius here

  Double_t dP = pmcp[0]; // side length of mcp divided by 2

  // uniform angle between detector faces==
  Double_t btta = 2 * TMath::ATan(dP / crad);

  // get noncompensated translation data
  Double_t grdin[6] = {-3, -2, -1, 1, 2, 3};
  Double_t gridpoints[6];
  for (Int_t i = 0; i < 6; i++) {
    gridpoints[i] = crad * TMath::Sin((1 - 1 / (2 * TMath::Abs(grdin[i]))) * grdin[i] * btta);
  }

  Double_t xi[Geometry::NCellsC] = {gridpoints[1], gridpoints[2], gridpoints[3], gridpoints[4], gridpoints[0],
                                    gridpoints[1], gridpoints[2], gridpoints[3], gridpoints[4], gridpoints[5],
                                    gridpoints[0], gridpoints[1], gridpoints[4], gridpoints[5], gridpoints[0],
                                    gridpoints[1], gridpoints[4], gridpoints[5], gridpoints[0], gridpoints[1],
                                    gridpoints[2], gridpoints[3], gridpoints[4], gridpoints[5], gridpoints[1],
                                    gridpoints[2], gridpoints[3], gridpoints[4]};
  Double_t yi[Geometry::NCellsC] = {gridpoints[5], gridpoints[5], gridpoints[5], gridpoints[5], gridpoints[4],
                                    gridpoints[4], gridpoints[4], gridpoints[4], gridpoints[4], gridpoints[4],
                                    gridpoints[3], gridpoints[3], gridpoints[3], gridpoints[3], gridpoints[2],
                                    gridpoints[2], gridpoints[2], gridpoints[2], gridpoints[1], gridpoints[1],
                                    gridpoints[1], gridpoints[1], gridpoints[1], gridpoints[1], gridpoints[0],
                                    gridpoints[0], gridpoints[0], gridpoints[0]};
  Double_t zi[Geometry::NCellsC];
  for (Int_t i = 0; i < Geometry::NCellsC; i++) {
    zi[i] = TMath::Sqrt(TMath::Power(crad, 2) - TMath::Power(xi[i], 2) - TMath::Power(yi[i], 2));
  }

  // get rotation data
  Double_t ac[Geometry::NCellsC], bc[Geometry::NCellsC], gc[Geometry::NCellsC];
  for (Int_t i = 0; i < Geometry::NCellsC; i++) {
    ac[i] = TMath::ATan(yi[i] / xi[i]) - TMath::Pi() / 2 + 2 * TMath::Pi();
    if (xi[i] < 0) {
      bc[i] = TMath::ACos(zi[i] / crad);
    } else {
      bc[i] = -1 * TMath::ACos(zi[i] / crad);
    }
  }
  Double_t xc2[Geometry::NCellsC], yc2[Geometry::NCellsC], zc2[Geometry::NCellsC];

  // compensation based on node position within individual detector geometries
  // determine compensated radius
  Double_t rcomp = crad + pstartC[2] / 2.0; //
  for (Int_t i = 0; i < Geometry::NCellsC; i++) {
    // Get compensated translation data
    xc2[i] = rcomp * TMath::Cos(ac[i] + TMath::Pi() / 2) * TMath::Sin(-1 * bc[i]);
    yc2[i] = rcomp * TMath::Sin(ac[i] + TMath::Pi() / 2) * TMath::Sin(-1 * bc[i]);
    zc2[i] = rcomp * TMath::Cos(bc[i]);

    // Convert angles to degrees
    ac[i] *= 180 / TMath::Pi();
    bc[i] *= 180 / TMath::Pi();
    gc[i] = -1 * ac[i];
  }
  // A Side

  Float_t xa[Geometry::NCellsA] = {-11.8, -5.9, 0, 5.9, 11.8, -11.8, -5.9, 0, 5.9, 11.8, -12.8, -6.9,
                                   6.9, 12.8, -11.8, -5.9, 0, 5.9, 11.8, -11.8, -5.9, 0, 5.9, 11.8};

  Float_t ya[Geometry::NCellsA] = {11.9, 11.9, 12.9, 11.9, 11.9, 6.0, 6.0, 7.0, 6.0, 6.0, -0.1, -0.1,
                                   0.1, 0.1, -6.0, -6.0, -7.0, -6.0, -6.0, -11.9, -11.9, -12.9, -11.9, -11.9};

  TGeoVolumeAssembly* stlinA = new TGeoVolumeAssembly("0STL"); // A side mother
  TGeoVolumeAssembly* stlinC = new TGeoVolumeAssembly("0STR"); // C side mother

  // FIT interior
  TVirtualMC::GetMC()->Gsvolu("0INS", "BOX", getMediumID(kAir), pinstart, 3);
  TGeoVolume* ins = gGeoManager->GetVolume("0INS");
  //
  TGeoTranslation* tr[Geometry::NCellsA + Geometry::NCellsC];
  TString nameTr;

  // A side Translations
  for (Int_t itr = 0; itr < Geometry::NCellsA; itr++) {
    nameTr = Form("0TR%i", itr + 1);
    z = -pstartA[2] + pinstart[2];
    tr[itr] = new TGeoTranslation(nameTr.Data(), xa[itr], ya[itr], z);
    tr[itr]->RegisterYourself();
    stlinA->AddNode(ins, itr, tr[itr]);
  }

  TGeoRotation* rot[Geometry::NCellsC];
  TString nameRot;

  TGeoCombiTrans* com[Geometry::NCellsC];
  TString nameCom;

  // C Side Transformations
  for (Int_t itr = Geometry::NCellsA; itr < Geometry::NCellsA + Geometry::NCellsC; itr++) {
    nameTr = Form("0TR%i", itr + 1);
    nameRot = Form("0Rot%i", itr + 1);
    int ic = itr - Geometry::NCellsA;
    // nameCom = Form("0Com%i",itr+1);
    rot[ic] = new TGeoRotation(nameRot.Data(), ac[ic], bc[ic], gc[ic]);
    rot[ic]->RegisterYourself();

    tr[itr] = new TGeoTranslation(nameTr.Data(), xc2[ic], yc2[ic], (zc2[ic] - 80.));
    tr[itr]->RegisterYourself();

    //   com[itr-Geometry::NCellsA] = new TGeoCombiTrans(tr[itr],rot[itr-Geometry::NCellsA]);
    com[ic] = new TGeoCombiTrans(xc2[ic], yc2[ic], (zc2[ic] - 80), rot[ic]);
    TGeoHMatrix hm = *com[ic];
    TGeoHMatrix* ph = new TGeoHMatrix(hm);
    stlinC->AddNode(ins, itr, ph);
  }

  stlinA->AddNode(ConstructFrameGeometry(), 1);
  TGeoVolume* alice = gGeoManager->GetVolume("cave");
  alice->AddNode(stlinA, 1, new TGeoTranslation(0, 0, zdetA));
  // alice->AddNode(stlinC,1,new TGeoTranslation(0,0, -zdetC ) );
  TGeoRotation* rotC = new TGeoRotation("rotC", 90., 0., 90., 90., 180., 0.);
  alice->AddNode(stlinC, 1, new TGeoCombiTrans(0., 0., -zdetC, rotC));

  // MCP + 4 x wrapped radiator + 4xphotocathod + MCP + Al top in front of radiators
  SetOneMCP(ins);
}

void Detector::ConstructOpGeometry()
{
  LOG(DEBUG) << "Creating FIT optical geometry properties";

  DefineOpticalProperties();
}

//_________________________________________
void Detector::SetOneMCP(TGeoVolume* ins)
{

  Double_t x, y, z;
  Double_t crad = 82.;         // Define concave c-side radius here
  Double_t dP = 3.31735114408; // Work in Progress side length

  Float_t pinstart[3] = {2.95, 2.95, 2.5};
  Float_t ptop[3] = {1.324, 1.324, 1.};      // Cherenkov radiator
  Float_t ptopref[3] = {1.3241, 1.3241, 1.}; // Cherenkov radiator wrapped with reflector
  Double_t prfv[3] = {0.0002, 1.323, 1.};    // Vertical refracting layer bettwen radiators and between radiator and not optical Air
  Double_t prfh[3] = {1.323, 0.0002, 1.};    // Horizontal refracting layer bettwen radiators and ...
  Float_t pmcp[3] = {2.949, 2.949, 1.};      // MCP
  Float_t pmcpinner[3] = {2.749, 2.979, 0.1};
  Float_t pmcpside[3] = {0.1, 2.949, 1};
  Float_t pmcpbase[3] = {2.949, 2.949, 0.1};
  Float_t pmcptopglass[3] = {2.949, 2.949, 0.1}; // MCP top glass optical

  Float_t preg[3] = {1.324, 1.324, 0.05}; // Photcathode
  Double_t pal[3] = {2.648, 2.648, 0.25}; // 5mm Al on top of each radiator
  // Entry window (glass)
  TVirtualMC::GetMC()->Gsvolu("0TOP", "BOX", getMediumID(kOpGlass), ptop, 3); // Glass radiator
  TGeoVolume* top = gGeoManager->GetVolume("0TOP");
  TVirtualMC::GetMC()->Gsvolu("0TRE", "BOX", getMediumID(kAir), ptopref, 3); // Air: wrapped  radiator
  TGeoVolume* topref = gGeoManager->GetVolume("0TRE");
  TVirtualMC::GetMC()->Gsvolu("0RFV", "BOX", getMediumID(kOpAir), prfv, 3); // Optical Air vertical
  TGeoVolume* rfv = gGeoManager->GetVolume("0RFV");
  TVirtualMC::GetMC()->Gsvolu("0RFH", "BOX", getMediumID(kOpAir), prfh, 3); // Optical Air horizontal
  TGeoVolume* rfh = gGeoManager->GetVolume("0RFH");

  TVirtualMC::GetMC()->Gsvolu("0PAL", "BOX", getMediumID(kAl), pal, 3); // 5mm Al on top of the radiator
  TGeoVolume* altop = gGeoManager->GetVolume("0PAL");

  Double_t thet = TMath::ATan(dP / crad);
  Double_t rat = TMath::Tan(thet) / 2.0;
  /*
  //Al housing definition
  Double_t mgon[16];

  mgon[0] = -45;
  mgon[1] = 360.0;
  mgon[2] = 4;
  mgon[3] = 4;

  z = -pinstart[2] + 2 * pal[2];
  mgon[4] = z;
  mgon[5] = 2 * ptop[0] + preg[2];
  mgon[6] = dP + rat * z * 4 / 3;

  z = -pinstart[2] + 2 * pal[2] + 2 * ptopref[2];
  mgon[7] = z;
  mgon[8] = mgon[5];
  mgon[9] = dP + z * rat;
  mgon[10] = z;
  mgon[11] = pmcp[0] + preg[2];
  mgon[12] = mgon[9];

  z = -pinstart[2] + 2 * pal[2] + 2 * ptopref[2] + 2 * preg[2] + 2 * pmcp[2];
  mgon[13] = z;
  mgon[14] = mgon[11];
  mgon[15] = dP + z * rat * pmcp[2] * 9 / 10;

  TVirtualMC::GetMC()->Gsvolu("0SUP", "PGON", getMediumID(kAl), mgon, 16); //Al Housing for Support Structure//
  TGeoVolume* alsup = gGeoManager->GetVolume("0SUP");
  */
  TVirtualMC::GetMC()->Gsvolu("0REG", "BOX", getMediumID(kOpGlassCathode), preg, 3);
  TGeoVolume* cat = gGeoManager->GetVolume("0REG");

  //wrapped radiator +  reflecting layers

  Int_t ntops = 0, nrfvs = 0, nrfhs = 0;
  Float_t xin = 0, yin = 0, xinv = 0, yinv = 0, xinh = 0, yinh = 0;
  x = y = z = 0;
  topref->AddNode(top, 1, new TGeoTranslation(0, 0, 0));
  xinv = -ptop[0] - prfv[0];
  topref->AddNode(rfv, 1, new TGeoTranslation(xinv, 0, 0));
  printf(" GEOGEO  refv %f ,  0,0 \n", xinv);
  xinv = ptop[0] + prfv[0];
  topref->AddNode(rfv, 2, new TGeoTranslation(xinv, 0, 0));
  printf(" GEOGEO  refv %f ,  0,0 \n", xinv);
  yinv = -ptop[1] - prfh[1];
  topref->AddNode(rfh, 1, new TGeoTranslation(0, yinv, 0));
  printf(" GEOGEO  refh  ,  0, %f, 0 \n", yinv);
  yinv = ptop[1] + prfh[1];
  topref->AddNode(rfh, 2, new TGeoTranslation(0, yinv, 0));

  //container for radiator, cathode
  for (Int_t ix = 0; ix < 2; ix++) {
    xin = -pinstart[0] + 0.3 + (ix + 0.5) * 2 * ptopref[0];
    for (Int_t iy = 0; iy < 2; iy++) {
      z = -pinstart[2] + 2 * pal[2] + ptopref[2];
      yin = -pinstart[1] + 0.3 + (iy + 0.5) * 2 * ptopref[1];
      ntops++;
      ins->AddNode(topref, ntops, new TGeoTranslation(xin, yin, z));
      printf(" 0TOP  full %i x %f y %f z %f \n", ntops, xin, yin, z);
      z += ptopref[2] + 2. * pmcptopglass[2] + preg[2];
      ins->AddNode(cat, ntops, new TGeoTranslation(xin, yin, z));
      cat->Print();
      printf(" GEOGEO  CATHOD x=%f , y= %f z= %f num  %i\n", xin, yin, z, ntops);
    }
  }
  //Al top
  z = -pinstart[2] + pal[2];
  ins->AddNode(altop, 1, new TGeoTranslation(0, 0, z));

  // MCP
  TVirtualMC::GetMC()->Gsvolu("0MTO", "BOX", getMediumID(kOpGlass), pmcptopglass, 3); //Op  Glass
  TGeoVolume* mcptop = gGeoManager->GetVolume("0MTO");
  z = -pinstart[2] + 2 * pal[2] + 2 * ptopref[2] + pmcptopglass[2];
  ins->AddNode(mcptop, 1, new TGeoTranslation(0, 0, z));

  TVirtualMC::GetMC()->Gsvolu("0MCP", "BOX", getMediumID(kAir), pmcp, 3); //glass
  TGeoVolume* mcp = gGeoManager->GetVolume("0MCP");
  z = -pinstart[2] + 2 * pal[2] + 2 * ptopref[2] + 2 * pmcptopglass[2] + 2 * preg[2] + pmcp[2];
  ins->AddNode(mcp, 1, new TGeoTranslation(0, 0, z));
  TVirtualMC::GetMC()->Gsvolu("0MIN", "BOX", getMediumID(kGlass), pmcpinner, 3); //glass
  TGeoVolume* mcpinner = gGeoManager->GetVolume("0MIN");
  mcp->AddNode(mcpinner, 1, new TGeoTranslation(0, 0, 0));

  TVirtualMC::GetMC()->Gsvolu("0MSI", "BOX", getMediumID(kGlass), pmcpside, 3); //glass
  TGeoVolume* mcpside = gGeoManager->GetVolume("0MSI");
  x = -pmcp[0] + pmcpside[0];
  y = -pmcp[1] + pmcpside[1];
  mcp->AddNode(mcpside, 1, new TGeoTranslation(x, y, 0));
  x = pmcp[0] - pmcpside[0];
  y = pmcp[1] - pmcpside[1];
  mcp->AddNode(mcpside, 2, new TGeoTranslation(x, y, 0));
  x = -pmcp[1] + pmcpside[1];
  y = -pmcp[0] + pmcpside[0];
  mcp->AddNode(mcpside, 3, new TGeoCombiTrans(x, y, 0, new TGeoRotation("R2", 90, 0, 0)));
  x = pmcp[1] - pmcpside[1];
  y = pmcp[0] - pmcpside[0];
  mcp->AddNode(mcpside, 4, new TGeoCombiTrans(x, y, 0, new TGeoRotation("R2", 90, 0, 0)));

  TVirtualMC::GetMC()->Gsvolu("0MBA", "BOX", getMediumID(kCeramic), pmcpbase, 3); //glass
  TGeoVolume* mcpbase = gGeoManager->GetVolume("0MBA");
  z = -pinstart[2] + 2 * pal[2] + 2 * ptopref[2] + pmcptopglass[2] + 2 * pmcp[2] + pmcpbase[2];
  ins->AddNode(mcpbase, 1, new TGeoTranslation(0, 0, z));

  // Al Housing for Support Structure
  //  ins->AddNode(alsup,1);
}

TGeoVolume* Detector::ConstructFrameGeometry()
{
  // define the media
  TGeoMedium* Vacuum = gGeoManager->GetMedium("FT0_Vacuum$");
  TGeoMedium* Al = gGeoManager->GetMedium("FT0_Aluminium$");

  // make a volume assembly for the frame
  TGeoVolumeAssembly* FT0_Frame = new TGeoVolumeAssembly("FT0_Frame");

  // define translations for the quartz radiators and PMTs
  defineTransformations();

  // approximate the frame with some rectangles
  TGeoBBox* frame1 = new TGeoBBox("frame1", frame1X / 2, frame1Y / 2, frameZ / 2);
  TGeoBBox* frame2 = new TGeoBBox("frame2", frame2X / 2, frame2Y / 2, frameZ / 2);
  TGeoBBox* quartzRadiator = new TGeoBBox("quartzRadiator", quartzRadiatorSide / 2, quartzRadiatorSide / 2, quartzRadiatorZ / 2);
  TGeoBBox* rect1 = new TGeoBBox("rect1", rect1X / 2, rect1Y / 2, frameZ / 2);
  TGeoBBox* rect2 = new TGeoBBox("rect2", rect2X / 2, rect2Y / 2 + eps, frameZ / 2 - mountZ / 2);
  TGeoBBox* rect3 = new TGeoBBox("rect3", rect3X / 2, rect3Y / 2, frameZ / 2);
  TGeoBBox* rect4 = new TGeoBBox("rect4", rect4X / 2, rect4Y / 2, frameZ / 2);
  TGeoBBox* rect5 = new TGeoBBox("rect5", rect5X / 2 + eps, rect5Y / 2, frameZ / 2);
  TGeoBBox* rect6 = new TGeoBBox("rect6", rect6X / 2 + eps, rect6Y / 2 + eps, frameZ / 2);
  TGeoBBox* rect7 = new TGeoBBox("rect7", rect7X / 2 + eps, rect7Y / 2 + eps, frameZ / 2 - mountZ / 2);
  TGeoBBox* rect8 = new TGeoBBox("rect8", rect8X / 2 + eps, rect8Y / 2 + eps, frameZ / 2);

  // PMT needs round edges
  TGeoBBox* pmtBox = new TGeoBBox("pmtBox", pmtSide / 2, pmtSide / 2, pmtZ / 2);
  TGeoBBox* pmtCornerRect = new TGeoBBox("pmtCornerRect", cornerRadius / 2, cornerRadius / 2, pmtZ / 2);
  TGeoTube* pmtCornerTube = new TGeoTube("pmtCornerTube", 0, cornerRadius, pmtZ / 2);
  TGeoVolume* PMTCorner = new TGeoVolume("PMTCorner", new TGeoCompositeShape("PMTCorner", pmtCornerCompositeShapeBoolean().c_str()), Al);
  TGeoVolume* PMT = new TGeoVolume("PMT", new TGeoCompositeShape("PMT", pmtCompositeShapeBoolean().c_str()), Vacuum);

  // add the plates on the bottom of the frame
  TGeoBBox* basicPlate = new TGeoBBox("basicPlate", plateSide / 2, plateSide / 2, basicPlateZ / 2);
  TGeoBBox* cablePlate = new TGeoBBox("cablePlate", plateSide / 2, plateSide / 2, cablePlateZ / 2);
  TGeoBBox* opticalFiberHead = new TGeoBBox("opticalFiberHead", fiberHeadX / 2, fiberHeadY / 2, cablePlateZ / 2);
  TGeoCompositeShape* opticalFiberPlate1 = new TGeoCompositeShape("opticalFiberPlate1", opticalFiberPlateCompositeShapeBoolean1().c_str());
  TGeoCompositeShape* opticalFiberPlate2 = new TGeoCompositeShape("opticalFiberPlate2", opticalFiberPlateCompositeShapeBoolean2().c_str());
  TGeoCompositeShape* plateBox = new TGeoCompositeShape("plateBox", plateBoxCompositeShapeBoolean().c_str());                                 // holds 2 basic plates and 2 cable plates
  TGeoVolume* plateGroup = new TGeoVolume("plateGroup", new TGeoCompositeShape("plateGroup", plateGroupCompositeShapeBoolean().c_str()), Al); // holds 3 plate boxes

  // remove the quartz radiators and PMTs from the frame
  TGeoCompositeShape* frameRemovedPMTandRadiators1 = new TGeoCompositeShape("frameRemovedPMTandRadiators1", frame1CompositeShapeBoolean().c_str());
  TGeoCompositeShape* frameRemovedPMTandRadiators2 = new TGeoCompositeShape("frameRemovedPMTandRadiators2", frame2CompositeShapeBoolean().c_str());

  // make the right side frame
  TGeoVolume* frame = new TGeoVolume("frame", new TGeoCompositeShape("frame", frameCompositeShapeBoolean().c_str()), Al);

  // reflection for the left side of the frame
  TGeoRotation* reflect = new TGeoRotation("reflect");
  reflect->ReflectX(true);
  reflect->ReflectY(true);
  reflect->RegisterYourself();

  // add the right and left sides to top volume
  FT0_Frame->AddNode(frame, 1);          // right side
  FT0_Frame->AddNode(frame, 2, reflect); // left side

  return FT0_Frame;
}
std::string Detector::frame1CompositeShapeBoolean()
{
  // create a string for the boolean operations for the composite frame shape
  std::string frame1CompositeShapeBoolean = "";
  frame1CompositeShapeBoolean += "((frame1";

  // remove the radiators
  frame1CompositeShapeBoolean += " - quartzRadiator:quartzRadiatorTr1";
  frame1CompositeShapeBoolean += " - quartzRadiator:quartzRadiatorTr2";
  frame1CompositeShapeBoolean += " - quartzRadiator:quartzRadiatorTr3";
  frame1CompositeShapeBoolean += " - quartzRadiator:quartzRadiatorTr4";
  frame1CompositeShapeBoolean += " - quartzRadiator:quartzRadiatorTr5";
  frame1CompositeShapeBoolean += " - quartzRadiator:quartzRadiatorTr6)";

  // remove the PMTs
  frame1CompositeShapeBoolean += " - PMT:PMTTr1";
  frame1CompositeShapeBoolean += " - PMT:PMTTr2";
  frame1CompositeShapeBoolean += " - PMT:PMTTr3";
  frame1CompositeShapeBoolean += " - PMT:PMTTr4";
  frame1CompositeShapeBoolean += " - PMT:PMTTr5";
  frame1CompositeShapeBoolean += " - PMT:PMTTr6)";

  return frame1CompositeShapeBoolean;
}

std::string Detector::frame2CompositeShapeBoolean()
{
  std::string frame2CompositeShapeBoolean = "";
  frame2CompositeShapeBoolean += "((frame2";

  // remove the radiators
  frame2CompositeShapeBoolean += " - quartzRadiator:quartzRadiatorTr7";
  frame2CompositeShapeBoolean += " - quartzRadiator:quartzRadiatorTr8";
  frame2CompositeShapeBoolean += " - quartzRadiator:quartzRadiatorTr9";
  frame2CompositeShapeBoolean += " - quartzRadiator:quartzRadiatorTr10";
  frame2CompositeShapeBoolean += " - quartzRadiator:quartzRadiatorTr11";
  frame2CompositeShapeBoolean += " - quartzRadiator:quartzRadiatorTr12)";

  // remove the PMTs
  frame2CompositeShapeBoolean += " - PMT:PMTTr7";
  frame2CompositeShapeBoolean += " - PMT:PMTTr8";
  frame2CompositeShapeBoolean += " - PMT:PMTTr9";
  frame2CompositeShapeBoolean += " - PMT:PMTTr10";
  frame2CompositeShapeBoolean += " - PMT:PMTTr11";
  frame2CompositeShapeBoolean += " - PMT:PMTTr12)";

  return frame2CompositeShapeBoolean;
}

std::string Detector::frameCompositeShapeBoolean()
{
  // create a string for the boolean operations for the composite plateGroup shape
  std::string frameCompositeShapeBoolean = "";

  // add frames 1 and 2
  frameCompositeShapeBoolean += "frameRemovedPMTandRadiators1:frameTr1";
  frameCompositeShapeBoolean += " + frameRemovedPMTandRadiators2:frameTr2";

  // add the plateGroups
  frameCompositeShapeBoolean += " + plateGroup:plateGroupTr1";
  frameCompositeShapeBoolean += " + plateGroup:plateGroupTr2";

  // subtract the extra Al
  frameCompositeShapeBoolean += " - rect1:rectTr1";
  frameCompositeShapeBoolean += " - rect2:rectTr2";
  frameCompositeShapeBoolean += " - rect3:rectTr3";
  frameCompositeShapeBoolean += " - rect4:rectTr4";
  frameCompositeShapeBoolean += " - rect5:rectTr5";
  frameCompositeShapeBoolean += " - rect6:rectTr6";
  frameCompositeShapeBoolean += " - rect7:rectTr7";
  frameCompositeShapeBoolean += " - rect8:rectTr8";

  return frameCompositeShapeBoolean;
}

std::string Detector::plateGroupCompositeShapeBoolean()
{
  // create a string for the boolean operations for the composite plateGroup shape
  std::string plateGroupCompositeShapeBoolean = "";

  // add the plateBoxes to the plateGroup
  plateGroupCompositeShapeBoolean += "plateBox:plateTr1";
  plateGroupCompositeShapeBoolean += " + plateBox:plateTr2";
  plateGroupCompositeShapeBoolean += " + plateBox:plateTr3";

  return plateGroupCompositeShapeBoolean;
}

std::string Detector::opticalFiberPlateCompositeShapeBoolean1()
{
  // create a string for the boolean operations for the composite opticalFiberPlate1 shape
  std::string opticalFiberPlateCompositeShapeBoolean1 = "";
  opticalFiberPlateCompositeShapeBoolean1 += "cablePlate";
  opticalFiberPlateCompositeShapeBoolean1 += " - opticalFiberHead:opticalFiberHeadTr1";
  opticalFiberPlateCompositeShapeBoolean1 += " - opticalFiberHead:opticalFiberHeadTr2";
  opticalFiberPlateCompositeShapeBoolean1 += " - opticalFiberHead:opticalFiberHeadTr3";
  opticalFiberPlateCompositeShapeBoolean1 += " - opticalFiberHead:opticalFiberHeadTr4";

  return opticalFiberPlateCompositeShapeBoolean1;
}

std::string Detector::opticalFiberPlateCompositeShapeBoolean2()
{
  // create a string for the boolean operations for the composite opticalFiberPlate2 shape
  std::string opticalFiberPlateCompositeShapeBoolean2 = "";

  // remove the opticalFiberHeads from the cablePlate
  opticalFiberPlateCompositeShapeBoolean2 += "cablePlate";
  opticalFiberPlateCompositeShapeBoolean2 += " - opticalFiberHead:opticalFiberHeadTr5";
  opticalFiberPlateCompositeShapeBoolean2 += " - opticalFiberHead:opticalFiberHeadTr6";
  opticalFiberPlateCompositeShapeBoolean2 += " - opticalFiberHead:opticalFiberHeadTr7";
  opticalFiberPlateCompositeShapeBoolean2 += " - opticalFiberHead:opticalFiberHeadTr8";

  return opticalFiberPlateCompositeShapeBoolean2;
}

std::string Detector::pmtCornerCompositeShapeBoolean()
{
  // create a string for the boolean operations for the composite pmtCorner shape
  std::string pmtCornerCompositeShapeBoolean = "";
  pmtCornerCompositeShapeBoolean += "pmtCornerRect:pmtCornerRectTr";
  pmtCornerCompositeShapeBoolean += " - pmtCornerTube:pmtCornerTubeTr";

  return pmtCornerCompositeShapeBoolean;
}

std::string Detector::pmtCompositeShapeBoolean()
{
  // create a string for the boolean operations for the composite PMT shape
  std::string pmtCompositeShapeBoolean = "";
  pmtCompositeShapeBoolean += "pmtBox";
  pmtCompositeShapeBoolean += " - PMTCorner:PMTCornerTr1";
  pmtCompositeShapeBoolean += " - PMTCorner:PMTCornerTr2";
  pmtCompositeShapeBoolean += " - PMTCorner:PMTCornerTr3";
  pmtCompositeShapeBoolean += " - PMTCorner:PMTCornerTr4";

  return pmtCompositeShapeBoolean;
}

std::string Detector::plateBoxCompositeShapeBoolean()
{
  // create a string for the boolean operations for the composite plateBox shape
  std::string plateBoxCompositeShapeBoolean = "";
  plateBoxCompositeShapeBoolean += "basicPlate";
  plateBoxCompositeShapeBoolean += " + basicPlate:basicPlateTr";
  plateBoxCompositeShapeBoolean += " + opticalFiberPlate1:opticalFiberPlateTr1";
  plateBoxCompositeShapeBoolean += " + opticalFiberPlate2:opticalFiberPlateTr2";

  return plateBoxCompositeShapeBoolean;
}

void Detector::defineTransformations()
{
  defineQuartzRadiatorTransformations();
  definePMTTransformations();
  definePlateTransformations();
  defineFrameTransformations();
}

void Detector::defineQuartzRadiatorTransformations()
{
  // translations for quartz radiators in frame 1
  TGeoTranslation* quartzRadiatorTr1 = new TGeoTranslation("quartzRadiatorTr1", pos1X[0], pos1Y[0], quartzHeight);
  quartzRadiatorTr1->RegisterYourself();
  TGeoTranslation* quartzRadiatorTr2 = new TGeoTranslation("quartzRadiatorTr2", pos1X[0], pos1Y[1], quartzHeight);
  quartzRadiatorTr2->RegisterYourself();
  TGeoTranslation* quartzRadiatorTr3 = new TGeoTranslation("quartzRadiatorTr3", pos1X[1], pos1Y[2], quartzHeight);
  quartzRadiatorTr3->RegisterYourself();
  TGeoTranslation* quartzRadiatorTr4 = new TGeoTranslation("quartzRadiatorTr4", pos1X[1], pos1Y[3], quartzHeight);
  quartzRadiatorTr4->RegisterYourself();
  TGeoTranslation* quartzRadiatorTr5 = new TGeoTranslation("quartzRadiatorTr5", pos1X[2], pos1Y[2], quartzHeight);
  quartzRadiatorTr5->RegisterYourself();
  TGeoTranslation* quartzRadiatorTr6 = new TGeoTranslation("quartzRadiatorTr6", pos1X[2], pos1Y[3], quartzHeight);
  quartzRadiatorTr6->RegisterYourself();

  // translations for quartz radiators in frame 2
  TGeoTranslation* quartzRadiatorTr7 = new TGeoTranslation("quartzRadiatorTr7", pos2X[0], pos2Y[0], quartzHeight);
  quartzRadiatorTr7->RegisterYourself();
  TGeoTranslation* quartzRadiatorTr8 = new TGeoTranslation("quartzRadiatorTr8", pos2X[1], pos2Y[0], quartzHeight);
  quartzRadiatorTr8->RegisterYourself();
  TGeoTranslation* quartzRadiatorTr9 = new TGeoTranslation("quartzRadiatorTr9", pos2X[2], pos2Y[1], quartzHeight);
  quartzRadiatorTr9->RegisterYourself();
  TGeoTranslation* quartzRadiatorTr10 = new TGeoTranslation("quartzRadiatorTr10", pos2X[3], pos2Y[1], quartzHeight);
  quartzRadiatorTr10->RegisterYourself();
  TGeoTranslation* quartzRadiatorTr11 = new TGeoTranslation("quartzRadiatorTr11", pos2X[2], pos2Y[2], quartzHeight);
  quartzRadiatorTr11->RegisterYourself();
  TGeoTranslation* quartzRadiatorTr12 = new TGeoTranslation("quartzRadiatorTr12", pos2X[3], pos2Y[2], quartzHeight);
  quartzRadiatorTr12->RegisterYourself();
}

void Detector::definePMTTransformations()
{
  // translations for PMTs in frame 1
  TGeoTranslation* PMTTr1 = new TGeoTranslation("PMTTr1", pos1X[0], pos1Y[0], PMTHeight);
  PMTTr1->RegisterYourself();
  TGeoTranslation* PMTTr2 = new TGeoTranslation("PMTTr2", pos1X[0], pos1Y[1], PMTHeight);
  PMTTr2->RegisterYourself();
  TGeoTranslation* PMTTr3 = new TGeoTranslation("PMTTr3", pos1X[1], pos1Y[2], PMTHeight);
  PMTTr3->RegisterYourself();
  TGeoTranslation* PMTTr4 = new TGeoTranslation("PMTTr4", pos1X[1], pos1Y[3], PMTHeight);
  PMTTr4->RegisterYourself();
  TGeoTranslation* PMTTr5 = new TGeoTranslation("PMTTr5", pos1X[2], pos1Y[2], PMTHeight);
  PMTTr5->RegisterYourself();
  TGeoTranslation* PMTTr6 = new TGeoTranslation("PMTTr6", pos1X[2], pos1Y[3], PMTHeight);
  PMTTr6->RegisterYourself();

  // translations for PMTs in frame 2
  TGeoTranslation* PMTTr7 = new TGeoTranslation("PMTTr7", pos2X[0], pos2Y[0], PMTHeight);
  PMTTr7->RegisterYourself();
  TGeoTranslation* PMTTr8 = new TGeoTranslation("PMTTr8", pos2X[1], pos2Y[0], PMTHeight);
  PMTTr8->RegisterYourself();
  TGeoTranslation* PMTTr9 = new TGeoTranslation("PMTTr9", pos2X[2], pos2Y[1], PMTHeight);
  PMTTr9->RegisterYourself();
  TGeoTranslation* PMTTr10 = new TGeoTranslation("PMTTr10", pos2X[3], pos2Y[1], PMTHeight);
  PMTTr10->RegisterYourself();
  TGeoTranslation* PMTTr11 = new TGeoTranslation("PMTTr11", pos2X[2], pos2Y[2], PMTHeight);
  PMTTr11->RegisterYourself();
  TGeoTranslation* PMTTr12 = new TGeoTranslation("PMTTr12", pos2X[3], pos2Y[2], PMTHeight);
  PMTTr12->RegisterYourself();

  // define pmtCorner transformations
  TGeoTranslation* pmtCornerTubeTr = new TGeoTranslation("pmtCornerTubeTr", pmtCornerTubePos, pmtCornerTubePos, 0);
  pmtCornerTubeTr->RegisterYourself();
  TGeoTranslation* pmtCornerRectTr = new TGeoTranslation("pmtCornerRectTr", 0, 0, 0);
  pmtCornerRectTr->RegisterYourself();
  TGeoTranslation* PMTCornerTr1 = new TGeoTranslation("PMTCornerTr1", pmtCornerPos, pmtCornerPos, 0);
  PMTCornerTr1->RegisterYourself();
  TGeoRotation* reflect2 = new TGeoRotation();
  reflect2->ReflectX(true);
  reflect2->RegisterYourself();
  TGeoCombiTrans* PMTCornerTr2 = new TGeoCombiTrans("PMTCornerTr2", -pmtCornerPos, pmtCornerPos, 0, reflect2);
  PMTCornerTr2->RegisterYourself();
  TGeoRotation* reflect3 = new TGeoRotation();
  reflect3->ReflectX(true);
  reflect3->ReflectY(true);
  reflect3->RegisterYourself();
  TGeoCombiTrans* PMTCornerTr3 = new TGeoCombiTrans("PMTCornerTr3", -pmtCornerPos, -pmtCornerPos, 0, reflect3);
  PMTCornerTr3->RegisterYourself();
  TGeoRotation* reflect4 = new TGeoRotation();
  reflect4->ReflectY(true);
  reflect4->RegisterYourself();
  TGeoCombiTrans* PMTCornerTr4 = new TGeoCombiTrans("PMTCornerTr4", pmtCornerPos, -pmtCornerPos, 0, reflect4);
  PMTCornerTr4->RegisterYourself();
  TGeoRotation* reflect5 = new TGeoRotation();
  reflect5->ReflectX(true);
  reflect5->ReflectY(true);
  reflect5->RegisterYourself();
  TGeoCombiTrans* edgeCornerTr = new TGeoCombiTrans("edgeCornerTr", edgeCornerPos[0], edgeCornerPos[1], 0, reflect5);
  edgeCornerTr->RegisterYourself();
}

void Detector::definePlateTransformations()
{
  // TODO: redefine fiber head transformations
  // define transformations for the fiber heads in opticalFiberPlate1
  TGeoTranslation* opticalFiberHeadTr1 = new TGeoTranslation("opticalFiberHeadTr1", 1.7384, 1.36, 0);
  opticalFiberHeadTr1->RegisterYourself();
  TGeoTranslation* opticalFiberHeadTr2 = new TGeoTranslation("opticalFiberHeadTr2", 1.7384, -1.36, 0);
  opticalFiberHeadTr2->RegisterYourself();
  TGeoCombiTrans* opticalFiberHeadTr3 = new TGeoCombiTrans("opticalFiberHeadTr3", -0.9252, -.9375, 0, new TGeoRotation("rot3", 15, 0, 0));
  opticalFiberHeadTr3->RegisterYourself();
  TGeoCombiTrans* opticalFiberHeadTr4 = new TGeoCombiTrans("opticalFiberHeadTr4", -0.9252, .9375, 0, new TGeoRotation("rot4", -15, 0, 0));
  opticalFiberHeadTr4->RegisterYourself();

  // make the transformations for the fiber heads in opticalFiberPlate2
  TGeoCombiTrans* opticalFiberHeadTr5 = new TGeoCombiTrans("opticalFiberHeadTr5", 1.6714, 1.525, 0, new TGeoRotation("rot5", 30, 0, 0));
  opticalFiberHeadTr5->RegisterYourself();
  TGeoCombiTrans* opticalFiberHeadTr6 = new TGeoCombiTrans("opticalFiberHeadTr6", 1.6714, -1.525, 0, new TGeoRotation("rot6", -30, 0, 0));
  opticalFiberHeadTr6->RegisterYourself();
  TGeoCombiTrans* opticalFiberHeadTr7 = new TGeoCombiTrans("opticalFiberHeadTr7", -0.9786, -1.125, 0, new TGeoRotation("rot7", 30, 0, 0));
  opticalFiberHeadTr7->RegisterYourself();
  TGeoCombiTrans* opticalFiberHeadTr8 = new TGeoCombiTrans("opticalFiberHeadTr8", -0.9786, 1.125, 0, new TGeoRotation("rot8", -30, 0, 0));
  opticalFiberHeadTr8->RegisterYourself();

  // define transformations to form a plateBox (2 basicPlates and 2 cablePlates)
  TGeoCombiTrans* basicPlateTr = new TGeoCombiTrans("basicPlateTr", 0, -plateSpacing, 0, new TGeoRotation("basicPlateRot", 90, 0, 0));
  basicPlateTr->RegisterYourself();
  TGeoCombiTrans* opticalFiberPlateTr1 = new TGeoCombiTrans("opticalFiberPlateTr1", 0, 0, opticalFiberPlateZ, new TGeoRotation("opticalFiberPlateRot1", 90, 0, 0));
  opticalFiberPlateTr1->RegisterYourself();
  TGeoCombiTrans* opticalFiberPlateTr2 = new TGeoCombiTrans("opticalFiberPlateTr2", 0, -plateSpacing, opticalFiberPlateZ, new TGeoRotation("opticalFiberPlateRot2", 90, 0, 0));
  opticalFiberPlateTr2->RegisterYourself();

  // define transformations to form a plateGroup
  TGeoTranslation* plateTr1 = new TGeoTranslation("plateTr1", -plateSpacing, plateDisplacementDeltaY, 0);
  plateTr1->RegisterYourself();
  TGeoTranslation* plateTr2 = new TGeoTranslation("plateTr2", 0, 0, 0);
  plateTr2->RegisterYourself();
  TGeoTranslation* plateTr3 = new TGeoTranslation("plateTr3", plateSpacing, 0, 0);
  plateTr3->RegisterYourself();

  // TODO: fix plateGroupTr2
  // define transformations for the plateGroups (6 basicPlates and 6 cablePlates)
  TGeoTranslation* plateGroupTr1 = new TGeoTranslation("plateGroupTr1", plateDisplacementX, plateDisplacementY, plateGroupZ);
  plateGroupTr1->RegisterYourself();
  TGeoCombiTrans* plateGroupTr2 = new TGeoCombiTrans("plateGroupTr2", 10.4358 + 1.5 * plateDisplacementDeltaY, -7.0747, plateGroupZ, new TGeoRotation("plateGroup2Rotation", -90, 0, 0));
  plateGroupTr2->RegisterYourself();
}

void Detector::defineFrameTransformations()
{
  // position of the two rectangles used to approximate the frame
  TGeoTranslation* frameTr1 = new TGeoTranslation("frameTr1", frame1PosX, frame1PosY, 0);
  frameTr1->RegisterYourself();
  TGeoTranslation* frameTr2 = new TGeoTranslation("frameTr2", frame2PosX, frame2PosY, 0);
  frameTr2->RegisterYourself();

  // remove the two smaller rectangles from the frame
  TGeoTranslation* rectTr1 = new TGeoTranslation("rectTr1", frame1PosX + 3.25, frame1PosY + 6.1875, 0);
  rectTr1->RegisterYourself();

  TGeoTranslation* rectTr2 = new TGeoTranslation("rectTr2", frame1PosX + 9.3, frame1PosY - 0.5775, mountZ / 2);
  rectTr2->RegisterYourself();

  TGeoTranslation* rectTr3 = new TGeoTranslation("rectTr3", frame1PosX + 10.75 - rect3X / 2, frame1PosY - 6.8525 + rect3Y / 2, 0);
  rectTr3->RegisterYourself();

  TGeoTranslation* rectTr4 = new TGeoTranslation("rectTr4", frame1PosX - 7.925, frame1PosY - 6.44, 0);
  rectTr4->RegisterYourself();

  TGeoTranslation* rectTr5 = new TGeoTranslation("rectTr5", frame2PosX + 6.965 - rect5X / 2, frame2PosY + 4.3625 - rect5Y / 2, 0);
  rectTr5->RegisterYourself();

  TGeoTranslation* rectTr6 = new TGeoTranslation("rectTr6", frame2PosX + 6.965 - rect6X / 2, frame2PosY - 10.7375 + rect6Y / 2, 0);
  rectTr6->RegisterYourself();

  TGeoTranslation* rectTr7 = new TGeoTranslation("rectTr7", frame2PosX + 6.965 - rect6X - rect7X / 2, frame2PosY - 10.7375 + rect7Y / 2, mountZ / 2);
  rectTr7->RegisterYourself();

  TGeoTranslation* rectTr8 = new TGeoTranslation("rectTr8", frame2PosX - 5.89 - rect8X / 2, frame2PosY + 5.1125 + rect8Y / 2, 0);
  rectTr8->RegisterYourself();
}

Bool_t Detector::ProcessHits(FairVolume* v)
{
  TVirtualMCStack* stack = fMC->GetStack();
  Int_t quadrant, mcp;
  if (fMC->IsTrackEntering()) {
    float x, y, z;
    fMC->TrackPosition(x, y, z);
    fMC->CurrentVolID(quadrant);
    fMC->CurrentVolOffID(1, mcp);
    float time = fMC->TrackTime() * 1.0e9; //time from seconds to ns
    int trackID = stack->GetCurrentTrackNumber();
    int detID = 4 * mcp + quadrant - 1;
    float etot = fMC->Etot();
    int iPart = fMC->TrackPid();
    float enDep = fMC->Edep();
    Int_t parentID = stack->GetCurrentTrack()->GetMother(0);
    if (fMC->TrackCharge()) { //charge particles for MCtrue
      AddHit(x, y, z, time, 10, trackID, detID);
    }
    if (iPart == 50000050) // If particles is photon then ...
    {
      if (RegisterPhotoE(etot)) {
        //        AddHit(x, y, z, time, enDep, trackID, detID);
        AddHit(x, y, z, time, enDep, parentID, detID);
        //	std::cout << trackID <<" parent "<<parentID<<std::endl;
      }
    }

    return kTRUE;
  }
  return kFALSE;
}

o2::ft0::HitType* Detector::AddHit(float x, float y, float z, float time, float energy, Int_t trackId, Int_t detId)
{
  mHits->emplace_back(x, y, z, time, energy, trackId, detId);
  auto stack = (o2::data::Stack*)fMC->GetStack();
  stack->addHit(GetDetId());
  return &(mHits->back());
}

void Detector::Register()
{
  // This will create a branch in the output tree called Hit, setting the last
  // parameter to kFALSE means that this collection will not be written to the file,
  // it will exist only during the simulation

  if (FairRootManager::Instance()) {
    FairRootManager::Instance()->RegisterAny(addNameTo("Hit").data(), mHits, kTRUE);
  }
}

void Detector::Reset()
{
  if (!o2::utils::ShmManager::Instance().isOperational()) {
    mHits->clear();
  }
}

void Detector::CreateMaterials()
{
  Int_t isxfld = 2;     // magneticField->Integ();
  Float_t sxmgmx = 10.; // magneticField->Max();
  // FIXME: use o2::base::Detector::initFieldTrack to init mag field params

  //   Float_t a,z,d,radl,absl,buf[1];
  // Int_t nbuf;
  // AIR

  Float_t aAir[4] = {12.0107, 14.0067, 15.9994, 39.948};
  Float_t zAir[4] = {6., 7., 8., 18.};
  Float_t wAir[4] = {0.000124, 0.755267, 0.231781, 0.012827};
  Float_t dAir = 1.20479E-3;
  Float_t dAir1 = 1.20479E-11;
  // Radiator  glass SiO2
  Float_t aglass[2] = {28.0855, 15.9994};
  Float_t zglass[2] = {14., 8.};
  Float_t wglass[2] = {1., 2.};
  Float_t dglass = 2.65;
  // MCP glass SiO2
  Float_t dglass_mcp = 1.3;
  // Ceramic   97.2% Al2O3 , 2.8% SiO2
  Float_t aCeramic[2] = {26.981539, 15.9994};
  Float_t zCeramic[2] = {13., 8.};
  Float_t wCeramic[2] = {2., 3.};
  Float_t denscer = 3.6;
  //*** Definition Of avaible FIT materials ***
  Material(11, "Aliminium$", 26.98, 13.0, 2.7, 8.9, 999);
  Mixture(1, "Vacuum$", aAir, zAir, dAir1, 4, wAir);
  Mixture(2, "Air$", aAir, zAir, dAir, 4, wAir);
  Mixture(4, "MCP glass   $", aglass, zglass, dglass_mcp, -2, wglass);
  Mixture(24, "Radiator Optical glass$", aglass, zglass, dglass, -2, wglass);
  Mixture(3, "Ceramic  $", aCeramic, zCeramic, denscer, -2, wCeramic);

  Medium(1, "Air$", 2, 0, isxfld, sxmgmx, 10., .1, 1., .003, .003);
  Medium(3, "Vacuum$", 1, 0, isxfld, sxmgmx, 10., .01, .1, .003, .003);
  Medium(4, "Ceramic$", 3, 0, isxfld, sxmgmx, 10., .01, .1, .003, .003);
  Medium(6, "Glass$", 4, 0, isxfld, sxmgmx, 10., .01, .1, .003, .003);
  Medium(7, "OpAir$", 2, 0, isxfld, sxmgmx, 10., .1, 1., .003, .003);
  Medium(15, "Aluminium$", 11, 0, isxfld, sxmgmx, 10., .01, 1., .003, .003);
  Medium(16, "OpticalGlass$", 24, 1, isxfld, sxmgmx, 10., .01, .1, .003, .003);
  Medium(19, "OpticalGlassCathode$", 24, 1, isxfld, sxmgmx, 10., .01, .1, .003, .003);
  Medium(22, "SensAir$", 2, 1, isxfld, sxmgmx, 10., .1, 1., .003, .003);
}

//-------------------------------------------------------------------
void Detector::DefineOpticalProperties()
{
  // Path of the optical properties input file
  TString inputDir;
  const char* aliceO2env = std::getenv("O2_ROOT");
  if (aliceO2env)
    inputDir = aliceO2env;
  inputDir += "/share/Detectors/FT0/files/";

  TString optPropPath = inputDir + "quartzOptProperties.txt";
  optPropPath = gSystem->ExpandPathName(optPropPath.Data()); // Expand $(ALICE_ROOT) into real system path

  Int_t result = ReadOptProperties(optPropPath.Data());
  if (result < 0) {
    // Error reading file
    LOG(ERROR) << "Could not read FIT optical properties " << result << " " << optPropPath.Data();
    return;
  }
  Int_t nBins = mPhotonEnergyD.size();
  // set QE
  mPMTeff = new TGraph(nBins, &(mPhotonEnergyD[0]), &(mQuantumEfficiency[0]));

  // Prepare pointers for arrays with constant and hardcoded values (independent on wavelength)
  FillOtherOptProperties();

  // Quick conversion from vector<Double_t> to Double_t*: photonEnergyD -> &(photonEnergyD[0])
  TVirtualMC::GetMC()->SetCerenkov(getMediumID(kOpGlass), nBins, &(mPhotonEnergyD[0]), &(mAbsorptionLength[0]),
                                   &(mEfficAll[0]), &(mRefractionIndex[0]));
  // TVirtualMC::GetMC()->SetCerenkov (getMediumID(kOpGlassCathode), kNbins, aPckov, aAbsSiO2, effCathode, rindexSiO2);
  TVirtualMC::GetMC()->SetCerenkov(getMediumID(kOpGlassCathode), nBins, &(mPhotonEnergyD[0]), &(mAbsorptionLength[0]),
                                   &(mEfficAll[0]), &(mRefractionIndex[0]));

  // Define a border for radiator optical properties
  TVirtualMC::GetMC()->DefineOpSurface("surfRd", kUnified /*kGlisur*/, kDielectric_metal, kPolished, 0.);
  TVirtualMC::GetMC()->SetMaterialProperty("surfRd", "EFFICIENCY", nBins, &(mPhotonEnergyD[0]), &(mEfficMet[0]));
  TVirtualMC::GetMC()->SetMaterialProperty("surfRd", "REFLECTIVITY", nBins, &(mPhotonEnergyD[0]), &(mReflMet[0]));
}

void Detector::FillOtherOptProperties()
{
  // Set constant values to the other arrays
  for (Int_t i = 0; i < mPhotonEnergyD.size(); i++) {
    mEfficAll.push_back(1.);
    mRindexAir.push_back(1.);
    mAbsorAir.push_back(0.3);
    mRindexCathodeNext.push_back(0.);
    mAbsorbCathodeNext.push_back(0.);
    mEfficMet.push_back(0.);
    mReflMet.push_back(1.);
  }
}

//------------------------------------------------------------------------
Bool_t Detector::RegisterPhotoE(float energy)
{
  float eff = mPMTeff->Eval(energy);
  float p = gRandom->Rndm();
  if (p > eff)
    return kFALSE;

  return kTRUE;
}

Int_t Detector::ReadOptProperties(const std::string filePath)
{
  std::ifstream infile;
  infile.open(filePath.c_str());

  // Check if file is opened correctly
  if (infile.fail() == true) {
    // AliFatal(Form("Error opening ascii file: %s", filePath.c_str()));
    return -1;
  }

  std::string comment;             // dummy, used just to read 4 first lines and move the cursor to the 5th, otherwise unused
  if (!getline(infile, comment)) { // first comment line
    //         AliFatal(Form("Error opening ascii file (it is probably a folder!): %s", filePath.c_str()));
    return -2;
  }
  getline(infile, comment); // 2nd comment line

  // Get number of elements required for the array
  Int_t nLines;
  infile >> nLines;
  if (nLines < 0 || nLines > 1e4) {
    //   AliFatal(Form("Input arraySize out of range 0..1e4: %i. Check input file: %s", kNbins, filePath.c_str()));
    return -4;
  }

  getline(infile, comment); // finish 3rd line after the nEntries are read
  getline(infile, comment); // 4th comment line

  // read the main body of the file (table of values: energy, absorption length and refractive index)
  Int_t iLine = 0;
  std::string sLine;
  getline(infile, sLine);
  while (!infile.eof()) {
    if (iLine >= nLines) {
      //      AliFatal(Form("Line number: %i reaches range of declared arraySize: %i. Check input file: %s", iLine,
      //      kNbins, filePath.c_str()));
      return -5;
    }
    std::stringstream ssLine(sLine);
    // First column:
    Double_t energy;
    ssLine >> energy;
    energy *= 1e-9; // Convert eV -> GeV immediately
    mPhotonEnergyD.push_back(energy);
    // Second column:
    Double_t absorption;
    ssLine >> absorption;
    mAbsorptionLength.push_back(absorption);
    // Third column:
    Double_t refraction;
    ssLine >> refraction;
    mRefractionIndex.push_back(refraction);
    // Fourth column:
    Double_t efficiency;
    ssLine >> efficiency;
    mQuantumEfficiency.push_back(efficiency);
    if (!(ssLine.good() || ssLine.eof())) { // check if there were problems with numbers conversion
      //    AliFatal(Form("Error while reading line %i: %s", iLine, ssLine.str().c_str()));
      return -6;
    }
    getline(infile, sLine);
    iLine++;
  }
  if (iLine != mPhotonEnergyD.size()) {
    //    AliFatal(Form("Total number of lines %i is different than declared %i. Check input file: %s", iLine, kNbins,
    //    filePath.c_str()));
    return -7;
  }

  //  AliInfo(Form("Optical properties taken from the file: %s. Number of lines read: %i",filePath.c_str(),iLine));
  return 0;
}
