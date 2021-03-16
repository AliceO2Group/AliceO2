// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <Buttons.h>
#include <TGeoCompositeShape.h>
#include <TGeoShape.h>
#include <TGeoBBox.h>
#include <TGeoTube.h>
#include <TGeoVolume.h>
#include <TMCManagerStack.h>
#include "TGeoManager.h" // for TGeoManager
#include "TMath.h"
#include "TGraph.h"
#include "TString.h"
#include "TSystem.h"
#include "TVirtualMC.h"
#include "TVector3.h"

#include "FairRootManager.h" // for FairRootManager
#include "FairLogger.h"
#include "FairVolume.h"

#include "FairRootManager.h"
#include "FairVolume.h"

#include <sstream>
#include <string>
#include "FT0Base/Geometry.h"
#include "FT0Simulation/Detector.h"
#include "SimulationDataFormat/Stack.h"

using namespace o2::ft0;
using o2::ft0::Geometry;

ClassImp(Detector);

Detector::Detector(Bool_t Active)
  : o2::base::DetImpl<Detector>("FT0", Active), mIdSens1(0), mPMTeff(nullptr), mHits(o2::utils::createSimVector<o2::ft0::HitType>()), mTrackIdTop(-1), mTrackIdMCPtop(-1)

{
  // Gegeo  = GetGeometry() ;

  //  TString gn(geo->GetName());
}

Detector::Detector(const Detector& rhs)
  : o2::base::DetImpl<Detector>(rhs), mIdSens1(rhs.mIdSens1), mPMTeff(rhs.mPMTeff), mHits(o2::utils::createSimVector<o2::ft0::HitType>()), mTrackIdTop(-1), mTrackIdMCPtop(-1)
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

  TGeoVolume* vrad = gGeoManager->GetVolume("0TOP");
  if (vrad == nullptr) {
    LOG(WARN) << "@@@@ Sensitive radiator not found!!!!!!!!";
  } else {
    AddSensitiveVolume(vrad);
  }
  TGeoVolume* vmcp = gGeoManager->GetVolume("0MTO");
  if (vmcp == nullptr) {
    LOG(WARN) << "@@@@ Sensitive MCP glass not found!!!!!!!!";
  } else {
    AddSensitiveVolume(vmcp);
  }
}

void Detector::ConstructGeometry()
{
  LOG(DEBUG) << "Creating FT0 geometry\n";
  CreateMaterials();

  Float_t zdetA = Geometry::ZdetA;
  Float_t zdetC = Geometry::ZdetC;

  Int_t idrotm[999];
  Double_t x, y, z;

  int nCellsA = Geometry::NCellsA;
  int nCellsC = Geometry::NCellsC;

  Geometry geometry;
  TVector3 centerMCP = geometry.centerMCP(2);
  Matrix(idrotm[901], 90., 0., 90., 90., 180., 0.);

  // C side Concave Geometry

  Double_t crad = Geometry::ZdetC; // define concave c-side radius here

  Double_t dP = mInStart[0]; // side length of mcp divided by 2

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
  Double_t rcomp = crad + mStartC[2] / 2.0; //
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
  TGeoVolumeAssembly* stlinA = new TGeoVolumeAssembly("0STL"); // A side mother
  TGeoVolumeAssembly* stlinC = new TGeoVolumeAssembly("0STR"); // C side mother

  // FIT interior
  TVirtualMC::GetMC()->Gsvolu("0INS", "BOX", getMediumID(kAir), mInStart, 3);
  TGeoVolume* ins = gGeoManager->GetVolume("0INS");
  //
  TGeoTranslation* tr[Geometry::NCellsA + Geometry::NCellsC];
  TString nameTr;

  // A side Translations
  for (Int_t itr = 0; itr < Geometry::NCellsA; itr++) {
    nameTr = Form("0TR%i", itr + 1);
    z = -mStartA[2] + mInStart[2];
    tr[itr] = new TGeoTranslation(nameTr.Data(), mPosModuleAx[itr], mPosModuleAy[itr], z);
    tr[itr]->RegisterYourself();
    stlinA->AddNode(ins, itr, tr[itr]);
  }
  SetCablesA(stlinA);

  TGeoRotation* rot[Geometry::NCellsC];
  TString nameRot;

  TGeoCombiTrans* com[Geometry::NCellsC];
  TGeoCombiTrans* comCable[Geometry::NCellsC];
  TString nameCom;

  // C Side Transformations
  for (Int_t itr = Geometry::NCellsA; itr < Geometry::NCellsA + Geometry::NCellsC; itr++) {
    nameTr = Form("0TR%i", itr + 1);
    nameRot = Form("0Rot%i", itr + 1);
    int ic = itr - Geometry::NCellsA;
    // nameCom = Form("0Com%i",itr+1);
    rot[ic] = new TGeoRotation(nameRot.Data(), ac[ic], bc[ic], gc[ic]);
    rot[ic]->RegisterYourself();

    //    tr[itr] = new TGeoTranslation(nameTr.Data(), xc2[ic], yc2[ic], (zc2[ic] - 80.));
    // tr[itr]->RegisterYourself();
    com[ic] = new TGeoCombiTrans(xc2[ic], yc2[ic], (zc2[ic] - 80), rot[ic]);
    //    com[ic] = new TGeoCombiTrans(tr[itr], rot[ic]);
    mPosModuleCx[ic] = xc2[ic];
    mPosModuleCy[ic] = yc2[ic];
    mPosModuleCz[ic] = zc2[ic] - 80;

    TGeoHMatrix hm = *com[ic];
    TGeoHMatrix* ph = new TGeoHMatrix(hm);
    stlinC->AddNode(ins, itr, ph);
    //cables
    TGeoVolume* cables = SetCablesSize(itr);
    comCable[ic] = new TGeoCombiTrans(mPosModuleCx[ic], mPosModuleCy[ic], mPosModuleCz[ic] + mInStart[2] + 0.2, rot[ic]);
    TGeoHMatrix hmCable = *comCable[ic];
    TGeoHMatrix* phCable = new TGeoHMatrix(hmCable);
    stlinC->AddNode(cables, itr, comCable[ic]);
  }

  //Add FT0-A support Structure to the geometry
  stlinA->AddNode(constructFrameGeometry(), 1, new TGeoTranslation(0, 0, -mStartA[2] + mInStart[2]));

  TGeoVolume* alice = gGeoManager->GetVolume("barrel");
  alice->AddNode(stlinA, 1, new TGeoTranslation(0, 30., zdetA));
  TGeoRotation* rotC = new TGeoRotation("rotC", 90., 0., 90., 90., 180., 0.);
  alice->AddNode(stlinC, 1, new TGeoCombiTrans(0., 30., -zdetC, rotC));

  // MCP + 4 x wrapped radiator + 4xphotocathod + MCP + Al top in front of radiators
  SetOneMCP(ins);
  //SetCablesC(stlinC);
}

void Detector::ConstructOpGeometry()
{
  LOG(DEBUG) << "Creating FIT optical geometry properties";

  DefineOpticalProperties();
  DefineSim2LUTindex();
}

//_________________________________________
void Detector::SetOneMCP(TGeoVolume* ins)
{

  Double_t x, y, z;

  Float_t ptop[3] = {1.324, 1.324, 1.};      // Cherenkov radiator
  Float_t ptopref[3] = {1.3241, 1.3241, 1.}; // Cherenkov radiator wrapped with reflector
  Double_t prfv[3] = {0.0002, 1.323, 1.};    // Vertical refracting layer bettwen radiators and between radiator and not optical Air
  Double_t prfh[3] = {1.323, 0.0002, 1.};    // Horizontal refracting layer bettwen radiators and ...
  Float_t pmcp[3] = {2.949, 2.949, 0.66};    // MCP
  Float_t pmcpinner[3] = {2.749, 2.749, 0.1};
  Float_t pmcpbase[3] = {2.949, 2.949, 0.675};
  Float_t pmcpside[3] = {0.15, 2.949, 0.65};
  Float_t pmcptopglass[3] = {2.949, 2.949, 0.1}; // MCP top glass optical
  Float_t preg[3] = {1.324, 1.324, 0.005};       // Photcathode
  Double_t pal[3] = {2.648, 2.648, 0.25};        // 5mm Al on top of each radiator

  // Entry window (glass)
  TVirtualMC::GetMC()->Gsvolu("0TOP", "BOX", getMediumID(kOpGlass), ptop, 3); // Glass radiator
  TGeoVolume* top = gGeoManager->GetVolume("0TOP");
  top->Print();
  // TVirtualMC::GetMC()->Gsvolu("0TBL", "BOX", getMediumID(kOptBlack), ptopblack, 3); // Glass radiator
  //  TGeoVolume* topblack = gGeoManager->GetVolume("0TBL");
  TVirtualMC::GetMC()->Gsvolu("0TRE", "BOX", getMediumID(kAir), ptopref, 3); // Air: wrapped  radiator
  TGeoVolume* topref = gGeoManager->GetVolume("0TRE");
  TVirtualMC::GetMC()->Gsvolu("0RFV", "BOX", getMediumID(kOptAl), prfv, 3); // Optical Air vertical
  TGeoVolume* rfv = gGeoManager->GetVolume("0RFV");
  TVirtualMC::GetMC()->Gsvolu("0RFH", "BOX", getMediumID(kOptAl), prfh, 3); // Optical Air horizontal
  TGeoVolume* rfh = gGeoManager->GetVolume("0RFH");

  TVirtualMC::GetMC()->Gsvolu("0REG", "BOX", getMediumID(kOpGlassCathode), preg, 3);
  TGeoVolume* cat = gGeoManager->GetVolume("0REG");

  //wrapped radiator +  reflecting layers

  Int_t ntops = 0, nrfvs = 0, nrfhs = 0;
  x = y = z = 0;
  topref->AddNode(top, 1, new TGeoTranslation(0, 0, 0));
  float xinv = -ptop[0] - prfv[0];
  topref->AddNode(rfv, 1, new TGeoTranslation(xinv, 0, 0));
  xinv = ptop[0] + prfv[0];
  topref->AddNode(rfv, 2, new TGeoTranslation(xinv, 0, 0));
  float yinv = -ptop[1] - prfh[1];
  topref->AddNode(rfh, 1, new TGeoTranslation(0, yinv, 0));
  yinv = ptop[1] + prfh[1];
  topref->AddNode(rfh, 2, new TGeoTranslation(0, yinv, 0));

  //container for radiator, cathode
  for (Int_t ix = 0; ix < 2; ix++) {
    float xin = -mInStart[0] + 0.3 + (ix + 0.5) * 2 * ptopref[0];
    for (Int_t iy = 0; iy < 2; iy++) {
      z = -mInStart[2] + ptopref[2];
      float yin = -mInStart[1] + 0.3 + (iy + 0.5) * 2 * ptopref[1];
      ntops++;
      ins->AddNode(topref, ntops, new TGeoTranslation(xin, yin, z));
      z += ptopref[2] + 2. * pmcptopglass[2] + preg[2];
      ins->AddNode(cat, ntops, new TGeoTranslation(xin, yin, z));
      LOG(INFO) << " n " << ntops << " x " << xin << " y " << yin;
    }
  }
  // MCP
  TVirtualMC::GetMC()->Gsvolu("0MTO", "BOX", getMediumID(kOpGlass), pmcptopglass, 3); //Op  Glass
  TGeoVolume* mcptop = gGeoManager->GetVolume("0MTO");
  mcptop->Print();
  z = -mInStart[2] + 2 * ptopref[2] + pmcptopglass[2];
  ins->AddNode(mcptop, 1, new TGeoTranslation(0, 0, z));

  TVirtualMC::GetMC()->Gsvolu("0MCP", "BOX", getMediumID(kAir), pmcp, 3); //glass
  TGeoVolume* mcp = gGeoManager->GetVolume("0MCP");
  z = -mInStart[2] + 2 * ptopref[2] + 2 * pmcptopglass[2] + 2 * preg[2] + pmcp[2];
  ins->AddNode(mcp, 1, new TGeoTranslation(0, 0, z));

  TVirtualMC::GetMC()->Gsvolu("0MSI", "BOX", getMediumID(kMCPwalls), pmcpside, 3); //glass
  TGeoVolume* mcpside = gGeoManager->GetVolume("0MSI");
  mcpside->Print();
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
  mcpbase->Print();
  z = -mInStart[2] + 2 * ptopref[2] + pmcptopglass[2] + 2 * pmcp[2] + pmcpbase[2];
  ins->AddNode(mcpbase, 1, new TGeoTranslation(0, 0, z));
}

//----------------------------------
void Detector::SetCablesA(TGeoVolume* stl)
{

  float pcableplane[3] = {20, 20, 0.25}; //

  TVirtualMC::GetMC()->Gsvolu("0CAA", "BOX", getMediumID(kAir), pcableplane, 3); //container for cables
  TGeoVolume* cableplane = gGeoManager->GetVolume("0CAA");
  //  float zcableplane = -mStartA[2] + 2 * mInStart[2] + pcableplane[2];
  int na = 0;

  double xcell[24], ycell[24];

  for (int imcp = 0; imcp < 24; imcp++) {
    xcell[na] = mPosModuleAx[imcp];
    ycell[na] = mPosModuleAy[imcp];
    TGeoVolume* vol = SetCablesSize(imcp);
    cableplane->AddNode(vol, na, new TGeoTranslation(xcell[na], ycell[na], 0));
    na++;
  }

  //12 cables extending beyond the frame
  Float_t pcablesextend[3] = {2, 15, 0.245};
  Float_t pcablesextendsmall[3] = {3, 2, 0.245};
  Float_t* ppcablesextend[] = {pcablesextend, pcablesextend, pcablesextendsmall, pcablesextendsmall};
  //left side
  double xcell_side[] = {-mStartA[0] + pcablesextend[0], mStartA[0] - pcablesextend[0], 0, 0};
  double ycell_side[] = {0, 0, -mStartA[1] + pcablesextendsmall[1], mStartA[1] - pcablesextendsmall[1]};

  for (int icab = 0; icab < 4; icab++) {
    const std::string volName = Form("CAB%2.i", 52 + icab);
    TVirtualMC::GetMC()->Gsvolu(volName.c_str(), " BOX", getMediumID(kCable), ppcablesextend[icab], 3); // cables
    TGeoVolume* vol = gGeoManager->GetVolume(volName.c_str());
    cableplane->AddNode(vol, 1, new TGeoTranslation(xcell_side[icab], ycell_side[icab], 0));
  }
  float zcableplane = mStartA[2] - pcableplane[2] - 3;
  stl->AddNode(cableplane, 1, new TGeoTranslation(0, 0, zcableplane));
}
//------------------------------------------

TGeoVolume* Detector::SetCablesSize(int mod)
{
  int na = 0;
  int ncells = Geometry::NCellsC;
  int mcpcables[52] = {2, 1, 2, 1, 2,
                       2, 1, 1, 1, 2,
                       2, 1, 1, 2,
                       2, 1, 1, 1, 2,
                       2, 1, 2, 1, 2,
                       2, 1, 1, 2,
                       3, 2, 1, 1, 2, 3,
                       2, 1, 1, 2,
                       2, 1, 1, 2,
                       3, 2, 1, 1, 2, 3,
                       2, 1, 1, 2};
  // cable D=0.257cm, Weight: 13 lbs/1000ft = 0.197g/cm; 1 piece 0.65cm
  //1st 8 pieces - tube  8*0.65cm = 5.2cm; V = 0.0531cm2 -> box {0.27*0.27*1}cm; W = 0.66g
  //2nd 24 pieces 24*0.65cm; V = 0.76 -> {0.44, 0.447 1}; W = 3.07g
  //3d  48  pieces  48*0.65cm;  V = 1.53cm^3; ->box {0.66, 0.66, 1.}; W= 6.14g
  double xcell[ncells], ycell[ncells], zcell[ncells];
  float xsize[3] = {1.8, 1.8, 2.6}; //
  float ysize[3] = {0.6, 1.7, 2.};
  float zsize[3] = {0.1, 0.1, 0.1};
  //  for (int imcp = 0; imcp < Geometry::NCellsC; imcp++) {
  int ic = mcpcables[mod];
  float calblesize[3];
  calblesize[0] = xsize[ic - 1];
  calblesize[1] = ysize[ic - 1];
  calblesize[2] = zsize[ic - 1];
  const std::string volName = Form("CAB%2.i", mod);
  TVirtualMC::GetMC()->Gsvolu(volName.c_str(), "BOX", getMediumID(kCable), calblesize, 3); // cables
  TGeoVolume* vol = gGeoManager->GetVolume(volName.c_str());
  //  vol->Print();
  //  vol->Weight();
  return vol;
}

// Class wrapper for construction of FT0-A support structure
// The frame is constructed by defining two aluminum boxes that are placed in an L-shape,
// with material sequentially removed to re-create the CAD drawings,
// including sockets defined by the parameters of the sensitive elements that they are placed into
// Two L-shaped elements form the full support structure, with one reflected about the axes of symmetry
// First written by Joe Crowley and revised by Jason Pruitt from Cal Poly in 2019-2021
TGeoVolume* Detector::constructFrameGeometry()
{
  // define the media
  TGeoMedium* Vacuum = gGeoManager->GetMedium("FT0_Vacuum$");
  TGeoMedium* Al = gGeoManager->GetMedium("FT0_Aluminium$");

  // make a volume assembly for the frame
  TGeoVolumeAssembly* FT0_Frame = new TGeoVolumeAssembly("FT0_Frame");

  // define translations for the quartz radiator and PMT sockets
  defineTransformations();

  // frame1 and frame2 are rectangles that approximate the outline of one L
  // shape of the frame
  TGeoBBox* frame1 = new TGeoBBox("frame1", sFrame1X / 2, sFrame1Y / 2, sFrameZ / 2);
  TGeoBBox* frame2 = new TGeoBBox("frame2", sFrame2X / 2, sFrame2Y / 2, sFrameZ / 2);

  // the following elements are subtracted from frame1 and frame2 to better
  // approximate the CAD shape
  TGeoBBox* rect1 = new TGeoBBox("rect1", sRect1X / 2, sRect1Y / 2, sFrameZ / 2);
  TGeoBBox* rect2 = new TGeoBBox("rect2", sRect2X / 2, sRect2Y / 2 + sEps, sFrameZ / 2 - sMountZ / 2);
  TGeoBBox* rect3 = new TGeoBBox("rect3", sRect3X / 2, sRect3Y / 2, sFrameZ / 2);
  TGeoBBox* rect4 = new TGeoBBox("rect4", sRect4X / 2, sRect4Y / 2, sFrameZ / 2);
  TGeoBBox* rect5 = new TGeoBBox("rect5", sRect5X / 2 + sEps, sRect5Y / 2 + sEps, sFrameZ / 2 + sEps);
  TGeoBBox* rect6 = new TGeoBBox("rect6", sRect6X / 2 + sEps, sRect6Y / 2 + sEps, sFrameZ / 2 + sEps);
  TGeoBBox* rect7 = new TGeoBBox("rect7", sRect7X / 2 + sEps, sRect7Y / 2 + sEps, sFrameZ / 2 - sMountZ / 2 + sEps);
  TGeoBBox* rect8 = new TGeoBBox("rect8", sRect8X / 2 + sEps, sRect8Y / 2 + sEps, sFrameZ / 2 + sEps);

  // Define a value to overcut the coincidence between the closure of the tube
  // and the edge of the rectangle to eliminate artifacts
  Double_t flopsErr = .00001;
  // PMT and quartz radiator shapes provide the dimensions of the sockets to be subtracted
  // from the frame that will make room for a sensitive element to fit
  TGeoBBox* quartzRadiator = new TGeoBBox("quartzRadiator", sQuartzRadiatorSide / 2, sQuartzRadiatorSide / 2, sQuartzRadiatorZ / 2);
  TGeoBBox* pmtBox = new TGeoBBox("pmtBox", sPmtSide / 2 + sEps, sPmtSide / 2 + sEps, sPmtZ / 2 + sEps);

  // these two shapes create a subtraction so that the corners of the holes that
  // seat the sens elements are rounded
  TGeoBBox* pmtCornerRect = new TGeoBBox("pmtCornerRect", sCornerRadius / 2 - flopsErr, sCornerRadius / 2 - flopsErr, sPmtZ / 2);
  TGeoTube* pmtCornerTube = new TGeoTube("pmtCornerTube", 0, sCornerRadius, sPmtZ / 2 + sEps);
  TGeoVolume* PMTCorner = new TGeoVolume("PMTCorner", new TGeoCompositeShape("PMTCorner", pmtCornerCompositeShapeBoolean().c_str()), Al);
  // TGeoVolume* PMT = new TGeoVolume("PMT", new TGeoCompositeShape("PMT", pmtCompositeShapeBoolean().c_str()), Vacuum);
  TGeoVolume* PMT = gGeoManager->MakeBox("PMT", Vacuum, sPmtSide / 2 + sEps, sPmtSide / 2 + sEps, sPmtZ / 2 + sEps);

  // add the plates on the bottom of the frame
  TGeoBBox* basicPlate = new TGeoBBox("basicPlate", sPlateSide / 2, sPlateSide / 2, sBasicPlateZ / 2);
  TGeoBBox* cablePlate = new TGeoBBox("cablePlate", sPlateSide / 2, sPlateSide / 2, sCablePlateZ / 2);
  TGeoBBox* opticalFiberHead = new TGeoBBox("opticalFiberHead", sFiberHeadX / 2, sFiberHeadY / 2, sCablePlateZ / 2);
  TGeoCompositeShape* opticalFiberPlate1 = new TGeoCompositeShape("opticalFiberPlate1", opticalFiberPlateCompositeShapeBoolean1().c_str());
  TGeoCompositeShape* opticalFiberPlate2 = new TGeoCompositeShape("opticalFiberPlate2", opticalFiberPlateCompositeShapeBoolean2().c_str());
  TGeoCompositeShape* plateBox = new TGeoCompositeShape("plateBox", plateBoxCompositeShapeBoolean().c_str());
  // holds 2 basic plates and 2 cable plates
  TGeoVolume* plateGroup = new TGeoVolume("plateGroup", new TGeoCompositeShape("plateGroup", plateGroupCompositeShapeBoolean().c_str()), Al); // holds 3 plate boxes
  // remove the material to form the sockets for the quartz radiators and PMTs
  TGeoCompositeShape* frameRemovedPMTandRadiators1 = new TGeoCompositeShape("frameRemovedPMTandRadiators1", frame1CompositeShapeBoolean().c_str());
  TGeoCompositeShape* frameRemovedPMTandRadiators2 = new TGeoCompositeShape("frameRemovedPMTandRadiators2", frame2CompositeShapeBoolean().c_str());

  // make the right side frame - L shape
  TGeoVolume* frame = new TGeoVolume("frame", new TGeoCompositeShape("frame", frameCompositeShapeBoolean().c_str()), Al);

  // reflection for the left side of the frame
  TGeoRotation* reflect = new TGeoRotation("reflect");
  reflect->ReflectX(true);
  reflect->ReflectY(true);
  reflect->RegisterYourself();

  // add a shift to eliminate overlaps between sens elements and frame sockets
  // this shift will apply to both sides of the frame
  TGeoTranslation* xshift = new TGeoTranslation("xshift", .1028, 0, 0);

  // add the right and left sides to top volume
  FT0_Frame->AddNode(frame, 1, xshift);  // right side
  FT0_Frame->AddNode(frame, 2, reflect); // left side

  return FT0_Frame;
}

// the following are continually concatenated strings that ROOT Geometry will
// read in order to piece together the objects and translations that are
// defined above (what ROOT Geometry calls Booleans)
// frame1 is a horizontal aluminum box piece of the L-shape
std::string Detector::frame1CompositeShapeBoolean()
{
  // create a string for the boolean operations for the composite frame shape
  std::string frame1CompositeShapeBoolean = "";
  frame1CompositeShapeBoolean += "((frame1";

  // remove the radiator shapes for the sockets
  // frame1 is the horizontal piece of the right-hand L-shape (looking from back)
  // with its own internal numbering for the sockets.  To more easily map between
  // the sensitive elements and their socket locations, we've included the correspondence
  // between them.  Within the horizontal piece, the sockets are numbered column by column
  // from left to right
  // ---------
  // |       |                 <-----Rectangle 1 removed here
  // |   1   |----------------- ^
  // |       |        |       | |
  // ---------    3   |   5   | |
  // |       |        |       | | Rectangle 2 removed here
  // |   2   |----------------- |
  // |       |        |       | |
  // --------|    4   |   6   | |
  //         |        |       | v
  //    ^    ------------------  <------Rectangle 3 removed here
  //    |
  //    |
  //    Rectangle 4 removed here
  //
  // internal numbering for each is mapped to the sensitive element numbering
  // for ease of comparison and identification
  // Since one L is reflected about the axes of symmetry, the correspondence with
  // sensitive element numbering for the left-side L-shape is also included here.
  frame1CompositeShapeBoolean += " - quartzRadiator:quartzRadiatorTr1";  //Sens Elmt 2,21
  frame1CompositeShapeBoolean += " - quartzRadiator:quartzRadiatorTr2";  //Sens Elmt 7,16
  frame1CompositeShapeBoolean += " - quartzRadiator:quartzRadiatorTr3";  //Sens Elmt 3,20
  frame1CompositeShapeBoolean += " - quartzRadiator:quartzRadiatorTr4";  //Sens Elmt 8,15
  frame1CompositeShapeBoolean += " - quartzRadiator:quartzRadiatorTr5";  //Sens Elmt 4,19
  frame1CompositeShapeBoolean += " - quartzRadiator:quartzRadiatorTr6)"; //Sens Elmt 9,14

  // remove the PMT shapes for the sockets
  frame1CompositeShapeBoolean += " - PMT:PMTTr1";
  frame1CompositeShapeBoolean += " - PMT:PMTTr2";
  frame1CompositeShapeBoolean += " - PMT:PMTTr3";
  frame1CompositeShapeBoolean += " - PMT:PMTTr4";
  frame1CompositeShapeBoolean += " - PMT:PMTTr5";
  frame1CompositeShapeBoolean += " - PMT:PMTTr6)";

  return frame1CompositeShapeBoolean;
}

// frame2 is the vertical aluminum box piece of the L-shape
std::string Detector::frame2CompositeShapeBoolean()
{
  std::string frame2CompositeShapeBoolean = "";
  frame2CompositeShapeBoolean += "((frame2";

  // remove the radiator shapes for the sockets
  // frame2 is the vertical piece of the right-hand L-shape (looking from back)
  // with its own internal numbering for the sockets.  To more easily map between
  // the sensitive elements and their socket locations, we've included the correspondence
  // between them.  Within the vertical piece, the sockets are numbered row by row
  // from right to left
  //                  -----------------
  //                  |       |       |
  //  Rectangle-->    |   8   |   7   |
  //     8            |       |       |
  //  removed      --------------------
  //    here       |       |       | ^
  //               |  10   |   9   | |
  //               |       |       | | Rectangle 5 removed here
  //               ----------------- |
  //               |       |       | |
  //               |  12   |  11   | v
  //               |       |       |  <-----Rectangle 6 removed here
  //               -----------------
  //              <---------------->
  //            Rectangle 7 removed here
  //
  // internal numbering for each is mapped to the sensitive element numbering
  // for ease of comparison and identification
  // Since one L is reflected about the axes of symmetry, the correspondence with
  // sensitive element numbering for the left-side L-shape is also included here.
  frame2CompositeShapeBoolean += " - quartzRadiator:quartzRadiatorTr7";   //Sens Elmt 13,10
  frame2CompositeShapeBoolean += " - quartzRadiator:quartzRadiatorTr8";   //Sens Elmt 12,11
  frame2CompositeShapeBoolean += " - quartzRadiator:quartzRadiatorTr9";   //Sens Elmt 18,14
  frame2CompositeShapeBoolean += " - quartzRadiator:quartzRadiatorTr10";  //Sens Elmt 17,15
  frame2CompositeShapeBoolean += " - quartzRadiator:quartzRadiatorTr11";  //Sens Elmt 23,0
  frame2CompositeShapeBoolean += " - quartzRadiator:quartzRadiatorTr12)"; //Sens Elmt 22,1

  // remove the PMT shapes for the sockets
  frame2CompositeShapeBoolean += " - PMT:PMTTr7";
  frame2CompositeShapeBoolean += " - PMT:PMTTr8";
  frame2CompositeShapeBoolean += " - PMT:PMTTr9";
  frame2CompositeShapeBoolean += " - PMT:PMTTr10";
  frame2CompositeShapeBoolean += " - PMT:PMTTr11";
  frame2CompositeShapeBoolean += " - PMT:PMTTr12)";

  return frame2CompositeShapeBoolean;
}
//Support structure L-shape element definition
std::string Detector::frameCompositeShapeBoolean()
{
  // create a string for the boolean operations for the composite plateGroup shape
  std::string frameCompositeShapeBoolean = "";

  // add the two pieces called frame 1 and 2 into a single L-shaped element
  frameCompositeShapeBoolean += "frameRemovedPMTandRadiators1:frameTr1";
  frameCompositeShapeBoolean += " + frameRemovedPMTandRadiators2:frameTr2";

  // add the plateGroups to the L-shaped elements
  frameCompositeShapeBoolean += " + plateGroup:plateGroupTr1";
  frameCompositeShapeBoolean += " + plateGroup:plateGroupTr2";

  // subtract the extra Al from the L-shaped elements
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

//Plate group elements
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

//Optical fiber plate for the first aluminum box in the L-shaped element
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

//Optical fiber plate for the second aluminum box in the L-shaped element
std::string Detector::opticalFiberPlateCompositeShapeBoolean2()
{
  // create a string for the boolean operations for the composite opticalFiberPlate2 shape
  std::string opticalFiberPlateCompositeShapeBoolean2 = "";

  // remove the opticalFiberHead shapes from the cablePlate
  opticalFiberPlateCompositeShapeBoolean2 += "cablePlate";
  opticalFiberPlateCompositeShapeBoolean2 += " - opticalFiberHead:opticalFiberHeadTr5";
  opticalFiberPlateCompositeShapeBoolean2 += " - opticalFiberHead:opticalFiberHeadTr6";
  opticalFiberPlateCompositeShapeBoolean2 += " - opticalFiberHead:opticalFiberHeadTr7";
  opticalFiberPlateCompositeShapeBoolean2 += " - opticalFiberHead:opticalFiberHeadTr8";

  return opticalFiberPlateCompositeShapeBoolean2;
}
//Create rounded PMT socket corners
std::string Detector::pmtCornerCompositeShapeBoolean()
{
  // create a string for the boolean operations for the composite pmtCorner shape
  std::string pmtCornerCompositeShapeBoolean = "";
  pmtCornerCompositeShapeBoolean += "pmtCornerRect:pmtCornerRectTr";
  pmtCornerCompositeShapeBoolean += " - pmtCornerTube:pmtCornerTubeTr";

  return pmtCornerCompositeShapeBoolean;
}

//Create PMT socket shape
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
//Plate composite structure
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

//Wrapper function to define all support structure transformations at once
void Detector::defineTransformations()
{
  defineQuartzRadiatorTransformations();
  definePmtTransformations();
  definePlateTransformations();
  defineFrameTransformations();
}

//Transformations for quartz radiator sockets
void Detector::defineQuartzRadiatorTransformations()
{
  // translations for quartz radiator shapes to be removed from the frame2 pice of the L-shaped element
  TGeoTranslation* quartzRadiatorTr1 = new TGeoTranslation("quartzRadiatorTr1", sPos1X[0], sPos1Y[0], sQuartzHeight);
  quartzRadiatorTr1->RegisterYourself();
  TGeoTranslation* quartzRadiatorTr2 = new TGeoTranslation("quartzRadiatorTr2", sPos1X[0], sPos1Y[1], sQuartzHeight);
  quartzRadiatorTr2->RegisterYourself();
  TGeoTranslation* quartzRadiatorTr3 = new TGeoTranslation("quartzRadiatorTr3", sPos1X[1], sPos1Y[2], sQuartzHeight);
  quartzRadiatorTr3->RegisterYourself();
  TGeoTranslation* quartzRadiatorTr4 = new TGeoTranslation("quartzRadiatorTr4", sPos1X[1], sPos1Y[3], sQuartzHeight);
  quartzRadiatorTr4->RegisterYourself();
  TGeoTranslation* quartzRadiatorTr5 = new TGeoTranslation("quartzRadiatorTr5", sPos1X[2], sPos1Y[2], sQuartzHeight);
  quartzRadiatorTr5->RegisterYourself();
  TGeoTranslation* quartzRadiatorTr6 = new TGeoTranslation("quartzRadiatorTr6", sPos1X[2], sPos1Y[3], sQuartzHeight);
  quartzRadiatorTr6->RegisterYourself();

  // translations for quartz radiator shapes to be removed from the frame1 piece of the L-shaped element
  TGeoTranslation* quartzRadiatorTr7 = new TGeoTranslation("quartzRadiatorTr7", sPos2X[0], sPos2Y[0], sQuartzHeight);
  quartzRadiatorTr7->RegisterYourself();
  TGeoTranslation* quartzRadiatorTr8 = new TGeoTranslation("quartzRadiatorTr8", sPos2X[1], sPos2Y[0], sQuartzHeight);
  quartzRadiatorTr8->RegisterYourself();
  TGeoTranslation* quartzRadiatorTr9 = new TGeoTranslation("quartzRadiatorTr9", sPos2X[2], sPos2Y[1], sQuartzHeight);
  quartzRadiatorTr9->RegisterYourself();
  TGeoTranslation* quartzRadiatorTr10 = new TGeoTranslation("quartzRadiatorTr10", sPos2X[3], sPos2Y[1], sQuartzHeight);
  quartzRadiatorTr10->RegisterYourself();
  TGeoTranslation* quartzRadiatorTr11 = new TGeoTranslation("quartzRadiatorTr11", sPos2X[2], sPos2Y[2], sQuartzHeight);
  quartzRadiatorTr11->RegisterYourself();
  TGeoTranslation* quartzRadiatorTr12 = new TGeoTranslation("quartzRadiatorTr12", sPos2X[3], sPos2Y[2], sQuartzHeight);
  quartzRadiatorTr12->RegisterYourself();
}
//Transformations for PMT sockets, including rounded corners
void Detector::definePmtTransformations()
{
  // translations for PMT shapes to be removed from the frame2 piece in the L-shaped element
  TGeoTranslation* PMTTr1 = new TGeoTranslation("PMTTr1", sPos1X[0], sPos1Y[0], sPmtHeight);
  PMTTr1->RegisterYourself();
  TGeoTranslation* PMTTr2 = new TGeoTranslation("PMTTr2", sPos1X[0], sPos1Y[1], sPmtHeight);
  PMTTr2->RegisterYourself();
  TGeoTranslation* PMTTr3 = new TGeoTranslation("PMTTr3", sPos1X[1], sPos1Y[2], sPmtHeight);
  PMTTr3->RegisterYourself();
  TGeoTranslation* PMTTr4 = new TGeoTranslation("PMTTr4", sPos1X[1], sPos1Y[3], sPmtHeight);
  PMTTr4->RegisterYourself();
  TGeoTranslation* PMTTr5 = new TGeoTranslation("PMTTr5", sPos1X[2], sPos1Y[2], sPmtHeight);
  PMTTr5->RegisterYourself();
  TGeoTranslation* PMTTr6 = new TGeoTranslation("PMTTr6", sPos1X[2], sPos1Y[3], sPmtHeight);
  PMTTr6->RegisterYourself();

  // translations for PMT shapes to be removed from the frame1 piece in the L-shaped element
  TGeoTranslation* PMTTr7 = new TGeoTranslation("PMTTr7", sPos2X[0], sPos2Y[0], sPmtHeight);
  PMTTr7->RegisterYourself();
  TGeoTranslation* PMTTr8 = new TGeoTranslation("PMTTr8", sPos2X[1], sPos2Y[0], sPmtHeight);
  PMTTr8->RegisterYourself();
  TGeoTranslation* PMTTr9 = new TGeoTranslation("PMTTr9", sPos2X[2], sPos2Y[1], sPmtHeight);
  PMTTr9->RegisterYourself();
  TGeoTranslation* PMTTr10 = new TGeoTranslation("PMTTr10", sPos2X[3], sPos2Y[1], sPmtHeight);
  PMTTr10->RegisterYourself();
  TGeoTranslation* PMTTr11 = new TGeoTranslation("PMTTr11", sPos2X[2], sPos2Y[2], sPmtHeight);
  PMTTr11->RegisterYourself();
  TGeoTranslation* PMTTr12 = new TGeoTranslation("PMTTr12", sPos2X[3], sPos2Y[2], sPmtHeight);
  PMTTr12->RegisterYourself();

  // define pmtCorner transformations
  TGeoTranslation* pmtCornerTubeTr = new TGeoTranslation("pmtCornerTubeTr", sPmtCornerTubePos, sPmtCornerTubePos, 0);
  pmtCornerTubeTr->RegisterYourself();
  TGeoTranslation* pmtCornerRectTr = new TGeoTranslation("pmtCornerRectTr", 0, 0, 0);
  pmtCornerRectTr->RegisterYourself();
  TGeoTranslation* PMTCornerTr1 = new TGeoTranslation("PMTCornerTr1", sPmtCornerPos, sPmtCornerPos, 0);
  PMTCornerTr1->RegisterYourself();
  TGeoRotation* reflect2 = new TGeoRotation();
  reflect2->ReflectX(true);
  reflect2->RegisterYourself();
  TGeoCombiTrans* PMTCornerTr2 = new TGeoCombiTrans("PMTCornerTr2", -sPmtCornerPos, sPmtCornerPos, 0, reflect2);
  PMTCornerTr2->RegisterYourself();
  TGeoRotation* reflect3 = new TGeoRotation();
  reflect3->ReflectX(true);
  reflect3->ReflectY(true);
  reflect3->RegisterYourself();
  TGeoCombiTrans* PMTCornerTr3 = new TGeoCombiTrans("PMTCornerTr3", -sPmtCornerPos, -sPmtCornerPos, 0, reflect3);
  PMTCornerTr3->RegisterYourself();
  TGeoRotation* reflect4 = new TGeoRotation();
  reflect4->ReflectY(true);
  reflect4->RegisterYourself();
  TGeoCombiTrans* PMTCornerTr4 = new TGeoCombiTrans("PMTCornerTr4", sPmtCornerPos, -sPmtCornerPos, 0, reflect4);
  PMTCornerTr4->RegisterYourself();
  TGeoRotation* reflect5 = new TGeoRotation();
  reflect5->ReflectX(true);
  reflect5->ReflectY(true);
  reflect5->RegisterYourself();
  TGeoCombiTrans* edgeCornerTr = new TGeoCombiTrans("edgeCornerTr", sEdgeCornerPos[0], sEdgeCornerPos[1], 0, reflect5);
  edgeCornerTr->RegisterYourself();
}
//Transformations for plate elements
void Detector::definePlateTransformations()
{
  // TODO: redefine fiber head transformations
  // TODO: move hard-coded numbers to be variables in the constants lists
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
  TGeoCombiTrans* basicPlateTr = new TGeoCombiTrans("basicPlateTr", 0, -sPlateSpacing, 0, new TGeoRotation("basicPlateRot", 90, 0, 0));
  basicPlateTr->RegisterYourself();
  TGeoCombiTrans* opticalFiberPlateTr1 = new TGeoCombiTrans("opticalFiberPlateTr1", 0, 0, sOpticalFiberPlateZ, new TGeoRotation("opticalFiberPlateRot1", 90, 0, 0));
  opticalFiberPlateTr1->RegisterYourself();
  TGeoCombiTrans* opticalFiberPlateTr2 = new TGeoCombiTrans("opticalFiberPlateTr2", 0, -sPlateSpacing, sOpticalFiberPlateZ, new TGeoRotation("opticalFiberPlateRot2", 90, 0, 0));
  opticalFiberPlateTr2->RegisterYourself();

  // define transformations to form a plateGroup
  TGeoTranslation* plateTr1 = new TGeoTranslation("plateTr1", -sPlateSpacing, sPlateDisplacementDeltaY, 0);
  plateTr1->RegisterYourself();
  TGeoTranslation* plateTr2 = new TGeoTranslation("plateTr2", 0, 0, 0);
  plateTr2->RegisterYourself();
  TGeoTranslation* plateTr3 = new TGeoTranslation("plateTr3", sPlateSpacing, 0, 0);
  plateTr3->RegisterYourself();

  // TODO: fix plateGroupTr2
  // TODO: Move hard-coded numbers to variables defined in the constants list
  // define transformations for the plateGroups (6 basicPlates and 6 cablePlates)
  TGeoTranslation* plateGroupTr1 = new TGeoTranslation("plateGroupTr1", sPlateDisplacementX, sPlateDisplacementY, sPlateGroupZ);
  plateGroupTr1->RegisterYourself();
  TGeoCombiTrans* plateGroupTr2 = new TGeoCombiTrans("plateGroupTr2", 10.4358 + 1.5 * sPlateDisplacementDeltaY, -7.0747, sPlateGroupZ, new TGeoRotation("plateGroup2Rotation", -90, 0, 0));
  plateGroupTr2->RegisterYourself();
}

//Transformations for the L-shaped elements
void Detector::defineFrameTransformations()
{

  // TODO: Confirm shifts that eliminate internal overlaps do not then cause
  //       overlaps with FV0 or other elements
  // TODO: Move these hard-coded numbers to be variables in the list of constants
  Float_t zshift = .2741;
  Float_t rectShift = .274101;
  Float_t frameXshift = -.1009;

  // position of the two rectangles used to approximate the L-shaped frame element
  TGeoTranslation* frameTr1 = new TGeoTranslation("frameTr1", sFrame1PosX + frameXshift, sFrame1PosY, 0 + zshift);
  frameTr1->RegisterYourself();
  TGeoTranslation* frameTr2 = new TGeoTranslation("frameTr2", sFrame2PosX + frameXshift, sFrame2PosY, 0 + zshift);
  frameTr2->RegisterYourself();

  // remove the two smaller rectangles from the L-shaped frame element
  TGeoTranslation* rectTr1 = new TGeoTranslation("rectTr1", sFrame1PosX + sXoffset + frameXshift + 3.25, sFrame1PosY + sYoffset + 6.1875, 0 + zshift);
  rectTr1->RegisterYourself();

  TGeoTranslation* rectTr2 = new TGeoTranslation("rectTr2", sFrame1PosX + sXoffset + frameXshift + 9.3, sFrame1PosY + sYoffset - 0.5775, sMountZ / 2 + zshift);
  rectTr2->RegisterYourself();

  TGeoTranslation* rectTr3 = new TGeoTranslation("rectTr3", sFrame1PosX + sXoffset + frameXshift + 10.75 - sRect3X / 2, sFrame1PosY + sYoffset - 6.8525 + sRect3Y / 2, 0 + zshift);
  rectTr3->RegisterYourself();

  TGeoTranslation* rectTr4 = new TGeoTranslation("rectTr4", sFrame1PosX + sXoffset + frameXshift - 7.925, sFrame1PosY + sYoffset - 6.44, 0 + zshift + 10);
  rectTr4->RegisterYourself();

  TGeoTranslation* rectTr5 = new TGeoTranslation("rectTr5", sFrame2PosX + sXoffset + frameXshift + 6.965 + sRect5X / 2, sFrame2PosY + sYoffset + 4.3625 - sRect5Y / 2, 0 + zshift + rectShift);
  rectTr5->RegisterYourself();

  TGeoTranslation* rectTr6 = new TGeoTranslation("rectTr6", sFrame2PosX + sXoffset + frameXshift + 6.965 - sRect6X / 2, sFrame2PosY + sYoffset - 10.7375 + sRect6Y / 2, 0 + zshift);
  rectTr6->RegisterYourself();

  TGeoTranslation* rectTr7 = new TGeoTranslation("rectTr7", sFrame2PosX + sXoffset + frameXshift + 6.965 - sRect6X - sRect7X / 2, sFrame2PosY + sYoffset - 10.7375 + sRect7Y / 2, sMountZ / 2 + zshift);
  rectTr7->RegisterYourself();

  TGeoTranslation* rectTr8 = new TGeoTranslation("rectTr8", sFrame2PosX + sXoffset + frameXshift - 5.89 - sRect8X / 2, sFrame2PosY + sYoffset + 5.1125 + sRect8Y / 2, 0 + zshift);
  rectTr8->RegisterYourself();
}

Bool_t Detector::ProcessHits(FairVolume* v)
{

  TString volname = fMC->CurrentVolName();

  TVirtualMCStack* stack = fMC->GetStack();
  Int_t quadrant, mcp;
  if (fMC->IsTrackEntering()) {
    float x, y, z;
    fMC->TrackPosition(x, y, z);
    fMC->CurrentVolID(quadrant);
    fMC->CurrentVolOffID(1, mcp);
    float time = fMC->TrackTime() * 1.0e9; //time from seconds to ns
    int trackID = stack->GetCurrentTrackNumber();
    int detID = mSim2LUT[4 * mcp + quadrant - 1];
    float etot = fMC->Etot();
    int iPart = fMC->TrackPid();
    float enDep = fMC->Edep();
    Int_t parentID = stack->GetCurrentTrack()->GetMother(0);
    if (fMC->TrackCharge() && volname.Contains("0REG")) { //charge particles for MCtrue
      AddHit(x, y, z, time, 10, trackID, detID);
    }
    if (iPart == 50000050) { // If particles is photon then ...
      if (volname.Contains("0TOP")) {
        if (!RegisterPhotoE(etot)) {
          fMC->StopTrack();
          return kFALSE;
        }
        mTrackIdTop = trackID;
      }

      if (volname.Contains("0MTO")) {
        if (trackID != mTrackIdTop) {
          if (!RegisterPhotoE(etot)) {
            fMC->StopTrack();
            return kFALSE;
          }
          mTrackIdMCPtop = trackID;
        }
      }

      if (volname.Contains("0REG")) {
        if (trackID != mTrackIdTop && trackID != mTrackIdMCPtop) {
          if (RegisterPhotoE(etot)) {
            AddHit(x, y, z, time, enDep, parentID, detID);
          }
        }
        if (trackID == mTrackIdTop || trackID == mTrackIdMCPtop) {
          AddHit(x, y, z, time, enDep, parentID, detID);
        }
      }
    }

    return kTRUE;
  }
  return kFALSE;
}

o2::ft0::HitType* Detector::AddHit(float x, float y, float z, float time, float energy, Int_t trackId, Int_t detId)
{
  mHits->emplace_back(x, y, z, time, energy, trackId, detId);
  if (energy == 10) {
    auto stack = (o2::data::Stack*)fMC->GetStack();
    stack->addHit(GetDetId());
  }
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
  Float_t dglass = 2.2;
  // MCP glass SiO2
  Float_t dglass_mcp = 1.3;
  /* Ceramic   97.2% Al2O3 , 2.8% SiO2 : average material for
   -  stack of 2 MCPs thickness 2mm with density 1.6 g/cm3
   -  back wall of MCP thickness 2 mm with density 2.4 g/cm3
   -  MCP electrods thickness 1 mm with density 4.2 g/cm3
   -  Backplane PCBs thickness 4.5 mm with density 1.85 g/cm3
   -  electromagnetic shielding 1 mm  with density 2.8 g/cm3
   -  Al back cover 5mm  2.7 g/cm3
  */
  Float_t aCeramic[2] = {26.981539, 15.9994};
  Float_t zCeramic[2] = {13., 8.};
  Float_t wCeramic[2] = {2., 3.};
  Float_t denscer = 2.37;

  //MCP walls Ceramic+Nickel (50//50)
  const Int_t nCeramicNice = 3;
  Float_t aCeramicNicel[3] = {26.981539, 15.9994, 58.6934};
  Float_t zCeramicNicel[3] = {13., 8., 28};
  Float_t wCeramicNicel[3] = {0.2, 0.3, 0.5};
  Float_t denscerCeramicNickel = 5.6;

  //Mixed Cables material simulated as plastic with density taken from description of Low Loss Microwave Coax24 AWG 0
  //  plastic + cooper (6%)
  const Int_t nPlast = 4;
  Float_t aPlast[nPlast] = {1.00784, 12.0107, 15.999, 63.54};
  Float_t zPlast[nPlast] = {1, 6, 8, 29};
  Float_t wPlast[nPlast] = {0.08, 0.53, 0.22, 0.17}; ////!!!!!
  const Float_t denCable = 3.66;

  //*** Definition Of avaible FIT materials ***
  Material(11, "Aliminium$", 26.98, 13.0, 2.7, 8.9, 999);
  Mixture(1, "Vacuum$", aAir, zAir, dAir1, 4, wAir);
  Mixture(2, "Air$", aAir, zAir, dAir, 4, wAir);
  Mixture(4, "MCP glass   $", aglass, zglass, dglass_mcp, -2, wglass);
  Mixture(24, "Radiator Optical glass$", aglass, zglass, dglass, -2, wglass);
  Mixture(3, "Ceramic$", aCeramic, zCeramic, denscer, -2, wCeramic);
  Mixture(23, "CablePlasticCooper$", aPlast, zPlast, denCable, 4, wPlast);
  Mixture(25, "MCPwalls $", aCeramicNicel, zCeramicNicel, denscerCeramicNickel, 3, wCeramicNicel);

  Medium(1, "Air$", 2, 0, isxfld, sxmgmx, 10., .1, 1., .003, .003);
  Medium(3, "Vacuum$", 1, 0, isxfld, sxmgmx, 10., .01, .1, .003, .003);
  Medium(4, "Ceramic$", 3, 0, isxfld, sxmgmx, 10., .01, .1, .003, .003);
  Medium(6, "Glass$", 4, 0, isxfld, sxmgmx, 10., .01, .1, .003, .003);
  //  Medium(7, "OpAir$", 2, 0, isxfld, sxmgmx, 10., .1, 1., .003, .003);
  //  Medium(18, "OpBlack$", 2, 0, isxfld, sxmgmx, 10., .1, 1., .003, .003);
  Medium(15, "Aluminium$", 11, 0, isxfld, sxmgmx, 10., .01, 1., .003, .003);
  Medium(17, "OptAluminium$", 11, 0, isxfld, sxmgmx, 10., .01, 1., .003, .003);
  Medium(16, "OpticalGlass$", 24, 1, isxfld, sxmgmx, 10., .01, .1, .003, .01);
  Medium(19, "OpticalGlassCathode$", 24, 1, isxfld, sxmgmx, 10., .01, .1, .003, .003);
  //  Medium(22, "SensAir$", 2, 1, isxfld, sxmgmx, 10., .1, 1., .003, .003);
  Medium(23, "Cables$", 23, 1, isxfld, sxmgmx, 10., .1, 1., .003, .003);
  Medium(25, "MCPWalls", 25, 1, isxfld, sxmgmx, 10., .1, 1., .003, .003);
}

//-------------------------------------------------------------------
void Detector::DefineOpticalProperties()
{
  // Path of the optical properties input file
  TString inputDir;
  const char* aliceO2env = std::getenv("O2_ROOT");
  if (aliceO2env) {
    inputDir = aliceO2env;
  }
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
                                   &(mQuantumEfficiency[0]), &(mRefractionIndex[0]));
  TVirtualMC::GetMC()->SetCerenkov(getMediumID(kOpGlassCathode), nBins, &(mPhotonEnergyD[0]), &(mAbsorptionLength[0]),
                                   &(mQuantumEfficiency[0]), &(mRefractionIndex[0]));
  /*
    TVirtualMC::GetMC()->SetCerenkov(getMediumID(kOptBlack), nBins, &(mPhotonEnergyD[0]), &(mAbsorAir[0]),
                                   &(mEfficAll[0]), &(mRindexAir[0]));
  TVirtualMC::GetMC()->SetCerenkov(getMediumID(kOptAl), nBins, &(mPhotonEnergyD[0]), &(mAbsorbCathodeNext[0]),
                                   &(mEfficMet[0]), &(mRindexCathodeNext[0]));

  */
  // Define a border for radiator optical properties
  TVirtualMC::GetMC()->DefineOpSurface("surfRd", kUnified, kDielectric_metal, kPolishedbackpainted, 0.);
  TVirtualMC::GetMC()->SetMaterialProperty("surfRd", "EFFICIENCY", nBins, &(mPhotonEnergyD[0]), &(mEfficMet[0]));
  TVirtualMC::GetMC()->SetMaterialProperty("surfRd", "REFLECTIVITY", nBins, &(mPhotonEnergyD[0]), &(mReflMet[0]));
  TVirtualMC::GetMC()->SetBorderSurface("surMirrorBorder0", "0TOP", 1, "0RFV", 1, "surfRd");
  TVirtualMC::GetMC()->SetBorderSurface("surMirrorBorder1", "0TOP", 1, "0RFH", 1, "surfRd");
  TVirtualMC::GetMC()->SetBorderSurface("surMirrorBorder2", "0TOP", 1, "0RFV", 2, "surfRd");
  TVirtualMC::GetMC()->SetBorderSurface("surMirrorBorder3", "0TOP", 1, "0RFH", 2, "surfRd");
  //Define black paper on the top of radiator
  TVirtualMC::GetMC()->DefineOpSurface("surBlack", kUnified, kDielectric_dielectric, kGroundbackpainted, 0.);
  // TVirtualMC::GetMC()->SetMaterialProperty("surBlack", "EFFICIENCY", nBins, &(mPhotonEnergyD[0]), &(mEffBlackPaper[0]));
  TVirtualMC::GetMC()->SetMaterialProperty("surBlack", "REFLECTIVITY", nBins, &(mPhotonEnergyD[0]), &(mReflBlackPaper[0]));
  TVirtualMC::GetMC()->SetBorderSurface("surBlackBorder", "0TOP", 1, "0PAL", 1, "surBlack");
  //between cathode and back of front MCP glass window
  TVirtualMC::GetMC()->DefineOpSurface("surFrontBWindow", kUnified, kDielectric_dielectric, kPolishedbackpainted, 0.);
  //  TVirtualMC::GetMC()->SetMaterialProperty("surFrontBWindow", "EFFICIENCY", nBins, &(mPhotonEnergyD[0]), &(mEfficAll[0]));
  TVirtualMC::GetMC()->SetMaterialProperty("surFrontBWindow", "REFLECTIVITY", nBins, &(mPhotonEnergyD[0]), &(mReflFrontWindow[0]));
  TVirtualMC::GetMC()->SetBorderSurface("surBorderFrontBWindow", "0REG", 1, "0MTO", 1, "surFrontBWindow");
  //between radiator and front MCP glass window
  TVirtualMC::GetMC()->DefineOpSurface("surFrontWindow", kUnified, kDielectric_dielectric, kPolishedbackpainted, 0.);
  //TVirtualMC::GetMC()->SetMaterialProperty("surFrontWindow", "EFFICIENCY", nBins, &(mPhotonEnergyD[0]), &(mEfficAll[0]));
  TVirtualMC::GetMC()->SetMaterialProperty("surFrontWindow", "REFLECTIVITY", nBins, &(mPhotonEnergyD[0]), &(mReflBlackPaper[0]));
  TVirtualMC::GetMC()->SetBorderSurface("surBorderFrontWindow", "0TOP", 1, "0MTO", 1, "surFrontWindow");
}

void Detector::FillOtherOptProperties()
{
  // Set constant values to the other arrays
  for (Int_t i = 0; i < mPhotonEnergyD.size(); i++) {
    mReflBlackPaper.push_back(0.);
    mEffBlackPaper.push_back(0);
    mAbsBlackPaper.push_back(1);

    mReflFrontWindow.push_back(0.5);

    mRindexAir.push_back(1.);
    mAbsorAir.push_back(0.3);
    mRindexCathodeNext.push_back(1);
    mAbsorbCathodeNext.push_back(1);
    mEfficMet.push_back(0);
    mRindexMet.push_back(0);
    mReflMet.push_back(0.9);
  }
}

//------------------------------------------------------------------------
Bool_t Detector::RegisterPhotoE(float energy)
{
  float eff = mPMTeff->Eval(energy);
  float p = gRandom->Rndm();
  if (p > eff) {
    return kFALSE;
  }

  return kTRUE;
}

Int_t Detector::ReadOptProperties(const std::string filePath)
{
  std::ifstream infile;
  infile.open(filePath.c_str());
  LOG(INFO) << " file " << filePath.c_str();
  // Check if file is opened correctly
  if (infile.fail() == true) {
    // AliFatal(Form("Error opening ascii file: %s", filePath.c_str()));
    return -1;
  }

  std::string comment;             // dummy, used just to read 4 first lines and move the cursor to the 5th, otherwise unused
  if (!getline(infile, comment)) { // first comment line
    LOG(ERROR) << "Error opening ascii file (it is probably a folder!): " << filePath.c_str();
    return -2;
  }
  getline(infile, comment); // 2nd comment line

  // Get number of elements required for the array
  Int_t nLines;
  infile >> nLines;
  if (nLines < 0 || nLines > 1e4) {
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
      //   LOG(ERROR) << "Line number: " << iLine << " reaches range of declared arraySize:" << kNbins << " Check input file:" << filePath.c_str();
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
    //    LOG(ERROR)(Form("Total number of lines %i is different than declared %i. Check input file: %s", iLine, kNbins,
    //    filePath.c_str()));
    return -7;
  }

  LOG(INFO) << "Optical properties taken from the file: " << filePath.c_str() << " Number of lines read: " << iLine;
  return 0;
}

void Detector::DefineSim2LUTindex()
{
  // Path of the LookUp table
  std::string inputDir;
  const char* aliceO2env = std::getenv("O2_ROOT");
  if (aliceO2env) {
    inputDir = aliceO2env;
  }
  inputDir += "/share/Detectors/FT0/files/";

  std::string indPath = inputDir + "Sim2DataChannels.txt";
  indPath = gSystem->ExpandPathName(indPath.data()); // Expand $(ALICE_ROOT) into real system path

  std::ifstream infile;
  infile.open(indPath.data());
  LOG(INFO) << " file  open " << indPath.data();
  // Check if file is opened correctly
  if (infile.fail() == true) {
    LOG(ERROR) << "Error opening ascii file (it is probably a folder!): " << indPath.c_str();
  }
  int fromfile;
  for (int iind = 0; iind < Geometry::Nchannels; iind++) {
    infile >> fromfile;
    mSim2LUT[iind] = fromfile;
  }
}
