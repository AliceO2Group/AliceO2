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

#include <Buttons.h>
#include <TGeoCompositeShape.h>
#include <TGeoShape.h>
#include <TGeoBBox.h>
#include <TGeoTube.h>
#include <TGeoSphere.h>
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
#include <fairlogger/Logger.h>
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
  TVirtualMC* fMC = TVirtualMC::GetMC();
  TGeoVolume* v = gGeoManager->GetVolume("0REG");
  if (v == nullptr) {
    LOG(warn) << "@@@@ Sensitive volume 0REG not found!!!!!!!!";
  } else {
    AddSensitiveVolume(v);
    mREGVolID = fMC->VolId("0REG");
  }
  TGeoVolume* vrad = gGeoManager->GetVolume("0TOP");
  if (vrad == nullptr) {
    LOG(warn) << "@@@@ Sensitive radiator not found!!!!!!!!";
  } else {
    AddSensitiveVolume(vrad);
    mTOPVolID = fMC->VolId("0TOP");
  }
  TGeoVolume* vmcp = gGeoManager->GetVolume("0MTO");
  if (vmcp == nullptr) {
    LOG(warn) << "@@@@ Sensitive MCP glass not found!!!!!!!!";
  } else {
    AddSensitiveVolume(vmcp);
    mMTOVolID = fMC->VolId("0MTO");
  }
}

void Detector::ConstructGeometry()
{
  LOG(debug) << "Creating FT0 geometry\n";
  CreateMaterials();

  TGeoVolumeAssembly* stlinA = new TGeoVolumeAssembly("FT0A"); // A side mother
  TGeoVolumeAssembly* stlinC = new TGeoVolumeAssembly("FT0C"); // C side mother

  Geometry geometry;
  Float_t zdetA = geometry.ZdetA;
  Float_t zdetC = geometry.ZdetC;
  int nCellsA = geometry.NCellsA;
  int nCellsC = geometry.NCellsC;

  for (int ipos = 0; ipos < nCellsA; ipos++) {
    mPosModuleAx[ipos] = geometry.centerMCP(ipos).X();
    mPosModuleAy[ipos] = geometry.centerMCP(ipos).Y();
  }

  // FIT interior
  TVirtualMC::GetMC()->Gsvolu("0MOD", "BOX", getMediumID(kAir), mInStart, 3);
  TGeoVolume* ins = gGeoManager->GetVolume("0MOD");
  //
  TGeoTranslation* tr[nCellsA + nCellsC];
  TString nameTr;
  // A side Translations
  for (Int_t itr = 0; itr < Geometry::NCellsA; itr++) {
    nameTr = Form("0TR%i", itr + 1);
    float z = -mStartA[2] + mInStart[2];
    tr[itr] = new TGeoTranslation(nameTr.Data(), mPosModuleAx[itr], mPosModuleAy[itr], z);
    tr[itr]->RegisterYourself();
    stlinA->AddNode(ins, itr, tr[itr]);
    LOG(debug) << " A geom " << itr << " " << mPosModuleAx[itr] << " " << mPosModuleAy[itr];
  }
  SetCablesA(stlinA);

  // Add FT0-A support Structure to the geometry
  stlinA->AddNode(constructFrameAGeometry(), 1, new TGeoTranslation(0, 0, -mStartA[2] + mInStart[2]));

  // C Side
  TGeoRotation* rot[nCellsC];
  TString nameRot;
  TGeoCombiTrans* com[nCellsC];
  TGeoCombiTrans* comCable[nCellsC];
  TString nameCom;

  //Additional elements for the C-side frame
  TGeoCombiTrans* plateCom[nCellsC];
  TGeoMedium* Al = gGeoManager->GetMedium("FT0_Aluminium$");
  TGeoCompositeShape* plateCompositeShape = new TGeoCompositeShape("plateCompositeShape", cPlateShapeString().c_str());
  TGeoVolume* plateVol = new TGeoVolume("plateVol", plateCompositeShape, Al);

  for (Int_t itr = Geometry::NCellsA; itr < Geometry::NCellsA + nCellsC; itr++) {
    nameTr = Form("0TR%i", itr + 1);
    nameRot = Form("0Rot%i", itr + 1);
    int ic = itr - Geometry::NCellsA;
    float ac1 = geometry.tiltMCP(ic).X();
    float bc1 = geometry.tiltMCP(ic).Y();
    float gc1 = geometry.tiltMCP(ic).Z();
    rot[ic] = new TGeoRotation(nameRot.Data(), ac1, bc1, gc1);
    LOG(debug) << " rot geom " << ic << " " << ac1 << " " << bc1 << " " << gc1;
    rot[ic]->RegisterYourself();
    mPosModuleCx[ic] = geometry.centerMCP(ic + nCellsA).X();
    mPosModuleCy[ic] = geometry.centerMCP(ic + nCellsA).Y();
    mPosModuleCz[ic] = geometry.centerMCP(ic + nCellsA).Z() - 80; // !!! fix later
    com[ic] = new TGeoCombiTrans(mPosModuleCx[ic], mPosModuleCy[ic], mPosModuleCz[ic], rot[ic]);
    TGeoHMatrix hm = *com[ic];
    TGeoHMatrix* ph = new TGeoHMatrix(hm);
    stlinC->AddNode(ins, itr, ph);
    // cables
    TGeoVolume* cables = SetCablesSize(itr);
    LOG(debug) << " C " << mPosModuleCx[ic] << " " << mPosModuleCy[ic];
    //    cables->Print();
    //Additional shift (+0.1) introduced to cable planes so they don't overlap the C-side frame
    comCable[ic] = new TGeoCombiTrans(mPosModuleCx[ic], mPosModuleCy[ic], mPosModuleCz[ic] + mInStart[2] + 0.2 + 0.1, rot[ic]);
    TGeoHMatrix hmCable = *comCable[ic];
    TGeoHMatrix* phCable = new TGeoHMatrix(hmCable);
    stlinC->AddNode(cables, itr, comCable[ic]);

    //C-side frame elements - module plates
    plateCom[ic] = new TGeoCombiTrans(mPosModuleCx[ic], mPosModuleCy[ic], (mPosModuleCz[ic] - 3), rot[ic]);
    TGeoHMatrix hmPlate = *plateCom[ic];
    TGeoHMatrix* phPlate = new TGeoHMatrix(hmPlate);
    stlinC->AddNode(plateVol, itr, phPlate);
  }
  //Add C-side frame
  stlinC->AddNode(constructFrameCGeometry(), nCellsA + nCellsC + 1);

  TGeoVolume* alice = gGeoManager->GetVolume("barrel");
  //Add A-side detector
  alice->AddNode(stlinA, 1, new TGeoTranslation(0, 30., zdetA + 0.63)); //offset to avoid overlap with FV0

  //Add C-side detector
  TGeoRotation* rotC = new TGeoRotation("rotC", 90., 0., 90., 90., 180., 0.);
  alice->AddNode(stlinC, 1, new TGeoCombiTrans(0., 30., -zdetC, rotC));

  // MCP + 4 x wrapped radiator + 4xphotocathod + MCP + Al top in front of radiators
  SetOneMCP(ins);
  // SetCablesC(stlinC);
}

void Detector::ConstructOpGeometry()
{
  LOG(debug) << "Creating FIT optical geometry properties";

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
  // Entry window (glass)
  TVirtualMC::GetMC()->Gsvolu("0TOP", "BOX", getMediumID(kOpGlass), ptop, 3); // Glass radiator
  TGeoVolume* top = gGeoManager->GetVolume("0TOP");
  TVirtualMC::GetMC()->Gsvolu("0TRE", "BOX", getMediumID(kAir), ptopref, 3); // Air: wrapped  radiator
  TGeoVolume* topref = gGeoManager->GetVolume("0TRE");
  TVirtualMC::GetMC()->Gsvolu("0RFV", "BOX", getMediumID(kOptAl), prfv, 3); // Optical Air vertical
  TGeoVolume* rfv = gGeoManager->GetVolume("0RFV");
  TVirtualMC::GetMC()->Gsvolu("0RFH", "BOX", getMediumID(kOptAl), prfh, 3); // Optical Air horizontal
  TGeoVolume* rfh = gGeoManager->GetVolume("0RFH");

  TVirtualMC::GetMC()->Gsvolu("0REG", "BOX", getMediumID(kOpGlassCathode), preg, 3);
  TGeoVolume* cat = gGeoManager->GetVolume("0REG");

  // wrapped radiator +  reflecting layers

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

  // container for radiator, cathode
  for (Int_t ix = 0; ix < 2; ix++) {
    float xin = -mInStart[0] + 0.3 + (ix + 0.5) * 2 * ptopref[0];
    for (Int_t iy = 0; iy < 2; iy++) {
      float yin = -mInStart[1] + 0.3 + (iy + 0.5) * 2 * ptopref[1];
      ntops++;
      z = -mInStart[2] + ptopref[2];
      ins->AddNode(topref, ntops, new TGeoTranslation(xin, yin, z));
      LOG(debug) << " n " << ntops << " x " << xin << " y " << yin << " z radiator " << z;
      z += ptopref[2] + 2. * pmcptopglass[2] + preg[2];
      ins->AddNode(cat, ntops, new TGeoTranslation(xin, yin, z));
      LOG(debug) << " n " << ntops << " x " << xin << " y " << yin << " z cathod " << z;
    }
  }
  // MCP
  TVirtualMC::GetMC()->Gsvolu("0MTO", "BOX", getMediumID(kOpGlass), pmcptopglass, 3); // Op  Glass
  TGeoVolume* mcptop = gGeoManager->GetVolume("0MTO");
  z = -mInStart[2] + 2 * ptopref[2] + pmcptopglass[2];
  ins->AddNode(mcptop, 1, new TGeoTranslation(0, 0, z));

  TVirtualMC::GetMC()->Gsvolu("0MCP", "BOX", getMediumID(kAir), pmcp, 3); // glass
  TGeoVolume* mcp = gGeoManager->GetVolume("0MCP");
  z = -mInStart[2] + 2 * ptopref[2] + 2 * pmcptopglass[2] + 2 * preg[2] + pmcp[2];
  ins->AddNode(mcp, 1, new TGeoTranslation(0, 0, z));

  TVirtualMC::GetMC()->Gsvolu("0MSI", "BOX", getMediumID(kMCPwalls), pmcpside, 3); // glass
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

  TVirtualMC::GetMC()->Gsvolu("0MBA", "BOX", getMediumID(kCeramic), pmcpbase, 3); // glass
  TGeoVolume* mcpbase = gGeoManager->GetVolume("0MBA");
  z = -mInStart[2] + 2 * ptopref[2] + pmcptopglass[2] + 2 * pmcp[2] + pmcpbase[2];
  ins->AddNode(mcpbase, 1, new TGeoTranslation(0, 0, z));
}

//----------------------------------
void Detector::SetCablesA(TGeoVolume* stl)
{

  float pcableplane[3] = {20, 20, 0.25}; //

  TVirtualMC::GetMC()->Gsvolu("0CAA", "BOX", getMediumID(kAir), pcableplane, 3); // container for cables
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

  // 12 cables extending beyond the frame
  Float_t pcablesextend[3] = {2, 15, 0.245};
  Float_t pcablesextendsmall[3] = {3, 2, 0.245};
  Float_t* ppcablesextend[] = {pcablesextend, pcablesextend, pcablesextendsmall, pcablesextendsmall};
  // left side
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
                       2, 2, 3, 3, 1,
                       1, 2, 2, 2, 2,
                       1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1,
                       2, 2, 2, 2, 2,
                       2, 3, 3};

  // cable D=0.257cm, Weight: 13 lbs/1000ft = 0.197g/cm; 1 piece 0.65cm
  // 1st 8 pieces - tube  8*0.65cm = 5.2cm; V = 0.0531cm2 -> box {0.27*0.27*1}cm; W = 0.66g
  // 2nd 24 pieces 24*0.65cm; V = 0.76 -> {0.44, 0.447 1}; W = 3.07g
  // 3d  48  pieces  48*0.65cm;  V = 1.53cm^3; ->box {0.66, 0.66, 1.}; W= 6.14g
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
  LOG(debug) << "C cables " << mod << " " << volName << " " << ic;
  return vol;
}

void Detector::addAlignableVolumes() const
{
  //
  // Creates entries for alignable volumes associating the symbolic volume
  // name with the corresponding volume path.
  //
  //  First version (mainly ported from AliRoot)
  //

  LOG(info) << "Add FT0 alignable volumes";

  if (!gGeoManager) {
    LOG(fatal) << "TGeoManager doesn't exist !";
    return;
  }

  TString volPath = Form("/cave_1/barrel_1");
  // set A side
  TString volPathA = volPath + Form("/FT0A_1");
  TString symNameA = "FT0A";
  LOG(info) << symNameA << " <-> " << volPathA;
  if (!gGeoManager->SetAlignableEntry(symNameA.Data(), volPathA.Data())) {
    LOG(fatal) << "Unable to set alignable entry ! " << symNameA << " : " << volPathA;
  }
  // set C side
  TString volPathC = volPath + Form("/FT0C_1");
  TString symNameC = "FT0C";
  LOG(info) << symNameC << " <-> " << volPathC;
  if (!gGeoManager->SetAlignableEntry(symNameC.Data(), volPathC.Data())) {
    LOG(fatal) << "Unable to set alignable entry ! " << symNameA << " : " << volPathA;
  }
  TString volPathMod, symNameMod;
  for (Int_t imod = 0; imod < Geometry::NCellsA + Geometry::NCellsC; imod++) {
    TString volPath = (imod < Geometry::NCellsA) ? volPathA : volPathC;
    volPathMod = volPath + Form("/0MOD_%d", imod);
    symNameMod = Form("0MOD_%d", imod);
    if (!gGeoManager->SetAlignableEntry(symNameMod.Data(), volPathMod.Data())) {
      LOG(fatal) << (Form("Alignable entry %s not created. Volume path %s not valid", symNameMod.Data(), volPathMod.Data()));
    }
  }
}

//Construction of FT0-A support structure
//The frame is constructed by first building a block of Aluminum as a mother volume from which details can
//be subtracted. First, 6 boxe shapes are subtracted from around the edges and 2 from the center of the frame
//to create the fin shapes at the edges of the detector and the cross shape at the center of the detector.
//These boxe shapes are then subtracted again but from positions reflected in both x and y which is
//reflects the symmetry of the frame. Then a loop is used to subtract out the PMT sockets which are also
//box shapes from the positions given in the Geometry.cxx file. In the loop, after a socket is subtracted,
//either an inner or an outer plate group is placed inside of it. Inner and outer plate groups are
//both composed of a cover plate and a cable plate with fiber heads subtracted from the cable plate.
//The orientation of these holes differs between the inner and outer plates making them distinct.
//Contributors: Joe Crowley (2019-20), Jason Pruitt (2020-21), Sam Christensen (2021-22),
//and Jennifer Klay (2019-22) from Cal Poly SLO.
TGeoVolume* Detector::constructFrameAGeometry()
{
  // define the media
  TGeoMedium* Vacuum = gGeoManager->GetMedium("FT0_Vacuum$");
  TGeoMedium* Al = gGeoManager->GetMedium("FT0_Aluminium$");

  // make a volume assembly for the frame
  TGeoVolumeAssembly* FT0_Frame = new TGeoVolumeAssembly("FT0_Frame");

  //Define the block of aluminum that forms the frame
  Double_t blockdX = 37.1;  //slightly larger in x
  Double_t blockdY = 36.85; //than y
  Double_t blockdZ = 6.95;  //thickness of frame and back plates
  TGeoBBox* block = new TGeoBBox("block", blockdX / 2, blockdY / 2, blockdZ / 2);

  //To form the outer frame shape with fins that bolt it to the FV0, remove
  //aluminum in six chunks (boxes) from two sides, then reflect these to remove
  //from the other sides.  As viewed from the back side of the detector, count
  //clockwise from bottom left for numbering.
  Double_t box1dX = 1.57;                          //narrower
  Double_t box1dY = 6.55;                          //than it is tall
  Double_t box1PosX = -(blockdX / 2 - box1dX / 2); //placement on the frame block
  Double_t box1PosY = 0;                           //starts at the middle
  TGeoBBox* box1 = new TGeoBBox("box1", box1dX / 2, box1dY / 2, blockdZ / 2);
  TGeoTranslation* box1Tr1 = new TGeoTranslation("box1Tr1", box1PosX, box1PosY, 0);
  box1Tr1->RegisterYourself();
  TGeoTranslation* box1Tr2 = new TGeoTranslation("box1Tr2", -box1PosX, -box1PosY, 0);
  box1Tr2->RegisterYourself();

  Double_t box2dX = 2.9;
  Double_t box2dY = 15.1;
  Double_t box2PosX = -(blockdX / 2 - box2dX / 2);
  Double_t box2PosY = blockdY / 2 - box2dY / 2;
  TGeoBBox* box2 = new TGeoBBox("box2", box2dX / 2, box2dY / 2, blockdZ / 2);
  TGeoTranslation* box2Tr1 = new TGeoTranslation("box2Tr1", box2PosX, box2PosY, 0);
  box2Tr1->RegisterYourself();
  TGeoTranslation* box2Tr2 = new TGeoTranslation("box2Tr2", -box2PosX, -box2PosY, 0);
  box2Tr2->RegisterYourself();

  //Box 3 is shallower than the others to preserve the aluminum fin where the
  //FT0 is bolted to the FV0
  Double_t box3dX = 12.7;
  Double_t box3dY = 3;
  Double_t box3dZ = 5.45;
  Double_t box3PosX = -(blockdX / 2 - box2dX - box3dZ / 2);
  Double_t box3PosY = blockdY / 2 - box3dY / 2;
  Double_t box3PosZ = blockdZ / 2 - box3dZ / 2; //subtract from the back, leaving fin on the front
  TGeoBBox* box3 = new TGeoBBox("box3", box3dX / 2, box3dY / 2, box3dZ / 2);
  TGeoTranslation* box3Tr1 = new TGeoTranslation("box3Tr1", box3PosX, box3PosY, box3PosZ);
  box3Tr1->RegisterYourself();
  TGeoTranslation* box3Tr2 = new TGeoTranslation("box3Tr2", -box3PosX, -box3PosY, box3PosZ);
  box3Tr2->RegisterYourself();

  Double_t box4dX = 6.6;
  Double_t box4dY = 1.67;
  Double_t box4PosX = 0;
  Double_t box4PosY = blockdY / 2 - box4dY / 2;
  TGeoBBox* box4 = new TGeoBBox("box4", box4dX / 2, box4dY / 2, blockdZ / 2);
  TGeoTranslation* box4Tr1 = new TGeoTranslation("box4Tr1", box4PosX, box4PosY, 0);
  box4Tr1->RegisterYourself();
  TGeoTranslation* box4Tr2 = new TGeoTranslation("box4Tr2", -box4PosX, -box4PosY, 0);
  box4Tr2->RegisterYourself();

  Double_t box5dX = 15;
  Double_t box5dY = 3;
  Double_t box5PosX = blockdX / 2 - box5dX / 2;
  Double_t box5PosY = blockdY / 2 - box5dY / 2;
  TGeoBBox* box5 = new TGeoBBox("box5", box5dX / 2, box5dY / 2, blockdZ / 2);
  TGeoTranslation* box5Tr1 = new TGeoTranslation("box5Tr1", box5PosX, box5PosY, 0);
  box5Tr1->RegisterYourself();
  TGeoTranslation* box5Tr2 = new TGeoTranslation("box5Tr2", -box5PosX, -box5PosY, 0);
  box5Tr2->RegisterYourself();

  //Similar to box 3, box 6 is shallower in z to leave aluminum for the fin that
  //bolts FT0 to FV0
  Double_t box6dX = 2.9;
  Double_t box6dY = 12.2;
  Double_t box6dZ = 5.45;
  Double_t box6PosX = blockdX / 2 - box6dX / 2;
  Double_t box6PosY = blockdY / 2 - box5dY - box6dY / 2;
  Double_t box6PosZ = blockdZ / 2 - box6dZ / 2; //subtract from the back, leaving fin at the front
  TGeoBBox* box6 = new TGeoBBox("box6", box6dX / 2, box6dY / 2, box6dZ / 2);
  TGeoTranslation* box6Tr1 = new TGeoTranslation("box6Tr1", box6PosX, box6PosY, box6PosZ);
  box6Tr1->RegisterYourself();
  TGeoTranslation* box6Tr2 = new TGeoTranslation("box6Tr2", -box6PosX, -box6PosY, box6PosZ);
  box6Tr2->RegisterYourself();

  //The central hole that accommodates the beam pipe is not the same on all four sides
  //so we define two rectangular boxes - one vertical and one horizontal - and copy/rotate them
  //to remove the aluminum in a "+" shape at the center
  // cbox is a central rectangle
  Double_t cbox1dX = 7.175; //horizontal center box
  Double_t cbox1dY = 5.5;
  Double_t cbox1Xoffset = 14.425;
  Double_t cbox1PosX = -(blockdX / 2 - cbox1Xoffset - cbox1dX / 2);
  Double_t cbox1PosY = 0;
  TGeoBBox* cbox1 = new TGeoBBox("cbox1", cbox1dX / 2, cbox1dY / 2, blockdZ / 2);
  TGeoTranslation* cbox1Tr1 = new TGeoTranslation("cbox1Tr1", cbox1PosX, cbox1PosY, 0);
  cbox1Tr1->RegisterYourself();
  TGeoTranslation* cbox1Tr2 = new TGeoTranslation("cbox1Tr2", -cbox1PosX, -cbox1PosY, 0);
  cbox1Tr2->RegisterYourself();

  Double_t cbox2dX = 5.75; //vertical center box

  Double_t cbox2dY = 6.575;
  Double_t cbox2Yoffset = 14.425;
  Double_t cbox2PosX = 0;
  Double_t cbox2PosY = blockdY / 2 - cbox2Yoffset - cbox2dY / 2;
  TGeoBBox* cbox2 = new TGeoBBox("cbox2", cbox2dX / 2, cbox2dY / 2, blockdZ / 2);
  TGeoTranslation* cbox2Tr1 = new TGeoTranslation("cbox2Tr1", cbox2PosX, cbox2PosY, 0);
  cbox2Tr1->RegisterYourself();
  TGeoTranslation* cbox2Tr2 = new TGeoTranslation("cbox2Tr2", -cbox2PosX, -cbox2PosY, 0);
  cbox2Tr2->RegisterYourself();

  //The two L-shaped pieces that form the frame have a small 1mm gap between them,
  //where they come together.  As viewed from the back, the gaps are on the upper
  //left and lower right, so that for the center column of modules, the upper two
  //are shifted slightly to the right (as viewed from the back) and the lower two
  //are shifted slightly to the left (as viewed from the back)
  Double_t gapBoxdX = 0.1;
  Double_t gapBoxdY = blockdY / 2;
  Double_t gapPosX = -(sPmtSide / 2 + sEps + gapBoxdX / 2);
  Double_t gapPosY = blockdY / 4;
  TGeoBBox* gapBox = new TGeoBBox("gapBox", gapBoxdX / 2, gapBoxdY / 2, blockdZ / 2);
  TGeoTranslation* gapBoxTr1 = new TGeoTranslation("gapBoxTr1", gapPosX, gapPosY, 0);
  gapBoxTr1->RegisterYourself();
  TGeoTranslation* gapBoxTr2 = new TGeoTranslation("gapBoxTr2", -gapPosX, -gapPosY, 0);
  gapBoxTr2->RegisterYourself();

  //Create a string to define the complete frame object shape
  //Start from the aluminum block then subtract the boxes
  std::string frameACompositeString = "block ";
  frameACompositeString += "- box1:box1Tr1 - box1:box1Tr2 ";
  frameACompositeString += "- box2:box2Tr1 - box2:box2Tr2 ";
  frameACompositeString += "- box3:box3Tr1 - box3:box3Tr2 ";
  frameACompositeString += "- box4:box4Tr1 - box4:box4Tr2 ";
  frameACompositeString += "- box5:box5Tr1 - box5:box5Tr2 ";
  frameACompositeString += "- box6:box6Tr1 - box6:box6Tr2 ";
  frameACompositeString += "- cbox1:cbox1Tr1 - cbox1:cbox1Tr2 ";
  frameACompositeString += "- cbox2:cbox2Tr1 - cbox2:cbox2Tr2 ";
  frameACompositeString += "- gapBox:gapBoxTr1 - gapBox:gapBoxTr2";

  //The next section defines the objects that form the sockets in the
  //frame for the sensitive elements and the individual cover plates
  //at the front of the detector which include the optical fiber
  //heads that permit the LED pulser light to reach the quartz radiator
  //surfaces of each module

  //There are two fiber head configurations, called "inner" and "outer"
  //with different locations and angles of the fiber heads.
  Double_t coverPlatedZ = 0.2; //top cover thickness
  Double_t fiberPlatedZ = 0.5; //fiberhead plate is underneath

  //Each fiber is guided to a small rectangular opening in the plate
  Double_t opticalFiberHeaddY = 0.52;  //narrow side
  Double_t opticalFiberHeaddX = 1.142; //long side

  //The "top" two fiber heads are positioned at slightly different
  //locations than the bottom two, which are also rotated

  //"Outer" fiberhead placements
  Double_t fh1TopPosX = -1.6;
  Double_t fh1TopPosY = 1.325;
  Double_t fh1BotPosX = 1.555;
  Double_t fh1BotPosY = 1.249;
  Double_t fh1BotAngle = 16; //degrees

  //"Inner" fiberhead placements
  //All of these are placed at an angle
  Double_t fh2TopPosX = -1.563;
  Double_t fh2TopPosY = 1.4625;
  Double_t fh2TopAngle = 60;

  Double_t fh2BotPosX = 1.084;
  Double_t fh2BotPosY = 1.186;
  Double_t fh2BotAngle = -30;

  // Define cover plate, fiber plate, and optical Fiber Head shapes
  TGeoBBox* coverPlate = new TGeoBBox("coverPlate", sPmtSide / 2 + sEps, sPmtSide / 2 + sEps, coverPlatedZ / 2);
  TGeoBBox* fiberPlate = new TGeoBBox("fiberPlate", sPmtSide / 2 + sEps, sPmtSide / 2 + sEps, fiberPlatedZ / 2);
  TGeoBBox* opticalFiberHead = new TGeoBBox("opticalFiberHead", opticalFiberHeaddX / 2, opticalFiberHeaddY / 2, fiberPlatedZ / 2);

  // Define transformations of optical fiber heads for outer plate
  TGeoTranslation* coverPlateTr = new TGeoTranslation("coverPlateTr", 0, 0, fiberPlatedZ / 2 + coverPlatedZ / 2);
  coverPlateTr->RegisterYourself();
  TGeoTranslation* fh1TopTr1 = new TGeoTranslation("fh1TopTr1", fh1TopPosX, fh1TopPosY, 0);
  fh1TopTr1->RegisterYourself();
  TGeoTranslation* fh1TopTr2 = new TGeoTranslation("fh1TopTr2", fh1TopPosX, -fh1TopPosY, 0);
  fh1TopTr2->RegisterYourself();
  TGeoCombiTrans* fh1BotTr1 = new TGeoCombiTrans("fh1BotTr1", fh1BotPosX, fh1BotPosY, 0, new TGeoRotation("fh1BotRot1", fh1BotAngle, 0, 0));
  fh1BotTr1->RegisterYourself();
  TGeoCombiTrans* fh1BotTr2 = new TGeoCombiTrans("fh1BotTr2", fh1BotPosX, -fh1BotPosY, 0, new TGeoRotation("fh1BotRot2", -fh1BotAngle, 0, 0));
  fh1BotTr2->RegisterYourself();
  TGeoCombiTrans* fh2TopTr1 = new TGeoCombiTrans("fh2TopTr1", fh2TopPosX, fh2TopPosY, 0, new TGeoRotation("fh2TopRot1", fh2TopAngle + 90, 0, 0));
  fh2TopTr1->RegisterYourself();
  TGeoCombiTrans* fh2TopTr2 = new TGeoCombiTrans("fh2TopTr2", fh2TopPosX, -fh2TopPosY, 0, new TGeoRotation("fh2TopRot2", -fh2TopAngle - 90, 0, 0));
  fh2TopTr2->RegisterYourself();
  TGeoCombiTrans* fh2BotTr1 = new TGeoCombiTrans("fh2BotTr1", fh2BotPosX, fh2BotPosY, 0, new TGeoRotation("fh2BotRot1", -fh2BotAngle, 0, 0));
  fh2BotTr1->RegisterYourself();
  TGeoCombiTrans* fh2BotTr2 = new TGeoCombiTrans("fh2BotTr2", fh2BotPosX, -fh2BotPosY, 0, new TGeoRotation("fh2BotRot2", fh2BotAngle, 0, 0));
  fh2BotTr2->RegisterYourself();

  //Create a string that defines the plate group for the outer plates
  std::string outerPlateGroupString = "fiberPlate ";
  outerPlateGroupString += "- opticalFiberHead:fh1TopTr1 ";
  outerPlateGroupString += "- opticalFiberHead:fh1TopTr2 ";
  outerPlateGroupString += "- opticalFiberHead:fh1BotTr1 ";
  outerPlateGroupString += "- opticalFiberHead:fh1BotTr2 ";
  outerPlateGroupString += "+ coverPlate:coverPlateTr";

  //Create the composite shape for the outer plates
  TGeoCompositeShape* outerPlateGroup = new TGeoCompositeShape("outerPlateGroup", outerPlateGroupString.c_str());

  //Create a string that defines the plate group for the inner plates
  std::string innerPlateGroupString = "fiberPlate ";
  innerPlateGroupString += "- opticalFiberHead:fh2TopTr1 ";
  innerPlateGroupString += "- opticalFiberHead:fh2TopTr2 ";
  innerPlateGroupString += "- opticalFiberHead:fh2BotTr1 ";
  innerPlateGroupString += "- opticalFiberHead:fh2BotTr2 ";
  innerPlateGroupString += "+ coverPlate:coverPlateTr";

  //Create the composite shape for the inner plates
  TGeoCompositeShape* innerPlateGroup = new TGeoCompositeShape("innerPlateGroup", innerPlateGroupString.c_str());

  //The sockets that are cut out of the aluminum block for the senitive elements
  //to fit into are offset slightly in z to leave a thin plate of aluminum at the
  //back - the back plate covers
  Double_t backPlanedZ = 0.25;
  Double_t socketdZ = blockdZ - backPlanedZ;

  //Define the socket volume as a box of vacuum
  TGeoVolume* socket = gGeoManager->MakeBox("Socket", Vacuum, sPmtSide / 2 + sEps, sPmtSide / 2 + sEps, socketdZ / 2);

  //Define the orientation angles of the plate groups that will cover
  //the sockets holding the sensitive elements
  Double_t rotAngle[Geometry::NCellsA] = {0, 0, -90, -90, -90, 0, 0, -90, -90, -90, 0, 0, 180, 180, 90, 90, 90, 180, 180, 90, 90, 90, 180, 180};
  //Define the socket and plate group translations
  TGeoTranslation* trSocket[Geometry::NCellsA];
  TString nameTrSocket;
  TGeoCombiTrans* trPlateGroup[Geometry::NCellsA];
  TString nameTrPlateGroup;
  TString namePGRot;

  //Loop over the number of modules, subtracting the sockets and adding back in the
  //plate groups at the position of each module
  for (Int_t itr = 0; itr < Geometry::NCellsA; itr++) {

    nameTrSocket = Form("trSocket%i", itr + 1);
    float z = -backPlanedZ / 4.0;
    trSocket[itr] = new TGeoTranslation(nameTrSocket.Data(), mPosModuleAx[itr], mPosModuleAy[itr], z);
    trSocket[itr]->RegisterYourself();
    frameACompositeString += "- Socket:";         //subtract it from the aluminum block
    frameACompositeString += nameTrSocket.Data(); //at its corresponding location

    nameTrPlateGroup = Form("trPlateGroup%i", itr + 1);
    namePGRot = Form("pgRot%i", itr + 1);
    float z2 = -blockdZ / 2 + (coverPlatedZ + fiberPlatedZ) / 2;
    trPlateGroup[itr] = new TGeoCombiTrans(nameTrPlateGroup.Data(), mPosModuleAx[itr], mPosModuleAy[itr], z2, new TGeoRotation(namePGRot.Data(), rotAngle[itr], 0, 0));
    trPlateGroup[itr]->RegisterYourself();

    if (itr == 0 || itr == 2 || itr == 3 || itr == 4 || itr == 5 || itr == 10 || itr == 13 || itr == 18 || itr == 19 || itr == 20 || itr == 21 || itr == 23) {
      frameACompositeString += " + outerPlateGroup:"; //add the outer plate group back on to these modules
      frameACompositeString += nameTrPlateGroup.Data();
      frameACompositeString += " ";
    } else {
      frameACompositeString += " + innerPlateGroup:"; //or add the inner plate group back on to all other modules
      frameACompositeString += nameTrPlateGroup.Data();
      frameACompositeString += " ";
    }
  }

  //Finally, define the A side frame object from the complete composite shape defined above
  TGeoVolume* frameA = new TGeoVolume("frameA", new TGeoCompositeShape("frameA", frameACompositeString.c_str()), Al);

  //Add the frame object to the mother volume
  FT0_Frame->AddNode(frameA, 1);

  return FT0_Frame;
}

//C-side Support Structure
//This code was written by Jason Pruitt and Sam Christensen of Cal Poly in 2021
//They followed a similar method as for the A-side frame but had to account for the
//spherical geometry of the C-side.
TGeoVolume* Detector::constructFrameCGeometry()
{

  // define the media
  TGeoMedium* Vacuum = gGeoManager->GetMedium("FT0_Vacuum$");
  TGeoMedium* Al = gGeoManager->GetMedium("FT0_Aluminium$");
  static constexpr Double_t sFrameZC = 5.5632;
  static constexpr Double_t frameHeightC = 2.5; // pinstart[2]  or l_s

  // quartz & PMT C-side transformations
  static constexpr Double_t sensShift = 0.5;
  static constexpr Double_t sQuartzRadiatorZC = 1.94360;                              // Dimension variable (l_{q}
  static constexpr Double_t sQuartzHeightC = (-sFrameZC / 2 + sQuartzRadiatorZC / 2); // placement variable )
  static constexpr Double_t sPmtZC = 3.600;                                           // Dimension variable (l_{p}
  static constexpr Double_t sPmtHeightC = (sFrameZC / 2 - sPmtZC / 2);                // placement variable

  Double_t crad = 82.;
  static constexpr Int_t NCellsC = Geometry::NCellsC;
  static constexpr Int_t NCellsA = Geometry::NCellsA;

  Float_t sweep = 3.5 * 2;
  Float_t rMin = 81.9791;
  Float_t rMax = rMin + sFrameZC;
  Float_t tMin = 0;
  Float_t tMax = 35;
  Float_t pMin = 0;
  Float_t pMax = 180;
  Float_t pinstart[3] = {2.9491, 2.9491, 2.5};
  Float_t pstartC[3] = {20., 20, 5};

  Float_t multCorn = 1.275; // multiplication factor for corners
  Double_t xCorn = multCorn * (-14.75272569);
  Double_t yCorn = multCorn * (14.9043284);
  Double_t zCorn = 79.27306024;

  Double_t xCorn2 = -xCorn;
  Double_t yCorn2 = yCorn;
  Double_t zCorn2 = zCorn;

  Double_t acCorn = TMath::ATan(yCorn / xCorn) - TMath::Pi() / 2 + 2 * TMath::Pi();
  Double_t bcCorn = /*(-1)**/ TMath::ACos(zCorn / crad);
  Double_t gcCorn = -1 * acCorn;

  // holepunch corners not implemented for quartzRadiatorSeat, rounded corners are
  // in place for PMT
  Double_t flopsErr = 0.00001;
  Double_t exag = 5;

  // highest overlap values
  Double_t errPMTZ = 10 * sEps;
  Double_t errPMTXY = 0.02;
  Double_t errQrdZ = 0.143 + 0.22;
  Double_t errQrdXY = 0.35;

  Float_t backPlateZ = 0.5;

  // sphere1 is the spherical shell that will be subtracted
  // to approximate the c-side support frame and the subsequent
  // spheres clip the shape with curvature preserved
  TGeoSphere* sphere1 = new TGeoSphere("sphere1", rMin, rMax, tMin, tMax, pMin, pMax);
  TGeoSphere* sphere2 = new TGeoSphere("sphere2", rMin - sweep, rMax + sweep, tMin, tMax, pMin, pMax);
  TGeoSphere* sphere3 = new TGeoSphere("sphere3", rMin, rMin + backPlateZ, tMin, tMax, pMin, pMax);
  TGeoSphere* sphere4 = new TGeoSphere("sphere4", rMin - sweep, rMax + backPlateZ + sweep, tMin, tMax, pMin, pMax);

  TGeoBBox* insSeat = new TGeoBBox("insSeat", pinstart[0] * 2, pinstart[1] * 2, pinstart[2] * 2);

  TGeoBBox* quartzRadiatorSeat = new TGeoBBox("quartzRadiatorSeat",
                                              sQuartzRadiatorSide / 2 + sEps + errQrdXY,
                                              sQuartzRadiatorSide / 2 + sEps + errQrdXY,
                                              sQuartzRadiatorZC / 2 + sEps + errQrdZ);

  TGeoBBox* pmtBoxSeat = new TGeoBBox("pmtBoxSeat",
                                      sPmtSide / 2 + sEps + errPMTXY,
                                      sPmtSide / 2 + sEps + errPMTXY,
                                      sPmtZ / 2 + sEps + errPMTZ);
  TGeoBBox* pmtCornerRect = new TGeoBBox("pmtCornerRect",
                                         sCornerRadius / 2 - flopsErr,
                                         sCornerRadius / 2 - flopsErr,
                                         sPmtZ / 2);

  TGeoBBox* framecornerBox = new TGeoBBox("framecornerBox", 5, 5, 10);

  // C-side transformations
  TGeoRotation* rot1 = new TGeoRotation("rot1", 90, 0, 0);
  rot1->RegisterYourself();
  TGeoCombiTrans* rotTr1 = new TGeoCombiTrans("rotTr1", -20, -1, -5, rot1); // cuts off left side of shell
  rotTr1->RegisterYourself();

  TGeoRotation* rot2 = new TGeoRotation("rot2", -90, 0, 0);
  rot2->RegisterYourself();
  TGeoCombiTrans* rotTr2 = new TGeoCombiTrans("rotTr2", 20, -1, -5, rot2);
  rotTr2->RegisterYourself();

  TGeoRotation* rot3 = new TGeoRotation("rot3", 0, 0, 0);
  rot3->RegisterYourself();
  TGeoCombiTrans* rotTr3 = new TGeoCombiTrans("rotTr3", 0, 20, -5, rot3);
  rotTr3->RegisterYourself();

  TGeoTranslation* centerTrans = new TGeoTranslation("centerTrans", 0, 0, 85);
  centerTrans->RegisterYourself();

  TGeoRotation* reflectC1 = new TGeoRotation("reflectC1", 0, 0, 0);
  reflectC1->ReflectX(true);
  reflectC1->ReflectY(true);
  reflectC1->RegisterYourself();

  TGeoRotation* rotCorners = new TGeoRotation("rotCorners", acCorn, bcCorn, gcCorn);
  rotCorners->RegisterYourself();

  TGeoCombiTrans* comCorners = new TGeoCombiTrans("comCorners", xCorn, yCorn, zCorn, rotCorners);
  comCorners->RegisterYourself();

  TGeoCombiTrans* comCorners2 = new TGeoCombiTrans("comCorners2", xCorn2, yCorn2, zCorn2, rotCorners);
  comCorners2->RegisterYourself();

  //Create a string that defines the composite shape
  std::string shellString = "";
  shellString += "sphere1";                      // start with spherical shell - this will be reflected
  shellString += "- sphere2:rotTr1";             // copy and combitrans a subtraction
  shellString += "- sphere2:rotTr2";             //
  shellString += "- sphere2:rotTr3";             //
  shellString += "- insSeat:centerTrans";        // subtract center
  shellString += "- framecornerBox:comCorners";  // subtract corners
  shellString += "- framecornerBox:comCorners2"; //

  //Create string that defines the back plate composite shape
  std::string backPlateString = "";
  backPlateString += "sphere3";
  backPlateString += "- sphere4:rotTr1";
  backPlateString += "- sphere4:rotTr2";
  backPlateString += "- sphere4:rotTr3";
  backPlateString += "- insSeat:centerTrans";
  backPlateString += "- framecornerBox:comCorners";
  backPlateString += "- framecornerBox:comCorners2";

  //These could be set up to use the values in the geometry file after some
  //investigation of subtle differences...
  static constexpr Double_t xi[NCellsC] = {-15.038271418735729, 15.038271418735729, -15.003757581112167, 15.003757581112167, -9.02690018974363, 9.02690018974363, -9.026897413747076, 9.026897413747076, -9.026896531935773, 9.026896531935773, -3.0004568618531313, 3.0004568618531313, -3.0270795197907225, 3.0270795197907225, 3.0003978432927543, -3.0003978432927543, 3.0270569670429572, -3.0270569670429572, 9.026750365564254, -9.026750365564254, 9.026837450695885, -9.026837450695885, 9.026849243816981, -9.026849243816981, 15.038129472387304, -15.038129472387304, 15.003621961057961, -15.003621961057961};
  static constexpr Double_t yi[NCellsC] = {3.1599494336464455, -3.1599494336464455, 9.165191680982874, -9.165191680982874, 3.1383331772537426, -3.1383331772537426, 9.165226363918643, -9.165226363918643, 15.141616002932361, -15.141616002932361, 9.16517861649866, -9.16517861649866, 15.188854859073416, -15.188854859073416, 9.165053319552113, -9.165053319552113, 15.188703787345304, -15.188703787345304, 3.138263189805292, -3.138263189805292, 9.165104089644917, -9.165104089644917, 15.141494417823818, -15.141494417823818, 3.1599158563428644, -3.1599158563428644, 9.165116302773846, -9.165116302773846};

  Double_t zi[NCellsC];
  for (Int_t ic = 0; ic < NCellsC; ic++) {
    zi[ic] = TMath::Sqrt(TMath::Power(crad, 2) - TMath::Power(xi[ic], 2) - TMath::Power(yi[ic], 2));
  }

  // get rotation data
  Double_t ac[NCellsC], bc[NCellsC], gc[NCellsC];
  for (Int_t i = 0; i < NCellsC; i++) {
    ac[i] = TMath::ATan(yi[i] / xi[i]) - TMath::Pi() / 2 + 2 * TMath::Pi();
    if (xi[i] < 0) {
      bc[i] = TMath::ACos(zi[i] / crad);
    } else {
      bc[i] = -1 * TMath::ACos(zi[i] / crad);
    }
  }

  Double_t xc2[NCellsC], yc2[NCellsC], zc2[NCellsC];

  // compensation based on node position within individual detector geometries
  // determine compensated radius
  Double_t rcomp = crad + pstartC[2] / 2.0;
  for (Int_t i = 0; i < NCellsC; i++) {
    // Get compensated translation data
    xc2[i] = rcomp * TMath::Cos(ac[i] + TMath::Pi() / 2) * TMath::Sin(-1 * bc[i]);
    yc2[i] = rcomp * TMath::Sin(ac[i] + TMath::Pi() / 2) * TMath::Sin(-1 * bc[i]);
    zc2[i] = rcomp * TMath::Cos(bc[i]);

    // Convert angles to degrees
    ac[i] *= 180 / TMath::Pi();
    bc[i] *= 180 / TMath::Pi();
    gc[i] = -1 * ac[i];
  }

  Double_t rmag = sqrt(xc2[0] * xc2[0] + yc2[0] * yc2[0] + zc2[0] * zc2[0]);

  Double_t scalePMT = (rmag + (frameHeightC / 2.0) - (sPmtHeightC / 2)) / rmag;
  Double_t scaleQrad = (rmag + (frameHeightC / 2.0) - sPmtHeightC - (sQuartzRadiatorZC / 2.0)) / rmag;

  Double_t xPMT[NCellsC];
  Double_t yPMT[NCellsC];
  Double_t zPMT[NCellsC];

  Double_t aPMT[NCellsC];
  Double_t bPMT[NCellsC];
  Double_t gPMT[NCellsC];

  Double_t xQrad[NCellsC];
  Double_t yQrad[NCellsC];
  Double_t zQrad[NCellsC];

  Double_t aQrad[NCellsC];
  Double_t bQrad[NCellsC];
  Double_t gQrad[NCellsC];

  Double_t rotC[NCellsC];
  Double_t comC[NCellsC];

  for (Int_t i = 0; i < NCellsC; i++) {
    // PMT Transformations
    xPMT[i] = scalePMT * xc2[i];
    yPMT[i] = scalePMT * yc2[i];
    zPMT[i] = scalePMT * zc2[i];

    aPMT[i] = TMath::ATan(yPMT[i] / xPMT[i]) - TMath::Pi() / 2 + 2 * TMath::Pi();
    if (xPMT[i] < 0) {
      bPMT[i] = TMath::ACos(zPMT[i] / crad);
    } else {
      bPMT[i] = -1 * TMath::ACos(zPMT[i] / crad);
    }

    aPMT[i] *= 180 / TMath::Pi();
    bPMT[i] *= 180 / TMath::Pi();
    gPMT[i] = -1 * aPMT[i];

    // Quartz radiator transformations
    xQrad[i] = scaleQrad * xc2[i];
    yQrad[i] = scaleQrad * yc2[i];
    zQrad[i] = scaleQrad * zc2[i];

    aQrad[i] = TMath::ATan(yQrad[i] / xQrad[i]) - TMath::Pi() / 2 + 2 * TMath::Pi();
    if (xQrad[i] < 0) {
      bQrad[i] = TMath::ACos(zQrad[i] / crad);
    } else {
      bQrad[i] = -1 * TMath::ACos(zQrad[i] / crad);
    }
    aQrad[i] *= 180 / TMath::Pi();
    bQrad[i] *= 180 / TMath::Pi();
    gQrad[i] = -1 * aQrad[i];
  }

  TString nameRot;
  TString nameComPMT;
  TString nameComQuartz;
  TString nameComPlates;
  TString nameComC;

  for (Int_t itr = NCellsA; itr < NCellsA + NCellsC; itr++) {
    nameRot = Form("0Rot%i", itr + 1);
    int ic = itr - NCellsA;
    nameComPMT = Form("0ComPMT%i", ic + 1);
    nameComQuartz = Form("0ComQuartz%i", ic + 1);

    // getting even indices to skip reflections -> reflections happen later in
    // frame construction
    if (ic % 2 == 0) {
      TGeoRotation* rotC = new TGeoRotation(nameRot.Data(), ac[ic], bc[ic], gc[ic]);
      rotC->RegisterYourself();

      TGeoCombiTrans* comC = new TGeoCombiTrans(nameComC.Data(), xc2[ic], yc2[ic], zc2[ic], rotC);
      comC->RegisterYourself();

      TGeoRotation* rotPMT = new TGeoRotation(nameRot.Data(), ac[ic], bc[ic], gc[ic]);
      rotPMT->RegisterYourself();

      TGeoCombiTrans* comPMT = new TGeoCombiTrans(nameComPMT.Data(),
                                                  xPMT[ic], yPMT[ic],
                                                  zPMT[ic], rotPMT);
      comPMT->RegisterYourself();

      TGeoRotation* rotQuartz = new TGeoRotation(nameRot.Data(),
                                                 ac[ic], bc[ic], gc[ic]);
      rotQuartz->RegisterYourself();

      TGeoCombiTrans* comQuartz = new TGeoCombiTrans(nameComQuartz.Data(),
                                                     xQrad[ic], yQrad[ic],
                                                     zQrad[ic] - (sQuartzRadiatorZC / 2 + 3 * sEps),
                                                     rotQuartz);
      comQuartz->RegisterYourself();

      TGeoRotation* rotPlates = new TGeoRotation(nameRot.Data(),
                                                 ac[ic], bc[ic], gc[ic]);
      rotPlates->RegisterYourself();
      TGeoCombiTrans* comPlates = new TGeoCombiTrans(nameComPlates.Data(),
                                                     xQrad[ic], yQrad[ic],
                                                     zQrad[ic],
                                                     rotPlates);
      comPlates->RegisterYourself();

      // Subtract the PMTs from the frame
      std::string pmtCombiString = "";
      pmtCombiString += "- ";
      pmtCombiString += "pmtBoxSeat:";
      pmtCombiString += nameComPMT.Data();
      shellString += pmtCombiString;

      // Subtract the QuartzRadiators from the frame
      std::string quartzCombiString = "";
      quartzCombiString += "- ";
      quartzCombiString += "quartzRadiatorSeat:";
      quartzCombiString += nameComQuartz.Data();
      shellString += quartzCombiString;
    }
  }

  // Construct composite shape from boolean
  TGeoCompositeShape* shellCompShape = new TGeoCompositeShape("shellCompShape", shellString.c_str());

  TGeoVolume* shellVol = new TGeoVolume("shellVol", shellCompShape, Al);

  // frame mother assembly
  TGeoVolumeAssembly* FT0_C_Frame = new TGeoVolumeAssembly("FT0_C_Frame");

  // placement and reflections of frame approxes
  TGeoTranslation* shellTr1 = new TGeoTranslation("shellTr1", 0, 0, -80);
  shellTr1->RegisterYourself();

  TGeoCombiTrans* shellTr2 = new TGeoCombiTrans("shellTr2", 0, 0, -80, reflectC1);
  shellTr2->RegisterYourself();

  FT0_C_Frame->AddNode(shellVol, 1, shellTr1);
  FT0_C_Frame->AddNode(shellVol, 2, shellTr2);

  TGeoTranslation* backPlateTr1 = new TGeoTranslation("backPlateTr1", 0, 0, -74);
  backPlateTr1->RegisterYourself();

  TGeoCombiTrans* backPlateTr2 = new TGeoCombiTrans("backPlateTr2", 0, 0, -74, reflectC1);
  backPlateTr2->RegisterYourself();

  TGeoCompositeShape* backPlateShape = new TGeoCompositeShape("backPlateShape", backPlateString.c_str());
  TGeoVolume* backPlateVol = new TGeoVolume("backPlateVol", backPlateShape, Al);

  FT0_C_Frame->AddNode(backPlateVol, 3, backPlateTr1);
  FT0_C_Frame->AddNode(backPlateVol, 4, backPlateTr2);

  return FT0_C_Frame;
}

std::string Detector::cPlateShapeString()
{
  Double_t prismHeight = 0.3895; //height of vertical edge of square prism part of base
  Double_t prismSide = 5.9;      //width and length of square prism part of base
  Double_t radCurve = 81.9469;   //radius of curvature of top part of base
  Double_t delHeight = radCurve * (1.0 - TMath::Sqrt(1.0 - 0.5 * TMath::Power(prismSide / radCurve, 2.0)));
  //height from top of square prism to center of curved top surface of base

  Double_t heightBase = prismHeight + delHeight; //from center of bottom to center of top
  Double_t sliceSide = 5.3;                      //side lengths of slice's flat top
  Double_t heightBaseBox = 2 * heightBase;
  Double_t totalHeight = 0.5;
  Double_t sliceHeight = 0.5 - heightBase;

  //cable dimensions and distances
  Double_t cableHoleWidth = 0.3503;
  Double_t cableHoleLength = 0.9003;
  Double_t cableHoleDepth = 1; //really big just to punch a hole

  //sholes denotes "straight holes" and rholes denote "rotated holes"
  //all distances measured from edges of slice
  //up and down sholes
  Double_t sHolesBottomEdge = 1.585;
  Double_t sHolesTopEdge = 0.515;
  Double_t sHolesAvgTopBottom = (sHolesBottomEdge + sHolesTopEdge) / 2.0;
  Double_t sHolesUpFromCenter = ((sliceSide / 2.0) - sHolesAvgTopBottom); //amount up in x the sholes need to move
  //left and right sholes
  Double_t sHolesFarEdge = 1.585;
  Double_t sHolesNearEdge = 1.065;
  Double_t sHolesAvgNearFar = (sHolesFarEdge + sHolesNearEdge) / 2.0;
  Double_t sHolesLateralFromCenter = ((sliceSide / 2.0) - sHolesAvgNearFar);

  // Create Boxes
  TGeoBBox* box = new TGeoBBox("BASE", prismSide / 2.0, heightBaseBox / 2.0, prismSide / 2.0);

  // Base raw box to be subtracted
  TGeoBBox* slice = new TGeoBBox("SLICE", sliceSide / 2.0, heightBaseBox / 2.0, sliceSide / 2.0);
  TGeoBBox* cableHole = new TGeoBBox("CABLE", cableHoleLength / 2.0, cableHoleDepth / 2.0, cableHoleWidth / 2.0);
  TGeoBBox* cableHole2 = new TGeoBBox("CABLE2", cableHoleWidth / 2.0, cableHoleLength / 2.0, cableHoleDepth / 2.0);

  TGeoSphere* baseShape = new TGeoSphere("BASE_SUBTRACTION", radCurve, radCurve + 5.0, 80, 100, 80, 100);

  TGeoTranslation* rTrans = new TGeoTranslation("rTrans", 0, radCurve, 0);
  rTrans->RegisterYourself();

  TGeoTranslation* rBackTrans = new TGeoTranslation("rBackTrans", 0, -1.0 * radCurve, 0);
  rBackTrans->RegisterYourself();

  TGeoTranslation* subSliceTrans = new TGeoTranslation("subSliceTrans", 0, (heightBaseBox / 2.0) + sliceHeight, 0);
  subSliceTrans->RegisterYourself();

  TGeoTranslation* sHolesTopLeftTrans = new TGeoTranslation("sHolesTopLeftTrans", sHolesUpFromCenter, 0, sHolesLateralFromCenter);
  sHolesTopLeftTrans->RegisterYourself();

  TGeoTranslation* sHolesTopRightTrans = new TGeoTranslation("sHolesTopRightTrans", sHolesUpFromCenter, 0, -1.0 * sHolesLateralFromCenter);
  sHolesTopRightTrans->RegisterYourself();

  TGeoTranslation* testTrans = new TGeoTranslation("testTrans", 0.1, 0.1, 0);
  testTrans->RegisterYourself();

  TGeoRotation* switchToZ = new TGeoRotation("switchToZ", 90, 90, 0);
  switchToZ->RegisterYourself();

  TGeoRotation* rotateHolesLeft = new TGeoRotation("rotateHolesLeft", 345, 0, 0);
  rotateHolesLeft->RegisterYourself();

  TGeoRotation* rotateHolesRight = new TGeoRotation("rotatetHolesRight", 15, 0, 0);
  rotateHolesRight->RegisterYourself();

  // Bottom holes rotation and translation with combitrans
  TGeoCombiTrans* rHolesBottomLeftTrans = new TGeoCombiTrans("rHolesBottomLeftTrans", -1.0 * sHolesLateralFromCenter, -1.0 * sHolesUpFromCenter, 0, rotateHolesLeft);
  rHolesBottomLeftTrans->RegisterYourself();

  TGeoCombiTrans* rHolesBottomRightTrans = new TGeoCombiTrans("rHolesBottomRightTrans", sHolesLateralFromCenter, -1.0 * sHolesUpFromCenter, 0, rotateHolesRight);
  rHolesBottomRightTrans->RegisterYourself();

  std::string plateString = " ";
  plateString += "(((BASE:rTrans";
  plateString += "- BASE_SUBTRACTION)";
  plateString += "+ (SLICE:rTrans))";
  plateString += ":rBackTrans";
  plateString += "- BASE:subSliceTrans";
  plateString += "- (CABLE:sHolesTopLeftTrans)";
  plateString += "- (CABLE:sHolesTopRightTrans))";
  plateString += ":switchToZ";
  plateString += "- (CABLE2:rHolesBottomLeftTrans)";
  plateString += "- (CABLE2:rHolesBottomRightTrans)";

  return plateString;
}
//End Support structure code
////////////////////////////////////////////

Bool_t Detector::ProcessHits(FairVolume* v)
{

  Int_t copy;
  Int_t volID = fMC->CurrentVolID(copy);

  TVirtualMCStack* stack = fMC->GetStack();
  Int_t quadrant, mcp;
  if (fMC->IsTrackEntering()) {
    float x, y, z;
    fMC->TrackPosition(x, y, z);
    fMC->CurrentVolID(quadrant);
    fMC->CurrentVolOffID(1, mcp);
    float time = fMC->TrackTime() * 1.0e9; // time from seconds to ns
    int trackID = stack->GetCurrentTrackNumber();
    int detID = mSim2LUT[4 * mcp + quadrant - 1];
    int iPart = fMC->TrackPid();
    if (fMC->TrackCharge() && volID == mREGVolID) { // charge particles for MCtrue
      AddHit(x, y, z, time, 10, trackID, detID);
    }
    if (iPart == 50000050) { // If particle is photon then ...
      float etot = fMC->Etot();
      float enDep = fMC->Edep();
      Int_t parentID = stack->GetCurrentTrack()->GetMother(0);
      if (volID == mTOPVolID) {
        if (!RegisterPhotoE(etot)) {
          fMC->StopTrack();
          return kFALSE;
        }
        mTrackIdTop = trackID;
      }

      if (volID == mMTOVolID) {
        if (trackID != mTrackIdTop) {
          if (!RegisterPhotoE(etot)) {
            fMC->StopTrack();
            return kFALSE;
          }
          mTrackIdMCPtop = trackID;
        }
      }

      if (volID == mREGVolID) {
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

  // MCP walls Ceramic+Nickel (50//50)
  const Int_t nCeramicNice = 3;
  Float_t aCeramicNicel[3] = {26.981539, 15.9994, 58.6934};
  Float_t zCeramicNicel[3] = {13., 8., 28};
  Float_t wCeramicNicel[3] = {0.2, 0.3, 0.5};
  Float_t denscerCeramicNickel = 5.6;

  // Mixed Cables material simulated as plastic with density taken from description of Low Loss Microwave Coax24 AWG 0
  //   plastic + cooper (6%)
  const Int_t nPlast = 4;
  Float_t aPlast[nPlast] = {1.00784, 12.0107, 15.999, 63.54};
  Float_t zPlast[nPlast] = {1, 6, 8, 29};
  Float_t wPlast[nPlast] = {0.08, 0.53, 0.22, 0.17}; ////!!!!!
  const Float_t denCable = 3.66;

  // Black paper
  // G4Element* elC = new G4Element("Carbon", "C", 6., 12.0107*g/mole);
  // G4Material* C = new G4Material("Carbon Material", 3.52*g/cm3, 1);
  // C->AddElement(elC, 1);

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
    LOG(error) << "Could not read FIT optical properties " << result << " " << optPropPath.Data();
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

  // Define a side mirror border for radiator optical properties
  TVirtualMC::GetMC()->DefineOpSurface("surfRd", kUnified, kDielectric_dielectric, kPolishedbackpainted, 0.);
  TVirtualMC::GetMC()->SetMaterialProperty("surfRd", "EFFICIENCY", nBins, &(mPhotonEnergyD[0]), &(mEfficMet[0]));
  TVirtualMC::GetMC()->SetMaterialProperty("surfRd", "REFLECTIVITY", nBins, &(mPhotonEnergyD[0]), &(mReflMet[0]));
  TVirtualMC::GetMC()->SetBorderSurface("surMirrorBorder0", "0TOP", 1, "0RFV", 1, "surfRd");
  TVirtualMC::GetMC()->SetBorderSurface("surMirrorBorder1", "0TOP", 1, "0RFH", 1, "surfRd");
  TVirtualMC::GetMC()->SetBorderSurface("surMirrorBorder2", "0TOP", 1, "0RFV", 2, "surfRd");
  TVirtualMC::GetMC()->SetBorderSurface("surMirrorBorder3", "0TOP", 1, "0RFH", 2, "surfRd");
  // between cathode and back of front MCP glass window
  TVirtualMC::GetMC()->DefineOpSurface("surFrontBWindow", kUnified, kDielectric_dielectric, kPolished, 0.);
  TVirtualMC::GetMC()->SetMaterialProperty("surFrontBWindow", "EFFICIENCY", nBins, &(mPhotonEnergyD[0]), &(mEffFrontWindow[0]));
  TVirtualMC::GetMC()->SetMaterialProperty("surFrontBWindow", "REFLECTIVITY", nBins, &(mPhotonEnergyD[0]), &(mReflFrontWindow[0]));
  TVirtualMC::GetMC()->SetBorderSurface("surBorderFrontBWindow", "0REG", 1, "0MTO", 1, "surFrontBWindow");
  // between radiator and front MCP glass window
  TVirtualMC::GetMC()->DefineOpSurface("surBackFrontWindow", kUnified, kDielectric_dielectric, kPolished, 0.);
  TVirtualMC::GetMC()->SetMaterialProperty("surBackFrontWindow", "EFFICIENCY", nBins, &(mPhotonEnergyD[0]), &(mEffFrontWindow[0]));
  TVirtualMC::GetMC()->SetMaterialProperty("surBackFrontWindow", "REFLECTIVITY", nBins, &(mPhotonEnergyD[0]), &(mReflFrontWindow[0]));
  TVirtualMC::GetMC()->SetBorderSurface("surBorderBackFrontWindow", "0TOP", 1, "0MTO", 1, "surBackFrontWindow");
}
void Detector::FillOtherOptProperties()
{
  // Set constant values to the other arrays
  for (Int_t i = 0; i < mPhotonEnergyD.size(); i++) {
    mReflBlackPaper.push_back(0.);
    mEffBlackPaper.push_back(0);
    mAbsBlackPaper.push_back(1);

    mReflFrontWindow.push_back(0.01);
    mEffFrontWindow.push_back(1);
    mRindexFrontWindow.push_back(1);

    mRindexAir.push_back(1.);
    mAbsorAir.push_back(0.3);
    mRindexCathodeNext.push_back(1);

    mAbsorbCathodeNext.push_back(1);
    mEfficMet.push_back(0);
    mRindexMet.push_back(1);
    mReflMet.push_back(1);
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
  LOG(info) << " file " << filePath.c_str();
  // Check if file is opened correctly
  if (infile.fail() == true) {
    // AliFatal(Form("Error opening ascii file: %s", filePath.c_str()));
    return -1;
  }

  std::string comment;             // dummy, used just to read 4 first lines and move the cursor to the 5th, otherwise unused
  if (!getline(infile, comment)) { // first comment line
    LOG(error) << "Error opening ascii file (it is probably a folder!): " << filePath.c_str();
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
      //   LOG(error) << "Line number: " << iLine << " reaches range of declared arraySize:" << kNbins << " Check input file:" << filePath.c_str();
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
    //    LOG(error)(Form("Total number of lines %i is different than declared %i. Check input file: %s", iLine, kNbins,
    //    filePath.c_str()));
    return -7;
  }

  LOG(info) << "Optical properties taken from the file: " << filePath.c_str() << " Number of lines read: " << iLine;
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
  LOG(info) << " file  open " << indPath.data();
  // Check if file is opened correctly
  if (infile.fail() == true) {
    LOG(error) << "Error opening ascii file (it is probably a folder!): " << indPath.c_str();
  }
  int fromfile;
  for (int iind = 0; iind < Geometry::Nchannels; iind++) {
    infile >> fromfile;
    mSim2LUT[iind] = fromfile;
  }
}
