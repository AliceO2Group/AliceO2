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
  Float_t pstartC[3] = {20., 20, 5};
  Float_t pstartA[3] = {20, 20, 5};
  Float_t pinstart[3] = {2.9491, 2.9491, 2.5};
  Float_t pmcp[3] = {2.949, 2.949, 1.}; // MCP

  int nCellsA = Geometry::NCellsA;
  int nCellsC = Geometry::NCellsC;

  Geometry geometry;
  TVector3 centerMCP = geometry.centerMCP(2);
  Matrix(idrotm[901], 90., 0., 90., 90., 180., 0.);

  // C side Concave Geometry

  Double_t crad = Geometry::ZdetC; // define concave c-side radius here

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
    std::cout << ic << " " << xc2[ic] << " " << yc2[ic] << std::endl;
    TGeoHMatrix hm = *com[ic];
    TGeoHMatrix* ph = new TGeoHMatrix(hm);
    stlinC->AddNode(ins, itr, ph);
  }

  TGeoVolume* alice = gGeoManager->GetVolume("cave");
  alice->AddNode(stlinA, 1, new TGeoTranslation(0, 0, zdetA));
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

  Float_t pinstart[3] = {2.9491, 2.9491, 2.5};
  Float_t ptop[3] = {1.324, 1.324, 1.}; // Cherenkov radiator
  //  Float_t ptopblack[3] = {1.324, 1.324, 0.0002}; // black paper on the top on radiator
  Float_t ptopref[3] = {1.3241, 1.3241, 1.}; // Cherenkov radiator wrapped with reflector
  Double_t prfv[3] = {0.0002, 1.323, 1.};    // Vertical refracting layer bettwen radiators and between radiator and not optical Air
  Double_t prfh[3] = {1.323, 0.0002, 1.};    // Horizontal refracting layer bettwen radiators and ...
  Float_t pmcp[3] = {2.949, 2.949, 1.};      // MCP
  Float_t pmcpinner[3] = {2.749, 2.749, 0.1};
  Float_t pmcpside[3] = {0.1, 2.949, 1};
  Float_t pmcpbase[3] = {2.949, 2.949, 0.1};
  Float_t pmcptopglass[3] = {2.949, 2.949, 0.1}; // MCP top glass optical

  Float_t preg[3] = {1.324, 1.324, 0.005}; // Photcathode
  Double_t pal[3] = {2.648, 2.648, 0.25};  // 5mm Al on top of each radiator
  // Entry window (glass)
  TVirtualMC::GetMC()->Gsvolu("0TOP", "BOX", getMediumID(kOpGlass), ptop, 3); // Glass radiator
  TGeoVolume* top = gGeoManager->GetVolume("0TOP");
  // TVirtualMC::GetMC()->Gsvolu("0TBL", "BOX", getMediumID(kOptBlack), ptopblack, 3); // Glass radiator
  //  TGeoVolume* topblack = gGeoManager->GetVolume("0TBL");
  TVirtualMC::GetMC()->Gsvolu("0TRE", "BOX", getMediumID(kAir), ptopref, 3); // Air: wrapped  radiator
  TGeoVolume* topref = gGeoManager->GetVolume("0TRE");
  TVirtualMC::GetMC()->Gsvolu("0RFV", "BOX", getMediumID(kOptAl), prfv, 3); // Optical Air vertical
  TGeoVolume* rfv = gGeoManager->GetVolume("0RFV");
  TVirtualMC::GetMC()->Gsvolu("0RFH", "BOX", getMediumID(kOptAl), prfh, 3); // Optical Air horizontal
  TGeoVolume* rfh = gGeoManager->GetVolume("0RFH");

  TVirtualMC::GetMC()->Gsvolu("0PAL", "BOX", getMediumID(kAl), pal, 3); // 5mm Al on top of the radiator
  TGeoVolume* altop = gGeoManager->GetVolume("0PAL");

  TVirtualMC::GetMC()->Gsvolu("0REG", "BOX", getMediumID(kOpGlassCathode), preg, 3);
  TGeoVolume* cat = gGeoManager->GetVolume("0REG");

  //wrapped radiator +  reflecting layers

  Int_t ntops = 0, nrfvs = 0, nrfhs = 0;
  //  Float_t yin = 0, xinv = 0, yinv = 0;
  x = y = z = 0;
  topref->AddNode(top, 1, new TGeoTranslation(0, 0, 0));
  float xinv = -ptop[0] - prfv[0];
  topref->AddNode(rfv, 1, new TGeoTranslation(xinv, 0, 0));
  printf(" GEOGEO  refv %f ,  0,0 \n", xinv);
  xinv = ptop[0] + prfv[0];
  topref->AddNode(rfv, 2, new TGeoTranslation(xinv, 0, 0));
  printf(" GEOGEO  refv %f ,  0,0 \n", xinv);
  float yinv = -ptop[1] - prfh[1];
  topref->AddNode(rfh, 1, new TGeoTranslation(0, yinv, 0));
  printf(" GEOGEO  refh  ,  0, %f, 0 \n", yinv);
  yinv = ptop[1] + prfh[1];
  topref->AddNode(rfh, 2, new TGeoTranslation(0, yinv, 0));
  //  zin = -ptop[2] - ptopblack[2];
  //  printf(" GEOGEO  refh  ,  0, 0, %f \n", zin);
  //  topref->AddNode(topblack, 1, new TGeoTranslation(0, 0, zin));

  //container for radiator, cathode
  for (Int_t ix = 0; ix < 2; ix++) {
    float xin = -pinstart[0] + 0.3 + (ix + 0.5) * 2 * ptopref[0];
    for (Int_t iy = 0; iy < 2; iy++) {
      z = -pinstart[2] + 2 * pal[2] + ptopref[2];
      float yin = -pinstart[1] + 0.3 + (iy + 0.5) * 2 * ptopref[1];
      ntops++;
      ins->AddNode(topref, ntops, new TGeoTranslation(xin, yin, z));
      z += ptopref[2] + 2. * pmcptopglass[2] + preg[2];
      ins->AddNode(cat, ntops, new TGeoTranslation(xin, yin, z));
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
    int detID = 4 * mcp + quadrant - 1;
    float etot = fMC->Etot();
    int iPart = fMC->TrackPid();
    float enDep = fMC->Edep();
    Int_t parentID = stack->GetCurrentTrack()->GetMother(0);
    if (fMC->TrackCharge()) { //charge particles for MCtrue
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
            //   std::cout<<" брысь "<<etot<<" track "<<trackID<<" parentID "<<parentID<<" "<<volname<<std::endl;
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
  Medium(18, "OpBlack$", 2, 0, isxfld, sxmgmx, 10., .1, 1., .003, .003);
  Medium(15, "Aluminium$", 11, 0, isxfld, sxmgmx, 10., .01, 1., .003, .003);
  Medium(17, "OptAluminium$", 11, 0, isxfld, sxmgmx, 10., .01, 1., .003, .003);
  Medium(16, "OpticalGlass$", 24, 1, isxfld, sxmgmx, 10., .01, .1, .003, .01);
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
  TVirtualMC::GetMC()->SetBorderSurface("surBorderFrontWindow", "0TOP", 1, "0MT0", 1, "surFrontWindow");
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
  if (p > eff)
    return kFALSE;

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
