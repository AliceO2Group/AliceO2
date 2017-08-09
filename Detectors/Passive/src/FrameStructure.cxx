// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DetectorsPassive/FrameStructure.h"
#include "DetectorsBase/Detector.h"
#include <TGeoBBox.h>
#include <TGeoCompositeShape.h>
#include <TGeoManager.h>
#include <TGeoMatrix.h>
#include <TGeoPgon.h>
#include <TGeoTrd1.h>
#include <TMath.h>
#include <TString.h>
#include <TSystem.h>
#include <TVirtualMC.h>

#include <cassert>

namespace o2
{
namespace Passive
{
const char* TOPNAME = "cave";
// in AliROOT this was TOPNAME="ALIC"

// Implementation based on AliFRAMEv2 in AliRoot
FrameStructure::FrameStructure(const char* name, const char* Title) : FairModule(name, Title)
{
  // for the moment we are using the VMC interfaces for geometry,material handling
  // so verify that this is correctly initialized
}

#define AliMatrix TVirtualMC::GetMC()->Matrix
#define kRaddeg TMath::RadToDeg()
#define kDegrad TMath::DegToRad()

void FrameStructure::MakeHeatScreen(const char* name, Float_t dyP, Int_t rot1, Int_t rot2)
{
  // Heat screen panel
  //
  const Int_t kAir = mAirMedID;
  const Int_t kAlu = mAluMedID;

  Float_t dx, dy;
  char mname[16];
  char cname[16];
  char t1name[16];
  char t2name[16];
  char t3name[16];
  char t4name[16];
  char t5name[16];

  //
  Float_t dxP = 2. * (287. * TMath::Sin(10. * TMath::Pi() / 180.) - 2.);
  Float_t dzP = 1.05;
  //
  // Mother volume
  Float_t thshM[3];
  thshM[0] = dxP / 2.;
  thshM[1] = dyP / 2.;
  thshM[2] = dzP / 2.;
  snprintf(mname, 16, "BTSH_%s", name);
  TVirtualMC::GetMC()->Gsvolu(mname, "BOX ", kAir, thshM, 3);
  //
  // Aluminum sheet
  thshM[2] = 0.025;
  snprintf(cname, 16, "BTSHA_%s", name);
  TVirtualMC::GetMC()->Gsvolu(cname, "BOX ", kAlu, thshM, 3);
  TVirtualMC::GetMC()->Gspos(cname, 1, mname, 0., 0., -0.5, 0);
  //
  // Tubes
  Float_t thshT[3];
  thshT[0] = 0.4;
  thshT[1] = 0.5;
  thshT[2] = (dyP / 2. - 8.);
  //
  snprintf(t1name, 16, "BTSHT1_%s", name);
  TVirtualMC::GetMC()->Gsvolu(t1name, "TUBE", kAlu, thshT, 3);
  dx = -dxP / 2. + 8. - 0.5;
  TVirtualMC::GetMC()->Gspos(t1name, 1, mname, dx, 0., 0.025, rot1);
  //
  snprintf(t2name, 16, "BTSHT2_%s", name);
  snprintf(t3name, 16, "BTSHT3_%s", name);
  snprintf(t4name, 16, "BTSHT4_%s", name);
  snprintf(t5name, 16, "BTSHT5_%s", name);
  thshT[2] = (thshM[1] - 12.);
  TVirtualMC::GetMC()->Gsvolu(t2name, "TUBE", kAlu, thshT, 3);
  thshT[2] = 7.9 / 2.;
  TVirtualMC::GetMC()->Gsvolu(t3name, "TUBE", kAlu, thshT, 3);
  thshT[2] = 23.9 / 2.;
  TVirtualMC::GetMC()->Gsvolu(t4name, "TUBE", kAlu, thshT, 3);

  Int_t sig = 1;
  Int_t ipo = 1;
  for (Int_t i = 0; i < 5; i++) {
    sig *= -1;
    dx += 8.00;
    dy = 4. * sig;
    Float_t dy1 = -(thshM[1] - 15.5) * sig;
    Float_t dy2 = -(thshM[1] - 7.5) * sig;

    TVirtualMC::GetMC()->Gspos(t2name, ipo++, mname, dx, dy, 0.025, rot1);
    dx += 6.9;
    TVirtualMC::GetMC()->Gspos(t2name, ipo++, mname, dx, dy, 0.025, rot1);

    TVirtualMC::GetMC()->Gspos(t3name, i + 1, mname, dx - 3.45, dy1, 0.025, rot2);
    TVirtualMC::GetMC()->Gspos(t4name, i + 1, mname, dx - 3.45, dy2, 0.025, rot2);
  }
  dx += 8.;
  TVirtualMC::GetMC()->Gspos(t1name, 2, mname, dx, 0., 0.025, rot1);
  TVirtualMC::GetMC()->Gspos(t3name, 6, mname, dx - 3.45, -(thshM[1] - 7.5), 0.025, rot2);
}

void FrameStructure::WebFrame(const char* name, Float_t dHz, Float_t theta0, Float_t phi0)
{
  //
  // Create a web frame element
  //
  const Float_t krad2deg = 180. / TMath::Pi();
  const Float_t kdeg2rad = 1. / krad2deg;
  const Int_t kAir = mAirMedID;
  const Int_t kSteel = mSteelMedID;

  Float_t ptrap[11];
  char nameA[16];
  snprintf(nameA, 16, "%sA", name);
  theta0 *= kdeg2rad;
  phi0 *= kdeg2rad;
  Float_t theta = TMath::ATan(TMath::Tan(theta0) / TMath::Sin(phi0));
  Float_t phi = TMath::ACos(TMath::Cos(theta0) * TMath::Cos(phi0));
  if (phi0 < 0)
    phi = -phi;

  phi *= krad2deg;
  theta *= krad2deg;

  ptrap[0] = dHz / 2;
  ptrap[2] = theta;
  ptrap[1] = phi;
  ptrap[3] = 6. / cos(theta0 * kdeg2rad) / 2.;
  ptrap[4] = 1.;
  ptrap[5] = ptrap[4];
  ptrap[6] = 0;
  ptrap[7] = ptrap[3];
  ptrap[8] = ptrap[4];
  ptrap[9] = ptrap[4];
  ptrap[10] = 0;
  TVirtualMC::GetMC()->Gsvolu(name, "TRAP", kSteel, ptrap, 11);
  ptrap[3] = (6. - 1.) / cos(theta0 * kdeg2rad) / 2.;
  ptrap[4] = 0.75;
  ptrap[5] = ptrap[4];
  ptrap[7] = ptrap[3];
  ptrap[8] = ptrap[4];
  ptrap[9] = ptrap[4];

  TVirtualMC::GetMC()->Gsvolu(nameA, "TRAP", kAir, ptrap, 11);
  TVirtualMC::GetMC()->Gspos(nameA, 1, name, 0.0, -0.25, 0., 0, "ONLY");
  gGeoManager->GetVolume(name)->SetVisibility(1);
}

void FrameStructure::ConstructGeometry()
{
  // TODO: we should be allowed to call this function only once
  if (TVirtualMC::GetMC() == nullptr) {
    throw std::runtime_error("VMC instance not initialized");
  }

  // verify that we have the world volume already setup
  if (gGeoManager != nullptr && gGeoManager->GetVolume(TOPNAME)) {
    mCaveIsAvailable = true;
  }

  // create materials
  CreateMaterials();

  Int_t idrotm[2299];

  AliMatrix(idrotm[2070], 90.0, 0.0, 90.0, 270.0, 0.0, 0.0);
  //
  AliMatrix(idrotm[2083], 170.0, 0.0, 90.0, 90.0, 80.0, 0.0);
  AliMatrix(idrotm[2084], 170.0, 180.0, 90.0, 90.0, 80.0, 180.0);
  AliMatrix(idrotm[2085], 90.0, 180.0, 90.0, 90.0, 0.0, 0.0);
  //
  AliMatrix(idrotm[2086], 90.0, 0.0, 90.0, 90., 0.0, 0.0);
  AliMatrix(idrotm[2087], 90.0, 180.0, 90.0, 270., 0.0, 0.0);
  AliMatrix(idrotm[2088], 90.0, 90.0, 90.0, 180., 0.0, 0.0);
  AliMatrix(idrotm[2089], 90.0, 90.0, 90.0, 0., 0.0, 0.0);
  //
  AliMatrix(idrotm[2090], 90.0, 0.0, 0.0, 0., 90.0, 90.0);
  AliMatrix(idrotm[2091], 0.0, 0.0, 90.0, 90., 90.0, 0.0);
  //
  // Matrices have been imported from Euclid. Some simplification
  // seems possible
  //

  AliMatrix(idrotm[2003], 0.0, 0.0, 90.0, 130.0, 90.0, 40.0);
  AliMatrix(idrotm[2004], 180.0, 0.0, 90.0, 130.0, 90.0, 40.0);
  AliMatrix(idrotm[2005], 180.0, 0.0, 90.0, 150.0, 90.0, 240.0);
  AliMatrix(idrotm[2006], 0.0, 0.0, 90.0, 150.0, 90.0, 240.0);
  AliMatrix(idrotm[2007], 0.0, 0.0, 90.0, 170.0, 90.0, 80.0);
  AliMatrix(idrotm[2008], 180.0, 0.0, 90.0, 190.0, 90.0, 280.0);
  AliMatrix(idrotm[2009], 180.0, 0.0, 90.0, 170.0, 90.0, 80.0);
  AliMatrix(idrotm[2010], 0.0, 0.0, 90.0, 190.0, 90.0, 280.0);
  AliMatrix(idrotm[2011], 0.0, 0.0, 90.0, 350.0, 90.0, 260.0);
  AliMatrix(idrotm[2012], 180.0, 0.0, 90.0, 350.0, 90.0, 260.0);
  AliMatrix(idrotm[2013], 180.0, 0.0, 90.0, 10.0, 90.0, 100.0);
  AliMatrix(idrotm[2014], 0.0, 0.0, 90.0, 10.0, 90.0, 100.0);
  AliMatrix(idrotm[2015], 0.0, 0.0, 90.0, 30.0, 90.0, 300.0);
  AliMatrix(idrotm[2016], 180.0, 0.0, 90.0, 30.0, 90.0, 300.0);
  AliMatrix(idrotm[2017], 180.0, 0.0, 90.0, 50.0, 90.0, 140.0);
  AliMatrix(idrotm[2018], 0.0, 0.0, 90.0, 50.0, 90.0, 140.0);

  AliMatrix(idrotm[2019], 180.0, 0.0, 90.0, 130.0, 90.0, 220.0);
  AliMatrix(idrotm[2020], 180.0, 0.0, 90.0, 50.0, 90.0, 320.0);
  AliMatrix(idrotm[2021], 180.0, 0.0, 90.0, 150.0, 90.0, 60.0);
  AliMatrix(idrotm[2022], 180.0, 0.0, 90.0, 30.0, 90.0, 120.0);
  AliMatrix(idrotm[2023], 180.0, 0.0, 90.0, 170.0, 90.0, 260.0);
  AliMatrix(idrotm[2024], 180.0, 0.0, 90.0, 190.0, 90.0, 100.0);
  AliMatrix(idrotm[2025], 180.0, 0.0, 90.0, 350.0, 90.0, 80.0);
  AliMatrix(idrotm[2026], 180.0, 0.0, 90.0, 10.0, 90.0, 280.0);

  AliMatrix(idrotm[2027], 0.0, 0.0, 90.0, 50.0, 90.0, 320.0);
  AliMatrix(idrotm[2028], 0.0, 0.0, 90.0, 150.0, 90.0, 60.0);
  AliMatrix(idrotm[2029], 0.0, 0.0, 90.0, 30.0, 90.0, 120.0);
  AliMatrix(idrotm[2030], 0.0, 0.0, 90.0, 10.0, 90.0, 280.0);
  AliMatrix(idrotm[2031], 0.0, 0.0, 90.0, 170.0, 90.0, 260.0);
  AliMatrix(idrotm[2032], 0.0, 0.0, 90.0, 190.0, 90.0, 100.0);
  AliMatrix(idrotm[2033], 0.0, 0.0, 90.0, 350.0, 90.0, 80.0);

  //
  // The Space frame
  //
  //
  Float_t pbox[3], ptrap[11], ptrd1[4], ppgon[10];

  Float_t dx, dy, dz;
  Int_t i, j, jmod;
  jmod = 0;
  //
  // Constants
  const Float_t kEps = 0.01;
  const Int_t kAir = mAirMedID;
  const Int_t kSteel = mSteelMedID;

  const Float_t krad2deg = 180. / TMath::Pi();
  const Float_t kdeg2rad = 1. / krad2deg;

  Float_t iFrH = 118.66; // Height of inner frame
  Float_t ringH = 6.00;  // Height of the ring bars
  Float_t ringW = 10.00; // Width  of the ring bars in z
  Float_t longH = 6.00;
  Float_t longW = 4.00;
  //
  Float_t dymodU[3] = { 70.0, 224.0, 340.2 };
  Float_t dymodL[3] = { 50.0, 175.0, 297.5 };
  //

  //
  // Frame mother volume
  //
  TGeoPgon* shB77A = new TGeoPgon(0., 360., 18, 2);
  shB77A->SetName("shB77A");
  shB77A->DefineSection(0, -376.5, 280., 415.7);
  shB77A->DefineSection(1, 376.5, 280., 415.7);
  TGeoBBox* shB77B = new TGeoBBox(3.42, 2., 375.5);
  shB77B->SetName("shB77B");
  TGeoTranslation* trB77A = new TGeoTranslation("trB77A", +283.32, 0., 0.);
  TGeoTranslation* trB77B = new TGeoTranslation("trB77B", -283.32, 0., 0.);
  trB77A->RegisterYourself();
  trB77B->RegisterYourself();
  TGeoCompositeShape* shB77 = new TGeoCompositeShape("shB77", "shB77A+shB77B:trB77A+shB77B:trB77B");
  TGeoVolume* voB77 = new TGeoVolume("B077", shB77, gGeoManager->GetMedium(mAirMedID));
  voB77->SetName("B077"); // just to avoid a warning

  if (mCaveIsAvailable) {
    TVirtualMC::GetMC()->Gspos("B077", 1, TOPNAME, 0., 0., 0., 0, "ONLY");
  }

  //
  // Reference plane #1 for TRD
  TGeoPgon* shBREFA = new TGeoPgon(0.0, 360., 18, 2);
  shBREFA->DefineSection(0, -376., 280., 280.1);
  shBREFA->DefineSection(1, 376., 280., 280.1);
  shBREFA->SetName("shBREFA");
  TGeoCompositeShape* shBREF1 = new TGeoCompositeShape("shBREF1", "shBREFA-(shB77B:trB77A+shB77B:trB77B)");
  TGeoVolume* voBREF = new TGeoVolume("BREF1", shBREF1, gGeoManager->GetMedium(mAirMedID));
  voBREF->SetVisibility(0);
  TVirtualMC::GetMC()->Gspos("BREF1", 1, "B077", 0., 0., 0., 0, "ONLY");
  //
  //  The outer Frame
  //

  Float_t dol = 4.;
  Float_t doh = 4.;
  Float_t ds = 0.63;
  //
  // Mother volume
  //
  ppgon[0] = 0.;
  ppgon[1] = 360.;
  ppgon[2] = 18.;

  ppgon[3] = 2.;

  ppgon[4] = -350.;
  ppgon[5] = 401.35;
  ppgon[6] = 415.6;

  ppgon[7] = -ppgon[4];
  ppgon[8] = ppgon[5];
  ppgon[9] = ppgon[6];
  TVirtualMC::GetMC()->Gsvolu("B076", "PGON", kAir, ppgon, 10);
  TVirtualMC::GetMC()->Gspos("B076", 1, "B077", 0., 0., 0., 0, "ONLY");
  //
  // Rings
  //
  dz = 2. * 410.2 * TMath::Sin(10. * kdeg2rad) - 2. * dol * TMath::Cos(10. * kdeg2rad) -
       2. * doh * TMath::Tan(10. * kdeg2rad);
  Float_t l1 = dz / 2.;
  Float_t l2 = dz / 2. + 2. * doh * TMath::Tan(10. * kdeg2rad);

  TGeoVolumeAssembly* asBI42 = new TGeoVolumeAssembly("BI42");
  // Horizontal
  ptrd1[0] = l2 - 0.6 * TMath::Tan(10. * kdeg2rad);
  ptrd1[1] = l2;
  ptrd1[2] = 8.0 / 2.;
  ptrd1[3] = 0.6 / 2.;
  TVirtualMC::GetMC()->Gsvolu("BIH142", "TRD1", kSteel, ptrd1, 4);
  ptrd1[0] = l1;
  ptrd1[1] = l1 + 0.6 * TMath::Tan(10. * kdeg2rad);
  ptrd1[2] = 8.0 / 2.;
  ptrd1[3] = 0.6 / 2.;
  TVirtualMC::GetMC()->Gsvolu("BIH242", "TRD1", kSteel, ptrd1, 4);

  // Vertical
  ptrd1[0] = l1 + 0.6 * TMath::Tan(10. * kdeg2rad);
  ptrd1[1] = l2 - 0.6 * TMath::Tan(10. * kdeg2rad);
  ptrd1[2] = 0.8 / 2.;
  ptrd1[3] = 6.8 / 2.;
  TVirtualMC::GetMC()->Gsvolu("BIV42", "TRD1", kSteel, ptrd1, 4);
  // Place
  asBI42->AddNode(gGeoManager->GetVolume("BIV42"), 1, new TGeoTranslation(0., 0., 0.));
  asBI42->AddNode(gGeoManager->GetVolume("BIH142"), 1, new TGeoTranslation(0., 0., 3.7));
  asBI42->AddNode(gGeoManager->GetVolume("BIH242"), 1, new TGeoTranslation(0., 0., -3.7));
  //
  // longitudinal bars
  //
  // 80 x 80 x 6.3
  //
  pbox[0] = dol;
  pbox[1] = doh;
  pbox[2] = 345.;
  TVirtualMC::GetMC()->Gsvolu("B033", "BOX", kSteel, pbox, 3);
  pbox[0] = dol - ds;
  pbox[1] = doh - ds;
  TVirtualMC::GetMC()->Gsvolu("B034", "BOX", kAir, pbox, 3);
  TVirtualMC::GetMC()->Gspos("B034", 1, "B033", 0., 0., 0., 0, "ONLY");

  //
  // TPC support
  //
  pbox[0] = 3.37;
  pbox[1] = 2.0;
  pbox[2] = 375.5;
  TVirtualMC::GetMC()->Gsvolu("B080", "BOX", kSteel, pbox, 3);
  pbox[0] = 2.78;
  pbox[1] = 1.4;
  pbox[2] = 375.5;
  TVirtualMC::GetMC()->Gsvolu("B081", "BOX", kAir, pbox, 3);
  TVirtualMC::GetMC()->Gspos("B081", 1, "B080", 0., 0., 0., 0, "ONLY");

  // Small 2nd reference plane elemenet
  pbox[0] = 0.05;
  pbox[1] = 2.0;
  pbox[2] = 375.5;
  TVirtualMC::GetMC()->Gsvolu("BREF2", "BOX", kAir, pbox, 3);
  TVirtualMC::GetMC()->Gspos("BREF2", 1, "B080", 3.37 - 0.05, 0., 0., 0, "ONLY");

  TVirtualMC::GetMC()->Gspos("B080", 1, "B077", 283.3, 0., 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("B080", 2, "B077", -283.3, 0., 0., idrotm[2087], "ONLY");

  //
  // Diagonal bars (1)
  //
  Float_t h, d, dq, x, theta;

  h = (dymodU[1] - dymodU[0] - 2. * dol) * .999;
  d = 2. * dol;
  dq = h * h + dz * dz;

  x = TMath::Sqrt((dz * dz - d * d) / dq + d * d * h * h / dq / dq) + d * h / dq;

  theta = krad2deg * TMath::ACos(x);

  ptrap[0] = dz / 2.;
  ptrap[1] = theta;
  ptrap[2] = 0.;
  ptrap[3] = doh;
  ptrap[4] = dol / x;
  ptrap[5] = ptrap[4];
  ptrap[6] = 0;
  ptrap[7] = ptrap[3];
  ptrap[8] = ptrap[4];
  ptrap[9] = ptrap[4];
  ptrap[10] = 0;

  TVirtualMC::GetMC()->Gsvolu("B047", "TRAP", kSteel, ptrap, 11);
  ptrap[3] = doh - ds;
  ptrap[4] = (dol - ds) / x;
  ptrap[5] = ptrap[4];
  ptrap[7] = ptrap[3];
  ptrap[8] = ptrap[4];
  ptrap[9] = ptrap[4];
  TVirtualMC::GetMC()->Gsvolu("B048", "TRAP", kAir, ptrap, 11);
  TVirtualMC::GetMC()->Gspos("B048", 1, "B047", 0.0, 0.0, 0., 0, "ONLY");

  /*
   Crosses (inner most)
         \\  //
          \\//
          //\\
         //  \\
  */
  h = (2. * dymodU[0] - 2. * dol) * .999;
  //
  // Mother volume
  //
  pbox[0] = h / 2;
  pbox[1] = doh;
  pbox[2] = dz / 2.;
  TVirtualMC::GetMC()->Gsvolu("BM49", "BOX ", kAir, pbox, 3);

  dq = h * h + dz * dz;
  x = TMath::Sqrt((dz * dz - d * d) / dq + d * d * h * h / dq / dq) + d * h / dq;
  theta = krad2deg * TMath::ACos(x);

  ptrap[0] = dz / 2. - kEps;
  ptrap[1] = theta;
  ptrap[2] = 0.;
  ptrap[3] = doh - kEps;
  ptrap[4] = dol / x;
  ptrap[5] = ptrap[4];
  ptrap[7] = ptrap[3];
  ptrap[8] = ptrap[4];
  ptrap[9] = ptrap[4];

  TVirtualMC::GetMC()->Gsvolu("B049", "TRAP", kSteel, ptrap, 11);
  ptrap[0] = ptrap[0] - kEps;
  ptrap[3] = (doh - ds);
  ptrap[4] = (dol - ds) / x;
  ptrap[5] = ptrap[4];
  ptrap[7] = ptrap[3];
  ptrap[8] = ptrap[4];
  ptrap[9] = ptrap[4];
  TVirtualMC::GetMC()->Gsvolu("B050", "TRAP", kAir, ptrap, 11);
  TVirtualMC::GetMC()->Gspos("B050", 1, "B049", 0.0, 0.0, 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("B049", 1, "BM49", 0.0, 0.0, 0., 0, "ONLY");

  Float_t dd1 = d * TMath::Tan(theta * kdeg2rad);
  Float_t dd2 = d / TMath::Tan(2. * theta * kdeg2rad);
  Float_t theta2 = TMath::ATan(TMath::Abs(dd2 - dd1) / d / 2.);

  ptrap[0] = dol;
  ptrap[1] = theta2 * krad2deg;
  ptrap[2] = 0.;
  ptrap[3] = doh;
  ptrap[4] = (dz / 2. / x - dd1 - dd2) / 2.;
  ptrap[5] = ptrap[4];
  ptrap[6] = 0.;
  ptrap[7] = ptrap[3];
  ptrap[8] = dz / 4. / x;
  ptrap[9] = ptrap[8];

  TVirtualMC::GetMC()->Gsvolu("B051", "TRAP", kSteel, ptrap, 11);
  Float_t ddx0 = ptrap[8];

  Float_t dd1s = dd1 * (1. - 2. * ds / d);
  Float_t dd2s = dd2 * (1. - 2. * ds / d);
  Float_t theta2s = TMath::ATan(TMath::Abs(dd2s - dd1s) / (d - 2. * ds) / 2.);

  ptrap[0] = dol - ds;
  ptrap[1] = theta2s * krad2deg;
  ptrap[2] = 0.;
  ptrap[3] = doh - ds;
  ptrap[4] = ptrap[4] + ds / d / 2. * (dd1 + dd2);
  ptrap[5] = ptrap[4];
  ptrap[6] = 0.;
  ptrap[7] = ptrap[3];
  ptrap[8] = ptrap[8] - ds / 2. / d * (dd1 + dd2);
  ptrap[9] = ptrap[8];

  TVirtualMC::GetMC()->Gsvolu("B052", "TRAP", kAir, ptrap, 11);
  TVirtualMC::GetMC()->Gspos("B052", 1, "B051", 0.0, 0.0, 0., 0, "ONLY");

  Float_t ddx, ddz, drx, drz, rtheta;

  AliMatrix(idrotm[2001], -theta + 180, 0.0, 90.0, 90.0, 90. - theta, 0.0);
  rtheta = (90. - theta) * kdeg2rad;
  ddx = -ddx0 - dol * TMath::Tan(theta2);
  ddz = -dol;

  drx = TMath::Cos(rtheta) * ddx + TMath::Sin(rtheta) * ddz + pbox[0];
  drz = -TMath::Sin(rtheta) * ddx + TMath::Cos(rtheta) * ddz - pbox[2];
  TVirtualMC::GetMC()->Gspos("B051", 1, "BM49", drx, 0.0, drz, idrotm[2001], "ONLY");

  AliMatrix(idrotm[2002], -theta, 0.0, 90.0, 90.0, 270. - theta, 0.0);
  rtheta = (270. - theta) * kdeg2rad;

  drx = TMath::Cos(rtheta) * ddx + TMath::Sin(rtheta) * ddz - pbox[0];
  drz = -TMath::Sin(rtheta) * ddx + TMath::Cos(rtheta) * ddz + pbox[2];
  TVirtualMC::GetMC()->Gspos("B051", 2, "BM49", drx, 0.0, drz, idrotm[2002], "ONLY");

  //
  // Diagonal bars (3)
  //
  h = ((dymodU[2] - dymodU[1]) - 2. * dol) * .999;
  dq = h * h + dz * dz;
  x = TMath::Sqrt((dz * dz - d * d) / dq + d * d * h * h / dq / dq) + d * h / dq;
  theta = krad2deg * TMath::ACos(x);

  ptrap[0] = dz / 2.;
  ptrap[1] = theta;
  ptrap[3] = doh;
  ptrap[4] = dol / x;
  ptrap[5] = ptrap[4];
  ptrap[7] = ptrap[3];
  ptrap[8] = ptrap[4];
  ptrap[9] = ptrap[4];

  TVirtualMC::GetMC()->Gsvolu("B045", "TRAP", kSteel, ptrap, 11);
  ptrap[3] = doh - ds;
  ptrap[4] = (dol - ds) / x;
  ptrap[5] = ptrap[4];
  ptrap[7] = ptrap[3];
  ptrap[8] = ptrap[4];
  ptrap[9] = ptrap[4];
  TVirtualMC::GetMC()->Gsvolu("B046", "TRAP", kAir, ptrap, 11);
  TVirtualMC::GetMC()->Gspos("B046", 1, "B045", 0.0, 0.0, 0., 0, "ONLY");

  //
  // Positioning of diagonal bars

  Float_t rd = 405.5;
  dz = (dymodU[1] + dymodU[0]) / 2.;
  Float_t dz2 = (dymodU[1] + dymodU[2]) / 2.;

  //
  //  phi = 40
  //
  Float_t phi = 40;
  dx = rd * TMath::Sin(phi * kdeg2rad);
  dy = rd * TMath::Cos(phi * kdeg2rad);

  TVirtualMC::GetMC()->Gspos("B045", 1, "B076", -dx, dy, dz2, idrotm[2019], "ONLY");
  TVirtualMC::GetMC()->Gspos("B045", 2, "B076", -dx, dy, -dz2, idrotm[2003], "ONLY"); // ?
  TVirtualMC::GetMC()->Gspos("B045", 3, "B076", dx, dy, dz2, idrotm[2020], "ONLY");
  TVirtualMC::GetMC()->Gspos("B045", 4, "B076", dx, dy, -dz2, idrotm[2027], "ONLY");

  //
  //  phi = 60
  //

  phi = 60;
  dx = rd * TMath::Sin(phi * kdeg2rad);
  dy = rd * TMath::Cos(phi * kdeg2rad);

  TVirtualMC::GetMC()->Gspos("B045", 5, "B076", -dx, dy, dz2, idrotm[2021], "ONLY");
  TVirtualMC::GetMC()->Gspos("B045", 6, "B076", -dx, dy, -dz2, idrotm[2028], "ONLY");
  TVirtualMC::GetMC()->Gspos("B045", 7, "B076", dx, dy, dz2, idrotm[2022], "ONLY");
  TVirtualMC::GetMC()->Gspos("B045", 8, "B076", dx, dy, -dz2, idrotm[2029], "ONLY");

  //
  //  phi = 80
  //

  phi = 80;
  dx = rd * TMath::Sin(phi * kdeg2rad);
  dy = rd * TMath::Cos(phi * kdeg2rad);

  TVirtualMC::GetMC()->Gspos("B047", 13, "B076", -dx, -dy, dz, idrotm[2008], "ONLY");
  TVirtualMC::GetMC()->Gspos("B047", 14, "B076", -dx, -dy, -dz, idrotm[2010], "ONLY");
  TVirtualMC::GetMC()->Gspos("B047", 15, "B076", dx, -dy, dz, idrotm[2012], "ONLY");
  TVirtualMC::GetMC()->Gspos("B047", 16, "B076", dx, -dy, -dz, idrotm[2011], "ONLY");

  TVirtualMC::GetMC()->Gspos("B045", 9, "B076", -dx, dy, dz2, idrotm[2023], "ONLY");
  TVirtualMC::GetMC()->Gspos("B045", 10, "B076", -dx, dy, -dz2, idrotm[2031], "ONLY");
  TVirtualMC::GetMC()->Gspos("B045", 11, "B076", dx, dy, dz2, idrotm[2026], "ONLY");
  TVirtualMC::GetMC()->Gspos("B045", 12, "B076", dx, dy, -dz2, idrotm[2030], "ONLY");

  TVirtualMC::GetMC()->Gspos("B045", 13, "B076", -dx, -dy, dz2, idrotm[2024], "ONLY");
  TVirtualMC::GetMC()->Gspos("B045", 14, "B076", -dx, -dy, -dz2, idrotm[2032], "ONLY");
  TVirtualMC::GetMC()->Gspos("B045", 15, "B076", dx, -dy, dz2, idrotm[2025], "ONLY");
  TVirtualMC::GetMC()->Gspos("B045", 16, "B076", dx, -dy, -dz2, idrotm[2033], "ONLY");

  TVirtualMC::GetMC()->Gspos("BM49", 7, "B076", dx, -dy, 0., idrotm[2025], "ONLY");
  TVirtualMC::GetMC()->Gspos("BM49", 8, "B076", -dx, -dy, 0., idrotm[2024], "ONLY");

  //
  // The internal frame
  //
  //
  //
  //  Mother Volumes
  //

  ptrd1[0] = 49.8;
  ptrd1[1] = 70.7;
  ptrd1[2] = 376.5;
  ptrd1[3] = iFrH / 2.;

  Float_t r = 342.0;
  Float_t rout1 = 405.5;
  Float_t rout2 = 411.55;
  TString module[18];

  for (i = 0; i < 18; i++) {
    // Create volume i
    char name[16];
    Int_t mod = i + 13;
    if (mod > 17)
      mod -= 18;
    snprintf(name, 16, "BSEGMO%d", mod);
    TVirtualMC::GetMC()->Gsvolu(name, "TRD1", kAir, ptrd1, 4);
    gGeoManager->GetVolume(name)->SetVisibility(kFALSE);

    module[i] = name;
    // Place volume i
    Float_t phi1 = i * 20.;
    Float_t phi2 = 270 + phi1;
    if (phi2 >= 360.)
      phi2 -= 360.;

    dx = TMath::Sin(phi1 * kdeg2rad) * r;
    dy = -TMath::Cos(phi1 * kdeg2rad) * r;

    char nameR[16];
    snprintf(nameR, 16, "B43_Rot_%d", i);
    TGeoRotation* rot = new TGeoRotation(nameR, 90.0, phi1, 0., 0., 90., phi2);
    AliMatrix(idrotm[2034 + i], 90.0, phi1, 0., 0., 90., phi2);
    TGeoVolume* vol77 = gGeoManager->GetVolume("B077");
    TGeoVolume* volS = gGeoManager->GetVolume(name);
    vol77->AddNode(volS, 1, new TGeoCombiTrans(dx, dy, 0., rot));

    //
    //    Position elements of outer Frame
    //
    dx = TMath::Sin(phi1 * kdeg2rad) * rout1;
    dy = -TMath::Cos(phi1 * kdeg2rad) * rout1;
    for (j = 0; j < 3; j++) {
      dz = dymodU[j];
      TGeoVolume* vol = gGeoManager->GetVolume("B076");
      vol->AddNode(asBI42, 6 * i + 2 * j + 1, new TGeoCombiTrans(dx, dy, dz, rot));
      vol->AddNode(asBI42, 6 * i + 2 * j + 2, new TGeoCombiTrans(dx, dy, -dz, rot));
    }

    phi1 = i * 20. + 10;
    phi2 = 270 + phi1;
    AliMatrix(idrotm[2052 + i], 90.0, phi1, 90., phi2, 0., 0.);

    dx = TMath::Sin(phi1 * kdeg2rad) * rout2;
    dy = -TMath::Cos(phi1 * kdeg2rad) * rout2;
    TVirtualMC::GetMC()->Gspos("B033", i + 1, "B076", dx, dy, 0., idrotm[2052 + i], "ONLY");
    //
  }
  // Internal Frame rings
  //
  //
  //            60x60x5x6  for inner rings (I-beam)
  //           100x60x5    for front and rear rings
  //
  // Front and rear
  ptrd1[0] = 287. * TMath::Sin(10. * kdeg2rad) - 2.1;
  ptrd1[1] = 293. * TMath::Sin(10. * kdeg2rad) - 2.1;
  ptrd1[2] = ringW / 2.;
  ptrd1[3] = ringH / 2.;

  TVirtualMC::GetMC()->Gsvolu("B072", "TRD1", kSteel, ptrd1, 4);

  ptrd1[0] = 287.5 * TMath::Sin(10. * kdeg2rad) - 2.1;
  ptrd1[1] = 292.5 * TMath::Sin(10. * kdeg2rad) - 2.1;
  ptrd1[2] = ringW / 2. - 0.5;
  ptrd1[3] = ringH / 2. - 0.5;

  TVirtualMC::GetMC()->Gsvolu("B073", "TRD1", kAir, ptrd1, 4);
  TVirtualMC::GetMC()->Gspos("B073", 1, "B072", 0., 0., 0., 0, "ONLY");
  //
  // I-Beam
  // Mother volume
  TGeoVolumeAssembly* asBI72 = new TGeoVolumeAssembly("BI72");
  // Horizontal
  ptrd1[0] = 292.5 * TMath::Sin(10. * kdeg2rad) - 2.1;
  ptrd1[1] = 293.0 * TMath::Sin(10. * kdeg2rad) - 2.1;
  ptrd1[2] = 6. / 2.;
  ptrd1[3] = 0.5 / 2.;
  TVirtualMC::GetMC()->Gsvolu("BIH172", "TRD1", kSteel, ptrd1, 4);
  ptrd1[0] = 287.0 * TMath::Sin(10. * kdeg2rad) - 2.1;
  ptrd1[1] = 287.5 * TMath::Sin(10. * kdeg2rad) - 2.1;
  ptrd1[2] = 6. / 2.;
  ptrd1[3] = 0.5 / 2.;
  TVirtualMC::GetMC()->Gsvolu("BIH272", "TRD1", kSteel, ptrd1, 4);

  // Vertical
  ptrd1[0] = 287.5 * TMath::Sin(10. * kdeg2rad) - 2.1;
  ptrd1[1] = 292.5 * TMath::Sin(10. * kdeg2rad) - 2.1;
  ptrd1[2] = 0.6 / 2.;
  ptrd1[3] = 5. / 2.;
  TVirtualMC::GetMC()->Gsvolu("BIV72", "TRD1", kSteel, ptrd1, 4);
  // Place
  asBI72->AddNode(gGeoManager->GetVolume("BIV72"), 1, new TGeoTranslation(0., 0., 0.));
  asBI72->AddNode(gGeoManager->GetVolume("BIH172"), 1, new TGeoTranslation(0., 0., 2.75));
  asBI72->AddNode(gGeoManager->GetVolume("BIH272"), 1, new TGeoTranslation(0., 0., -2.75));

  // Web frame 0-degree
  //
  // h x w x s = 60x40x5
  // (attention: element is are half bars, "U" shaped)
  //
  Float_t dHz = 112.66;

  WebFrame("B063", dHz, 10.0, 10.);
  WebFrame("B063I", dHz, 10.0, -10.);

  WebFrame("B163", dHz, -40.0, 10.);
  WebFrame("B163I", dHz, -40.0, -10.);

  WebFrame("B263", dHz, 20.0, 10.);
  WebFrame("B263I", dHz, 20.0, -10.);

  WebFrame("B363", dHz, -27.1, 10.);
  WebFrame("B363I", dHz, -27.1, -10.);

  WebFrame("B463", dHz, 18.4, 10.);
  WebFrame("B463I", dHz, 18.4, -10.);

  dz = -iFrH / 2. + ringH / 2. + kEps;
  Float_t dz0 = 3.;
  Float_t dx0 = 49.8 + dHz / 2. * TMath::Tan(10. * kdeg2rad) + 0.035;

  for (jmod = 0; jmod < 18; jmod++) {
    // ring bars
    for (i = 0; i < 3; i++) {
      //	if ((i == 2) || (jmod ==0) || (jmod == 8)) {
      if (i == 2) {
        TVirtualMC::GetMC()->Gspos("B072", 6 * jmod + i + 1, module[jmod], 0, dymodL[i], dz, 0, "ONLY");
        TVirtualMC::GetMC()->Gspos("B072", 6 * jmod + i + 4, module[jmod], 0, -dymodL[i], dz, idrotm[2070], "ONLY");
      } else {
        TGeoVolume* vol = gGeoManager->GetVolume(module[jmod]);
        vol->AddNode(asBI72, 6 * jmod + i + 1, new TGeoTranslation(0, dymodL[i], dz));
        vol->AddNode(asBI72, 6 * jmod + i + 4, new TGeoTranslation(0, -dymodL[i], dz));
      }
    }
  }

  // outer diagonal web

  dy = dymodL[0] + (dHz / 2. - 4.) * TMath::Tan(10. * kdeg2rad);

  for (jmod = 0; jmod < 18; jmod++) {
    TVirtualMC::GetMC()->Gspos("B063", 4 * jmod + 1, module[jmod], dx0, dy, dz0, idrotm[2086], "ONLY");
    TVirtualMC::GetMC()->Gspos("B063I", 4 * jmod + 2, module[jmod], dx0, -dy, dz0, idrotm[2087], "ONLY");
    TVirtualMC::GetMC()->Gspos("B063", 4 * jmod + 3, module[jmod], -dx0, -dy, dz0, idrotm[2087], "ONLY");
    TVirtualMC::GetMC()->Gspos("B063I", 4 * jmod + 4, module[jmod], -dx0, dy, dz0, idrotm[2086], "ONLY");
  }

  dy = 73.6 + (dHz / 2. + 4.) * TMath::Tan(40. * kdeg2rad);

  for (jmod = 0; jmod < 18; jmod++) {
    TVirtualMC::GetMC()->Gspos("B163", 4 * jmod + 1, module[jmod], dx0, dy, dz0, idrotm[2086], "ONLY");
    TVirtualMC::GetMC()->Gspos("B163I", 4 * jmod + 2, module[jmod], dx0, -dy, dz0, idrotm[2087], "ONLY");
    TVirtualMC::GetMC()->Gspos("B163", 4 * jmod + 3, module[jmod], -dx0, -dy, dz0, idrotm[2087], "ONLY");
    TVirtualMC::GetMC()->Gspos("B163I", 4 * jmod + 4, module[jmod], -dx0, dy, dz0, idrotm[2086], "ONLY");
  }

  dy = 224.5 - (dHz / 2 + 4.) * TMath::Tan(20. * kdeg2rad);

  for (jmod = 0; jmod < 18; jmod++) {
    TVirtualMC::GetMC()->Gspos("B263", 4 * jmod + 1, module[jmod], dx0, dy, dz0, idrotm[2086], "ONLY");
    TVirtualMC::GetMC()->Gspos("B263I", 4 * jmod + 2, module[jmod], dx0, -dy, dz0, idrotm[2087], "ONLY");
    TVirtualMC::GetMC()->Gspos("B263", 4 * jmod + 3, module[jmod], -dx0, -dy, dz0, idrotm[2087], "ONLY");
    TVirtualMC::GetMC()->Gspos("B263I", 4 * jmod + 4, module[jmod], -dx0, dy, dz0, idrotm[2086], "ONLY");
  }

  dy = 231.4 + (dHz / 2. + 4.) * TMath::Tan(27.1 * kdeg2rad);

  for (jmod = 0; jmod < 18; jmod++) {
    TVirtualMC::GetMC()->Gspos("B363", 4 * jmod + 1, module[jmod], dx0, dy, dz0, idrotm[2086], "ONLY");
    TVirtualMC::GetMC()->Gspos("B363I", 4 * jmod + 2, module[jmod], dx0, -dy, dz0, idrotm[2087], "ONLY");
    TVirtualMC::GetMC()->Gspos("B363", 4 * jmod + 3, module[jmod], -dx0, -dy, dz0, idrotm[2087], "ONLY");
    TVirtualMC::GetMC()->Gspos("B363I", 4 * jmod + 4, module[jmod], -dx0, dy, dz0, idrotm[2086], "ONLY");
  }

  dy = 340.2 - (dHz / 2. + 4.) * TMath::Tan(18.4 * kdeg2rad);

  for (jmod = 0; jmod < 18; jmod++) {
    TVirtualMC::GetMC()->Gspos("B463", 4 * jmod + 1, module[jmod], dx0, dy, dz0, idrotm[2086], "ONLY");
    TVirtualMC::GetMC()->Gspos("B463I", 4 * jmod + 2, module[jmod], dx0, -dy, dz0, idrotm[2087], "ONLY");
    TVirtualMC::GetMC()->Gspos("B463", 4 * jmod + 3, module[jmod], -dx0, -dy, dz0, idrotm[2087], "ONLY");
    TVirtualMC::GetMC()->Gspos("B463I", 4 * jmod + 4, module[jmod], -dx0, dy, dz0, idrotm[2086], "ONLY");
  }

  // longitudinal bars (TPC rails attached)
  //  new specs:
  //  h x w x s = 100 x 75 x 6
  //  current:
  //  Attention: 2 "U" shaped half rods per cell
  //
  //  not yet used
  //
  ptrap[0] = 2.50;
  ptrap[1] = 10.00;
  ptrap[2] = 0.00;
  ptrap[3] = 350.00;
  ptrap[4] = 3.75;
  ptrap[5] = ptrap[4];
  ptrap[6] = 0;
  ptrap[7] = ptrap[3];
  ptrap[8] = ptrap[4];
  ptrap[9] = ptrap[4];
  ptrap[10] = 0;
  //  TVirtualMC::GetMC()->Gsvolu("B059", "TRAP", kSteel, ptrap, 11);
  ptrap[0] = 2.2;
  ptrap[4] = 2.15;
  ptrap[5] = ptrap[4];
  ptrap[7] = ptrap[3];
  ptrap[8] = ptrap[4];
  ptrap[9] = ptrap[4];
  // TVirtualMC::GetMC()->Gsvolu("B062", "TRAP", kAir, ptrap, 11);
  // TVirtualMC::GetMC()->Gspos("B062", 1, "B059", 0.0, 0., 0., 0, "ONLY");

  //
  // longitudinal bars (no TPC rails attached)
  // new specs: h x w x s = 40 x 60 x 5
  //
  //
  //
  ptrap[0] = longW / 4.;
  ptrap[4] = longH / 2.;
  ptrap[5] = ptrap[4];
  ptrap[7] = ptrap[3];
  ptrap[8] = ptrap[4];
  ptrap[9] = ptrap[4];

  TVirtualMC::GetMC()->Gsvolu("BA59", "TRAP", kSteel, ptrap, 11);
  ptrap[0] = longW / 4. - 0.25;
  ptrap[4] = longH / 2. - 0.50;
  ptrap[5] = ptrap[4];
  ptrap[7] = ptrap[3];
  ptrap[8] = ptrap[4];
  ptrap[9] = ptrap[4];
  TVirtualMC::GetMC()->Gsvolu("BA62", "TRAP", kAir, ptrap, 11);
  TVirtualMC::GetMC()->Gspos("BA62", 1, "BA59", 0.0, 0.0, -0.15, 0, "ONLY");

  dz = -iFrH / 2. + longH / 2.;

  for (jmod = 0; jmod < 18; jmod++) {
    TVirtualMC::GetMC()->Gspos("BA59", 2 * jmod + 1, module[jmod], 49.31, 0.0, dz, idrotm[2084], "ONLY");
    TVirtualMC::GetMC()->Gspos("BA59", 2 * jmod + 2, module[jmod], -49.31, 0.0, dz, idrotm[2083], "ONLY");
  }

  //
  // Thermal shield
  //

  Float_t dyM = 99.0;
  MakeHeatScreen("M", dyM, idrotm[2090], idrotm[2091]);
  Float_t dyAM = 119.5;
  MakeHeatScreen("AM", dyAM, idrotm[2090], idrotm[2091]);
  Float_t dyA = 122.5 - 5.5;
  MakeHeatScreen("A", dyA, idrotm[2090], idrotm[2091]);

  //
  //
  //
  dz = -57.2 + 0.6;
  for (i = 0; i < 18; i++) {
    char nameMo[16];
    snprintf(nameMo, 16, "BSEGMO%d", i);
    // M
    TVirtualMC::GetMC()->Gspos("BTSH_M", i + 1, nameMo, 0., 0., dz, 0, "ONLY");
    // AM, CM
    dy = dymodL[0] + dyAM / 2. + 3.;
    TVirtualMC::GetMC()->Gspos("BTSH_AM", i + 1, nameMo, 0., dy, dz, 0, "ONLY");
    TVirtualMC::GetMC()->Gspos("BTSH_AM", i + 19, nameMo, 0., -dy, dz, 0, "ONLY");
    // A, C
    dy = dymodL[1] + dyA / 2 + 0.4;
    TVirtualMC::GetMC()->Gspos("BTSH_A", i + 1, nameMo, 0., dy, dz, 0, "ONLY");
    TVirtualMC::GetMC()->Gspos("BTSH_A", i + 19, nameMo, 0., -dy, dz, 0, "ONLY");
  }

  //
  // TRD mother volumes
  //

  ptrd1[0] = 47.4405; // CBL 28/6/2006
  ptrd1[1] = 61.1765; // CBL
  ptrd1[2] = 375.5;   // CBL
  ptrd1[3] = 38.95;   // CBL

  for (i = 0; i < 18; i++) {
    char nameCh[16];
    snprintf(nameCh, 16, "BTRD%d", i);
    char nameMo[16];
    snprintf(nameMo, 16, "BSEGMO%d", i);
    TVirtualMC::GetMC()->Gsvolu(nameCh, "TRD1", kAir, ptrd1, 4);
    gGeoManager->GetVolume(nameCh)->SetVisibility(kFALSE);
    TVirtualMC::GetMC()->Gspos(nameCh, 1, nameMo, 0., 0., -12.62, 0, "ONLY"); // CBL 28/6/2006
  }

  //
  // TOF mother volumes as modified by B.Guerzoni
  // to remove overlaps/extrusions in case of aligned TOF SMs
  //
  ptrd1[0] = 62.2500;
  ptrd1[1] = 64.25;
  ptrd1[2] = 372.6;
  ptrd1[3] = 14.525 / 2;
  char nameChA[16];
  snprintf(nameChA, 16, "BTOFA");
  TGeoTrd1* trd1 = new TGeoTrd1(nameChA, ptrd1[0], ptrd1[1], ptrd1[2], ptrd1[3]);
  trd1->SetName("BTOFA"); // just to avoid a warning
  char nameChB[16];
  snprintf(nameChB, 16, "BTOFB");
  TGeoBBox* box1 = new TGeoBBox(nameChB, 64.25, 372.6, 14.525 / 2);
  box1->SetName("BTOFB"); // just to avoid a warning
  TGeoTranslation* tr1 = new TGeoTranslation("trnsl1", 0, 0, -14.525 / 2);
  tr1->RegisterYourself();
  TGeoTranslation* tr2 = new TGeoTranslation("trnsl2", 0, 0, +14.525 / 2);
  tr2->RegisterYourself();
  TGeoCompositeShape* btofcs = new TGeoCompositeShape("Btofcs", "(BTOFA:trnsl1)+(BTOFB:trnsl2)");

  for (i = 0; i < 18; i++) {
    char nameCh[16];
    snprintf(nameCh, 16, "BTOF%d", i);
    char nameMo[16];
    snprintf(nameMo, 16, "BSEGMO%d", i);
    TGeoVolume* btf = new TGeoVolume(nameCh, btofcs, gGeoManager->GetMedium(mAirMedID));
    btf->SetName(nameCh);
    gGeoManager->GetVolume(nameCh)->SetVisibility(kFALSE);
    TVirtualMC::GetMC()->Gspos(nameCh, 1, nameMo, 0., 0., 43.525, 0, "ONLY");
  }
  //
  //    Geometry of Rails starts here
  //
  //
  //
  //    Rails for space-frame
  //
  Float_t rbox[3];

  rbox[0] = 25.00;
  rbox[1] = 27.50;
  rbox[2] = 600.00;
  TVirtualMC::GetMC()->Gsvolu("BRS1", "BOX", kAir, rbox, 3);

  rbox[0] = 25.00;
  rbox[1] = 3.75;
  TVirtualMC::GetMC()->Gsvolu("BRS2", "BOX", kSteel, rbox, 3);

  rbox[0] = 3.00;
  rbox[1] = 20.00;
  TVirtualMC::GetMC()->Gsvolu("BRS3", "BOX", kSteel, rbox, 3);

  TVirtualMC::GetMC()->Gspos("BRS2", 1, "BRS1", 0., -27.5 + 3.75, 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("BRS2", 2, "BRS1", 0., 27.5 - 3.75, 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("BRS3", 1, "BRS1", 0., 0., 0., 0, "ONLY");

  if (mCaveIsAvailable) {
    TVirtualMC::GetMC()->Gspos("BRS1", 1, TOPNAME, -430. - 3., -190., 0., 0, "ONLY");
    TVirtualMC::GetMC()->Gspos("BRS1", 2, TOPNAME, 430. + 3., -190., 0., 0, "ONLY");
  }

  rbox[0] = 3.0;
  rbox[1] = 145. / 4.;
  rbox[2] = 25.0;
  TVirtualMC::GetMC()->Gsvolu("BRS4", "BOX", kSteel, rbox, 3);

  /*
    TVirtualMC::GetMC()->Gspos("BRS4", 1, TOPNAME,  430.+3.,    -190.+55./2.+rbox[1],  224., 0, "ONLY");
    TVirtualMC::GetMC()->Gspos("BRS4", 2, TOPNAME,  430.+3.,    -190.+55./2.+rbox[1], -224., 0, "ONLY");
  */
  //  TVirtualMC::GetMC()->Gspos("BRS4", 3, TOPNAME, -430.+3,    -180.+55./2.+rbox[1],  224., 0, "ONLY");
  //  TVirtualMC::GetMC()->Gspos("BRS4", 4, TOPNAME, -430.+3,    -180.+55./2.+rbox[1], -224., 0, "ONLY");

  //
  // The Backframe
  //
  // Inner radius
  Float_t kBFMRin = 270.0;
  // Outer Radius
  Float_t kBFMRou = 417.5;
  // Width
  Float_t kBFMdz = 118.0;
  //
  //
  // Rings
  Float_t kBFRdr = 7.5;
  Float_t kBFRdz = 8.0;
  //
  //
  // Bars and Spokes
  //
  Float_t kBFBd = 8.0;
  Float_t kBFBdd = 0.6;

  // The Mother volume
  Float_t tpar[3];
  tpar[0] = kBFMRin;
  tpar[1] = kBFMRou;
  tpar[2] = kBFMdz / 2.;
  TVirtualMC::GetMC()->Gsvolu("BFMO", "TUBE", kAir, tpar, 3);

  // CBL ////////////////////////////////////////////////////////
  //
  // TRD mother volume
  //

  ptrd1[0] = 47.4405 - 0.3;
  ptrd1[1] = 61.1765 - 0.3;
  ptrd1[2] = kBFMdz / 2.;
  ptrd1[3] = 38.95;
  TVirtualMC::GetMC()->Gsvolu("BFTRD", "TRD1", kAir, ptrd1, 4);
  gGeoManager->GetVolume("BFTRD")->SetVisibility(kFALSE);

  for (i = 0; i < 18; i++) {
    Float_t phiBF = i * 20.0;
    dx = TMath::Sin(phiBF * kdeg2rad) * (342.0 - 12.62);
    dy = -TMath::Cos(phiBF * kdeg2rad) * (342.0 - 12.62);
    TVirtualMC::GetMC()->Gspos("BFTRD", i, "BFMO", dx, dy, 0.0, idrotm[2034 + i], "ONLY");
  }

  // CBL ////////////////////////////////////////////////////////

  // Rings
  //
  // Inner Ring
  tpar[0] = kBFMRin;
  tpar[1] = tpar[0] + kBFRdr;
  tpar[2] = kBFRdz / 2.;

  TVirtualMC::GetMC()->Gsvolu("BFIR", "TUBE", kSteel, tpar, 3);

  tpar[0] = tpar[0] + kBFBdd;
  tpar[1] = tpar[1] - kBFBdd;
  tpar[2] = (kBFRdz - 2. * kBFBdd) / 2.;

  TVirtualMC::GetMC()->Gsvolu("BFII", "TUBE", kAir, tpar, 3);
  TVirtualMC::GetMC()->Gspos("BFII", 1, "BFIR", 0., 0., 0., 0, "ONLY");

  //
  // Outer RING
  tpar[0] = kBFMRou - kBFRdr + 0.1;
  tpar[1] = kBFMRou;
  tpar[2] = kBFRdz / 2.;

  TVirtualMC::GetMC()->Gsvolu("BFOR", "TUBE", kSteel, tpar, 3);

  tpar[0] = tpar[0] + kBFBdd;
  tpar[1] = tpar[1] - kBFBdd;
  tpar[2] = (kBFRdz - 2. * kBFBdd) / 2.;

  TVirtualMC::GetMC()->Gsvolu("BFOO", "TUBE", kAir, tpar, 3);
  TVirtualMC::GetMC()->Gspos("BFOO", 1, "BFOR", 0., 0., 0., 0, "ONLY");

  dz = kBFMdz / 2. - kBFRdz / 2.;
  TVirtualMC::GetMC()->Gspos("BFIR", 1, "BFMO", 0., 0., dz, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("BFIR", 2, "BFMO", 0., 0., -dz, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("BFOR", 1, "BFMO", 0., 0., dz, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("BFOR", 2, "BFMO", 0., 0., -dz, 0, "ONLY");

  //
  // Longitudinal Bars
  //
  Float_t bpar[3];

  bpar[0] = kBFBd / 2;
  bpar[1] = bpar[0];
  bpar[2] = kBFMdz / 2. - kBFBd;
  TVirtualMC::GetMC()->Gsvolu("BFLB", "BOX ", kSteel, bpar, 3);

  bpar[0] = bpar[0] - kBFBdd;
  bpar[1] = bpar[1] - kBFBdd;
  bpar[2] = bpar[2] - kBFBdd;
  TVirtualMC::GetMC()->Gsvolu("BFLL", "BOX ", kAir, bpar, 3);
  TVirtualMC::GetMC()->Gspos("BFLL", 1, "BFLB", 0., 0., 0., 0, "ONLY");

  for (i = 0; i < 18; i++) {
    Float_t ro = kBFMRou - kBFBd / 2. - 0.02;
    Float_t ri = kBFMRin + kBFBd / 2.;

    Float_t phi0 = Float_t(i) * 20.;

    Float_t xb = ri * TMath::Cos(phi0 * kDegrad);
    Float_t yb = ri * TMath::Sin(phi0 * kDegrad);
    AliMatrix(idrotm[2090 + i], 90.0, phi0, 90.0, phi0 + 270., 0., 0.);

    TVirtualMC::GetMC()->Gspos("BFLB", i + 1, "BFMO", xb, yb, 0., idrotm[2090 + i], "ONLY");

    xb = ro * TMath::Cos(phi0 * kDegrad);
    yb = ro * TMath::Sin(phi0 * kDegrad);

    TVirtualMC::GetMC()->Gspos("BFLB", i + 19, "BFMO", xb, yb, 0., idrotm[2090 + i], "ONLY");
  }

  //
  // Radial Bars
  //
  bpar[0] = (kBFMRou - kBFMRin - 2. * kBFRdr) / 2.;
  bpar[1] = kBFBd / 2;
  bpar[2] = bpar[1];
  //
  // Avoid overlap with circle
  Float_t rr = kBFMRou - kBFRdr;
  Float_t delta = rr - TMath::Sqrt(rr * rr - kBFBd * kBFBd / 4.) + 0.01;
  bpar[0] -= delta / 2.;

  TVirtualMC::GetMC()->Gsvolu("BFRB", "BOX ", kSteel, bpar, 3);

  bpar[0] = bpar[0] - kBFBdd;
  bpar[1] = bpar[1] - kBFBdd;
  bpar[2] = bpar[2] - kBFBdd;
  TVirtualMC::GetMC()->Gsvolu("BFRR", "BOX ", kAir, bpar, 3);
  TVirtualMC::GetMC()->Gspos("BFRR", 1, "BFRB", 0., 0., 0., 0, "ONLY");

  Int_t iphi[10] = { 0, 1, 3, 6, 8, 9, 10, 12, 15, 17 };

  for (i = 0; i < 10; i++) {
    Float_t rb = (kBFMRin + kBFMRou) / 2.;
    Float_t phib = Float_t(iphi[i]) * 20.;

    Float_t xb = rb * TMath::Cos(phib * kDegrad);
    Float_t yb = rb * TMath::Sin(phib * kDegrad);

    TVirtualMC::GetMC()->Gspos("BFRB", i + 1, "BFMO", xb, yb, dz, idrotm[2034 + iphi[i]], "ONLY");
    TVirtualMC::GetMC()->Gspos("BFRB", i + 11, "BFMO", xb, yb, -dz, idrotm[2034 + iphi[i]], "ONLY");
  }

  if (mCaveIsAvailable) {
    TVirtualMC::GetMC()->Gspos("BFMO", i + 19, TOPNAME, 0, 0, -376. - kBFMdz / 2. - 0.5, 0, "ONLY");
  }

  //
  //
  //  The Baby Frame
  //
  //
  //
  // Inner radius
  Float_t kBBMRin = 278.0;
  // Outer Radius
  Float_t kBBMRou = 410.5;
  // Width
  Float_t kBBMdz = 223.0;
  Float_t kBBBdz = 6.0;
  Float_t kBBBdd = 0.6;

  // The Mother volume

  ppgon[0] = 0.;
  ppgon[1] = 360.;
  ppgon[2] = 18.;

  ppgon[3] = 2.;
  ppgon[4] = -kBBMdz / 2.;
  ppgon[5] = kBBMRin;
  ppgon[6] = kBBMRou;

  ppgon[7] = -ppgon[4];
  ppgon[8] = ppgon[5];
  ppgon[9] = ppgon[6];

  TVirtualMC::GetMC()->Gsvolu("BBMO", "PGON", kAir, ppgon, 10);
  TVirtualMC::GetMC()->Gsdvn("BBCE", "BBMO", 18, 2);

  // CBL ////////////////////////////////////////////////////////
  //
  // TRD mother volume
  //

  AliMatrix(idrotm[2092], 90.0, 90.0, 0.0, 0.0, 90.0, 0.0);

  ptrd1[0] = 47.4405 - 2.5;
  ptrd1[1] = 61.1765 - 2.5;
  ptrd1[2] = kBBMdz / 2.;
  ptrd1[3] = 38.95;
  TVirtualMC::GetMC()->Gsvolu("BBTRD", "TRD1", kAir, ptrd1, 4);
  gGeoManager->GetVolume("BBTRD")->SetVisibility(kFALSE);
  TVirtualMC::GetMC()->Gspos("BBTRD", 1, "BBCE", 342.0 - 12.62, 0.0, 0.0, idrotm[2092], "ONLY");

  // CBL ////////////////////////////////////////////////////////

  // Longitudinal bars
  bpar[0] = kBBBdz / 2.;
  bpar[1] = bpar[0];
  bpar[2] = kBBMdz / 2. - kBBBdz;
  TVirtualMC::GetMC()->Gsvolu("BBLB", "BOX ", kSteel, bpar, 3);
  bpar[0] -= kBBBdd;
  bpar[1] -= kBBBdd;
  bpar[2] -= kBBBdd;
  TVirtualMC::GetMC()->Gsvolu("BBLL", "BOX ", kAir, bpar, 3);
  TVirtualMC::GetMC()->Gspos("BBLL", 1, "BBLB", 0., 0., 0., 0, "ONLY");

  dx = kBBMRin + kBBBdz / 2. + (bpar[1] + kBBBdd) * TMath::Sin(10. * kDegrad);
  dy = dx * TMath::Tan(10. * kDegrad) - kBBBdz / 2. / TMath::Cos(10. * kDegrad);
  TVirtualMC::GetMC()->Gspos("BBLB", 1, "BBCE", dx, dy, 0., idrotm[2052], "ONLY");

  dx = kBBMRou - kBBBdz / 2. - (bpar[1] + kBBBdd) * TMath::Sin(10. * kDegrad);
  dy = dx * TMath::Tan(10. * kDegrad) - kBBBdz / 2. / TMath::Cos(10. * kDegrad);

  TVirtualMC::GetMC()->Gspos("BBLB", 2, "BBCE", dx, dy, 0., idrotm[2052], "ONLY");

  //
  // Radial Bars
  //
  bpar[0] = (kBBMRou - kBBMRin) / 2. - kBBBdz;
  bpar[1] = kBBBdz / 2;
  bpar[2] = bpar[1];

  TVirtualMC::GetMC()->Gsvolu("BBRB", "BOX ", kSteel, bpar, 3);
  bpar[0] -= kBBBdd;
  bpar[1] -= kBBBdd;
  bpar[2] -= kBBBdd;
  TVirtualMC::GetMC()->Gsvolu("BBRR", "BOX ", kAir, bpar, 3);
  TVirtualMC::GetMC()->Gspos("BBRR", 1, "BBRB", 0., 0., 0., 0, "ONLY");

  dx = (kBBMRou + kBBMRin) / 2.;
  dy = ((kBBMRou + kBBMRin) / 2) * TMath::Tan(10 * kDegrad) - kBBBdz / 2. / TMath::Cos(10 * kDegrad);
  dz = kBBMdz / 2. - kBBBdz / 2.;

  TVirtualMC::GetMC()->Gspos("BBRB", 1, "BBCE", dx, dy, dz, idrotm[2052], "ONLY");
  TVirtualMC::GetMC()->Gspos("BBRB", 2, "BBCE", dx, dy, -dz, idrotm[2052], "ONLY");
  TVirtualMC::GetMC()->Gspos("BBRB", 3, "BBCE", dx, dy, 0., idrotm[2052], "ONLY");

  //
  // Circular bars
  //
  //  Inner

  bpar[1] = kBBMRin * TMath::Sin(10. * kDegrad);
  bpar[0] = kBBBdz / 2;
  bpar[2] = bpar[0];
  TVirtualMC::GetMC()->Gsvolu("BBC1", "BOX ", kSteel, bpar, 3);
  bpar[0] -= kBBBdd;
  bpar[1] -= kBBBdd;
  bpar[2] -= kBBBdd;
  TVirtualMC::GetMC()->Gsvolu("BBC2", "BOX ", kAir, bpar, 3);
  TVirtualMC::GetMC()->Gspos("BBC2", 1, "BBC1", 0., 0., 0., 0, "ONLY");
  dx = kBBMRin + kBBBdz / 2;
  dy = 0.;
  TVirtualMC::GetMC()->Gspos("BBC1", 1, "BBCE", dx, dy, dz, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("BBC1", 2, "BBCE", dx, dy, -dz, 0, "ONLY");
  //
  // Outer
  bpar[1] = (kBBMRou - kBBBdz) * TMath::Sin(10. * kDegrad);
  bpar[0] = kBBBdz / 2;
  bpar[2] = bpar[0];
  TVirtualMC::GetMC()->Gsvolu("BBC3", "BOX ", kSteel, bpar, 3);
  bpar[0] -= kBBBdd;
  bpar[1] -= kBBBdd;
  bpar[2] -= kBBBdd;
  TVirtualMC::GetMC()->Gsvolu("BBC4", "BOX ", kAir, bpar, 3);
  TVirtualMC::GetMC()->Gspos("BBC4", 1, "BBC3", 0., 0., 0., 0, "ONLY");
  dx = kBBMRou - kBBBdz / 2;
  dy = 0.;
  TVirtualMC::GetMC()->Gspos("BBC3", 1, "BBCE", dx, dy, dz, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("BBC3", 2, "BBCE", dx, dy, -dz, 0, "ONLY");
  //
  // Diagonal Bars
  //
  h = (kBBMRou - kBBMRin - 2. * kBBBdz);
  ;
  d = kBBBdz;
  dz = kBBMdz / 2. - 1.6 * kBBBdz;
  dq = h * h + dz * dz;

  x = TMath::Sqrt((dz * dz - d * d) / dq + d * d * h * h / dq / dq) + d * h / dq;

  theta = kRaddeg * TMath::ACos(x);

  ptrap[0] = dz / 2.;
  ptrap[1] = theta;
  ptrap[2] = 0.;
  ptrap[3] = d / 2;
  ptrap[4] = d / x / 2;
  ptrap[5] = ptrap[4];
  ptrap[6] = 0;
  ptrap[7] = ptrap[3];
  ptrap[8] = ptrap[4];
  ptrap[9] = ptrap[4];
  ptrap[10] = 0;
  TVirtualMC::GetMC()->Gsvolu("BBD1", "TRAP", kSteel, ptrap, 11);
  ptrap[3] = d / 2 - kBBBdd;
  ptrap[4] = (d / 2 - kBBBdd) / x;
  ptrap[5] = ptrap[4];
  ptrap[7] = ptrap[3];
  ptrap[8] = ptrap[4];
  ptrap[9] = ptrap[4];
  TVirtualMC::GetMC()->Gsvolu("BBD3", "TRAP", kAir, ptrap, 11);
  TVirtualMC::GetMC()->Gspos("BBD3", 1, "BBD1", 0.0, 0.0, 0., 0, "ONLY");
  dx = (kBBMRou + kBBMRin) / 2.;
  dy = ((kBBMRou + kBBMRin) / 2) * TMath::Tan(10 * kDegrad) - kBBBdz / 2. / TMath::Cos(10 * kDegrad);
  TVirtualMC::GetMC()->Gspos("BBD1", 1, "BBCE", dx, dy, dz / 2. + kBBBdz / 2., idrotm[2052], "ONLY");

  ptrap[0] = dz / 2.;
  ptrap[1] = -theta;
  ptrap[2] = 0.;
  ptrap[3] = d / 2;
  ptrap[4] = d / 2 / x;
  ptrap[5] = ptrap[4];
  ptrap[6] = 0;
  ptrap[7] = ptrap[3];
  ptrap[8] = ptrap[4];
  ptrap[9] = ptrap[4];
  ptrap[10] = 0;
  TVirtualMC::GetMC()->Gsvolu("BBD2", "TRAP", kSteel, ptrap, 11);
  ptrap[3] = d / 2 - kBBBdd;
  ptrap[4] = (d / 2 - kBBBdd) / x;
  ptrap[5] = ptrap[4];
  ptrap[7] = ptrap[3];
  ptrap[8] = ptrap[4];
  ptrap[9] = ptrap[4];
  TVirtualMC::GetMC()->Gsvolu("BBD4", "TRAP", kAir, ptrap, 11);
  TVirtualMC::GetMC()->Gspos("BBD4", 1, "BBD2", 0.0, 0.0, 0., 0, "ONLY");
  dx = (kBBMRou + kBBMRin) / 2.;
  dy = ((kBBMRou + kBBMRin) / 2) * TMath::Tan(10 * kDegrad) - kBBBdz / 2. / TMath::Cos(10 * kDegrad);
  TVirtualMC::GetMC()->Gspos("BBD2", 1, "BBCE", dx, dy, -dz / 2. - kBBBdz / 2., idrotm[2052], "ONLY");

  if (mCaveIsAvailable) {
    TVirtualMC::GetMC()->Gspos("BBMO", 1, TOPNAME, 0., 0., +376. + kBBMdz / 2. + 0.5, 0, "ONLY");
  }
}

namespace
{
// only here temporarily, I would like to harmonize Material treatment (outside of base detector)
int Material(Int_t imat, const char* name, Float_t a, Float_t z, Float_t dens, Float_t radl, Float_t absl,
             Float_t* buf = nullptr, Int_t nwbuf = 0)
{
  int kmat = -1;
  TVirtualMC::GetMC()->Material(kmat, name, a, z, dens, radl, absl, buf, nwbuf);
  return kmat;
}

int Mixture(Int_t imat, const char* name, Float_t* a, Float_t* z, Float_t dens, Int_t nlmat, Float_t* wmat = nullptr)
{
  // Check this!!!
  int kmat = -1;
  TVirtualMC::GetMC()->Mixture(kmat, name, a, z, dens, nlmat, wmat);
  return kmat;
}

int Medium(Int_t numed, const char* name, Int_t nmat, Int_t isvol, Int_t ifield, Float_t fieldm, Float_t tmaxfd,
           Float_t stemax, Float_t deemax, Float_t epsil, Float_t stmin, Float_t* ubuf = nullptr, Int_t nbuf = 0)
{
  // Check this!!!
  int kmed = -1;
  TVirtualMC::GetMC()->Medium(kmed, name, nmat, isvol, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin, ubuf,
                              nbuf);
  return kmed;
}
}

void FrameStructure::CreateMaterials()
{
  // Creates the materials
  Float_t epsil, stemax, tmaxfd, deemax, stmin;

  epsil = 1.e-4;  // Tracking precision,
  stemax = -0.01; // Maximum displacement for multiple scat
  tmaxfd = -20.;  // Maximum angle due to field deflection
  deemax = -.3;   // Maximum fractional energy loss, DLS
  stmin = -.8;
  Int_t isxfld = 2.;
  Float_t sxmgmx = 10.;
  o2::Base::Detector::initFieldTrackingParams(isxfld, sxmgmx);

  Float_t asteel[4] = { 55.847, 51.9961, 58.6934, 28.0855 };
  Float_t zsteel[4] = { 26., 24., 28., 14. };
  Float_t wsteel[4] = { .715, .18, .1, .005 };

  // Air
  Float_t aAir[4] = { 12.0107, 14.0067, 15.9994, 39.948 };
  Float_t zAir[4] = { 6., 7., 8., 18. };
  Float_t wAir[4] = { 0.000124, 0.755267, 0.231781, 0.012827 };
  Float_t dAir = 1.20479E-3;

  auto kSteelMatId = Mixture(65, "STAINLESS STEEL$", asteel, zsteel, 7.88, 4, wsteel);
  auto kAirMatId = Mixture(5, "AIR$      ", aAir, zAir, dAir, 4, wAir);
  auto kAluMatId = Material(9, "ALU      ", 26.98, 13., 2.7, 8.9, 37.2);

  mSteelMedID = Medium(65, "FRAME_Stainless Steel", kSteelMatId, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  mAirMedID = Medium(5, "FRAME_Air", kAirMatId, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  mAluMedID = Medium(9, "FRAME_Aluminum", kAluMatId, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

  // do a cross check
  assert(gGeoManager->GetMedium("FRAME_Air")->GetId() == mAirMedID);
  assert(gGeoManager->GetMedium("FRAME_Aluminum")->GetId() == mAluMedID);
  assert(gGeoManager->GetMedium("FRAME_Stainless Steel")->GetId() == mSteelMedID);
}
}
}

ClassImp(o2::Passive::FrameStructure)
