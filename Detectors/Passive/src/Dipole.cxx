// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <DetectorsBase/Detector.h>
#include <DetectorsPassive/Dipole.h>
#include <TGeoArb8.h>
#include <TGeoCompositeShape.h>
#include <TGeoCone.h>
#include <TGeoManager.h>
#include <TGeoMaterial.h>
#include <TGeoMatrix.h>
#include <TGeoMedium.h>
#include <TGeoPcon.h>
#include <TGeoPgon.h>
#include <TGeoTube.h>
#include <TGeoVolume.h>
#include <TGeoXtru.h>
#include <TVirtualMC.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

using namespace o2::passive;

Dipole::~Dipole() = default;

Dipole::Dipole() : FairModule("Dipole", "") {}
Dipole::Dipole(const char* name, const char* Title) : FairModule(name, Title) {}
Dipole::Dipole(const Dipole& rhs) = default;

Dipole& Dipole::operator=(const Dipole& rhs)
{
  // self assignment
  if (this == &rhs)
    return *this;

  // base class assignment
  FairModule::operator=(rhs);

  return *this;
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

void Dipole::createMaterials()
{
  //
  // Create Materials for Magnetic Dipole
  //
  Int_t isxfld1 = 2.;
  Float_t sxmgmx = 10.;
  o2::Base::Detector::initFieldTrackingParams(isxfld1, sxmgmx);

  Int_t isxfld2 = 2; // TODO: set this properly ((AliMagF*)TGeoGlobalMagField::Instance()->GetField())->PrecInteg();

  Float_t asteel[4] = { 55.847, 51.9961, 58.6934, 28.0855 };
  Float_t zsteel[4] = { 26., 24., 28., 14. };
  Float_t wsteel[4] = { .715, .18, .1, .005 };

  Float_t acoil[3] = { 26.98, 1.01, 16. };
  Float_t zcoil[3] = { 13., 1., 8. };
  Float_t wcoil[3] = { .66, .226, .114 };

  Float_t aresi[3] = { 1.01, 12.011, 16. };
  Float_t zresi[3] = { 1., 6., 8. };
  Float_t wresi[3] = { .0644, .7655, .1701 };

  Float_t aG10[5] = { 1.01, 12.011, 16., 28.085, 79.904 };
  Float_t zG10[5] = { 1., 6., 8., 14., 35. };
  Float_t wG10[5] = { .02089, .22338, .28493, .41342, .05738 };

  // AIR
  Float_t aAir[4] = { 12.0107, 14.0067, 15.9994, 39.948 };
  Float_t zAir[4] = { 6., 7., 8., 18. };
  Float_t wAir[4] = { 0.000124, 0.755267, 0.231781, 0.012827 };
  Float_t dAir = 1.20479E-3;
  Float_t dAir1 = 1.20479E-10;

  // ****************
  //     Defines tracking media parameters.
  //     Les valeurs sont commentees pour laisser le defaut
  //     a GEANT (version 3-21, page CONS200), f.m.
  Float_t epsil, stmin, deemax, tmaxfd, stemax;
  epsil = .001;  // Tracking precision,
  stemax = -1.;  // Maximum displacement for multiple scat
  tmaxfd = -20.; // Maximum angle due to field deflection
  deemax = -.3;  // Maximum fractional energy loss, DLS
  stmin = -.8;
  // ***************

  // --- Define the various materials + tracking media for GEANT ---
  //     Aluminum
  auto matAl0 = Material(9, "ALUMINIUM0", 26.98, 13., 2.7, 8.9, 37.2);
  Medium(9, "DIPO_ALU_C0", matAl0, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  auto matAl1 = Material(29, "ALUMINIUM1", 26.98, 13., 2.7, 8.9, 37.2);
  Medium(29, "DIPO_ALU_C1", matAl1, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  auto matAl2 = Material(49, "ALUMINIUM2", 26.98, 13., 2.7, 8.9, 37.2);
  Medium(49, "DIPO_ALU_C2", matAl2, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

  //    Iron
  auto matIr0 = Material(10, "IRON0", 55.85, 26., 7.87, 1.76, 17.1);
  Medium(10, "DIPO_FE_C0", matIr0, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  auto matIr1 = Material(30, "IRON1", 55.85, 26., 7.87, 1.76, 17.1);
  Medium(30, "DIPO_FE_C1", matIr1, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  auto matIr2 = Material(50, "IRON2", 55.85, 26., 7.87, 1.76, 17.1);
  Medium(50, "DIPO_FE_C2", matIr2, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

  //     Air
  auto matair0 = Mixture(15, "AIR0", aAir, zAir, dAir, 4, wAir);
  auto matair1 = Mixture(35, "AIR1", aAir, zAir, dAir, 4, wAir);
  auto matair2 = Mixture(55, "AIR2", aAir, zAir, dAir, 4, wAir);
  auto matairmuon = Mixture(75, "AIR_MUON", aAir, zAir, dAir, 4, wAir);
  Medium(15, "DIPO_AIR_C0", matair0, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  Medium(35, "DIPO_AIR_C1", matair1, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  Medium(55, "DIPO_AIR_C2", matair2, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  Medium(75, "DIPO_AIR_MUON", matairmuon, 0, isxfld2, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

  //    Vacuum
  auto matvac0 = Mixture(16, "VACUUM0", aAir, zAir, dAir1, 4, wAir);
  auto matvac1 = Mixture(36, "VACUUM1", aAir, zAir, dAir1, 4, wAir);
  auto matvac2 = Mixture(56, "VACUUM2", aAir, zAir, dAir1, 4, wAir);
  Medium(16, "DIPO_VA_C0", matvac0, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  Medium(36, "DIPO_VA_C1", matvac1, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  Medium(56, "DIPO_VA_C2", matvac2, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

  //     stainless Steel
  auto matst0 = Mixture(19, "STAINLESS STEEL0$", asteel, zsteel, 7.88, 4, wsteel);
  auto matst1 = Mixture(39, "STAINLESS STEEL1$", asteel, zsteel, 7.88, 4, wsteel);
  auto matst2 = Mixture(59, "STAINLESS STEEL2$", asteel, zsteel, 7.88, 4, wsteel);
  Medium(19, "DIPO_ST_C0", matst0, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  Medium(39, "DIPO_ST_C1", matst1, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  Medium(59, "DIPO_ST_C3", matst2, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

  //    Coil
  auto matcoil0 = Mixture(14, "Al0", acoil, zcoil, 2.122, 3, wcoil);
  auto matcoil1 = Mixture(34, "Al1", acoil, zcoil, 2.122, 3, wcoil);
  auto matcoil2 = Mixture(54, "Al2", acoil, zcoil, 2.122, 3, wcoil);
  Medium(14, "DIPO_Coil_C1", matcoil0, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  Medium(34, "DIPO_Coil_C2", matcoil1, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  Medium(54, "DIPO_Coil_C3", matcoil2, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

  //    Resin
  auto matresin0 = Mixture(13, "RESIN0", aresi, zresi, 1.05, 3, wresi);
  auto matresin1 = Mixture(33, "RESIN1", aresi, zresi, 1.05, 3, wresi);
  auto matresin2 = Mixture(53, "RESIN2", aresi, zresi, 1.05, 3, wresi);
  Medium(13, "DIPO_RESIN_C0", matresin0, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  Medium(33, "DIPO_RESIN_C1", matresin1, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  Medium(53, "DIPO_RESIN_C2", matresin2, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

  //    G10
  auto matG100 = Mixture(11, "G100", aG10, zG10, 1.7, 5, wG10);
  auto matG101 = Mixture(31, "G101", aG10, zG10, 1.7, 5, wG10);
  auto matG102 = Mixture(51, "G102", aG10, zG10, 1.7, 5, wG10);
  Medium(11, "DIPO_G10_C0", matG100, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  Medium(31, "DIPO_G10_C1", matG101, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  Medium(51, "DIPO_G10_C2", matG102, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

  //    Copper
  auto matCu1 = Material(17, "COPPER0", 63.55, 29., 8.96, 1.43, 15.1);
  auto matCu2 = Material(37, "COPPER1", 63.55, 29., 8.96, 1.43, 15.1);
  auto matCu3 = Material(57, "COPPER2", 63.55, 29., 8.96, 1.43, 15.1);
  Medium(17, "DIPO_Cu_C0", matCu1, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  Medium(37, "DIPO_Cu_C1", matCu2, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  Medium(57, "DIPO_Cu_C2", matCu3, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
}

auto GetMedium = [](const char* x) {
  assert(gGeoManager);
  auto med = gGeoManager->GetMedium(x);
  assert(med);
  return med;
};

void Dipole::ConstructGeometry()
{
  createMaterials();
  createSpectrometerDipole();
  createCompensatorDipole();
}

#define kDegrad TMath::DegToRad()

void Dipole::createSpectrometerDipole()
{
  // Detailed dipole geometry as built
  //
  // Drawing: ALIP2A__0026
  // Geometer measurements: EDMS 596079
  //                        EDMS 584963

  //
  // The top volume
  //
  TGeoVolume* top = gGeoManager->GetVolume("cave");

  //
  // Media
  //
  TGeoMedium* kMedSteel = GetMedium("DIPO_ST_C3");
  TGeoMedium* kMedCoil = GetMedium("DIPO_Coil_C1");
  TGeoMedium* kMedCoilSh = GetMedium("DIPO_Coil_C3");
  TGeoMedium* kMedCable = GetMedium("DIPO_ALU_C2");
  TGeoMedium* kMedAlu = GetMedium("DIPO_ALU_C2");
  TGeoMedium* kMedAir = GetMedium("DIPO_AIR_MUON");
  //
  // Rotations
  //
  Float_t alhc = 0.794;

  TGeoRotation* rotxz = new TGeoRotation("rotxz", 270., 0., 90., 90., 180., 0.);
  TGeoRotation* rotiz = new TGeoRotation("rotiz", 90., 0., 90., 90., 180., 0.);
  TGeoRotation* rotxzlhc = new TGeoRotation("rotxzlhc", 180., 180. + alhc, 0.);

  TGeoRotation* rotxz108 = new TGeoRotation("rotxz108", 90., 108., 90., 198., 180., 0.);
  TGeoRotation* rotxz180 = new TGeoRotation("rotxz180", 90., 180., 90., 270., 180., 0.);
  TGeoRotation* rotxz288 = new TGeoRotation("rotxz288", 90., 288., 90., 18., 180., 0.);

  TGeoRotation* rotxy180 = new TGeoRotation("rotxy180", 90., 180., 90., 270., 0., 0.);
  TGeoRotation* rotxy108 = new TGeoRotation("rotxy108", 90., 108., 90., 198., 0., 0.);
  TGeoRotation* rotxy288 = new TGeoRotation("rotxy288", 90., 288., 90., 18., 0., 0.);

  TGeoRotation* rot00 = new TGeoRotation("rot00", 180., 0., 90., 151., 90., 61.);
  TGeoRotation* rot01 = new TGeoRotation("rot01", 180., 0., 90., 29., -90., -61.);
  TGeoRotation* rot02 = new TGeoRotation("rot02", 0., 0., 90., 151., 90., 61.);
  TGeoRotation* rot03 = new TGeoRotation("rot03", 0., 0., 90., 29., -90., -61.);
  TGeoRotation* rot04 = new TGeoRotation("rot04", 90., 61., 90., 151., 0., 0.);
  TGeoRotation* rot05 = new TGeoRotation("rot05", 90., -61., 90., -151., 0., 0.);
  TGeoRotation* rot06 = new TGeoRotation("rot06", 90., 119., 90., 209., 0., 0.);
  TGeoRotation* rot07 = new TGeoRotation("rot07", 90., -119., 90., -209., 0., 0.);

  const Float_t dipoleL = 498.;
  const Float_t kZDipoleR = 1244.;
  const Float_t kZDipole = kZDipoleR - dipoleL / 2.;
  const Float_t kZDipoleF = kZDipoleR - dipoleL;
  const Float_t yokeLength = 309.4;
  const Float_t blockLength = yokeLength / 7.;
  const Float_t gapWidthFront = 297.6;
  const Float_t gapWidthRear = 395.4;
  const Float_t dGap = (gapWidthRear - gapWidthFront) / 12.;
  const Float_t gapHeight = 609.1;
  const Float_t blockHeight = 145.45;
  const Float_t dzCoil = 4.45;

  Float_t dx, dy, dz;

  //
  // Mother volume for muon spectrometer tracking station 3

  Float_t z30 = 825.;
  Float_t zst = 1052.;

  Float_t rcD0 = (kZDipoleF - 5.) * TMath::Tan(9. * kDegrad);
  Float_t rcD1 = kZDipole * TMath::Tan(9. * kDegrad);
  Float_t rcD2 = rcD1 + dipoleL / 2. * TMath::Tan(10.1 * kDegrad);
  Float_t rc30 = z30 * TMath::Tan(9. * kDegrad);
  Float_t rcst = rcD1 + (zst - kZDipole) * TMath::Tan(10.1 * kDegrad);

  Float_t riD0 = (kZDipoleF - 5.) * TMath::Tan(2. * kDegrad) + 0.2;
  Float_t riD1 = 28.9;
  Float_t riD2 = 35.8;
  Float_t riD3 = riD2 + (kZDipoleR - zst) * TMath::Tan(2. * kDegrad);
  Float_t riD4 = riD2 + (kZDipoleR - zst + 5.) * TMath::Tan(2. * kDegrad);

  TGeoPcon* shDDIP1 = new TGeoPcon("shDDIP1", 0., 360., 7);

  shDDIP1->DefineSection(0, (kZDipoleF - 5.), riD0, rcD0);
  shDDIP1->DefineSection(1, z30, riD1, rc30);
  shDDIP1->DefineSection(2, kZDipole, riD1, rcD1);
  shDDIP1->DefineSection(3, zst, riD1, rcst);
  shDDIP1->DefineSection(4, zst, riD2, rcst);
  shDDIP1->DefineSection(5, kZDipoleR, riD3, rcD2);
  shDDIP1->DefineSection(6, (kZDipoleR + 5.), riD4, rcD2);

  // JC Ch6 is 2x5cm longer than Ch5
  //    TGeoBBox* shDDIP2 =  new TGeoBBox(164., 182., 36.);
  Double_t xD0 = 162.;
  Double_t xD1 = 171.;
  Double_t yD0 = 182.;
  Double_t zD0 = 36.;

  Double_t xy[16] = { 0 };
  xy[0] = -xD0;
  xy[1] = -yD0;
  xy[2] = -xD0;
  xy[3] = yD0;
  xy[4] = xD0;
  xy[5] = yD0;
  xy[6] = xD0;
  xy[7] = -yD0;
  xy[8] = -xD1;
  xy[9] = -yD0;
  xy[10] = -xD1;
  xy[11] = yD0;
  xy[12] = xD1;
  xy[13] = yD0;
  xy[14] = xD1;
  xy[15] = -yD0;
  TGeoArb8* shDDIP2 = new TGeoArb8(zD0, xy);
  shDDIP2->SetName("shDDIP2");
  TGeoTranslation* trDDIP2 = new TGeoTranslation("trDDIP2", 0., 0., kZDipole - 12.);
  trDDIP2->RegisterYourself();

  TGeoTube* shDDIP3 = new TGeoTube(0., 30., 40.);
  shDDIP3->SetName("shDDIP3");

  TGeoCompositeShape* shDDIP = new TGeoCompositeShape("shDDIP", "shDDIP1+(shDDIP2:trDDIP2-shDDIP3:trDDIP2)");
  TGeoVolume* voDDIP = new TGeoVolume("DDIP", shDDIP, kMedAir);
  //
  // Yoke
  //

  TGeoVolumeAssembly* asYoke = new TGeoVolumeAssembly("DYoke");
  // Base
  char name[16];
  Float_t lx0 = gapWidthFront + 2. * blockHeight;
  Float_t lx = lx0;

  TGeoVolumeAssembly* asYokeBase = new TGeoVolumeAssembly("DYokeBase");
  for (Int_t i = 0; i < 7; i++) {
    snprintf(name, 16, "DYokeBaseBlock%1d", i);
    TGeoVolume* voBaseBlock =
      new TGeoVolume(name, new TGeoBBox(lx / 2., blockHeight / 2., blockLength / 2.), kMedSteel);
    asYokeBase->AddNode(voBaseBlock, 1, new TGeoTranslation(0., 0., Float_t(i - 3) * blockLength));
    lx += 2. * dGap;
  }

  asYoke->AddNode(asYokeBase, 1, new TGeoTranslation(0., -(gapHeight + blockHeight) / 2., 0.));
  asYoke->AddNode(asYokeBase, 2, new TGeoTranslation(0., +(gapHeight + blockHeight) / 2., 0.));

  // Side Wall
  TGeoVolumeAssembly* asYokeSide = new TGeoVolumeAssembly("DYokeSide");
  TGeoVolume* voSideBlock =
    new TGeoVolume("DSideBlock", new TGeoBBox(blockHeight / 2., gapHeight / 2., blockLength / 2.), kMedSteel);

  for (Int_t i = 0; i < 7; i++) {
    asYokeSide->AddNode(voSideBlock, i, new TGeoTranslation(Float_t(i - 3) * dGap, 0., Float_t(i - 3) * blockLength));
  }

  asYoke->AddNode(asYokeSide, 1, new TGeoTranslation(+lx0 / 2. + 3. * dGap - blockHeight / 2., 0., 0.));
  asYoke->AddNode(asYokeSide, 2, new TGeoCombiTrans(-lx0 / 2. - 3. * dGap + blockHeight / 2., 0., 0., rotiz));

  //
  // Coils
  //
  Float_t coilRi = 206.;
  Float_t coilD = 70.;
  Float_t coilRo = coilRi + coilD;
  Float_t coilH = 77.;
  Float_t phiMin = -61.;
  Float_t phiMax = 61.;
  Float_t lengthSt = 240. + 33.9;
  Float_t phiKnee = phiMax * kDegrad;
  Float_t rKnee = 31.5;

  //  Circular sections
  TGeoVolumeAssembly* asCoil = new TGeoVolumeAssembly("DCoil");

  TGeoVolume* voDC1 = new TGeoVolume("DC1", new TGeoTubeSeg(coilRi, coilRo, coilH / 2., phiMin, phiMax), kMedCoil);
  TGeoVolume* voDC2 =
    new TGeoVolume("DC2", new TGeoTubeSeg(coilRi + 5., coilRo - 5., coilH / 2., phiMin, phiMax), kMedCoilSh);

  voDC1->AddNode(voDC2, 1, gGeoIdentity);
  voDC2->SetVisibility(0);

  dz = lengthSt / 2. + coilH / 2. + rKnee;
  dx = 0.;

  asCoil->AddNode(voDC1, 1, new TGeoTranslation(-dx, 0., -dz));
  asCoil->AddNode(voDC1, 2, new TGeoCombiTrans(dx, 0., -dz, rotxy180));
  asCoil->AddNode(voDC1, 3, new TGeoTranslation(-dx, 0., dz));
  asCoil->AddNode(voDC1, 4, new TGeoCombiTrans(dx, 0., dz, rotxz180));

  // 90deg Knees

  TGeoVolume* voDC11 = new TGeoVolume("DC11", new TGeoTubeSeg(rKnee, rKnee + coilH, coilD / 2., 270., 360.), kMedCoil);

  dx = -TMath::Cos(phiKnee) * (coilRi + coilD / 2.);
  dy = -TMath::Sin(phiKnee) * (coilRi + coilD / 2.);
  dz = lengthSt / 2.;

  asCoil->AddNode(voDC11, 1, new TGeoCombiTrans(dx, dy, -dz, rot00));
  asCoil->AddNode(voDC11, 2, new TGeoCombiTrans(dx, dy, dz, rot02));
  asCoil->AddNode(voDC11, 3, new TGeoCombiTrans(-dx, dy, -dz, rot01));
  asCoil->AddNode(voDC11, 4, new TGeoCombiTrans(-dx, dy, dz, rot03));

  TGeoVolume* voDC12 = new TGeoVolume("DC12", new TGeoTubeSeg(rKnee, rKnee + coilH, coilD / 2., 0., 90.), kMedCoil);

  asCoil->AddNode(voDC12, 1, new TGeoCombiTrans(dx, -dy, -dz, rot01));
  asCoil->AddNode(voDC12, 2, new TGeoCombiTrans(dx, -dy, dz, rot03));
  asCoil->AddNode(voDC12, 3, new TGeoCombiTrans(-dx, -dy, -dz, rot00));
  asCoil->AddNode(voDC12, 4, new TGeoCombiTrans(-dx, -dy, dz, rot02));

  // Straight sections

  TGeoVolume* voDL0 = new TGeoVolume("DL0", new TGeoBBox(coilD / 2. + 2., coilH / 2. + 2., lengthSt / 2.), kMedCoil);

  TGeoVolume* voDL1 = new TGeoVolume("DL1", new TGeoBBox(coilD / 2., coilH / 2., lengthSt / 2.), kMedCoil);

  TGeoVolume* voDL2 =
    new TGeoVolume("DL2", new TGeoBBox(coilD / 2. - 5., coilH / 2. - 5., lengthSt / 2. - 5.), kMedCoilSh);
  // Sleeves
  TGeoVolume* voDL3 = new TGeoVolume("DL3", new TGeoBBox(1., coilH / 2., 120.), kMedAlu);

  TGeoVolume* voDL4 = new TGeoVolume("DL4", new TGeoBBox(coilD / 2., 1., 120.), kMedAlu);

  voDL0->SetVisibility(0);
  voDL1->AddNode(voDL2, 1, gGeoIdentity);
  voDL0->AddNode(voDL1, 1, gGeoIdentity);
  voDL0->AddNode(voDL3, 1, new TGeoTranslation(-coilD / 2. - 1., 0., 0.));
  voDL0->AddNode(voDL3, 2, new TGeoTranslation(+coilD / 2. + 1., 0., 0.));
  voDL0->AddNode(voDL4, 1, new TGeoTranslation(0., -coilH / 2. - 1., 0.));
  voDL0->AddNode(voDL4, 2, new TGeoTranslation(0., +coilH / 2. + 1., 0.));

  dx += (rKnee + coilH / 2.) * TMath::Sin(phiKnee);
  dy -= (rKnee + coilH / 2.) * TMath::Cos(phiKnee);
  dz = 0.;

  asCoil->AddNode(voDL0, 1, new TGeoCombiTrans(dx, dy, dz, rot04));
  asCoil->AddNode(voDL0, 2, new TGeoCombiTrans(dx, -dy, dz, rot05));
  asCoil->AddNode(voDL0, 3, new TGeoCombiTrans(-dx, dy, dz, rot06));
  asCoil->AddNode(voDL0, 4, new TGeoCombiTrans(-dx, -dy, dz, rot07));

  // Contactor
  // Outer face planes

  TGeoVolumeAssembly* asContactor = new TGeoVolumeAssembly("DContactor");
  dx = -5.;
  TGeoVolume* voDC10 = new TGeoVolume("DC10", new TGeoTubeSeg(coilRo + 5.1, coilRo + 73.5, 1., -20., 20.), kMedCable);
  asContactor->AddNode(voDC10, 1, new TGeoTranslation(dx, 0, -32.325));
  asContactor->AddNode(voDC10, 2, new TGeoTranslation(dx, 0, +32.325));

  // Coil Support
  //
  Float_t sW = 83.;

  TGeoVolumeAssembly* asDCoilSupport = new TGeoVolumeAssembly("DCoilSupport");

  // Steel fixed to the yoke
  TGeoVolume* voDCS01 = new TGeoVolume("DCS01", new TGeoTubeSeg(coilRo, 325., 1., 21., 51.), kMedAlu);

  // Steel on the coil
  TGeoVolume* voDCS02 = new TGeoVolume("DCS02", new TGeoTubeSeg(coilRo, coilRo + 3.125, sW / 2., 21., 51.), kMedAlu);
  TGeoVolume* voDCS021 = new TGeoVolume(
    "DCS021", new TGeoConeSeg(sW / 2., coilRo + 3.124, 320., coilRo + 3.125, coilRo + 5.125, 21., 21.4), kMedAlu);

  // Sleeves
  TGeoVolume* voDCS03 =
    new TGeoVolume("DCS03", new TGeoTubeSeg(coilRi - 3.125, coilRo + 3.125, 3.125 / 2., 21., 51.), kMedAlu);

  TGeoVolume* voDCS04 = new TGeoVolume("DCS04", new TGeoTubeSeg(coilRi - 3.125, coilRi, coilH / 2., 21., 51.), kMedAlu);

  TGeoVolume* voDCS05 = new TGeoVolume("DCS05", new TGeoTubeSeg(coilRi - 3.125, coilRo, 3.125 / 2., 21., 51.), kMedAlu);
  //
  asDCoilSupport->AddNode(voDCS02, 1, new TGeoTranslation(0., 0., -(sW - coilH) / 2.));
  asDCoilSupport->AddNode(voDCS04, 1, gGeoIdentity);
  for (Int_t i = 0; i < 9; i++) {
    char nameR[16];
    snprintf(nameR, 16, "rotdcs%1d", i);
    Float_t phi = Float_t(i) * 3.75;
    TGeoRotation* rot = new TGeoRotation(nameR, 90., phi, 90., 90. + phi, 0., 0.);
    asDCoilSupport->AddNode(voDCS021, i, new TGeoCombiTrans(0., 0.004, -(sW - coilH) / 2., rot));
  }

  asDCoilSupport->AddNode(voDCS01, 1, new TGeoTranslation(0., 0., -sW / 2. - (sW - coilH) / 2. - 3.125 / 2.));
  asDCoilSupport->AddNode(voDCS03, 1, new TGeoTranslation(0., 0., +coilH / 2. + 3.125 / 2.));
  asDCoilSupport->AddNode(voDCS05, 1, new TGeoTranslation(0., 0., -coilH / 2. - 3.125 / 2.));

  //
  // SAA1 Support: Hanger 1
  //
  TGeoTranslation* trHanger = new TGeoTranslation("trHanger", 0., 250., 0.);
  trHanger->RegisterYourself();

  Float_t rmin1, rmin2, rmax1, rmax2;

  Float_t zHanger1 = 811.9;
  TGeoBBox* shHanger11 = new TGeoBBox(2.5 / 2., 250., 25. / 2.);
  shHanger11->SetName("shHanger11");

  rmin1 = (zHanger1 - 13.) * TMath::Tan(2. * kDegrad);
  rmin2 = rmin1 + 26. * TMath::Tan(2.0 * kDegrad);

  rmax1 = (zHanger1 - 13.) * TMath::Tan(9. * kDegrad);
  rmax2 = rmax1 + 26. * TMath::Tan(9. * kDegrad);

  TGeoCone* shHanger12 = new TGeoCone(13., rmin1, rmax1, rmin2, rmax2);
  shHanger12->SetName("shHanger12");
  TGeoCompositeShape* shHanger1 = new TGeoCompositeShape("shHanger1", "shHanger12*shHanger11:trHanger");
  TGeoVolume* voHanger1 = new TGeoVolume("DHanger1", shHanger1, kMedSteel);
  //
  // SAA1 Support: Hanger 2
  //
  Float_t zHanger2 = 1171.9;
  TGeoBBox* shHanger21 = new TGeoBBox(3.5 / 2., 250., 25. / 2.);
  shHanger21->SetName("shHanger21");

  rmin1 = 35.8 + (zHanger2 - 13. - zst) * TMath::Tan(2. * kDegrad);
  rmin2 = rmin1 + 26. * TMath::Tan(2.0 * kDegrad);

  rmax1 = rcD1 + (zHanger2 - 13. - kZDipole) * TMath::Tan(10.1 * kDegrad);
  rmax2 = rmax1 + 26. * TMath::Tan(10.1 * kDegrad);
  TGeoCone* shHanger22 = new TGeoCone(13., rmin1, rmax1, rmin2, rmax2);
  shHanger22->SetName("shHanger22");

  TGeoCompositeShape* shHanger2 = new TGeoCompositeShape("shHanger2", "shHanger22*shHanger21:trHanger");

  TGeoVolume* voHanger2 = new TGeoVolume("DHanger2", shHanger2, kMedSteel);
  //
  // Hanger support
  Float_t hsLength = yokeLength + (zHanger2 - kZDipole - yokeLength / 2.) + 25. / 2.;

  TGeoVolume* voHS1 = new TGeoVolume("DHS1", new TGeoBBox(1.5, 12.5, hsLength / 2.), kMedSteel);
  TGeoVolume* voHS2 = new TGeoVolume("DHS2", new TGeoBBox(12.5, 1.5, hsLength / 2.), kMedSteel);
  Float_t hsH = gapHeight / 2. + blockHeight - (rmax1 + rmax2) / 2. - 2.;

  TGeoVolume* voHS3 = new TGeoVolume("DHS3", new TGeoBBox(3.5 / 2., hsH / 2., 25. / 2.), kMedSteel);

  TGeoVolumeAssembly* asHS = new TGeoVolumeAssembly("asHS");
  asHS->AddNode(voHS1, 1, gGeoIdentity);
  asHS->AddNode(voHS2, 1, new TGeoTranslation(0., +14., 0.));
  asHS->AddNode(voHS2, 2, new TGeoTranslation(0., -14., 0.));
  asHS->AddNode(voHS3, 1, new TGeoTranslation(0., -hsH / 2. - 14. - 1.5, hsLength / 2. - 25. / 2.));

  dz = zHanger1;
  voDDIP->AddNode(voHanger1, 1, new TGeoTranslation(0., 0., dz));

  dz = zHanger2;
  voDDIP->AddNode(voHanger2, 1, new TGeoTranslation(0., 0., dz));

  // Assembly everything

  TGeoVolumeAssembly* asDipole = new TGeoVolumeAssembly("Dipole");
  // Yoke
  asDipole->AddNode(asYoke, 1, new TGeoTranslation(0., 0., -dzCoil));
  asDipole->AddNode(asCoil, 1, gGeoIdentity);
  // Contactor
  dz = lengthSt / 2. + coilH / 2. + rKnee;
  asDipole->AddNode(asContactor, 1, new TGeoTranslation(0., 0., dz + dzCoil));
  asDipole->AddNode(asContactor, 2, new TGeoCombiTrans(0., 0., dz - dzCoil, rotxy180));
  // Coil support
  asDipole->AddNode(asDCoilSupport, 1, new TGeoTranslation(0., 0., dz));
  asDipole->AddNode(asDCoilSupport, 2, new TGeoCombiTrans(0., 0., dz, rotxy180));
  asDipole->AddNode(asDCoilSupport, 3, new TGeoCombiTrans(0., 0., dz, rotxy108));
  asDipole->AddNode(asDCoilSupport, 4, new TGeoCombiTrans(0., 0., dz, rotxy288));

  asDipole->AddNode(asDCoilSupport, 5, new TGeoCombiTrans(0., 0., -dz, rotiz));
  asDipole->AddNode(asDCoilSupport, 6, new TGeoCombiTrans(0., 0., -dz, rotxz108));
  asDipole->AddNode(asDCoilSupport, 7, new TGeoCombiTrans(0., 0., -dz, rotxz180));
  asDipole->AddNode(asDCoilSupport, 8, new TGeoCombiTrans(0., 0., -dz, rotxz288));

  // Hanger (Support)
  dy = gapHeight / 2. + blockHeight + 14.;

  asDipole->AddNode(asHS, 1,
                    new TGeoTranslation(0., dy + 1.5, ((zHanger2 - kZDipole - yokeLength / 2.) + 25. / 2.) / 2.));

  asDipole->SetVisContainers(1);
  voDDIP->SetVisibility(0);

  top->AddNode(asDipole, 1, new TGeoCombiTrans(0., dipoleL / 2. * TMath::Tan(alhc * kDegrad), -kZDipole, rotxzlhc));
  top->AddNode(voDDIP, 1, new TGeoCombiTrans(0., 0., 0., rotxz));
}

void Dipole::createCompensatorDipole()
{
  auto top = gGeoManager->GetVolume("cave");
  top->AddNode(createMagnetYoke(), 1, new TGeoTranslation(0., 0., 1075.));
}

TGeoVolume* Dipole::createMagnetYoke()
{
  TGeoVolumeAssembly* voMagnet = new TGeoVolumeAssembly("DCM0");
  voMagnet->SetName("DCM0");
  TGeoRotation* Ry180 = new TGeoRotation("Ry180", 180., 180., 0.);
  TGeoMedium* kMedAlu = GetMedium("DIPO_ALU_C0");
  TGeoMedium* kMedCooper = GetMedium("DIPO_Cu_C0");
  TGeoMedium* kMedIron = GetMedium("DIPO_FE_C0");

  new TGeoBBox("shMagnetYokeOuter", 116.4 / 2.0, 90.2 / 2.0, 250.0 / 2.0);
  new TGeoBBox("shMagnetYokeInnerUp", 8.0 / 2.0, 32.2 / 2.0, 250.0 / 1.0);
  new TGeoBBox("shMagnetYokeInnerDw", 46.0 / 2.0, 23.0 / 2.0, 250.0 / 1.0);
  (new TGeoTranslation("trMagnetYokeOuter", 0.0, -29.1, 0.0))->RegisterYourself();
  (new TGeoTranslation("trMagnetYokeInnerUp", 0.0, 0.0, 0.0))->RegisterYourself();
  (new TGeoTranslation("trMagnetYokeInnerDw", 0.0, -27.5, 0.0))->RegisterYourself();

  TGeoCompositeShape* shMagnetYoke =
    new TGeoCompositeShape("shMagnet",
                           "shMagnetYokeOuter:trMagnetYokeOuter-(shMagnetYokeInnerUp:trMagnetYokeInnerUp+"
                           "shMagnetYokeInnerDw:trMagnetYokeInnerDw)");
  TGeoVolume* voMagnetYoke = new TGeoVolume("voMagnetYoke", shMagnetYoke, kMedIron);

  // Make the coils:
  TGeoVolume* voCoilH = gGeoManager->MakeBox("voCoilH", kMedCooper, 12.64 / 2.0, 21.46 / 2.0, 310.5 / 2.0);
  TGeoVolume* voCoilV = gGeoManager->MakeBox("voCoilV", kMedCooper, 12.64 / 2.0, 35.80 / 2.0, 26.9 / 2.0);

  // Make the top coil supports:
  // Polygone Coordinates (x,y)
  Double_t x, y;
  const Double_t kDegToRad = TMath::Pi() / 180.;
  const Double_t AngleInner = 4.5 * kDegToRad;
  const Double_t AngleOuter = 56.0 * kDegToRad;
  const Double_t ArcStart = 90. - AngleOuter / kDegToRad;
  const Double_t ArcEnd = 90. + AngleInner / kDegToRad;
  const Double_t b = 13.6;
  const Double_t Lx = 37.2;
  const Double_t Ly = 25.7;
  const Double_t LxV = 14.9;
  const Double_t R = 9.50;
  const Double_t dz = 2.00 / 2.0;
  const Int_t npoints = 8;
  Double_t CenterX;
  Double_t CenterY;
  Double_t PointsX[npoints] = { 0. };
  Double_t PointsY[npoints] = { 0. };
  Int_t ip = 0;
  // Start point:
  x = 0.0;
  y = 0.0;
  PointsX[ip] = x;
  PointsY[ip] = y;
  ip++;
  // 1st step:
  x = 0.00;
  y = 1.95;
  PointsX[ip] = x;
  PointsY[ip] = y;
  ip++;
  // 2nd step:
  x += b;
  y += b * TMath::Tan(AngleInner);
  PointsX[ip] = x;
  PointsY[ip] = y;
  ip++;
  // Center of Arc:
  x += R * TMath::Sin(AngleInner);
  y -= R * TMath::Cos(AngleInner);
  CenterX = x;
  CenterY = y;
  TGeoTubeSeg* shPolygonArc = new TGeoTubeSeg("shPolygonArc", R - 2.0, R, dz, ArcStart, ArcEnd);
  (new TGeoTranslation("trPolygonArc", x, y, 0.))->RegisterYourself();
  // 3rd Step:
  x += R * TMath::Sin(AngleOuter);
  y += R * TMath::Cos(AngleOuter);
  PointsX[ip] = x;
  PointsY[ip] = y;
  ip++;
  // 4th Step:
  Double_t a = Lx - b - R * TMath::Sin(AngleInner) - R * TMath::Sin(AngleOuter);
  x = Lx;
  y -= a * TMath::Tan(AngleOuter);
  PointsX[ip] = x;
  PointsY[ip] = y;
  ip++;
  // 5th Step:
  x = Lx;
  y = -Ly;
  PointsX[ip] = x;
  PointsY[ip] = y;
  ip++;
  // 6th Step:
  x = LxV;
  y = -Ly;
  PointsX[ip] = x;
  PointsY[ip] = y;
  ip++;
  // 7th Step:
  x = LxV;
  y = 0.0;
  PointsX[ip] = x;
  PointsY[ip] = y;
  ip++;
  //
  //
  //
  TGeoXtru* shPolygon = new TGeoXtru(2);
  shPolygon->SetNameTitle("shPolygon", "shPolygon");
  shPolygon->DefinePolygon(npoints, PointsX, PointsY);
  shPolygon->DefineSection(0, -dz, 0., 0., 1.0); // index, Z position, offset (x,y) and scale for first section
  shPolygon->DefineSection(1, +dz, 0., 0., 1.0); // idem, second section

  TGeoCompositeShape* shCoilSupportV = new TGeoCompositeShape("shCoilSupportV", "shPolygon+shPolygonArc:trPolygonArc");
  TGeoVolume* voCoilSupportV = new TGeoVolume("voCoilSupportV", shCoilSupportV, kMedAlu);

  const Double_t MagCoilDx = 12.64 / 2.;
  const Double_t MagCoilDy = 21.46 / 2.;
  const Double_t SqOuterDx = MagCoilDx + 2.8;
  const Double_t SqInnerDx = MagCoilDx + 0.6;
  const Double_t SqOuterDy = 29.2 / 2.;
  const Double_t SqInnerDy = 24.8 / 2.;
  const Double_t SqOuterDz = 15.5 / 2.;
  const Double_t SqInnerDz = SqOuterDz * 2.;
  TGeoBBox* shCoilSupportSqOuter = new TGeoBBox("shCoilSupportSqOuter", SqOuterDx, SqOuterDy, SqOuterDz);
  TGeoBBox* shCoilSupportSqInner = new TGeoBBox("shCoilSupportSqInner", SqInnerDx, SqInnerDy, SqInnerDz);
  TGeoCompositeShape* shCoilSupportSq =
    new TGeoCompositeShape("shCoilSupportSq", "shCoilSupportSqOuter - shCoilSupportSqInner");
  TGeoVolume* voCoilSupportSq = new TGeoVolume("voCoilSupportSq", shCoilSupportSq, kMedAlu);

  const Double_t HSuppDx = (Lx - LxV + 0.6) / 2.0;
  const Double_t HSuppDy = 2.2 / 2.0;
  const Double_t HSuppDz = SqOuterDz;

  TGeoVolume* voCoilSupportH = gGeoManager->MakeBox("voCoilSupportH", kMedAlu, HSuppDx, HSuppDy, HSuppDz);

  TGeoVolumeAssembly* voCoilSupport = new TGeoVolumeAssembly("voCoilSupport");
  voCoilSupportV->SetLineColor(kViolet + 9);
  voCoilSupportSq->SetLineColor(kBlue - 5);
  voCoilSupportH->SetLineColor(kPink);
  // voCoilSupportH  -> SetTransparency(16);
  voCoilSupport->AddNode(voCoilSupportV, 1, new TGeoTranslation(SqOuterDx - LxV, SqOuterDy, 0.));
  voCoilSupport->AddNode(voCoilSupportSq, 1, new TGeoTranslation(0., 0., 0.));
  voCoilSupport->AddNode(voCoilSupportH, 1, new TGeoTranslation(SqOuterDx + HSuppDx, SqOuterDy - Ly - HSuppDy, 0.));

  // Make the Top Support for Geodesic reference points:
  TGeoVolume* voSupportHTop = gGeoManager->MakeBox("voSupportHTop", kMedAlu, 66.0 / 2.0, 2.0 / 2.0, 17.0 / 2.0);
  TGeoVolume* voSupportHBot = gGeoManager->MakeBox("voSupportHBot", kMedAlu, 14.0 / 2.0, 2.0 / 2.0, 17.0 / 2.0);
  TGeoVolume* voSupportVert = gGeoManager->MakeBox("voSupportVert", kMedAlu, 3.0 / 2.0, 25.0 / 2.0, 17.0 / 2.0);

  TGeoVolumeAssembly* voSupportGeoRefPoint = new TGeoVolumeAssembly("voSupportGeoRefPoint");
  voSupportHTop->SetLineColor(kGreen);
  voSupportHBot->SetLineColor(kGreen);
  voSupportVert->SetLineColor(kGreen);
  voSupportGeoRefPoint->AddNode(voSupportHTop, 1, new TGeoTranslation(0.0, 28.0, 0.));
  voSupportGeoRefPoint->AddNode(voSupportHBot, 1, new TGeoTranslation(+33.0, 1.0, 0.));
  voSupportGeoRefPoint->AddNode(voSupportHBot, 2, new TGeoTranslation(-33.0, 1.0, 0.));
  voSupportGeoRefPoint->AddNode(voSupportVert, 1, new TGeoTranslation(+31.5, 14.5, 0.));
  voSupportGeoRefPoint->AddNode(voSupportVert, 2, new TGeoTranslation(-31.5, 14.5, 0.));

  // Add some color:
  voMagnetYoke->SetLineColor(kAzure - 7);
  voCoilH->SetLineColor(kOrange - 3);
  voCoilV->SetLineColor(kOrange - 3);
  // Assembling:

  voMagnet->AddNode(voMagnetYoke, 1, new TGeoTranslation(0., 0., 0.0));
  voMagnet->AddNode(voCoilH, 1, new TGeoTranslation(+16.14, +29.83, 0.0));
  voMagnet->AddNode(voCoilH, 2, new TGeoTranslation(-16.14, +29.83, 0.0));
  voMagnet->AddNode(voCoilH, 3, new TGeoTranslation(+16.14, -27.43, 0.0));
  voMagnet->AddNode(voCoilH, 4, new TGeoTranslation(-16.14, -27.43, 0.0));
  voMagnet->AddNode(voCoilV, 1, new TGeoTranslation(+16.14, 1.20, +141.8));
  voMagnet->AddNode(voCoilV, 2, new TGeoTranslation(-16.14, 1.20, +141.8));
  voMagnet->AddNode(voCoilV, 3, new TGeoTranslation(+16.14, 1.20, -141.8));
  voMagnet->AddNode(voCoilV, 4, new TGeoTranslation(-16.14, 1.20, -141.8));
  Double_t zGeoRef = 74.0 / 2. + SqOuterDz + 9.0 + 17.0 / 2.0;
  voMagnet->AddNode(voSupportGeoRefPoint, 1, new TGeoTranslation(0., 16.0, +zGeoRef));
  voMagnet->AddNode(voSupportGeoRefPoint, 2, new TGeoTranslation(0., 16.0, -zGeoRef));
  Double_t zCoilSupp = 29.83 - MagCoilDy - 0.6 + SqInnerDy;
  voMagnet->AddNode(voCoilSupport, 1, new TGeoTranslation(+16.14, zCoilSupp, 74.0 * 0.5));
  voMagnet->AddNode(voCoilSupport, 2, new TGeoTranslation(+16.14, zCoilSupp, -74.0 * 0.5));
  voMagnet->AddNode(voCoilSupport, 3, new TGeoTranslation(+16.14, zCoilSupp, 74.0 * 1.5));
  voMagnet->AddNode(voCoilSupport, 4, new TGeoTranslation(+16.14, zCoilSupp, -74.0 * 1.5));
  //
  voMagnet->AddNode(voCoilSupport, 5, new TGeoCombiTrans(-16.14, zCoilSupp, 74.0 * 0.5, Ry180));
  voMagnet->AddNode(voCoilSupport, 6, new TGeoCombiTrans(-16.14, zCoilSupp, -74.0 * 0.5, Ry180));
  voMagnet->AddNode(voCoilSupport, 7, new TGeoCombiTrans(-16.14, zCoilSupp, 74.0 * 1.5, Ry180));
  voMagnet->AddNode(voCoilSupport, 8, new TGeoCombiTrans(-16.14, zCoilSupp, -74.0 * 1.5, Ry180));

  return (TGeoVolume*)voMagnet;
}

FairModule* Dipole::CloneModule() const { return new Dipole(*this); }
ClassImp(o2::passive::Dipole);
