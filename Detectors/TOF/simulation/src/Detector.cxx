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
#include "TLorentzVector.h"
#include "TMath.h"
#include "TString.h"

#include "FairLogger.h"
#include "FairRootManager.h"
#include "FairVolume.h"

#include "TOFSimulation/Detector.h"

#include "DetectorsBase/GeometryManager.h"
#include "SimulationDataFormat/Stack.h"
#include <TVirtualMC.h> // for TVirtualMC, gMC

using namespace o2::tof;

ClassImp(Detector);

Detector::Detector(Bool_t active)
  : o2::Base::DetImpl<Detector>("TOF", active),
    mEventNr(0),
    mTOFHoles(kTRUE),
    mHits(new std::vector<HitType>)
{
  for (Int_t i = 0; i < Geo::NSECTORS; i++)
    mTOFSectors[i] = 1;
}

void Detector::Initialize() { o2::Base::Detector::Initialize(); }
Bool_t Detector::ProcessHits(FairVolume* v)
{
  static auto* refMC = TVirtualMC::GetMC();

  static TLorentzVector position2;
  refMC->TrackPosition(position2);
  Float_t radius = TMath::Sqrt(position2.X() * position2.X() + position2.Y() * position2.Y());
  LOG(DEBUG) << "Process hit in TOF volume ar R=" << radius << " - Z=" << position2.Z() << FairLogger::endl;

  // This method is called from the MC stepping for the sensitive volume only

  if (static_cast<int>(refMC->TrackCharge()) == 0) {
    // set a very large step size for neutral particles
    return kFALSE; // take only charged particles
  }

  Float_t enDep = refMC->Edep();
  if (enDep < 1E-8)
    return kFALSE; // wo se need a threshold?

  // ADD HIT
  static TLorentzVector position;
  refMC->TrackPosition(position);
  float time = refMC->TrackTime() * 1.0e09;
  auto stack = static_cast<o2::Data::Stack*>(refMC->GetStack());
  int trackID = stack->GetCurrentTrackNumber();
  int sensID = v->getMCid();

  addHit(position.X(), position.Y(), position.Z(), time, enDep, trackID, sensID);
  stack->addHit(GetDetId());

  return kTRUE;
}

HitType* Detector::addHit(Float_t x, Float_t y, Float_t z, Float_t time, Float_t energy, Int_t trackId, Int_t detId)
{
  mHits->emplace_back(x, y, z, time, energy, trackId, detId);
  return &mHits->back();
}

void Detector::Register()
{
  auto* mgr = FairRootManager::Instance();
  mgr->RegisterAny(addNameTo("Hit").data(), mHits, kTRUE);
}

void Detector::Reset() { mHits->clear(); }
void Detector::CreateMaterials()
{
  Int_t isxfld = 2;
  Float_t sxmgmx = 10.;
  o2::Base::Detector::initFieldTrackingParams(isxfld, sxmgmx);

  //--- Quartz (SiO2) ---
  Float_t aq[2] = { 28.0855, 15.9994 };
  Float_t zq[2] = { 14., 8. };
  Float_t wq[2] = { 1., 2. };
  Float_t dq = 2.7; // (+5.9%)
  Int_t nq = -2;

  // --- Nomex (C14H22O2N2) ---
  Float_t anox[4] = { 12.011, 1.00794, 15.9994, 14.00674 };
  Float_t znox[4] = { 6., 1., 8., 7. };
  Float_t wnox[4] = { 14., 22., 2., 2. };
  // Float_t dnox  = 0.048; //old value
  Float_t dnox = 0.22; // (x 4.6)
  Int_t nnox = -4;

  // --- G10  {Si, O, C, H, O} ---
  Float_t we[7], na[7];

  Float_t ag10[5] = { 28.0855, 15.9994, 12.011, 1.00794, 15.9994 };
  Float_t zg10[5] = { 14., 8., 6., 1., 8. };
  Float_t wmatg10[5];
  Int_t nlmatg10 = 5;
  na[0] = 1., na[1] = 2., na[2] = 0., na[3] = 0., na[4] = 0.;
  MaterialMixer(we, ag10, na, 5);
  wmatg10[0] = we[0] * 0.6;
  wmatg10[1] = we[1] * 0.6;
  na[0] = 0., na[1] = 0., na[2] = 14., na[3] = 20., na[4] = 3.;
  MaterialMixer(we, ag10, na, 5);
  wmatg10[2] = we[2] * 0.4;
  wmatg10[3] = we[3] * 0.4;
  wmatg10[4] = we[4] * 0.4;
  // Float_t densg10 = 1.7; //old value
  Float_t densg10 = 2.0; // (+17.8%)

  // --- Water ---
  Float_t awa[2] = { 1.00794, 15.9994 };
  Float_t zwa[2] = { 1., 8. };
  Float_t wwa[2] = { 2., 1. };
  Float_t dwa = 1.0;
  Int_t nwa = -2;

  // --- Air ---
  Float_t aAir[4] = { 12.011, 14.00674, 15.9994, 39.948 };
  Float_t zAir[4] = { 6., 7., 8., 18. };
  Float_t wAir[4] = { 0.000124, 0.755267, 0.231781, 0.012827 };
  Float_t dAir = 1.20479E-3;

  // --- Fibre Glass ---
  Float_t afg[4] = { 28.0855, 15.9994, 12.011, 1.00794 };
  Float_t zfg[4] = { 14., 8., 6., 1. };
  Float_t wfg[4] = { 0.12906, 0.29405, 0.51502, 0.06187 };
  // Float_t dfg    = 1.111;
  Float_t dfg = 2.05; // (x1.845)
  Int_t nfg = 4;

  // --- Freon C2F4H2 + SF6 ---
  Float_t afre[4] = { 12.011, 1.00794, 18.9984032, 32.0065 };
  Float_t zfre[4] = { 6., 1., 9., 16. };
  Float_t wfre[4] = { 0.21250, 0.01787, 0.74827, 0.021355 };
  Float_t densfre = 0.00375;
  Int_t nfre = 4;

  // --- Cables and tubes {Al, Cu} ---
  Float_t acbt[2] = { 26.981539, 63.546 };
  Float_t zcbt[2] = { 13., 29. };
  Float_t wcbt[2] = { 0.407, 0.593 };
  Float_t decbt = 0.68;

  // --- Cable {CH2, Al, Cu} ---
  Float_t asc[4] = { 12.011, 1.00794, 26.981539, 63.546 };
  Float_t zsc[4] = { 6., 1., 13., 29. };
  Float_t wsc[4];
  for (Int_t ii = 0; ii < 4; ii++)
    wsc[ii] = 0.;

  Float_t wDummy[4], nDummy[4];
  for (Int_t ii = 0; ii < 4; ii++)
    wDummy[ii] = 0.;
  for (Int_t ii = 0; ii < 4; ii++)
    nDummy[ii] = 0.;
  nDummy[0] = 1.;
  nDummy[1] = 2.;
  MaterialMixer(wDummy, asc, nDummy, 2);
  wsc[0] = 0.4375 * wDummy[0];
  wsc[1] = 0.4375 * wDummy[1];
  wsc[2] = 0.3244;
  wsc[3] = 0.2381;
  Float_t dsc = 1.223;

  // --- Crates boxes {Al, Cu, Fe, Cr, Ni} ---
  Float_t acra[5] = { 26.981539, 63.546, 55.845, 51.9961, 58.6934 };
  Float_t zcra[5] = { 13., 29., 26., 24., 28. };
  Float_t wcra[5] = { 0.7, 0.2, 0.07, 0.018, 0.012 };
  Float_t dcra = 0.77;

  // --- Polietilene CH2 ---
  Float_t aPlastic[2] = { 12.011, 1.00794 };
  Float_t zPlastic[2] = { 6., 1. };
  Float_t wPlastic[2] = { 1., 2. };
  // Float_t dPlastic = 0.92; // PDB value
  Float_t dPlastic = 0.93; // (~+1.1%)
  Int_t nwPlastic = -2;

  Mixture(0, "Air$", aAir, zAir, dAir, 4, wAir);
  Mixture(1, "Nomex$", anox, znox, dnox, nnox, wnox);
  Mixture(2, "G10$", ag10, zg10, densg10, nlmatg10, wmatg10);
  Mixture(3, "fibre glass$", afg, zfg, dfg, nfg, wfg);
  Material(4, "Al $", 26.981539, 13., 2.7, -8.9, 999.);
  Float_t factor = 0.4 / 1.5 * 2. / 3.;
  Material(5, "Al honeycomb$", 26.981539, 13., 2.7 * factor, -8.9 / factor, 999.);
  Mixture(6, "Freon$", afre, zfre, densfre, nfre, wfre);
  Mixture(7, "Glass$", aq, zq, dq, nq, wq);
  Mixture(8, "Water$", awa, zwa, dwa, nwa, wwa);
  Mixture(9, "cables+tubes$", acbt, zcbt, decbt, 2, wcbt);
  Material(10, "Cu $", 63.546, 29., 8.96, -1.43, 999.);
  Mixture(11, "cable$", asc, zsc, dsc, 4, wsc);
  Mixture(12, "Al+Cu+steel$", acra, zcra, dcra, 5, wcra);
  Mixture(13, "plastic$", aPlastic, zPlastic, dPlastic, nwPlastic, wPlastic);
  Float_t factorHoles = 1. / 36.5;
  Material(14, "Al honey for holes$", 26.981539, 13., 2.7 * factorHoles, -8.9 / factorHoles, 999.);

  Float_t epsil, stmin, deemax, stemax;

  //   STD data
  //  EPSIL  = 0.1   ! Tracking precision,
  //  STEMAX = 0.1   ! Maximum displacement for multiple scattering
  //  DEEMAX = 0.1   ! Maximum fractional energy loss, DLS
  //  STMIN  = 0.1

  // TOF data
  epsil = .001; // Tracking precision,
  stemax = -1.; // Maximum displacement for multiple scattering
  deemax = -.3; // Maximum fractional energy loss, DLS
  stmin = -.8;

  Medium(kAir, "Air$", 0, 0, isxfld, sxmgmx, 10., stemax, deemax, epsil, stmin);
  Medium(kNomex, "Nomex$", 1, 0, isxfld, sxmgmx, 10., stemax, deemax, epsil, stmin);
  Medium(kG10, "G10$", 2, 0, isxfld, sxmgmx, 10., stemax, deemax, epsil, stmin);
  Medium(kFiberGlass, "fibre glass$", 3, 0, isxfld, sxmgmx, 10., stemax, deemax, epsil, stmin);
  Medium(kAlFrame, "Al Frame$", 4, 0, isxfld, sxmgmx, 10., stemax, deemax, epsil, stmin);
  Medium(kHoneycomb, "honeycomb$", 5, 0, isxfld, sxmgmx, 10., stemax, deemax, epsil, stmin);
  Medium(kFre, "Fre$", 6, 0, isxfld, sxmgmx, 10., stemax, deemax, epsil, stmin);
  Medium(kCuS, "Cu-S$", 10, 1, isxfld, sxmgmx, 10., stemax, deemax, epsil, stmin);
  Medium(kGlass, "Glass$", 7, 0, isxfld, sxmgmx, 10., stemax, deemax, epsil, stmin);
  Medium(kWater, "Water$", 8, 0, isxfld, sxmgmx, 10., stemax, deemax, epsil, stmin);
  Medium(kCable, "Cable$", 11, 0, isxfld, sxmgmx, 10., stemax, deemax, epsil, stmin);
  Medium(kCableTubes, "Cables+Tubes$", 9, 0, isxfld, sxmgmx, 10., stemax, deemax, epsil, stmin);
  Medium(kCopper, "Copper$", 10, 0, isxfld, sxmgmx, 10., stemax, deemax, epsil, stmin);
  Medium(kPlastic, "Plastic$", 13, 0, isxfld, sxmgmx, 10., stemax, deemax, epsil, stmin);
  Medium(kCrates, "Crates$", 12, 0, isxfld, sxmgmx, 10., stemax, deemax, epsil, stmin);
  Medium(kHoneyHoles, "honey_holes$", 14, 0, isxfld, sxmgmx, 10., stemax, deemax, epsil, stmin);
}

void Detector::MaterialMixer(Float_t* p, const Float_t* const a, const Float_t* const m, Int_t n) const
{
  // a[] atomic weights vector      (in)
  //     (atoms present in more compound appear separately)
  // m[] number of corresponding atoms in the compound  (in)
  Float_t t = 0.;
  for (Int_t i = 0; i < n; ++i) {
    p[i] = a[i] * m[i];
    t += p[i];
  }
  for (Int_t i = 0; i < n; ++i) {
    p[i] = p[i] / t;
  }
}

void Detector::ConstructGeometry()
{
  CreateMaterials();

  /*
    xTof = 124.5;//fTOFGeometry->StripLength()+2.*(0.3+0.03); // cm,  x-dimension of FTOA volume
    yTof = fTOFGeometry->Rmax()-fTOFGeometry->Rmin(); // cm,  y-dimension of FTOA volume
    Float_t zTof = fTOFGeometry->ZlenA();             // cm,  z-dimension of FTOA volume
   */

  Float_t xTof = Geo::STRIPLENGTH + 2.5, yTof = Geo::RMAX - Geo::RMIN, zTof = Geo::ZLENA;
  DefineGeometry(xTof, yTof, zTof);

  LOG(INFO) << "Loaded TOF geometry" << FairLogger::endl;

  TGeoVolume* v = gGeoManager->GetVolume("FPAD");
  if (v == nullptr)
    printf("Sensitive volume FSEN not found!!!!!!!!");
  else {
    AddSensitiveVolume(v);
  }
}

void Detector::EndOfEvent() { Reset(); }
void Detector::DefineGeometry(Float_t xtof, Float_t ytof, Float_t zlenA)
{
  //
  // Definition of the Time Of Fligh Resistive Plate Chambers
  //

  Float_t xFLT, yFLT, zFLTA;
  xFLT = xtof - 2. * Geo::MODULEWALLTHICKNESS;
  yFLT = ytof * 0.5 - Geo::MODULEWALLTHICKNESS;
  zFLTA = zlenA - 2. * Geo::MODULEWALLTHICKNESS;

  createModules(xtof, ytof, zlenA, xFLT, yFLT, zFLTA);
  makeStripsInModules(ytof, zlenA);

  createModuleCovers(xtof, zlenA);

  createBackZone(xtof, ytof, zlenA);
  makeFrontEndElectronics(xtof);
  makeFEACooling(xtof);
  makeNinoMask(xtof);
  makeSuperModuleCooling(xtof, ytof, zlenA);
  makeSuperModuleServices(xtof, ytof, zlenA);

  makeModulesInBTOFvolumes(ytof, zlenA);
  makeCoversInBTOFvolumes();
  makeBackInBTOFvolumes(ytof);

  makeReadoutCrates(ytof);
}

void Detector::createModules(Float_t xtof, Float_t ytof, Float_t zlenA, Float_t xFLT, Float_t yFLT, Float_t zFLTA) const
{
  //
  // Create supermodule volume
  // and wall volumes to separate 5 modules
  //

  Int_t idrotm[8];
  for (Int_t ii = 0; ii < 8; ii++)
    idrotm[ii] = 0;

  // Definition of the of fibre glass modules (FTOA, FTOB and FTOC)
  Float_t par[3];
  par[0] = xtof * 0.5;
  par[1] = ytof * 0.25;
  par[2] = zlenA * 0.5;
  TVirtualMC::GetMC()->Gsvolu("FTOA", "BOX ", getMediumID(kFiberGlass), par, 3); // Fibre glass

  if (mTOFHoles) {
    par[0] = xtof * 0.5;
    par[1] = ytof * 0.25;
    par[2] = (zlenA * 0.5 - Geo::INTERCENTRMODBORDER1) * 0.5;
    TVirtualMC::GetMC()->Gsvolu("FTOB", "BOX ", getMediumID(kFiberGlass), par, 3); // Fibre glass
    TVirtualMC::GetMC()->Gsvolu("FTOC", "BOX ", getMediumID(kFiberGlass), par, 3); // Fibre glass
  }

  // Definition and positioning
  // of the not sensitive volumes with Insensitive Freon (FLTA, FLTB and FLTC)
  par[0] = xFLT * 0.5;
  par[1] = yFLT * 0.5;
  par[2] = zFLTA * 0.5;
  TVirtualMC::GetMC()->Gsvolu("FLTA", "BOX ", getMediumID(kFre), par, 3); // Freon mix

  Float_t xcoor, ycoor, zcoor;
  xcoor = 0.;
  ycoor = Geo::MODULEWALLTHICKNESS * 0.5;
  zcoor = 0.;
  TVirtualMC::GetMC()->Gspos("FLTA", 0, "FTOA", xcoor, ycoor, zcoor, 0, "ONLY");

  if (mTOFHoles) {
    par[2] = (zlenA * 0.5 - 2. * Geo::MODULEWALLTHICKNESS - Geo::INTERCENTRMODBORDER1) * 0.5;
    TVirtualMC::GetMC()->Gsvolu("FLTB", "BOX ", getMediumID(kFre), par, 3); // Freon mix
    TVirtualMC::GetMC()->Gsvolu("FLTC", "BOX ", getMediumID(kFre), par, 3); // Freon mix

    // xcoor = 0.;
    // ycoor = Geo::MODULEWALLTHICKNESS*0.5;
    zcoor = Geo::MODULEWALLTHICKNESS;
    TVirtualMC::GetMC()->Gspos("FLTB", 0, "FTOB", xcoor, ycoor, zcoor, 0, "ONLY");
    TVirtualMC::GetMC()->Gspos("FLTC", 0, "FTOC", xcoor, ycoor, -zcoor, 0, "ONLY");
  }

  // Definition and positioning
  // of the fibre glass walls between central and intermediate modules (FWZ1 and FWZ2)
  Float_t alpha, tgal, beta, tgbe, trpa[11];
  // tgal  = (yFLT - 2.*Geo::LENGTHINCEMODBORDER)/(Geo::INTERCENTRMODBORDER2 - Geo::INTERCENTRMODBORDER1);
  tgal = (yFLT - Geo::LENGTHINCEMODBORDERU - Geo::LENGTHINCEMODBORDERD) /
         (Geo::INTERCENTRMODBORDER2 - Geo::INTERCENTRMODBORDER1);
  alpha = TMath::ATan(tgal);
  beta = (TMath::Pi() * 0.5 - alpha) * 0.5;
  tgbe = TMath::Tan(beta);
  trpa[0] = xFLT * 0.5;
  trpa[1] = 0.;
  trpa[2] = 0.;
  trpa[3] = 2. * Geo::MODULEWALLTHICKNESS;
  // trpa[4]  = (Geo::LENGTHINCEMODBORDER - 2.*Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
  // trpa[5]  = (Geo::LENGTHINCEMODBORDER + 2.*Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
  trpa[4] = (Geo::LENGTHINCEMODBORDERD - 2. * Geo::MODULEWALLTHICKNESS * tgbe) * 0.5;
  trpa[5] = (Geo::LENGTHINCEMODBORDERD + 2. * Geo::MODULEWALLTHICKNESS * tgbe) * 0.5;
  trpa[6] =
    TMath::ATan(tgbe * 0.5) * TMath::RadToDeg(); // TMath::ATan((trpa[5] - trpa[4])/(2.*trpa[3]))*TMath::RadToDeg();
  trpa[7] = 2. * Geo::MODULEWALLTHICKNESS;
  trpa[8] = (Geo::LENGTHINCEMODBORDERD - 2. * Geo::MODULEWALLTHICKNESS * tgbe) * 0.5;
  trpa[9] = (Geo::LENGTHINCEMODBORDERD + 2. * Geo::MODULEWALLTHICKNESS * tgbe) * 0.5;
  // trpa[8]  = (Geo::LENGTHINCEMODBORDER - 2.*Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
  // trpa[9]  = (Geo::LENGTHINCEMODBORDER + 2.*Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
  trpa[10] =
    TMath::ATan(tgbe * 0.5) * TMath::RadToDeg(); // TMath::ATan((trpa[5] - trpa[4])/(2.*trpa[3]))*TMath::RadToDeg();
  TVirtualMC::GetMC()->Gsvolu("FWZ1D", "TRAP", getMediumID(kFiberGlass), trpa, 11); // Fibre glass

  Matrix(idrotm[0], 90., 90., 180., 0., 90., 180.);
  Matrix(idrotm[1], 90., 90., 0., 0., 90., 0.);

  // xcoor = 0.;
  // ycoor = -(yFLT - Geo::LENGTHINCEMODBORDER)*0.5;
  ycoor = -(yFLT - Geo::LENGTHINCEMODBORDERD) * 0.5;
  zcoor = Geo::INTERCENTRMODBORDER1;
  TVirtualMC::GetMC()->Gspos("FWZ1D", 1, "FLTA", xcoor, ycoor, zcoor, idrotm[0], "ONLY");
  TVirtualMC::GetMC()->Gspos("FWZ1D", 2, "FLTA", xcoor, ycoor, -zcoor, idrotm[1], "ONLY");

  Float_t y0B, ycoorB, zcoorB;

  if (mTOFHoles) {
    // y0B = Geo::LENGTHINCEMODBORDER - Geo::MODULEWALLTHICKNESS*tgbe;
    y0B = Geo::LENGTHINCEMODBORDERD - Geo::MODULEWALLTHICKNESS * tgbe;
    trpa[0] = xFLT * 0.5;
    trpa[1] = 0.;
    trpa[2] = 0.;
    trpa[3] = Geo::MODULEWALLTHICKNESS;
    trpa[4] = (y0B - Geo::MODULEWALLTHICKNESS * tgbe) * 0.5;
    trpa[5] = (y0B + Geo::MODULEWALLTHICKNESS * tgbe) * 0.5;
    trpa[6] =
      TMath::ATan(tgbe * 0.5) * TMath::RadToDeg(); // TMath::ATan((trpa[5] - trpa[4])/(2.*trpa[3]))*TMath::RadToDeg();
    trpa[7] = Geo::MODULEWALLTHICKNESS;
    trpa[8] = (y0B - Geo::MODULEWALLTHICKNESS * tgbe) * 0.5;
    trpa[9] = (y0B + Geo::MODULEWALLTHICKNESS * tgbe) * 0.5;
    trpa[10] =
      TMath::ATan(tgbe * 0.5) * TMath::RadToDeg(); // TMath::ATan((trpa[5] - trpa[4])/(2.*trpa[3]))*TMath::RadToDeg();
    // xcoor = 0.;
    ycoorB = ycoor - Geo::MODULEWALLTHICKNESS * 0.5 * tgbe;
    zcoorB =
      (zlenA * 0.5 - 2. * Geo::MODULEWALLTHICKNESS - Geo::INTERCENTRMODBORDER1) * 0.5 - 2. * Geo::MODULEWALLTHICKNESS;
    TVirtualMC::GetMC()->Gsvolu("FWZAD", "TRAP", getMediumID(kFiberGlass), trpa, 11); // Fibre glass
    TVirtualMC::GetMC()->Gspos("FWZAD", 1, "FLTB", xcoor, ycoorB, zcoorB, idrotm[1], "ONLY");
    TVirtualMC::GetMC()->Gspos("FWZAD", 2, "FLTC", xcoor, ycoorB, -zcoorB, idrotm[0], "ONLY");
  }

  tgal = (yFLT - Geo::LENGTHINCEMODBORDERU - Geo::LENGTHINCEMODBORDERD) /
         (Geo::INTERCENTRMODBORDER2 - Geo::INTERCENTRMODBORDER1);
  alpha = TMath::ATan(tgal);
  beta = (TMath::Pi() * 0.5 - alpha) * 0.5;
  tgbe = TMath::Tan(beta);
  trpa[0] = xFLT * 0.5;
  trpa[1] = 0.;
  trpa[2] = 0.;
  trpa[3] = 2. * Geo::MODULEWALLTHICKNESS;
  // trpa[4]  = (Geo::LENGTHINCEMODBORDER - 2.*Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
  // trpa[5]  = (Geo::LENGTHINCEMODBORDER + 2.*Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
  trpa[4] = (Geo::LENGTHINCEMODBORDERU - 2. * Geo::MODULEWALLTHICKNESS * tgbe) * 0.5;
  trpa[5] = (Geo::LENGTHINCEMODBORDERU + 2. * Geo::MODULEWALLTHICKNESS * tgbe) * 0.5;
  trpa[6] =
    TMath::ATan(tgbe * 0.5) * TMath::RadToDeg(); // TMath::ATan((trpa[5] - trpa[4])/(2.*trpa[3]))*TMath::RadToDeg();
  trpa[7] = 2. * Geo::MODULEWALLTHICKNESS;
  trpa[8] = (Geo::LENGTHINCEMODBORDERU - 2. * Geo::MODULEWALLTHICKNESS * tgbe) * 0.5;
  trpa[9] = (Geo::LENGTHINCEMODBORDERU + 2. * Geo::MODULEWALLTHICKNESS * tgbe) * 0.5;
  // trpa[8]  = (Geo::LENGTHINCEMODBORDER - 2.*Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
  // trpa[9]  = (Geo::LENGTHINCEMODBORDER + 2.*Geo::MODULEWALLTHICKNESS*tgbe)*0.5;
  trpa[10] =
    TMath::ATan(tgbe * 0.5) * TMath::RadToDeg(); // TMath::ATan((trpa[5] - trpa[4])/(2.*trpa[3]))*TMath::RadToDeg();
  TVirtualMC::GetMC()->Gsvolu("FWZ1U", "TRAP", getMediumID(kFiberGlass), trpa, 11); // Fibre glass

  Matrix(idrotm[2], 90., 270., 0., 0., 90., 180.);
  Matrix(idrotm[3], 90., 270., 180., 0., 90., 0.);

  // xcoor = 0.;
  // ycoor = (yFLT - Geo::LENGTHINCEMODBORDER)*0.5;
  ycoor = (yFLT - Geo::LENGTHINCEMODBORDERU) * 0.5;
  zcoor = Geo::INTERCENTRMODBORDER2;
  TVirtualMC::GetMC()->Gspos("FWZ1U", 1, "FLTA", xcoor, ycoor, zcoor, idrotm[2], "ONLY");
  TVirtualMC::GetMC()->Gspos("FWZ1U", 2, "FLTA", xcoor, ycoor, -zcoor, idrotm[3], "ONLY");

  if (mTOFHoles) {
    // y0B = Geo::LENGTHINCEMODBORDER + Geo::MODULEWALLTHICKNESS*tgbe;
    y0B = Geo::LENGTHINCEMODBORDERU + Geo::MODULEWALLTHICKNESS * tgbe;
    trpa[0] = xFLT * 0.5;
    trpa[1] = 0.;
    trpa[2] = 0.;
    trpa[3] = Geo::MODULEWALLTHICKNESS;
    trpa[4] = (y0B - Geo::MODULEWALLTHICKNESS * tgbe) * 0.5;
    trpa[5] = (y0B + Geo::MODULEWALLTHICKNESS * tgbe) * 0.5;
    trpa[6] =
      TMath::ATan(tgbe * 0.5) * TMath::RadToDeg(); // TMath::ATan((trpa[5] - trpa[4])/(2.*trpa[3]))*TMath::RadToDeg();
    trpa[7] = Geo::MODULEWALLTHICKNESS;
    trpa[8] = (y0B - Geo::MODULEWALLTHICKNESS * tgbe) * 0.5;
    trpa[9] = (y0B + Geo::MODULEWALLTHICKNESS * tgbe) * 0.5;
    trpa[10] =
      TMath::ATan(tgbe * 0.5) * TMath::RadToDeg(); // TMath::ATan((trpa[5] - trpa[4])/(2.*trpa[3]))*TMath::RadToDeg();
    TVirtualMC::GetMC()->Gsvolu("FWZBU", "TRAP", getMediumID(kFiberGlass), trpa, 11); // Fibre glass
    // xcoor = 0.;
    ycoorB = ycoor - Geo::MODULEWALLTHICKNESS * 0.5 * tgbe;
    zcoorB = (zlenA * 0.5 - 2. * Geo::MODULEWALLTHICKNESS - Geo::INTERCENTRMODBORDER1) * 0.5 -
             (Geo::INTERCENTRMODBORDER2 - Geo::INTERCENTRMODBORDER1) - 2. * Geo::MODULEWALLTHICKNESS;
    TVirtualMC::GetMC()->Gspos("FWZBU", 1, "FLTB", xcoor, ycoorB, zcoorB, idrotm[3], "ONLY");
    TVirtualMC::GetMC()->Gspos("FWZBU", 2, "FLTC", xcoor, ycoorB, -zcoorB, idrotm[2], "ONLY");
  }

  trpa[0] = 0.5 * (Geo::INTERCENTRMODBORDER2 - Geo::INTERCENTRMODBORDER1) / TMath::Cos(alpha);
  trpa[1] = 2. * Geo::MODULEWALLTHICKNESS;
  trpa[2] = xFLT * 0.5;
  trpa[3] = -beta * TMath::RadToDeg();
  trpa[4] = 0.;
  trpa[5] = 0.;
  TVirtualMC::GetMC()->Gsvolu("FWZ2", "PARA", getMediumID(kFiberGlass), trpa, 6); // Fibre glass

  Matrix(idrotm[4], alpha * TMath::RadToDeg(), 90., 90. + alpha * TMath::RadToDeg(), 90., 90., 180.);
  Matrix(idrotm[5], 180. - alpha * TMath::RadToDeg(), 90., 90. - alpha * TMath::RadToDeg(), 90., 90., 0.);

  // xcoor = 0.;
  // ycoor = 0.;
  ycoor = (Geo::LENGTHINCEMODBORDERD - Geo::LENGTHINCEMODBORDERU) * 0.5;
  zcoor = (Geo::INTERCENTRMODBORDER2 + Geo::INTERCENTRMODBORDER1) * 0.5;
  TVirtualMC::GetMC()->Gspos("FWZ2", 1, "FLTA", xcoor, ycoor, zcoor, idrotm[4], "ONLY");
  TVirtualMC::GetMC()->Gspos("FWZ2", 2, "FLTA", xcoor, ycoor, -zcoor, idrotm[5], "ONLY");

  if (mTOFHoles) {
    trpa[0] = 0.5 * (Geo::INTERCENTRMODBORDER2 - Geo::INTERCENTRMODBORDER1) / TMath::Cos(alpha);
    trpa[1] = Geo::MODULEWALLTHICKNESS;
    trpa[2] = xFLT * 0.5;
    trpa[3] = -beta * TMath::RadToDeg();
    trpa[4] = 0.;
    trpa[5] = 0.;
    TVirtualMC::GetMC()->Gsvolu("FWZC", "PARA", getMediumID(kFiberGlass), trpa, 6); // Fibre glass
    // xcoor = 0.;
    ycoorB = ycoor - Geo::MODULEWALLTHICKNESS * tgbe;
    zcoorB = (zlenA * 0.5 - 2. * Geo::MODULEWALLTHICKNESS - Geo::INTERCENTRMODBORDER1) * 0.5 -
             (Geo::INTERCENTRMODBORDER2 - Geo::INTERCENTRMODBORDER1) * 0.5 - 2. * Geo::MODULEWALLTHICKNESS;
    TVirtualMC::GetMC()->Gspos("FWZC", 1, "FLTB", xcoor, ycoorB, zcoorB, idrotm[5], "ONLY");
    TVirtualMC::GetMC()->Gspos("FWZC", 2, "FLTC", xcoor, ycoorB, -zcoorB, idrotm[4], "ONLY");
  }

  // Definition and positioning
  // of the fibre glass walls between intermediate and lateral modules (FWZ3 and FWZ4)
  tgal = (yFLT - 2. * Geo::LENGTHEXINMODBORDER) / (Geo::EXTERINTERMODBORDER2 - Geo::EXTERINTERMODBORDER1);
  alpha = TMath::ATan(tgal);
  beta = (TMath::Pi() * 0.5 - alpha) * 0.5;
  tgbe = TMath::Tan(beta);
  trpa[0] = xFLT * 0.5;
  trpa[1] = 0.;
  trpa[2] = 0.;
  trpa[3] = 2. * Geo::MODULEWALLTHICKNESS;
  trpa[4] = (Geo::LENGTHEXINMODBORDER - 2. * Geo::MODULEWALLTHICKNESS * tgbe) * 0.5;
  trpa[5] = (Geo::LENGTHEXINMODBORDER + 2. * Geo::MODULEWALLTHICKNESS * tgbe) * 0.5;
  trpa[6] =
    TMath::ATan(tgbe * 0.5) * TMath::RadToDeg(); // TMath::ATan((trpa[5] - trpa[4])/(2.*trpa[3]))*TMath::RadToDeg();
  trpa[7] = 2. * Geo::MODULEWALLTHICKNESS;
  trpa[8] = (Geo::LENGTHEXINMODBORDER - 2. * Geo::MODULEWALLTHICKNESS * tgbe) * 0.5;
  trpa[9] = (Geo::LENGTHEXINMODBORDER + 2. * Geo::MODULEWALLTHICKNESS * tgbe) * 0.5;
  trpa[10] =
    TMath::ATan(tgbe * 0.5) * TMath::RadToDeg(); // TMath::ATan((trpa[5] - trpa[4])/(2.*trpa[3]))*TMath::RadToDeg();
  TVirtualMC::GetMC()->Gsvolu("FWZ3", "TRAP", getMediumID(kFiberGlass), trpa, 11); // Fibre glass

  // xcoor = 0.;
  ycoor = (yFLT - Geo::LENGTHEXINMODBORDER) * 0.5;
  zcoor = Geo::EXTERINTERMODBORDER1;
  TVirtualMC::GetMC()->Gspos("FWZ3", 1, "FLTA", xcoor, ycoor, zcoor, idrotm[3], "ONLY");
  TVirtualMC::GetMC()->Gspos("FWZ3", 2, "FLTA", xcoor, ycoor, -zcoor, idrotm[2], "ONLY");

  if (mTOFHoles) {
    // xcoor = 0.;
    // ycoor = (yFLT - Geo::LENGTHEXINMODBORDER)*0.5;
    zcoor =
      -Geo::EXTERINTERMODBORDER1 + (zlenA * 0.5 + Geo::INTERCENTRMODBORDER1 - 2. * Geo::MODULEWALLTHICKNESS) * 0.5;
    TVirtualMC::GetMC()->Gspos("FWZ3", 5, "FLTB", xcoor, ycoor, zcoor, idrotm[2], "ONLY");
    TVirtualMC::GetMC()->Gspos("FWZ3", 6, "FLTC", xcoor, ycoor, -zcoor, idrotm[3], "ONLY");
  }

  // xcoor = 0.;
  ycoor = -(yFLT - Geo::LENGTHEXINMODBORDER) * 0.5;
  zcoor = Geo::EXTERINTERMODBORDER2;
  TVirtualMC::GetMC()->Gspos("FWZ3", 3, "FLTA", xcoor, ycoor, zcoor, idrotm[1], "ONLY");
  TVirtualMC::GetMC()->Gspos("FWZ3", 4, "FLTA", xcoor, ycoor, -zcoor, idrotm[0], "ONLY");

  if (mTOFHoles) {
    // xcoor = 0.;
    // ycoor = -(yFLT - Geo::LENGTHEXINMODBORDER)*0.5;
    zcoor =
      -Geo::EXTERINTERMODBORDER2 + (zlenA * 0.5 + Geo::INTERCENTRMODBORDER1 - 2. * Geo::MODULEWALLTHICKNESS) * 0.5;
    TVirtualMC::GetMC()->Gspos("FWZ3", 7, "FLTB", xcoor, ycoor, zcoor, idrotm[0], "ONLY");
    TVirtualMC::GetMC()->Gspos("FWZ3", 8, "FLTC", xcoor, ycoor, -zcoor, idrotm[1], "ONLY");
  }

  trpa[0] = 0.5 * (Geo::EXTERINTERMODBORDER2 - Geo::EXTERINTERMODBORDER1) / TMath::Cos(alpha);
  trpa[1] = 2. * Geo::MODULEWALLTHICKNESS;
  trpa[2] = xFLT * 0.5;
  trpa[3] = -beta * TMath::RadToDeg();
  trpa[4] = 0.;
  trpa[5] = 0.;
  TVirtualMC::GetMC()->Gsvolu("FWZ4", "PARA", getMediumID(kFiberGlass), trpa, 6); // Fibre glass

  Matrix(idrotm[6], alpha * TMath::RadToDeg(), 90., 90. + alpha * TMath::RadToDeg(), 90., 90., 180.);
  Matrix(idrotm[7], 180. - alpha * TMath::RadToDeg(), 90., 90. - alpha * TMath::RadToDeg(), 90., 90., 0.);

  // xcoor = 0.;
  ycoor = 0.;
  zcoor = (Geo::EXTERINTERMODBORDER2 + Geo::EXTERINTERMODBORDER1) * 0.5;
  TVirtualMC::GetMC()->Gspos("FWZ4", 1, "FLTA", xcoor, ycoor, zcoor, idrotm[7], "ONLY");
  TVirtualMC::GetMC()->Gspos("FWZ4", 2, "FLTA", xcoor, ycoor, -zcoor, idrotm[6], "ONLY");

  if (mTOFHoles) {
    // xcoor = 0.;
    // ycoor = 0.;
    zcoor = -(Geo::EXTERINTERMODBORDER2 + Geo::EXTERINTERMODBORDER1) * 0.5 +
            (zlenA * 0.5 + Geo::INTERCENTRMODBORDER1 - 2. * Geo::MODULEWALLTHICKNESS) * 0.5;
    TVirtualMC::GetMC()->Gspos("FWZ4", 3, "FLTB", xcoor, ycoor, zcoor, idrotm[6], "ONLY");
    TVirtualMC::GetMC()->Gspos("FWZ4", 4, "FLTC", xcoor, ycoor, -zcoor, idrotm[7], "ONLY");
  }
}

void Detector::makeStripsInModules(Float_t ytof, Float_t zlenA) const
{
  //
  // Define MRPC strip volume, called FSTR
  // Insert FSTR volume in FLTA/B/C volumes
  //
  // ciao
  Float_t yFLT = ytof * 0.5 - Geo::MODULEWALLTHICKNESS;

  ///////////////// Detector itself //////////////////////

  // new description for strip volume -double stack strip-
  // -- all constants are expressed in cm
  // height of different layers
  constexpr Float_t HGLFY = Geo::HFILIY + 2. * Geo::HGLASSY; // height of GLASS Layer

  constexpr Float_t LSENSMX = Geo::NPADX * Geo::XPAD; // length of Sensitive Layer
  constexpr Float_t HSENSMY = Geo::HSENSMY;           // height of Sensitive Layer
  constexpr Float_t WSENSMZ = Geo::NPADZ * Geo::ZPAD; // width of Sensitive Layer

  // height of the FSTR Volume (the strip volume)
  constexpr Float_t HSTRIPY = 2. * Geo::HHONY + 2. * Geo::HPCBY + 4. * Geo::HRGLY + 2. * HGLFY + Geo::HCPCBY;

  // width  of the FSTR Volume (the strip volume)
  constexpr Float_t WSTRIPZ = Geo::WCPCBZ;
  // length of the FSTR Volume (the strip volume)
  constexpr Float_t LSTRIPX = Geo::STRIPLENGTH;

  // FSTR volume definition-filling this volume with non sensitive Gas Mixture
  Float_t parfp[3] = { static_cast<Float_t>(LSTRIPX * 0.5), static_cast<Float_t>(HSTRIPY * 0.5),
                       static_cast<Float_t>(WSTRIPZ * 0.5) };
  TVirtualMC::GetMC()->Gsvolu("FSTR", "BOX", getMediumID(kFre), parfp, 3); // Freon mix

  Float_t posfp[3] = { 0., 0., 0. };

  // NOMEX (HONEYCOMB) Layer definition
  // parfp[0] = LSTRIPX*0.5;
  parfp[1] = Geo::HHONY * 0.5;
  parfp[2] = Geo::WHONZ * 0.5;
  TVirtualMC::GetMC()->Gsvolu("FHON", "BOX", getMediumID(kNomex), parfp, 3); // Nomex (Honeycomb)
  // positioning 2 NOMEX Layers on FSTR volume
  // posfp[0] = 0.;
  posfp[1] = -HSTRIPY * 0.5 + parfp[1];
  // posfp[2] = 0.;
  TVirtualMC::GetMC()->Gspos("FHON", 1, "FSTR", 0., posfp[1], 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FHON", 2, "FSTR", 0., -posfp[1], 0., 0, "ONLY");

  // Lower PCB Layer definition
  // parfp[0] = LSTRIPX*0.5;
  parfp[1] = Geo::HPCBY * 0.5;
  parfp[2] = Geo::WPCBZ1 * 0.5;
  TVirtualMC::GetMC()->Gsvolu("FPC1", "BOX", getMediumID(kG10), parfp, 3); // G10

  // Upper PCB Layer definition
  // parfp[0] = LSTRIPX*0.5;
  // parfp[1] =  Geo::HPCBY*0.5;
  parfp[2] = Geo::WPCBZ2 * 0.5;
  TVirtualMC::GetMC()->Gsvolu("FPC2", "BOX", getMediumID(kG10), parfp, 3); // G10

  // positioning 2 external PCB Layers in FSTR volume
  // posfp[0] = 0.;
  posfp[1] = -HSTRIPY * 0.5 + Geo::HHONY + parfp[1];
  // posfp[2] = 0.;
  TVirtualMC::GetMC()->Gspos("FPC1", 1, "FSTR", 0., -posfp[1], 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FPC2", 1, "FSTR", 0., posfp[1], 0., 0, "ONLY");

  // Central PCB layer definition
  // parfp[0] = LSTRIPX*0.5;
  parfp[1] = Geo::HCPCBY * 0.5;
  parfp[2] = Geo::WCPCBZ * 0.5;
  TVirtualMC::GetMC()->Gsvolu("FPCB", "BOX", getMediumID(kG10), parfp, 3); // G10
  gGeoManager->GetVolume("FPCB")->VisibleDaughters(kFALSE);

  // positioning the central PCB layer
  TVirtualMC::GetMC()->Gspos("FPCB", 1, "FSTR", 0., 0., 0., 0, "ONLY");

  // Sensitive volume definition
  Float_t parfs[3] = { static_cast<Float_t>(LSENSMX * 0.5), static_cast<Float_t>(HSENSMY * 0.5),
                       static_cast<Float_t>(WSENSMZ * 0.5) };
  TVirtualMC::GetMC()->Gsvolu("FSEN", "BOX", getMediumID(kCuS), parfs, 3); // Cu sensitive

  // printf("check material\n");
  // printf("ID used = %i\n",getMediumID(kCuS));
  // printf("ID needed = %i\n",gGeoManager->GetMedium("TOF_Cu-S$")->GetId());
  // getchar();

  // dividing FSEN along z in Geo::NPADZ=2 and along x in Geo::NPADX=48
  TVirtualMC::GetMC()->Gsdvn("FSEZ", "FSEN", Geo::NPADZ, 3);
  TVirtualMC::GetMC()->Gsdvn("FPAD", "FSEZ", Geo::NPADX, 1);
  // positioning sensitive layer inside FPCB
  TVirtualMC::GetMC()->Gspos("FSEN", 1, "FPCB", 0., 0., 0., 0, "ONLY");

  // RED GLASS Layer definition
  // parfp[0] = LSTRIPX*0.5;
  parfp[1] = Geo::HRGLY * 0.5;
  parfp[2] = Geo::WRGLZ * 0.5;
  TVirtualMC::GetMC()->Gsvolu("FRGL", "BOX", getMediumID(kGlass), parfp, 3); // red glass
  // positioning 4 RED GLASS Layers in FSTR volume
  // posfp[0] = 0.;
  posfp[1] = -HSTRIPY * 0.5 + Geo::HHONY + Geo::HPCBY + parfp[1];
  // posfp[2] = 0.;
  TVirtualMC::GetMC()->Gspos("FRGL", 1, "FSTR", 0., posfp[1], 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FRGL", 4, "FSTR", 0., -posfp[1], 0., 0, "ONLY");
  // posfp[0] = 0.;
  posfp[1] = (Geo::HCPCBY + Geo::HRGLY) * 0.5;
  // posfp[2] = 0.;
  TVirtualMC::GetMC()->Gspos("FRGL", 2, "FSTR", 0., -posfp[1], 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FRGL", 3, "FSTR", 0., posfp[1], 0., 0, "ONLY");

  // GLASS Layer definition
  // parfp[0] = LSTRIPX*0.5;
  parfp[1] = Geo::HGLASSY;
  parfp[2] = Geo::WGLFZ * 0.5;
  TVirtualMC::GetMC()->Gsvolu("FGLF", "BOX", getMediumID(kGlass), parfp, 3); // glass
  // positioning 2 GLASS Layers in FSTR volume
  // posfp[0] = 0.;
  posfp[1] = (Geo::HCPCBY + HGLFY) * 0.5 + Geo::HRGLY;
  // posfp[2] = 0.;
  TVirtualMC::GetMC()->Gspos("FGLF", 1, "FSTR", 0., -posfp[1], 0., 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FGLF", 2, "FSTR", 0., posfp[1], 0., 0, "ONLY");

  // Positioning the Strips (FSTR volumes) in the FLT volumes
  Int_t maxStripNumbers[5] = { Geo::NSTRIPC, Geo::NSTRIPB, Geo::NSTRIPA, Geo::NSTRIPB, Geo::NSTRIPC };

  Int_t idrotm[Geo::NSTRIPXSECTOR];
  for (Int_t ii = 0; ii < Geo::NSTRIPXSECTOR; ii++)
    idrotm[ii] = 0;

  Int_t totalStrip = 0;
  Float_t xpos, zpos, ypos, ang;
  for (Int_t iplate = 0; iplate < Geo::NPLATES; iplate++) {
    if (iplate > 0)
      totalStrip += maxStripNumbers[iplate - 1];
    for (Int_t istrip = 0; istrip < maxStripNumbers[iplate]; istrip++) {
      ang = Geo::getAngles(iplate, istrip);

      if (ang > 0.)
        Matrix(idrotm[istrip + totalStrip], 90., 0., 90. + ang, 90., ang, 90.);
      else if (ang == 0.)
        Matrix(idrotm[istrip + totalStrip], 90., 0., 90., 90., 0., 0.);
      else if (ang < 0.)
        Matrix(idrotm[istrip + totalStrip], 90., 0., 90. + ang, 90., -ang, 270.);

      xpos = 0.;
      ypos = Geo::getHeights(iplate, istrip) + yFLT * 0.5;
      zpos = Geo::getDistances(iplate, istrip);
      TVirtualMC::GetMC()->Gspos("FSTR", istrip + totalStrip + 1, "FLTA", xpos, ypos, -zpos,
                                 idrotm[istrip + totalStrip], "ONLY");

      if (mTOFHoles) {
        if (istrip + totalStrip + 1 > 53)
          TVirtualMC::GetMC()->Gspos(
            "FSTR", istrip + totalStrip + 1, "FLTC", xpos, ypos,
            -zpos - (zlenA * 0.5 - 2. * Geo::MODULEWALLTHICKNESS + Geo::INTERCENTRMODBORDER1) * 0.5,
            idrotm[istrip + totalStrip], "ONLY");
        if (istrip + totalStrip + 1 < 39)
          TVirtualMC::GetMC()->Gspos(
            "FSTR", istrip + totalStrip + 1, "FLTB", xpos, ypos,
            -zpos + (zlenA * 0.5 - 2. * Geo::MODULEWALLTHICKNESS + Geo::INTERCENTRMODBORDER1) * 0.5,
            idrotm[istrip + totalStrip], "ONLY");
      }
    }
  }
}

void Detector::createModuleCovers(Float_t xtof, Float_t zlenA) const
{
  //
  // Create covers for module:
  //   per each module zone, defined according to
  //   fgkInterCentrModBorder2, fgkExterInterModBorder1 and zlenA+2 values,
  //   there is a frame of thickness 2cm in Al
  //   and the contained zones in honeycomb of Al.
  //   There is also an interface layer (1.6mm thichness)
  //   and plastic and Cu corresponding to the flat cables.
  //

  Float_t par[3];
  par[0] = xtof * 0.5 + 2.;
  par[1] = Geo::MODULECOVERTHICKNESS * 0.5;
  par[2] = zlenA * 0.5 + 2.;
  TVirtualMC::GetMC()->Gsvolu("FPEA", "BOX ", getMediumID(kAir), par, 3); // Air
  if (mTOFHoles)
    TVirtualMC::GetMC()->Gsvolu("FPEB", "BOX ", getMediumID(kAir), par, 3); // Air

  constexpr Float_t ALCOVERTHICKNESS = 1.5;
  constexpr Float_t INTERFACECARDTHICKNESS = 0.16;
  constexpr Float_t ALSKINTHICKNESS = 0.1;
  constexpr Float_t PLASTICFLATCABLETHICKNESS = 0.25;
  constexpr Float_t COPPERFLATCABLETHICKNESS = 0.01;

  // par[0] = xtof*0.5 + 2.;
  par[1] = ALCOVERTHICKNESS * 0.5;
  // par[2] = zlenA*0.5 + 2.;
  TVirtualMC::GetMC()->Gsvolu("FALT", "BOX ", getMediumID(kAlFrame), par, 3); // Al
  if (mTOFHoles)
    TVirtualMC::GetMC()->Gsvolu("FALB", "BOX ", getMediumID(kAlFrame), par, 3); // Al
  Float_t xcoor, ycoor, zcoor;
  xcoor = 0.;
  ycoor = 0.;
  zcoor = 0.;
  TVirtualMC::GetMC()->Gspos("FALT", 0, "FPEA", xcoor, ycoor, zcoor, 0, "ONLY");
  if (mTOFHoles)
    TVirtualMC::GetMC()->Gspos("FALB", 0, "FPEB", xcoor, ycoor, zcoor, 0, "ONLY");

  par[0] = xtof * 0.5;
  // par[1] = ALCOVERTHICKNESS*0.5;
  par[2] = Geo::INTERCENTRMODBORDER2 - 2.;
  TVirtualMC::GetMC()->Gsvolu("FPE1", "BOX ", getMediumID(kHoneycomb), par, 3); // Al honeycomb
  // xcoor = 0.;
  // ycoor = 0.;
  // zcoor = 0.;
  TVirtualMC::GetMC()->Gspos("FPE1", 0, "FALT", xcoor, ycoor, zcoor, 0, "ONLY");

  if (mTOFHoles) {
    // par[0] = xtof*0.5;
    par[1] = ALCOVERTHICKNESS * 0.5 - ALSKINTHICKNESS;
    // par[2] = Geo::INTERCENTRMODBORDER2 - 2.;
    TVirtualMC::GetMC()->Gsvolu("FPE4", "BOX ", getMediumID(kHoneyHoles), par, 3); // Al honeycomb for holes
    // xcoor = 0.;
    // ycoor = 0.;
    // zcoor = 0.;
    TVirtualMC::GetMC()->Gspos("FPE4", 0, "FALB", xcoor, ycoor, zcoor, 0, "ONLY");
  }

  // par[0] = xtof*0.5;
  // par[1] = ALCOVERTHICKNESS*0.5;
  par[2] = (Geo::EXTERINTERMODBORDER1 - Geo::INTERCENTRMODBORDER2) * 0.5 - 2.;
  TVirtualMC::GetMC()->Gsvolu("FPE2", "BOX ", getMediumID(kHoneycomb), par, 3); // Al honeycomb
  // xcoor = 0.;
  // ycoor = 0.;
  zcoor = (Geo::EXTERINTERMODBORDER1 + Geo::INTERCENTRMODBORDER2) * 0.5;
  TVirtualMC::GetMC()->Gspos("FPE2", 1, "FALT", xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FPE2", 2, "FALT", xcoor, ycoor, -zcoor, 0, "ONLY");

  if (mTOFHoles) {
    // xcoor = 0.;
    // ycoor = 0.;
    // zcoor = (Geo::EXTERINTERMODBORDER1 + Geo::INTERCENTRMODBORDER2)*0.5;
    TVirtualMC::GetMC()->Gspos("FPE2", 1, "FALB", xcoor, ycoor, zcoor, 0, "ONLY");
    TVirtualMC::GetMC()->Gspos("FPE2", 2, "FALB", xcoor, ycoor, -zcoor, 0, "ONLY");
  }

  // par[0] = xtof*0.5;
  // par[1] = ALCOVERTHICKNESS*0.5;
  par[2] = (zlenA * 0.5 + 2. - Geo::EXTERINTERMODBORDER1) * 0.5 - 2.;
  TVirtualMC::GetMC()->Gsvolu("FPE3", "BOX ", getMediumID(kHoneycomb), par, 3); // Al honeycomb
  // xcoor = 0.;
  // ycoor = 0.;
  zcoor = (zlenA * 0.5 + 2. + Geo::EXTERINTERMODBORDER1) * 0.5;
  TVirtualMC::GetMC()->Gspos("FPE3", 1, "FALT", xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FPE3", 2, "FALT", xcoor, ycoor, -zcoor, 0, "ONLY");

  if (mTOFHoles) {
    // xcoor = 0.;
    // ycoor = 0.;
    zcoor = (zlenA * 0.5 + 2. + Geo::EXTERINTERMODBORDER1) * 0.5;
    TVirtualMC::GetMC()->Gspos("FPE3", 1, "FALB", xcoor, ycoor, zcoor, 0, "ONLY");
    TVirtualMC::GetMC()->Gspos("FPE3", 2, "FALB", xcoor, ycoor, -zcoor, 0, "ONLY");
  }

  // volumes for Interface cards
  par[0] = xtof * 0.5;
  par[1] = INTERFACECARDTHICKNESS * 0.5;
  par[2] = Geo::INTERCENTRMODBORDER2 - 2.;
  TVirtualMC::GetMC()->Gsvolu("FIF1", "BOX ", getMediumID(kG10), par, 3); // G10
  // xcoor = 0.;
  ycoor = ALCOVERTHICKNESS * 0.5 + INTERFACECARDTHICKNESS * 0.5;
  zcoor = 0.;
  TVirtualMC::GetMC()->Gspos("FIF1", 0, "FPEA", xcoor, ycoor, zcoor, 0, "ONLY");

  // par[0] = xtof*0.5;
  // par[1] = INTERFACECARDTHICKNESS*0.5;
  par[2] = (Geo::EXTERINTERMODBORDER1 - Geo::INTERCENTRMODBORDER2) * 0.5 - 2.;
  TVirtualMC::GetMC()->Gsvolu("FIF2", "BOX ", getMediumID(kG10), par, 3); // G10
  // xcoor = 0.;
  // ycoor = ALCOVERTHICKNESS*0.5 + INTERFACECARDTHICKNESS*0.5;
  zcoor = (Geo::EXTERINTERMODBORDER1 + Geo::INTERCENTRMODBORDER2) * 0.5;
  TVirtualMC::GetMC()->Gspos("FIF2", 1, "FPEA", xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FIF2", 2, "FPEA", xcoor, ycoor, -zcoor, 0, "ONLY");
  if (mTOFHoles) {
    TVirtualMC::GetMC()->Gspos("FIF2", 1, "FPEB", xcoor, ycoor, zcoor, 0, "ONLY");
    TVirtualMC::GetMC()->Gspos("FIF2", 2, "FPEB", xcoor, ycoor, -zcoor, 0, "ONLY");
  }

  // par[0] = xtof*0.5;
  // par[1] = INTERFACECARDTHICKNESS*0.5;
  par[2] = (zlenA * 0.5 + 2. - Geo::EXTERINTERMODBORDER1) * 0.5 - 2.;
  TVirtualMC::GetMC()->Gsvolu("FIF3", "BOX ", getMediumID(kG10), par, 3); // G10
  // xcoor = 0.;
  // ycoor = ALCOVERTHICKNESS*0.5 + INTERFACECARDTHICKNESS*0.5;
  zcoor = (zlenA * 0.5 + 2. + Geo::EXTERINTERMODBORDER1) * 0.5;
  TVirtualMC::GetMC()->Gspos("FIF3", 1, "FPEA", xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FIF3", 2, "FPEA", xcoor, ycoor, -zcoor, 0, "ONLY");
  if (mTOFHoles) {
    TVirtualMC::GetMC()->Gspos("FIF3", 1, "FPEB", xcoor, ycoor, zcoor, 0, "ONLY");
    TVirtualMC::GetMC()->Gspos("FIF3", 2, "FPEB", xcoor, ycoor, -zcoor, 0, "ONLY");
  }

  // volumes for flat cables
  // plastic
  par[0] = xtof * 0.5;
  par[1] = PLASTICFLATCABLETHICKNESS * 0.5;
  par[2] = Geo::INTERCENTRMODBORDER2 - 2.;
  TVirtualMC::GetMC()->Gsvolu("FFC1", "BOX ", getMediumID(kPlastic), par, 3); // Plastic (CH2)
  // xcoor = 0.;
  ycoor = -ALCOVERTHICKNESS * 0.5 - PLASTICFLATCABLETHICKNESS * 0.5;
  zcoor = 0.;
  TVirtualMC::GetMC()->Gspos("FFC1", 0, "FPEA", xcoor, ycoor, zcoor, 0, "ONLY");

  // par[0] = xtof*0.5;
  // par[1] = PLASTICFLATCABLETHICKNESS*0.5;
  par[2] = (Geo::EXTERINTERMODBORDER1 - Geo::INTERCENTRMODBORDER2) * 0.5 - 2.;
  TVirtualMC::GetMC()->Gsvolu("FFC2", "BOX ", getMediumID(kPlastic), par, 3); // Plastic (CH2)
  // xcoor = 0.;
  // ycoor = -ALCOVERTHICKNESS*0.5 - PLASTICFLATCABLETHICKNESS*0.5;
  zcoor = (Geo::EXTERINTERMODBORDER1 + Geo::INTERCENTRMODBORDER2) * 0.5;
  TVirtualMC::GetMC()->Gspos("FFC2", 1, "FPEA", xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FFC2", 2, "FPEA", xcoor, ycoor, -zcoor, 0, "ONLY");
  if (mTOFHoles) {
    TVirtualMC::GetMC()->Gspos("FFC2", 1, "FPEB", xcoor, ycoor, zcoor, 0, "ONLY");
    TVirtualMC::GetMC()->Gspos("FFC2", 2, "FPEB", xcoor, ycoor, -zcoor, 0, "ONLY");
  }

  // par[0] = xtof*0.5;
  // par[1] = PLASTICFLATCABLETHICKNESS*0.5;
  par[2] = (zlenA * 0.5 + 2. - Geo::EXTERINTERMODBORDER1) * 0.5 - 2.;
  TVirtualMC::GetMC()->Gsvolu("FFC3", "BOX ", getMediumID(kPlastic), par, 3); // Plastic (CH2)
  // xcoor = 0.;
  // ycoor = -ALCOVERTHICKNESS*0.5 - PLASTICFLATCABLETHICKNESS*0.5;
  zcoor = (zlenA * 0.5 + 2. + Geo::EXTERINTERMODBORDER1) * 0.5;
  TVirtualMC::GetMC()->Gspos("FFC3", 1, "FPEA", xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FFC3", 2, "FPEA", xcoor, ycoor, -zcoor, 0, "ONLY");
  if (mTOFHoles) {
    TVirtualMC::GetMC()->Gspos("FFC3", 1, "FPEB", xcoor, ycoor, zcoor, 0, "ONLY");
    TVirtualMC::GetMC()->Gspos("FFC3", 2, "FPEB", xcoor, ycoor, -zcoor, 0, "ONLY");
  }

  // Cu
  par[0] = xtof * 0.5;
  par[1] = COPPERFLATCABLETHICKNESS * 0.5;
  par[2] = Geo::INTERCENTRMODBORDER2 - 2.;
  TVirtualMC::GetMC()->Gsvolu("FCC1", "BOX ", getMediumID(kCopper), par, 3); // Cu
  TVirtualMC::GetMC()->Gspos("FCC1", 0, "FFC1", 0., 0., 0., 0, "ONLY");

  // par[0] = xtof*0.5;
  // par[1] = COPPERFLATCABLETHICKNESS*0.5;
  par[2] = (Geo::EXTERINTERMODBORDER1 - Geo::INTERCENTRMODBORDER2) * 0.5 - 2.;
  TVirtualMC::GetMC()->Gsvolu("FCC2", "BOX ", getMediumID(kCopper), par, 3); // Cu
  TVirtualMC::GetMC()->Gspos("FCC2", 0, "FFC2", 0., 0., 0., 0, "ONLY");

  // par[0] = xtof*0.5;
  // par[1] = COPPERFLATCABLETHICKNESS*0.5;
  par[2] = (zlenA * 0.5 + 2. - Geo::EXTERINTERMODBORDER1) * 0.5 - 2.;
  TVirtualMC::GetMC()->Gsvolu("FCC3", "BOX ", getMediumID(kCopper), par, 3); // Cu
  TVirtualMC::GetMC()->Gspos("FCC3", 0, "FFC3", 0., 0., 0., 0, "ONLY");
}

void Detector::createBackZone(Float_t xtof, Float_t ytof, Float_t zlenA) const
{
  //
  // Define:
  //        - containers for FEA cards, cooling system
  //          signal cables and supermodule support structure
  //          (volumes called FAIA/B/C),
  //        - containers for FEA cards and some cooling
  //          elements for a FEA (volumes called FCA1/2).
  //

  Int_t idrotm[1] = { 0 };

  // Definition of the air card containers (FAIA, FAIC and FAIB)

  Float_t par[3];
  par[0] = xtof * 0.5;
  par[1] = (ytof * 0.5 - Geo::MODULECOVERTHICKNESS) * 0.5;
  par[2] = zlenA * 0.5;
  TVirtualMC::GetMC()->Gsvolu("FAIA", "BOX ", getMediumID(kAir), par, 3); // Air
  if (mTOFHoles)
    TVirtualMC::GetMC()->Gsvolu("FAIB", "BOX ", getMediumID(kAir), par, 3); // Air
  TVirtualMC::GetMC()->Gsvolu("FAIC", "BOX ", getMediumID(kAir), par, 3);   // Air

  Float_t feaParam[3] = { Geo::FEAPARAMETERS[0], Geo::FEAPARAMETERS[1], Geo::FEAPARAMETERS[2] };
  Float_t feaRoof1[3] = { Geo::ROOF1PARAMETERS[0], Geo::ROOF1PARAMETERS[1], Geo::ROOF1PARAMETERS[2] };
  Float_t al3[3] = { Geo::AL3PARAMETERS[0], Geo::AL3PARAMETERS[1], Geo::AL3PARAMETERS[2] };
  // Float_t feaRoof2[3] = {Geo::ROOF2PARAMETERS[0], Geo::ROOF2PARAMETERS[1], Geo::ROOF2PARAMETERS[2]};

  // FEA card mother-volume definition
  Float_t carpar[3] = { static_cast<Float_t>(xtof * 0.5 - Geo::CBLW - Geo::SAWTHICKNESS),
                        static_cast<Float_t>(feaParam[1] + feaRoof1[1] + Geo::ROOF2PARAMETERS[1] * 0.5),
                        static_cast<Float_t>(feaRoof1[2] + Geo::BETWEENLANDMASK * 0.5 + al3[2]) };
  TVirtualMC::GetMC()->Gsvolu("FCA1", "BOX ", getMediumID(kAir), carpar, 3); // Air
  TVirtualMC::GetMC()->Gsvolu("FCA2", "BOX ", getMediumID(kAir), carpar, 3); // Air

  // rotation matrix
  Matrix(idrotm[0], 90., 180., 90., 90., 180., 0.);

  // FEA card mother-volume positioning
  Float_t rowstep = 6.66;
  Float_t rowgap[5] = { 13.5, 22.9, 16.94, 23.8, 20.4 };
  Int_t rowb[5] = { 6, 7, 6, 19, 7 };
  Float_t carpos[3] = { 0., static_cast<Float_t>(-(ytof * 0.5 - Geo::MODULECOVERTHICKNESS) * 0.5 + carpar[1]), -0.8 };
  TVirtualMC::GetMC()->Gspos("FCA1", 91, "FAIA", carpos[0], carpos[1], carpos[2], 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FCA2", 91, "FAIC", carpos[0], carpos[1], carpos[2], 0, "MANY");

  Int_t row = 1;
  Int_t nrow = 0;
  for (Int_t sg = -1; sg < 2; sg += 2) {
    carpos[2] = sg * zlenA * 0.5 - 0.8;
    for (Int_t nb = 0; nb < 5; ++nb) {
      carpos[2] = carpos[2] - sg * (rowgap[nb] - rowstep);
      nrow = row + rowb[nb];
      for (; row < nrow; ++row) {
        carpos[2] -= sg * rowstep;

        if (nb == 4) {
          TVirtualMC::GetMC()->Gspos("FCA1", row, "FAIA", carpos[0], carpos[1], carpos[2], 0, "ONLY");
          TVirtualMC::GetMC()->Gspos("FCA2", row, "FAIC", carpos[0], carpos[1], carpos[2], 0, "ONLY");

        } else {
          switch (sg) {
            case 1:
              TVirtualMC::GetMC()->Gspos("FCA1", row, "FAIA", carpos[0], carpos[1], carpos[2], 0, "ONLY");
              TVirtualMC::GetMC()->Gspos("FCA2", row, "FAIC", carpos[0], carpos[1], carpos[2], 0, "ONLY");
              break;
            case -1:
              TVirtualMC::GetMC()->Gspos("FCA1", row, "FAIA", carpos[0], carpos[1], carpos[2], idrotm[0], "ONLY");
              TVirtualMC::GetMC()->Gspos("FCA2", row, "FAIC", carpos[0], carpos[1], carpos[2], idrotm[0], "ONLY");
              break;
          }
        }
      }
    }
  }

  if (mTOFHoles) {
    row = 1;
    for (Int_t sg = -1; sg < 2; sg += 2) {
      carpos[2] = sg * zlenA * 0.5 - 0.8;
      for (Int_t nb = 0; nb < 4; ++nb) {
        carpos[2] = carpos[2] - sg * (rowgap[nb] - rowstep);
        nrow = row + rowb[nb];
        for (; row < nrow; ++row) {
          carpos[2] -= sg * rowstep;

          switch (sg) {
            case 1:
              TVirtualMC::GetMC()->Gspos("FCA1", row, "FAIB", carpos[0], carpos[1], carpos[2], 0, "ONLY");
              break;
            case -1:
              TVirtualMC::GetMC()->Gspos("FCA1", row, "FAIB", carpos[0], carpos[1], carpos[2], idrotm[0], "ONLY");
              break;
          }
        }
      }
    }
  }
}

void Detector::makeFrontEndElectronics(Float_t xtof) const
{
  //
  // Fill FCA1/2 volumes with FEA cards (FFEA volumes).
  //

  // FEA card volume definition
  Float_t feaParam[3] = { Geo::FEAPARAMETERS[0], Geo::FEAPARAMETERS[1], Geo::FEAPARAMETERS[2] };
  TVirtualMC::GetMC()->Gsvolu("FFEA", "BOX ", getMediumID(kG10), feaParam, 3); // G10

  Float_t al1[3] = { Geo::AL1PARAMETERS[0], Geo::AL1PARAMETERS[1], Geo::AL1PARAMETERS[2] };
  Float_t al3[3] = { Geo::AL3PARAMETERS[0], Geo::AL3PARAMETERS[1], Geo::AL3PARAMETERS[2] };
  Float_t feaRoof1[3] = { Geo::ROOF1PARAMETERS[0], Geo::ROOF1PARAMETERS[1], Geo::ROOF1PARAMETERS[2] };
  // Float_t feaRoof2[3] = {Geo::ROOF2PARAMETERS[0], Geo::ROOF2PARAMETERS[1], Geo::ROOF2PARAMETERS[2]};

  Float_t carpar[3] = { static_cast<Float_t>(xtof * 0.5 - Geo::CBLW - Geo::SAWTHICKNESS),
                        static_cast<Float_t>(feaParam[1] + feaRoof1[1] + Geo::ROOF2PARAMETERS[1] * 0.5),
                        static_cast<Float_t>(feaRoof1[2] + Geo::BETWEENLANDMASK * 0.5 + al3[2]) };

  // FEA card volume positioning
  Float_t xCoor = xtof * 0.5 - 25.;
  Float_t yCoor = -carpar[1] + feaParam[1];
  Float_t zCoor = -carpar[2] + (2. * feaRoof1[2] - 2. * al1[2] - feaParam[2]);
  TVirtualMC::GetMC()->Gspos("FFEA", 1, "FCA1", -xCoor, yCoor, zCoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FFEA", 4, "FCA1", xCoor, yCoor, zCoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FFEA", 1, "FCA2", -xCoor, yCoor, zCoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FFEA", 4, "FCA2", xCoor, yCoor, zCoor, 0, "ONLY");
  xCoor = feaParam[0] + (Geo::FEAWIDTH2 * 0.5 - Geo::FEAWIDTH1);
  TVirtualMC::GetMC()->Gspos("FFEA", 2, "FCA1", -xCoor, yCoor, zCoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FFEA", 3, "FCA1", xCoor, yCoor, zCoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FFEA", 2, "FCA2", -xCoor, yCoor, zCoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FFEA", 3, "FCA2", xCoor, yCoor, zCoor, 0, "ONLY");
}

void Detector::makeFEACooling(Float_t xtof) const
{
  //
  // Make cooling system attached to each FEA card
  // (FAL1, FRO1 and FBAR/1/2 volumes)
  // in FCA1/2 volume containers.
  //

  // first FEA cooling element definition
  Float_t al1[3] = { Geo::AL1PARAMETERS[0], Geo::AL1PARAMETERS[1], Geo::AL1PARAMETERS[2] };
  TVirtualMC::GetMC()->Gsvolu("FAL1", "BOX ", getMediumID(kAlFrame), al1, 3); // Al

  // second FEA cooling element definition
  Float_t feaRoof1[3] = { Geo::ROOF1PARAMETERS[0], Geo::ROOF1PARAMETERS[1], Geo::ROOF1PARAMETERS[2] };
  TVirtualMC::GetMC()->Gsvolu("FRO1", "BOX ", getMediumID(kAlFrame), feaRoof1, 3); // Al

  Float_t al3[3] = { Geo::AL3PARAMETERS[0], Geo::AL3PARAMETERS[1], Geo::AL3PARAMETERS[2] };
  // Float_t feaRoof2[3] = {Geo::ROOF2PARAMETERS[0], Geo::ROOF2PARAMETERS[1], Geo::ROOF2PARAMETERS[2]};

  // definition and positioning of a small air groove in the FRO1 volume
  Float_t airHole[3] = { Geo::ROOF2PARAMETERS[0], static_cast<Float_t>(Geo::ROOF2PARAMETERS[1] * 0.5), feaRoof1[2] };
  TVirtualMC::GetMC()->Gsvolu("FREE", "BOX ", getMediumID(kAir), airHole, 3); // Air
  TVirtualMC::GetMC()->Gspos("FREE", 1, "FRO1", 0., feaRoof1[1] - airHole[1], 0., 0, "ONLY");
  gGeoManager->GetVolume("FRO1")->VisibleDaughters(kFALSE);

  // third FEA cooling element definition
  Float_t bar[3] = { Geo::BAR[0], Geo::BAR[1], Geo::BAR[2] };
  TVirtualMC::GetMC()->Gsvolu("FBAR", "BOX ", getMediumID(kAlFrame), bar, 3); // Al

  Float_t feaParam[3] = { Geo::FEAPARAMETERS[0], Geo::FEAPARAMETERS[1], Geo::FEAPARAMETERS[2] };

  Float_t carpar[3] = { static_cast<Float_t>(xtof * 0.5 - Geo::CBLW - Geo::SAWTHICKNESS),
                        static_cast<Float_t>(feaParam[1] + feaRoof1[1] + Geo::ROOF2PARAMETERS[1] * 0.5),
                        static_cast<Float_t>(feaRoof1[2] + Geo::BETWEENLANDMASK * 0.5 + al3[2]) };

  // fourth FEA cooling element definition
  Float_t bar1[3] = { Geo::BAR1[0], Geo::BAR1[1], Geo::BAR1[2] };
  TVirtualMC::GetMC()->Gsvolu("FBA1", "BOX ", getMediumID(kAlFrame), bar1, 3); // Al

  // fifth FEA cooling element definition
  Float_t bar2[3] = { Geo::BAR2[0], Geo::BAR2[1], Geo::BAR2[2] };
  TVirtualMC::GetMC()->Gsvolu("FBA2", "BOX ", getMediumID(kAlFrame), bar2, 3); // Al

  // first FEA cooling element positioning
  Float_t xcoor = xtof * 0.5 - 25.;
  Float_t ycoor = carpar[1] - 2. * Geo::ROOF2PARAMETERS[1] * 0.5 - 2. * feaRoof1[1] - al1[1];
  Float_t zcoor = -carpar[2] + 2. * feaRoof1[2] - al1[2];
  TVirtualMC::GetMC()->Gspos("FAL1", 1, "FCA1", -xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FAL1", 4, "FCA1", xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FAL1", 1, "FCA2", -xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FAL1", 4, "FCA2", xcoor, ycoor, zcoor, 0, "ONLY");
  xcoor = feaParam[0] + (Geo::FEAWIDTH2 * 0.5 - Geo::FEAWIDTH1);
  TVirtualMC::GetMC()->Gspos("FAL1", 2, "FCA1", -xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FAL1", 3, "FCA1", xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FAL1", 2, "FCA2", -xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FAL1", 3, "FCA2", xcoor, ycoor, zcoor, 0, "ONLY");

  // second FEA cooling element positioning
  xcoor = xtof * 0.5 - 25.;
  ycoor = carpar[1] - 2. * Geo::ROOF2PARAMETERS[1] * 0.5 - feaRoof1[1];
  zcoor = -carpar[2] + feaRoof1[2];
  TVirtualMC::GetMC()->Gspos("FRO1", 1, "FCA1", -xcoor, ycoor, zcoor, 0, "MANY"); // (AdC)
  TVirtualMC::GetMC()->Gspos("FRO1", 4, "FCA1", xcoor, ycoor, zcoor, 0, "MANY");  // (AdC)
  TVirtualMC::GetMC()->Gspos("FRO1", 1, "FCA2", -xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FRO1", 4, "FCA2", xcoor, ycoor, zcoor, 0, "ONLY");
  xcoor = feaParam[0] + (Geo::FEAWIDTH2 * 0.5 - Geo::FEAWIDTH1);
  TVirtualMC::GetMC()->Gspos("FRO1", 2, "FCA1", -xcoor, ycoor, zcoor, 0, "MANY"); // (AdC)
  TVirtualMC::GetMC()->Gspos("FRO1", 3, "FCA1", xcoor, ycoor, zcoor, 0, "MANY");  // (AdC)
  TVirtualMC::GetMC()->Gspos("FRO1", 2, "FCA2", -xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FRO1", 3, "FCA2", xcoor, ycoor, zcoor, 0, "ONLY");

  // third FEA cooling element positioning
  xcoor = xtof * 0.5 - 25.;
  ycoor = carpar[1] - 2. * Geo::ROOF2PARAMETERS[1] * 0.5 - 2. * feaRoof1[1] - bar[1];
  zcoor = -carpar[2] + bar[2];
  TVirtualMC::GetMC()->Gspos("FBAR", 1, "FCA1", -xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBAR", 4, "FCA1", xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBAR", 1, "FCA2", -xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBAR", 4, "FCA2", xcoor, ycoor, zcoor, 0, "ONLY");
  xcoor = feaParam[0] + (Geo::FEAWIDTH2 * 0.5 - Geo::FEAWIDTH1);
  TVirtualMC::GetMC()->Gspos("FBAR", 2, "FCA1", -xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBAR", 3, "FCA1", xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBAR", 2, "FCA2", -xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBAR", 3, "FCA2", xcoor, ycoor, zcoor, 0, "ONLY");

  // fourth FEA cooling element positioning
  Float_t tubepar[3] = { 0., 0.4, static_cast<Float_t>(xtof * 0.5 - Geo::CBLW) };
  xcoor = xtof * 0.5 - 25.;
  ycoor = carpar[1] - 2. * Geo::ROOF2PARAMETERS[1] * 0.5 - 2. * feaRoof1[1] - bar[1];
  zcoor = -carpar[2] + 2. * bar[2] + 2. * tubepar[1] + bar1[2];
  TVirtualMC::GetMC()->Gspos("FBA1", 1, "FCA1", -xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBA1", 4, "FCA1", xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBA1", 1, "FCA2", -xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBA1", 4, "FCA2", xcoor, ycoor, zcoor, 0, "ONLY");
  xcoor = feaParam[0] + (Geo::FEAWIDTH2 * 0.5 - Geo::FEAWIDTH1);
  TVirtualMC::GetMC()->Gspos("FBA1", 2, "FCA1", -xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBA1", 3, "FCA1", xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBA1", 2, "FCA2", -xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBA1", 3, "FCA2", xcoor, ycoor, zcoor, 0, "ONLY");

  // fifth FEA cooling element positioning
  xcoor = xtof * 0.5 - 25.;
  ycoor = carpar[1] - 2. * Geo::ROOF2PARAMETERS[1] * 0.5 - 2. * feaRoof1[1] - bar2[1];
  zcoor = -carpar[2] + 2. * bar[2] + bar2[2];
  TVirtualMC::GetMC()->Gspos("FBA2", 1, "FCA1", -xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBA2", 4, "FCA1", xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBA2", 1, "FCA2", -xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBA2", 4, "FCA2", xcoor, ycoor, zcoor, 0, "ONLY");
  xcoor = feaParam[0] + (Geo::FEAWIDTH2 * 0.5 - Geo::FEAWIDTH1);
  TVirtualMC::GetMC()->Gspos("FBA2", 2, "FCA1", -xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBA2", 3, "FCA1", xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBA2", 2, "FCA2", -xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBA2", 3, "FCA2", xcoor, ycoor, zcoor, 0, "ONLY");

  xcoor = xtof * 0.5 - 25.;
  ycoor = carpar[1] - 2. * Geo::ROOF2PARAMETERS[1] * 0.5 - 2. * feaRoof1[1] - 2. * bar2[1] - 2. * tubepar[1] - bar2[1];
  zcoor = -carpar[2] + 2. * bar[2] + bar2[2];
  TVirtualMC::GetMC()->Gspos("FBA2", 5, "FCA1", -xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBA2", 8, "FCA1", xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBA2", 5, "FCA2", -xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBA2", 8, "FCA2", xcoor, ycoor, zcoor, 0, "ONLY");
  xcoor = feaParam[0] + (Geo::FEAWIDTH2 * 0.5 - Geo::FEAWIDTH1);
  TVirtualMC::GetMC()->Gspos("FBA2", 6, "FCA1", -xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBA2", 7, "FCA1", xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBA2", 6, "FCA2", -xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBA2", 7, "FCA2", xcoor, ycoor, zcoor, 0, "ONLY");
}

void Detector::makeNinoMask(Float_t xtof) const
{
  //
  // Make cooling Nino mask
  // for each FEA card (FAL2/3 and FRO2 volumes)
  // in FCA1 volume container.
  //

  // first Nino ASIC mask volume definition
  Float_t al2[3] = { Geo::AL2PARAMETERS[0], Geo::AL2PARAMETERS[1], Geo::AL2PARAMETERS[2] };
  TVirtualMC::GetMC()->Gsvolu("FAL2", "BOX ", getMediumID(kAlFrame), al2, 3); // Al

  // second Nino ASIC mask volume definition
  Float_t al3[3] = { Geo::AL3PARAMETERS[0], Geo::AL3PARAMETERS[1], Geo::AL3PARAMETERS[2] };
  TVirtualMC::GetMC()->Gsvolu("FAL3", "BOX ", getMediumID(kAlFrame), al3, 3); // Al

  // third Nino ASIC mask volume definition
  Float_t feaRoof2[3] = { Geo::ROOF2PARAMETERS[0], Geo::ROOF2PARAMETERS[1], Geo::ROOF2PARAMETERS[2] };
  TVirtualMC::GetMC()->Gsvolu("FRO2", "BOX ", getMediumID(kAlFrame), feaRoof2, 3); // Al

  Float_t feaRoof1[3] = { Geo::ROOF1PARAMETERS[0], Geo::ROOF1PARAMETERS[1], Geo::ROOF1PARAMETERS[2] };
  Float_t feaParam[3] = { Geo::FEAPARAMETERS[0], Geo::FEAPARAMETERS[1], Geo::FEAPARAMETERS[2] };

  Float_t carpar[3] = { static_cast<Float_t>(xtof * 0.5 - Geo::CBLW - Geo::SAWTHICKNESS),
                        static_cast<Float_t>(feaParam[1] + feaRoof1[1] + Geo::ROOF2PARAMETERS[1] * 0.5),
                        static_cast<Float_t>(feaRoof1[2] + Geo::BETWEENLANDMASK * 0.5 + al3[2]) };

  // first Nino ASIC mask volume positioning
  Float_t xcoor = xtof * 0.5 - 25.;
  Float_t ycoor = carpar[1] - 2. * al3[1];
  Float_t zcoor = carpar[2] - 2. * al3[2] - al2[2];
  TVirtualMC::GetMC()->Gspos("FAL2", 1, "FCA1", -xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FAL2", 4, "FCA1", xcoor, ycoor, zcoor, 0, "ONLY");
  xcoor = feaParam[0] + (Geo::FEAWIDTH2 * 0.5 - Geo::FEAWIDTH1);
  TVirtualMC::GetMC()->Gspos("FAL2", 2, "FCA1", -xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FAL2", 3, "FCA1", xcoor, ycoor, zcoor, 0, "ONLY");

  // second Nino ASIC mask volume positioning
  xcoor = xtof * 0.5 - 25.;
  ycoor = carpar[1] - al3[1];
  zcoor = carpar[2] - al3[2];
  TVirtualMC::GetMC()->Gspos("FAL3", 1, "FCA1", -xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FAL3", 4, "FCA1", xcoor, ycoor, zcoor, 0, "ONLY");
  xcoor = feaParam[0] + (Geo::FEAWIDTH2 * 0.5 - Geo::FEAWIDTH1);
  TVirtualMC::GetMC()->Gspos("FAL3", 2, "FCA1", -xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FAL3", 3, "FCA1", xcoor, ycoor, zcoor, 0, "ONLY");

  // third Nino ASIC mask volume positioning
  xcoor = xtof * 0.5 - 25.;
  ycoor = carpar[1] - Geo::ROOF2PARAMETERS[1];
  zcoor = carpar[2] - 2. * al3[2] - Geo::ROOF2PARAMETERS[2];
  TVirtualMC::GetMC()->Gspos("FRO2", 1, "FCA1", -xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FRO2", 4, "FCA1", xcoor, ycoor, zcoor, 0, "ONLY");
  xcoor = feaParam[0] + (Geo::FEAWIDTH2 * 0.5 - Geo::FEAWIDTH1);
  TVirtualMC::GetMC()->Gspos("FRO2", 2, "FCA1", -xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FRO2", 3, "FCA1", xcoor, ycoor, zcoor, 0, "ONLY");
}

void Detector::makeSuperModuleCooling(Float_t xtof, Float_t ytof, Float_t zlenA) const
{
  //
  // Make cooling tubes (FTUB volume)
  // and cooling bars (FTLN and FLO1/2/3 volumes)
  // in FAIA/B/C volume containers.
  //

  Int_t idrotm[1] = { 0 };

  // cooling tube volume definition
  Float_t tubepar[3] = { 0., 0.4, static_cast<Float_t>(xtof * 0.5 - Geo::CBLW - Geo::SAWTHICKNESS) };
  TVirtualMC::GetMC()->Gsvolu("FTUB", "TUBE", getMediumID(kCopper), tubepar, 3); // Cu

  // water cooling tube volume definition
  Float_t tubeparW[3] = { 0., 0.3, tubepar[2] };
  TVirtualMC::GetMC()->Gsvolu("FITU", "TUBE", getMediumID(kWater), tubeparW, 3); // H2O

  // Positioning of the water tube into the steel one
  TVirtualMC::GetMC()->Gspos("FITU", 1, "FTUB", 0., 0., 0., 0, "ONLY");

  // definition of transverse components of SM cooling system
  Float_t trapar[3] = { tubepar[2], 6.175 /*6.15*/, 0.7 };
  TVirtualMC::GetMC()->Gsvolu("FTLN", "BOX ", getMediumID(kAlFrame), trapar, 3); // Al

  // rotation matrix
  Matrix(idrotm[0], 180., 90., 90., 90., 90., 0.);

  Float_t feaParam[3] = { Geo::FEAPARAMETERS[0], Geo::FEAPARAMETERS[1], Geo::FEAPARAMETERS[2] };
  Float_t feaRoof1[3] = { Geo::ROOF1PARAMETERS[0], Geo::ROOF1PARAMETERS[1], Geo::ROOF1PARAMETERS[2] };
  Float_t bar[3] = { Geo::BAR[0], Geo::BAR[1], Geo::BAR[2] };
  Float_t bar2[3] = { Geo::BAR2[0], Geo::BAR2[1], Geo::BAR2[2] };
  Float_t al3[3] = { Geo::AL3PARAMETERS[0], Geo::AL3PARAMETERS[1], Geo::AL3PARAMETERS[2] };
  // Float_t feaRoof2[3] = {Geo::ROOF2PARAMETERS[0], Geo::ROOF2PARAMETERS[1], Geo::ROOF2PARAMETERS[2]};

  Float_t carpar[3] = { static_cast<Float_t>(xtof * 0.5 - Geo::CBLW - Geo::SAWTHICKNESS),
                        static_cast<Float_t>(feaParam[1] + feaRoof1[1] + Geo::ROOF2PARAMETERS[1] * 0.5),
                        static_cast<Float_t>(feaRoof1[2] + Geo::BETWEENLANDMASK * 0.5 + al3[2]) };

  Float_t ytub = -(ytof * 0.5 - Geo::MODULECOVERTHICKNESS) * 0.5 + carpar[1] + carpar[1] -
                 2. * Geo::ROOF2PARAMETERS[1] * 0.5 - 2. * feaRoof1[1] - 2. * bar2[1] - tubepar[1];

  // Positioning of tubes for the SM cooling system
  Float_t ycoor = carpar[1] - 2. * Geo::ROOF2PARAMETERS[1] * 0.5 - 2. * feaRoof1[1] - 2. * bar2[1] - tubepar[1];
  Float_t zcoor = -carpar[2] + 2. * bar[2] + tubepar[1];
  TVirtualMC::GetMC()->Gspos("FTUB", 1, "FCA1", 0., ycoor, zcoor, idrotm[0], "ONLY");
  TVirtualMC::GetMC()->Gspos("FTUB", 1, "FCA2", 0., ycoor, zcoor, idrotm[0], "ONLY");
  gGeoManager->GetVolume("FTUB")->VisibleDaughters(kFALSE);

  Float_t yFLTN = trapar[1] - (ytof * 0.5 - Geo::MODULECOVERTHICKNESS) * 0.5;
  for (Int_t sg = -1; sg < 2; sg += 2) {
    // Positioning of transverse components for the SM cooling system
    TVirtualMC::GetMC()->Gspos("FTLN", 5 + 4 * sg, "FAIA", 0., yFLTN, 369.9 * sg, 0, "MANY");
    TVirtualMC::GetMC()->Gspos("FTLN", 5 + 3 * sg, "FAIA", 0., yFLTN, 366.9 * sg, 0, "MANY");
    TVirtualMC::GetMC()->Gspos("FTLN", 5 + 2 * sg, "FAIA", 0., yFLTN, 198.8 * sg, 0, "MANY");
    TVirtualMC::GetMC()->Gspos("FTLN", 5 + sg, "FAIA", 0., yFLTN, 56.82 * sg, 0, "MANY");
    TVirtualMC::GetMC()->Gspos("FTLN", 5 + 4 * sg, "FAIC", 0., yFLTN, 369.9 * sg, 0, "MANY");
    TVirtualMC::GetMC()->Gspos("FTLN", 5 + 3 * sg, "FAIC", 0., yFLTN, 366.9 * sg, 0, "MANY");
    TVirtualMC::GetMC()->Gspos("FTLN", 5 + 2 * sg, "FAIC", 0., yFLTN, 198.8 * sg, 0, "MANY");
    TVirtualMC::GetMC()->Gspos("FTLN", 5 + sg, "FAIC", 0., yFLTN, 56.82 * sg, 0, "MANY");
  }

  // definition of longitudinal components of SM cooling system
  Float_t lonpar1[3] = { 2., 0.5, static_cast<Float_t>(56.82 - trapar[2]) };
  Float_t lonpar2[3] = { lonpar1[0], lonpar1[1], static_cast<Float_t>((198.8 - 56.82) * 0.5 - trapar[2]) };
  Float_t lonpar3[3] = { lonpar1[0], lonpar1[1], static_cast<Float_t>((366.9 - 198.8) * 0.5 - trapar[2]) };
  TVirtualMC::GetMC()->Gsvolu("FLO1", "BOX ", getMediumID(kAlFrame), lonpar1, 3); // Al
  TVirtualMC::GetMC()->Gsvolu("FLO2", "BOX ", getMediumID(kAlFrame), lonpar2, 3); // Al
  TVirtualMC::GetMC()->Gsvolu("FLO3", "BOX ", getMediumID(kAlFrame), lonpar3, 3); // Al

  // Positioning of longitudinal components for the SM cooling system
  ycoor = ytub + (tubepar[1] + 2. * bar2[1] + lonpar1[1]);
  TVirtualMC::GetMC()->Gspos("FLO1", 4, "FAIA", -24., ycoor, 0., 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO1", 2, "FAIA", 24., ycoor, 0., 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO1", 4, "FAIC", -24., ycoor, 0., 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO1", 2, "FAIC", 24., ycoor, 0., 0, "MANY");

  zcoor = (198.8 + 56.82) * 0.5;
  TVirtualMC::GetMC()->Gspos("FLO2", 4, "FAIA", -24., ycoor, -zcoor, 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO2", 2, "FAIA", 24., ycoor, -zcoor, 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO2", 4, "FAIC", -24., ycoor, -zcoor, 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO2", 2, "FAIC", 24., ycoor, -zcoor, 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO2", 8, "FAIA", -24., ycoor, zcoor, 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO2", 6, "FAIA", 24., ycoor, zcoor, 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO2", 8, "FAIC", -24., ycoor, zcoor, 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO2", 6, "FAIC", 24., ycoor, zcoor, 0, "MANY");

  zcoor = (366.9 + 198.8) * 0.5;
  TVirtualMC::GetMC()->Gspos("FLO3", 4, "FAIA", -24., ycoor, -zcoor, 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO3", 2, "FAIA", 24., ycoor, -zcoor, 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO3", 4, "FAIC", -24., ycoor, -zcoor, 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO3", 2, "FAIC", 24., ycoor, -zcoor, 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO3", 8, "FAIA", -24., ycoor, zcoor, 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO3", 6, "FAIA", 24., ycoor, zcoor, 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO3", 8, "FAIC", -24., ycoor, zcoor, 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO3", 6, "FAIC", 24., ycoor, zcoor, 0, "MANY");

  ycoor = ytub - (tubepar[1] + 2. * bar2[1] + lonpar1[1]);
  TVirtualMC::GetMC()->Gspos("FLO1", 3, "FAIA", -24., ycoor, 0., 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO1", 1, "FAIA", 24., ycoor, 0., 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO1", 3, "FAIC", -24., ycoor, 0., 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO1", 1, "FAIC", 24., ycoor, 0., 0, "MANY");

  zcoor = (198.8 + 56.82) * 0.5;
  TVirtualMC::GetMC()->Gspos("FLO2", 3, "FAIA", -24., ycoor, -zcoor, 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO2", 1, "FAIA", 24., ycoor, -zcoor, 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO2", 3, "FAIC", -24., ycoor, -zcoor, 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO2", 1, "FAIC", 24., ycoor, -zcoor, 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO2", 7, "FAIA", -24., ycoor, zcoor, 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO2", 5, "FAIA", 24., ycoor, zcoor, 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO2", 7, "FAIC", -24., ycoor, zcoor, 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO2", 5, "FAIC", 24., ycoor, zcoor, 0, "MANY");

  zcoor = (366.9 + 198.8) * 0.5;
  TVirtualMC::GetMC()->Gspos("FLO3", 3, "FAIA", -24., ycoor, -zcoor, 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO3", 1, "FAIA", 24., ycoor, -zcoor, 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO3", 3, "FAIC", -24., ycoor, -zcoor, 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO3", 1, "FAIC", 24., ycoor, -zcoor, 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO3", 7, "FAIA", -24., ycoor, zcoor, 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO3", 5, "FAIA", 24., ycoor, zcoor, 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO3", 7, "FAIC", -24., ycoor, zcoor, 0, "MANY");
  TVirtualMC::GetMC()->Gspos("FLO3", 5, "FAIC", 24., ycoor, zcoor, 0, "MANY");

  Float_t carpos[3] = { static_cast<Float_t>(25. - xtof * 0.5),
                        static_cast<Float_t>((11.5 - (ytof * 0.5 - Geo::MODULECOVERTHICKNESS)) * 0.5), 0. };
  if (mTOFHoles) {
    for (Int_t sg = -1; sg < 2; sg += 2) {
      carpos[2] = sg * zlenA * 0.5;
      TVirtualMC::GetMC()->Gspos("FTLN", 5 + 4 * sg, "FAIB", 0., yFLTN, 369.9 * sg, 0, "MANY");
      TVirtualMC::GetMC()->Gspos("FTLN", 5 + 3 * sg, "FAIB", 0., yFLTN, 366.9 * sg, 0, "MANY");
      TVirtualMC::GetMC()->Gspos("FTLN", 5 + 2 * sg, "FAIB", 0., yFLTN, 198.8 * sg, 0, "MANY");
      TVirtualMC::GetMC()->Gspos("FTLN", 5 + sg, "FAIB", 0., yFLTN, 56.82 * sg, 0, "MANY");
    }

    ycoor = ytub + (tubepar[1] + 2. * bar2[1] + lonpar1[1]);
    zcoor = (198.8 + 56.82) * 0.5;
    TVirtualMC::GetMC()->Gspos("FLO2", 2, "FAIB", -24., ycoor, -zcoor, 0, "MANY");
    TVirtualMC::GetMC()->Gspos("FLO2", 1, "FAIB", -24., ycoor, zcoor, 0, "MANY");
    zcoor = (366.9 + 198.8) * 0.5;
    TVirtualMC::GetMC()->Gspos("FLO3", 2, "FAIB", -24., ycoor, -zcoor, 0, "MANY");
    TVirtualMC::GetMC()->Gspos("FLO3", 1, "FAIB", -24., ycoor, zcoor, 0, "MANY");
    ycoor = ytub - (tubepar[1] + 2. * bar2[1] + lonpar1[1]);
    zcoor = (198.8 + 56.82) * 0.5;
    TVirtualMC::GetMC()->Gspos("FLO2", 4, "FAIB", 24., ycoor, -zcoor, 0, "MANY");
    TVirtualMC::GetMC()->Gspos("FLO2", 3, "FAIB", 24., ycoor, zcoor, 0, "MANY");
    zcoor = (366.9 + 198.8) * 0.5;
    TVirtualMC::GetMC()->Gspos("FLO3", 4, "FAIB", 24., ycoor, -zcoor, 0, "MANY");
    TVirtualMC::GetMC()->Gspos("FLO3", 3, "FAIB", 24., ycoor, zcoor, 0, "MANY");
  }

  Float_t barS[3] = { Geo::BARS[0], Geo::BARS[1], Geo::BARS[2] };
  TVirtualMC::GetMC()->Gsvolu("FBAS", "BOX ", getMediumID(kAlFrame), barS, 3); // Al

  Float_t barS1[3] = { Geo::BARS1[0], Geo::BARS1[1], Geo::BARS1[2] };
  TVirtualMC::GetMC()->Gsvolu("FBS1", "BOX ", getMediumID(kAlFrame), barS1, 3); // Al

  Float_t barS2[3] = { Geo::BARS2[0], Geo::BARS2[1], Geo::BARS2[2] };
  TVirtualMC::GetMC()->Gsvolu("FBS2", "BOX ", getMediumID(kAlFrame), barS2, 3); // Al

  Float_t ytubBis = carpar[1] - 2. * Geo::ROOF2PARAMETERS[1] * 0.5 - 2. * feaRoof1[1] - 2. * barS2[1] - tubepar[1];
  ycoor = ytubBis;
  zcoor = -carpar[2] + barS[2];
  TVirtualMC::GetMC()->Gspos("FBAS", 1, "FCA1", -24., ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBAS", 2, "FCA1", 24., ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBAS", 1, "FCA2", -24., ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBAS", 2, "FCA2", 24., ycoor, zcoor, 0, "ONLY");

  zcoor = -carpar[2] + 2. * barS[2] + 2. * tubepar[1] + barS1[2];
  TVirtualMC::GetMC()->Gspos("FBS1", 1, "FCA1", -24., ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBS1", 2, "FCA1", 24., ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBS1", 1, "FCA2", -24., ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBS1", 2, "FCA2", 24., ycoor, zcoor, 0, "ONLY");

  ycoor = ytubBis + (tubepar[1] + barS2[1]);
  zcoor = -carpar[2] + 2. * barS[2] + barS2[2];
  TVirtualMC::GetMC()->Gspos("FBS2", 1, "FCA1", -24., ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBS2", 2, "FCA1", 24., ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBS2", 1, "FCA2", -24., ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBS2", 2, "FCA2", 24., ycoor, zcoor, 0, "ONLY");

  ycoor = ytubBis - (tubepar[1] + barS2[1]);
  // zcoor =-carpar[2] + 2.*barS[2] + barS2[2];
  TVirtualMC::GetMC()->Gspos("FBS2", 3, "FCA1", -24., ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBS2", 4, "FCA1", 24., ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBS2", 3, "FCA2", -24., ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FBS2", 4, "FCA2", 24., ycoor, zcoor, 0, "ONLY");
}

//_____________________________________________________________________________
void Detector::makeSuperModuleServices(Float_t xtof, Float_t ytof, Float_t zlenA) const
{
  //
  // Make signal cables (FCAB/L and FCBL/B volumes),
  // supemodule cover (FCOV volume) and wall (FSAW volume)
  // in FAIA/B/C volume containers.
  //

  Int_t idrotm[3] = { 0, 0, 0 };

  Float_t tubepar[3] = { 0., 0.4, static_cast<Float_t>(xtof * 0.5 - Geo::CBLW - Geo::SAWTHICKNESS) };
  Float_t al1[3] = { Geo::AL1PARAMETERS[0], Geo::AL1PARAMETERS[1], Geo::AL1PARAMETERS[2] };
  Float_t al3[3] = { Geo::AL3PARAMETERS[0], Geo::AL3PARAMETERS[1], Geo::AL3PARAMETERS[2] };
  Float_t feaRoof1[3] = { Geo::ROOF1PARAMETERS[0], Geo::ROOF1PARAMETERS[1], Geo::ROOF1PARAMETERS[2] };
  // Float_t feaRoof2[3] = {Geo::ROOF2PARAMETERS[0], Geo::ROOF2PARAMETERS[1], Geo::ROOF2PARAMETERS[2]};
  Float_t feaParam[3] = { Geo::FEAPARAMETERS[0], Geo::FEAPARAMETERS[1], Geo::FEAPARAMETERS[2] };

  // FEA cables definition
  Float_t cbpar[3] = { 0., 0.5,
                       static_cast<Float_t>((tubepar[2] - (Geo::FEAWIDTH2 - Geo::FEAWIDTH1 / 6.) * 0.5) * 0.5) };
  TVirtualMC::GetMC()->Gsvolu("FCAB", "TUBE", getMediumID(kCable), cbpar, 3); // copper+alu

  Float_t cbparS[3] = { cbpar[0], cbpar[1],
                        static_cast<Float_t>(
                          (tubepar[2] - (xtof * 0.5 - 25. + (Geo::FEAWIDTH1 - Geo::FEAWIDTH1 / 6.) * 0.5)) * 0.5) };
  TVirtualMC::GetMC()->Gsvolu("FCAL", "TUBE", getMediumID(kCable), cbparS, 3); // copper+alu

  // rotation matrix
  Matrix(idrotm[0], 180., 90., 90., 90., 90., 0.);

  Float_t carpar[3] = { static_cast<Float_t>(xtof * 0.5 - Geo::CBLW - Geo::SAWTHICKNESS),
                        static_cast<Float_t>(feaParam[1] + feaRoof1[1] + Geo::ROOF2PARAMETERS[1] * 0.5),
                        static_cast<Float_t>(feaRoof1[2] + Geo::BETWEENLANDMASK * 0.5 + al3[2]) };

  Float_t bar2[3] = { Geo::BAR2[0], Geo::BAR2[1], Geo::BAR2[2] };
  Float_t ytub = -(ytof * 0.5 - Geo::MODULECOVERTHICKNESS) * 0.5 + carpar[1] + carpar[1] -
                 2. * Geo::ROOF2PARAMETERS[1] * 0.5 - 2. * feaRoof1[1] - 2. * bar2[1] - tubepar[1];

  // FEA cables positioning
  Float_t xcoor = (tubepar[2] + (Geo::FEAWIDTH2 - Geo::FEAWIDTH1 / 6.) * 0.5) * 0.5;
  Float_t ycoor = ytub - 3.;
  Float_t zcoor = -carpar[2] + (2. * feaRoof1[2] - 2. * al1[2] - 2. * feaParam[2] - cbpar[1]);
  TVirtualMC::GetMC()->Gspos("FCAB", 1, "FCA1", -xcoor, ycoor, zcoor, idrotm[0], "ONLY");
  TVirtualMC::GetMC()->Gspos("FCAB", 2, "FCA1", xcoor, ycoor, zcoor, idrotm[0], "ONLY");
  TVirtualMC::GetMC()->Gspos("FCAB", 1, "FCA2", -xcoor, ycoor, zcoor, idrotm[0], "ONLY");
  TVirtualMC::GetMC()->Gspos("FCAB", 2, "FCA2", xcoor, ycoor, zcoor, idrotm[0], "ONLY");
  xcoor = (tubepar[2] + (xtof * 0.5 - 25. + (Geo::FEAWIDTH1 - Geo::FEAWIDTH1 / 6.) * 0.5)) * 0.5;
  ycoor -= 2. * cbpar[1];
  TVirtualMC::GetMC()->Gspos("FCAL", 1, "FCA1", -xcoor, ycoor, zcoor, idrotm[0], "ONLY");
  TVirtualMC::GetMC()->Gspos("FCAL", 2, "FCA1", xcoor, ycoor, zcoor, idrotm[0], "ONLY");
  TVirtualMC::GetMC()->Gspos("FCAL", 1, "FCA2", -xcoor, ycoor, zcoor, idrotm[0], "ONLY");
  TVirtualMC::GetMC()->Gspos("FCAL", 2, "FCA2", xcoor, ycoor, zcoor, idrotm[0], "ONLY");

  // Cables and tubes on the side blocks
  // constants definition
  Float_t kCBLl = zlenA * 0.5;                              // length of block
  Float_t kCBLlh = zlenA * 0.5 - Geo::INTERCENTRMODBORDER2; // length  of block in case of holes
  // constexpr Float_t Geo::CBLW   = 13.5;      // width of block
  // constexpr Float_t Geo::CBLH1  = 2.;        // min. height of block
  // constexpr Float_t Geo::CBLH2  = 12.3;      // max. height of block
  // constexpr Float_t Geo::SAWTHICKNESS = 1.; // Al wall thickness

  // lateral cable and tube volume definition
  Float_t tgal = (Geo::CBLH2 - Geo::CBLH1) / (2. * kCBLl);
  Float_t cblpar[11];
  cblpar[0] = Geo::CBLW * 0.5;
  cblpar[1] = 0.;
  cblpar[2] = 0.;
  cblpar[3] = kCBLl * 0.5;
  cblpar[4] = Geo::CBLH1 * 0.5;
  cblpar[5] = Geo::CBLH2 * 0.5;
  cblpar[6] = TMath::ATan(tgal) * TMath::RadToDeg();
  cblpar[7] = kCBLl * 0.5;
  cblpar[8] = Geo::CBLH1 * 0.5;
  cblpar[9] = Geo::CBLH2 * 0.5;
  cblpar[10] = cblpar[6];
  TVirtualMC::GetMC()->Gsvolu("FCBL", "TRAP", getMediumID(kCableTubes), cblpar, 11); // cables and tubes mix

  // Side Al Walls definition
  Float_t sawpar[3] = { static_cast<Float_t>(Geo::SAWTHICKNESS * 0.5), static_cast<Float_t>(Geo::CBLH2 * 0.5), kCBLl };
  TVirtualMC::GetMC()->Gsvolu("FSAW", "BOX ", getMediumID(kAlFrame), sawpar, 3); // Al

  Matrix(idrotm[1], 90., 90., 180., 0., 90., 180.);
  Matrix(idrotm[2], 90., 90., 0., 0., 90., 0.);

  // lateral cable and tube volume positioning
  xcoor = (xtof - Geo::CBLW) * 0.5 - 2. * sawpar[0];
  ycoor = (Geo::CBLH1 + Geo::CBLH2) * 0.25 - (ytof * 0.5 - Geo::MODULECOVERTHICKNESS) * 0.5;
  zcoor = kCBLl * 0.5;
  TVirtualMC::GetMC()->Gspos("FCBL", 1, "FAIA", -xcoor, ycoor, -zcoor, idrotm[1], "ONLY");
  TVirtualMC::GetMC()->Gspos("FCBL", 2, "FAIA", xcoor, ycoor, -zcoor, idrotm[1], "ONLY");
  TVirtualMC::GetMC()->Gspos("FCBL", 3, "FAIA", -xcoor, ycoor, zcoor, idrotm[2], "ONLY");
  TVirtualMC::GetMC()->Gspos("FCBL", 4, "FAIA", xcoor, ycoor, zcoor, idrotm[2], "ONLY");
  TVirtualMC::GetMC()->Gspos("FCBL", 1, "FAIC", -xcoor, ycoor, -zcoor, idrotm[1], "ONLY");
  TVirtualMC::GetMC()->Gspos("FCBL", 2, "FAIC", xcoor, ycoor, -zcoor, idrotm[1], "ONLY");
  TVirtualMC::GetMC()->Gspos("FCBL", 3, "FAIC", -xcoor, ycoor, zcoor, idrotm[2], "ONLY");
  TVirtualMC::GetMC()->Gspos("FCBL", 4, "FAIC", xcoor, ycoor, zcoor, idrotm[2], "ONLY");

  if (mTOFHoles) {
    cblpar[3] = kCBLlh * 0.5;
    cblpar[5] = Geo::CBLH1 * 0.5 + kCBLlh * tgal;
    cblpar[7] = kCBLlh * 0.5;
    cblpar[9] = cblpar[5];
    TVirtualMC::GetMC()->Gsvolu("FCBB", "TRAP", getMediumID(kCableTubes), cblpar, 11); // cables and tubes mix

    xcoor = (xtof - Geo::CBLW) * 0.5 - 2. * sawpar[0];
    ycoor = (Geo::CBLH1 + 2. * cblpar[5]) * 0.25 - (ytof * 0.5 - Geo::MODULECOVERTHICKNESS) * 0.5;
    zcoor = kCBLl - kCBLlh * 0.5;
    TVirtualMC::GetMC()->Gspos("FCBB", 1, "FAIB", -xcoor, ycoor, -zcoor, idrotm[1], "ONLY");
    TVirtualMC::GetMC()->Gspos("FCBB", 2, "FAIB", xcoor, ycoor, -zcoor, idrotm[1], "ONLY");
    TVirtualMC::GetMC()->Gspos("FCBB", 3, "FAIB", -xcoor, ycoor, zcoor, idrotm[2], "ONLY");
    TVirtualMC::GetMC()->Gspos("FCBB", 4, "FAIB", xcoor, ycoor, zcoor, idrotm[2], "ONLY");
  }

  // lateral cable and tube volume positioning
  xcoor = xtof * 0.5 - sawpar[0];
  ycoor = (Geo::CBLH2 - ytof * 0.5 + Geo::MODULECOVERTHICKNESS) * 0.5;
  zcoor = 0.;
  TVirtualMC::GetMC()->Gspos("FSAW", 1, "FAIA", -xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FSAW", 2, "FAIA", xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FSAW", 1, "FAIC", -xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FSAW", 2, "FAIC", xcoor, ycoor, zcoor, 0, "ONLY");

  if (mTOFHoles) {
    xcoor = xtof * 0.5 - sawpar[0];
    ycoor = (Geo::CBLH2 - ytof * 0.5 + Geo::MODULECOVERTHICKNESS) * 0.5;
    TVirtualMC::GetMC()->Gspos("FSAW", 1, "FAIB", -xcoor, ycoor, 0., 0, "ONLY");
    TVirtualMC::GetMC()->Gspos("FSAW", 2, "FAIB", xcoor, ycoor, 0., 0, "ONLY");
  }

  // TOF Supermodule cover definition and positioning
  Float_t covpar[3] = { static_cast<Float_t>(xtof * 0.5), 0.075, static_cast<Float_t>(zlenA * 0.5) };
  TVirtualMC::GetMC()->Gsvolu("FCOV", "BOX ", getMediumID(kAlFrame), covpar, 3); // Al
  if (mTOFHoles) {
    covpar[2] = (zlenA * 0.5 - Geo::INTERCENTRMODBORDER2) * 0.5;
    TVirtualMC::GetMC()->Gsvolu("FCOB", "BOX ", getMediumID(kAlFrame), covpar, 3); // Al
    covpar[2] = Geo::INTERCENTRMODBORDER2;
    TVirtualMC::GetMC()->Gsvolu("FCOP", "BOX ", getMediumID(kPlastic), covpar, 3); // Plastic (CH2)
  }

  xcoor = 0.;
  ycoor = (ytof * 0.5 - Geo::MODULECOVERTHICKNESS) * 0.5 - covpar[1];
  zcoor = 0.;
  TVirtualMC::GetMC()->Gspos("FCOV", 0, "FAIA", xcoor, ycoor, zcoor, 0, "ONLY");
  TVirtualMC::GetMC()->Gspos("FCOV", 0, "FAIC", xcoor, ycoor, zcoor, 0, "ONLY");
  if (mTOFHoles) {
    zcoor = (zlenA * 0.5 + Geo::INTERCENTRMODBORDER2) * 0.5;
    TVirtualMC::GetMC()->Gspos("FCOB", 1, "FAIB", xcoor, ycoor, zcoor, 0, "ONLY");
    TVirtualMC::GetMC()->Gspos("FCOB", 2, "FAIB", xcoor, ycoor, -zcoor, 0, "ONLY");
    zcoor = 0.;
    TVirtualMC::GetMC()->Gspos("FCOP", 0, "FAIB", xcoor, ycoor, zcoor, 0, "ONLY");
  }
}

//_____________________________________________________________________________
void Detector::makeReadoutCrates(Float_t ytof) const
{
  // Services Volumes

  // Empty crate weight: 50 Kg, electronics cards + cables ~ 52 Kg.
  // Per each side (A and C) the total weight is: 2x102 ~ 204 Kg.
  // ... + weight of the connection pannel for the steel cooling system (Cr 18%, Ni 12%, Fe 70%)
  // + other remaining elements + various supports

  // Each FEA card weight + all supports
  // (including all bolts and not including the cable connectors)
  //  353.1 g.
  // Per each strip there are 4 FEA cards, then
  // the total weight of the front-end electonics section is: 353.1 g x 4 = 1412.4 g.

  // Services Volumes

  // Empty crate weight: 50 Kg, electronics cards + cables ~ 52 Kg.
  // Per each side (A and C) the total weight is: 2x102 ~ 204 Kg.
  // ... + weight of the connection pannel for the steel cooling system (Cr 18%, Ni 12%, Fe 70%)
  // + other remaining elements + various supports

  // Each FEA card weight + all supports
  // (including all bolts and not including the cable connectors)
  //  353.1 g.
  // Per each strip there are 4 FEA cards, then
  // the total weight of the front-end electonics section is: 353.1 g x 4 = 1412.4 g.
  //

  Int_t idrotm[Geo::NSECTORS];
  for (Int_t ii = 0; ii < Geo::NSECTORS; ii++)
    idrotm[ii] = 0;

  // volume definition
  Float_t serpar[3] = { 29. * 0.5, 121. * 0.5, 90. * 0.5 };
  TVirtualMC::GetMC()->Gsvolu("FTOS", "BOX ", getMediumID(kCrates), serpar, 3); // Al + Cu + steel

  Float_t xcoor, ycoor, zcoor;
  zcoor = (118. - 90.) * 0.5;
  Float_t phi = -10., ra = Geo::RMIN + ytof * 0.5;
  for (Int_t i = 0; i < Geo::NSECTORS; i++) {
    phi += Geo::PHISEC;
    xcoor = ra * TMath::Cos(phi * TMath::DegToRad());
    ycoor = ra * TMath::Sin(phi * TMath::DegToRad());
    Matrix(idrotm[i], 90., phi, 90., phi + 270., 0., 0.);
    TVirtualMC::GetMC()->Gspos("FTOS", i, "BFMO", xcoor, ycoor, zcoor, idrotm[i], "ONLY");
  }

  zcoor = (90. - 223.) * 0.5;
  TVirtualMC::GetMC()->Gspos("FTOS", 1, "BBCE", ra, -3., zcoor, 0, "ONLY");
}

void Detector::makeModulesInBTOFvolumes(Float_t ytof, Float_t zlenA) const
{
  //
  // Fill BTOF_%i (for i=0,...17) volumes
  // with volumes FTOA (MRPC strip container),
  // In case of TOF holes, three sectors (i.e. 13th, 14th and 15th)
  // are filled with volumes: FTOB and FTOC (MRPC containers),
  //

  constexpr Int_t SIZESTR = 16;

  Int_t idrotm[1] = { 0 };

  // Matrix(idrotm[0], 90.,  0., 0., 0., 90.,-90.);
  Matrix(idrotm[0], 90., 0., 0., 0., 90., 270.);

  Float_t xcoor, ycoor, zcoor;
  xcoor = 0.;

  // Positioning of fibre glass modules (FTOA, FTOB and FTOC)
  for (Int_t isec = 0; isec < Geo::NSECTORS; isec++) {
    if (mTOFSectors[isec] == -1)
      continue;

    char name[SIZESTR];
    snprintf(name, SIZESTR, "BTOF%d", isec);
    if (mTOFHoles && (isec == 13 || isec == 14 || isec == 15)) {
      // xcoor = 0.;
      ycoor = (zlenA * 0.5 + Geo::INTERCENTRMODBORDER1) * 0.5;
      zcoor = -ytof * 0.25;
      TVirtualMC::GetMC()->Gspos("FTOB", 0, name, xcoor, ycoor, zcoor, idrotm[0], "ONLY");
      TVirtualMC::GetMC()->Gspos("FTOC", 0, name, xcoor, -ycoor, zcoor, idrotm[0], "ONLY");
    } else {
      // xcoor = 0.;
      ycoor = 0.;
      zcoor = -ytof * 0.25;
      TVirtualMC::GetMC()->Gspos("FTOA", 0, name, xcoor, ycoor, zcoor, idrotm[0], "ONLY");
    }
  }

  // float par[3] = {100,500,10};
  // TVirtualMC::GetMC()->Gsvolu("FTEM", "BOX ", getMediumID(kAlFrame), par, 3); // Fibre glass
  // ycoor = 0.;
  // zcoor = 350;
  // TVirtualMC::GetMC()->Gspos("FTEM", 0, "cave", xcoor, ycoor, zcoor, idrotm[0], "ONLY");

  //  TVirtualMC::GetMC()->Gspos("FTOA", 0, "cave", xcoor, ycoor, zcoor, idrotm[0], "ONLY");
}

void Detector::makeCoversInBTOFvolumes() const
{
  //
  // Fill BTOF_%i (for i=0,...17) volumes
  // with volumes FPEA (to separate strips from FEA cards)
  // In case of TOF holes, three sectors (i.e. 13th, 14th and 15th)
  // are filled with FPEB volumes
  // (to separate MRPC strips from FEA cards)
  //

  constexpr Int_t SIZESTR = 16;

  Int_t idrotm[1] = { 0 };

  // Matrix(idrotm[0], 90.,  0., 0., 0., 90.,-90.);
  Matrix(idrotm[0], 90., 0., 0., 0., 90., 270.);

  Float_t xcoor, ycoor, zcoor;
  xcoor = 0.;
  ycoor = 0.;
  zcoor = Geo::MODULECOVERTHICKNESS * 0.5;

  char name[SIZESTR];

  // Positioning of module covers (FPEA, FPEB)
  for (Int_t isec = 0; isec < Geo::NSECTORS; isec++) {
    if (mTOFSectors[isec] == -1)
      continue;
    snprintf(name, SIZESTR, "BTOF%d", isec);
    if (mTOFHoles && (isec == 13 || isec == 14 || isec == 15))
      TVirtualMC::GetMC()->Gspos("FPEB", 0, name, xcoor, ycoor, zcoor, idrotm[0], "ONLY");
    else
      TVirtualMC::GetMC()->Gspos("FPEA", 0, name, xcoor, ycoor, zcoor, idrotm[0], "ONLY");
  }
}

//_____________________________________________________________________________
void Detector::makeBackInBTOFvolumes(Float_t ytof) const
{
  //
  // Fill BTOF_%i (for i=0,...17) volumes with volumes called FAIA and
  // FAIC (FEA cards and services container).
  // In case of TOF holes, three sectors (i.e. 13th, 14th and 15th) are
  // filled with volumes FAIB (FEA cards and services container).
  //

  constexpr Int_t SIZESTR = 16;

  Int_t idrotm[1] = { 0 };

  // Matrix(idrotm[0], 90.,  0., 0., 0., 90.,-90.);
  Matrix(idrotm[0], 90., 0., 0., 0., 90., 270.);

  Float_t xcoor, ycoor, zcoor;
  xcoor = 0.;
  ycoor = 0.;
  zcoor = Geo::MODULECOVERTHICKNESS + (ytof * 0.5 - Geo::MODULECOVERTHICKNESS) * 0.5;

  char name[SIZESTR];

  // Positioning of FEA cards and services containers (FAIA, FAIC and FAIB)
  for (Int_t isec = 0; isec < Geo::NSECTORS; isec++) {
    if (mTOFSectors[isec] == -1)
      continue;
    snprintf(name, SIZESTR, "BTOF%d", isec);
    if (Geo::FEAWITHMASKS[isec])
      TVirtualMC::GetMC()->Gspos("FAIA", 0, name, xcoor, ycoor, zcoor, idrotm[0], "ONLY");
    else {
      if (mTOFHoles && (isec == 13 || isec == 14 || isec == 15))
        TVirtualMC::GetMC()->Gspos("FAIB", 0, name, xcoor, ycoor, zcoor, idrotm[0], "ONLY");
      else
        TVirtualMC::GetMC()->Gspos("FAIC", 0, name, xcoor, ycoor, zcoor, idrotm[0], "ONLY");
    }
  }
}

void Detector::addAlignableVolumes() const
{
  //
  // Create entries for alignable volumes associating the symbolic volume
  // name with the corresponding volume path. Needs to be syncronized with
  // eventual changes in the geometry.
  //

  o2::Base::DetID::ID idTOF = o2::Base::DetID::TOF;
  Int_t modUID, modnum = 0;

  TString volPath;
  TString symName;

  TString vpL0 = "cave/B077_1/BSEGMO";
  TString vpL1 = "_1/BTOF";
  TString vpL2 = "_1";
  TString vpL3 = "/FTOA_0";
  TString vpL4 = "/FLTA_0/FSTR_";

  TString snSM = "TOF/sm";
  TString snSTRIP = "/strip";

  //
  // The TOF MRPC Strips
  // The symbolic names are: TOF/sm00/strip01
  //                           ...
  //                         TOF/sm17/strip91

  Int_t imod = 0;

  for (Int_t isect = 0; isect < Geo::NSECTORS; isect++) {
    for (Int_t istr = 1; istr <= Geo::NSTRIPXSECTOR; istr++) {
      modUID = o2::Base::GeometryManager::getSensID(idTOF, modnum++);
      LOG(INFO)<<"modUID: "<<modUID<<"\n";

      if (mTOFSectors[isect] == -1)
        continue;

      if (mTOFHoles && (isect == 13 || isect == 14 || isect == 15)) {
        if (istr < 39) {
          vpL3 = "/FTOB_0";
          vpL4 = "/FLTB_0/FSTR_";
        } else if (istr > 53) {
          vpL3 = "/FTOC_0";
          vpL4 = "/FLTC_0/FSTR_";
        } else
          continue;
      } else {
        vpL3 = "/FTOA_0";
        vpL4 = "/FLTA_0/FSTR_";
      }

      volPath = vpL0;
      volPath += isect;
      volPath += vpL1;
      volPath += isect;
      volPath += vpL2;
      volPath += vpL3;
      volPath += vpL4;
      volPath += istr;


      symName = snSM;
      symName += Form("%02d", isect);
      symName += snSTRIP;
      symName += Form("%02d", istr);

      LOG(DEBUG)<< "--------------------------------------------"<<"\n";
      LOG(DEBUG)<< "Alignable object"<< imod<<"\n";
      LOG(DEBUG)<< "volPath="<<volPath<<"\n";
      LOG(DEBUG)<< "symName="<<symName<<"\n";
      LOG(DEBUG)<< "--------------------------------------------"<<"\n";

      LOG(INFO)<<"Check for alignable entry: "<<symName<<"\n";

      if (!gGeoManager->SetAlignableEntry(symName.Data(), volPath.Data(), modUID))
        LOG(ERROR)<<"Alignable entry "<<symName<<" NOT set\n";
        LOG(INFO)<<"Alignable entry "<<symName<<" set\n";

      // T2L matrices for alignment
      TGeoPNEntry* e = gGeoManager->GetAlignableEntryByUID(modUID);
      LOG(INFO)<<"Got TGeoPNEntry "<<e<<"\n";
      
      if (e) {
        TGeoHMatrix* globMatrix = e->GetGlobalOrig();
        Double_t phi = Geo::PHISEC * (isect % Geo::NSECTORS) + Geo::PHISEC * 0.5;
        TGeoHMatrix* t2l = new TGeoHMatrix();
        t2l->RotateZ(phi);
        t2l->MultiplyLeft(&(globMatrix->Inverse()));
        e->SetMatrix(t2l);
      } else {
        // AliError(Form("Alignable entry %s is not valid!",symName.Data()));
      }
      imod++;
    }
  }

  //
  // The TOF supermodules
  // The symbolic names are: TOF/sm00
  //                           ...
  //                         TOF/sm17
  //
  for (Int_t isect = 0; isect < Geo::NSECTORS; isect++) {
    volPath = vpL0;
    volPath += isect;
    volPath += vpL1;
    volPath += isect;
    volPath += vpL2;

    symName = snSM;
    symName += Form("%02d", isect);

    // AliDebug(2,"--------------------------------------------");
    // AliDebug(2,Form("Alignable object %d", isect+imod));
    // AliDebug(2,Form("volPath=%s\n",volPath.Data()));
    // AliDebug(2,Form("symName=%s\n",symName.Data()));
    // AliDebug(2,"--------------------------------------------");

    gGeoManager->SetAlignableEntry(symName.Data(), volPath.Data());
  }
}
