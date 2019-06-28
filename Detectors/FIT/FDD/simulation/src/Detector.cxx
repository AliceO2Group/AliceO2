// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Detector.cxx
/// \brief Implementation of the Detector class
/// \author michal broz
/// \date 01/08/2016

#include "DataFormatsFDD/Hit.h"

#include "FDDBase/Geometry.h"

#include "FDDSimulation/Detector.h"

#include "SimulationDataFormat/Stack.h"
#include "Field/MagneticField.h"

#include "TVirtualMC.h"
#include "TLorentzVector.h"
#include "TVector3.h"
#include "TGeoManager.h"
#include "TRandom.h"

#include "FairLogger.h"
#include "FairRootManager.h"
#include "FairVolume.h"
#include "FairRootManager.h"

using namespace o2::fdd;
using o2::fdd::Geometry;
using o2::fdd::Hit;

ClassImp(o2::fdd::Detector);

//_____________________________________________________________________________
Detector::Detector(Bool_t Active)
  : o2::base::DetImpl<Detector>("FDD", Active),
    mHits(o2::utils::createSimVector<o2::fdd::Hit>()),
    mGeometry(nullptr)
{
}

//_____________________________________________________________________________
Detector::Detector(const Detector& src)
  : o2::base::DetImpl<Detector>(src),
    mHits(o2::utils::createSimVector<o2::fdd::Hit>())
{
}

//_____________________________________________________________________________
Detector& Detector::operator=(const Detector& src)
{

  if (this == &src) {
    return *this;
  }
  // base class assignment
  base::Detector::operator=(src);

  mHits = nullptr;
  return *this;
}

//_____________________________________________________________________________
Detector::~Detector()
{

  if (mHits) {
    o2::utils::freeSimVector(mHits);
  }
}

//_____________________________________________________________________________
void Detector::InitializeO2Detector()
{

  TGeoVolume* vol;

  vol = gGeoManager->GetVolume("FDApad");
  if (!vol) {
    LOG(FATAL) << "can't find volume FDApad";
  } else {
    AddSensitiveVolume(vol);
  }

  vol = gGeoManager->GetVolume("FDCpad");
  if (!vol) {
    LOG(FATAL) << "can't find volume FDCpad";
  } else {
    AddSensitiveVolume(vol);
  }
}

//_____________________________________________________________________________
Bool_t Detector::ProcessHits(FairVolume* vol)
{
  // This is step manager
  // do not track neutral particles
  if (!(fMC->TrackCharge())) {
    return kFALSE;
  }

  Int_t copy;
  Int_t current_volid = fMC->CurrentVolID(copy);

  // Get sensitive volumes id (scintillator pads)
  static Int_t idFDA = fMC->VolId("FDApad");
  static Int_t idFDC = fMC->VolId("FDCpad");

  // Get sector copy (1,2,3,4) ( 1 level up from pad )
  Int_t sect;
  fMC->CurrentVolOffID(1, sect);

  // Get Detector copy (1,2) ( 2 levels up from pad )
  Int_t detc;
  fMC->CurrentVolOffID(2, detc);

  // Set detector type: FDA or FDC
  Int_t ADlayer = (current_volid == idFDC) ? 0 : 2;

  sect--;          //     sector within layer [0-3]
  detc--;          //     detector copy       [0-1]
  ADlayer += detc; //     global layer number [0-3]

  Int_t ADsector = ADlayer * 4 + sect; // Global AD sector number [0-15]
  // Layer    Sector Number
  // FDC 0  =   0- 3
  // FDC 1  =   4- 7
  // FDA 2  =   8-11
  // FDA 3  =  12-15

  Float_t fFDDLightYield(12.8e6);
  //BC420 yield is 0.64 of antracene which is 20k photons/MeV = 12800/MeV = 12.8e6/GeV

  Float_t destep_ad = fMC->Edep();
  Int_t nPhotonsInStep_ad = Int_t(fFDDLightYield * destep_ad);
  nPhotonsInStep_ad = gRandom->Poisson(nPhotonsInStep_ad);

  static Float_t eloss_ad = 0.;
  static Float_t tlength_ad = 0.;
  static Int_t nPhotons_ad = 0;

  Float_t x, y, z;

  eloss_ad += destep_ad;

  if (fMC->IsTrackEntering()) {
    nPhotons_ad = nPhotonsInStep_ad;
    fMC->TrackPosition(x, y, z);

    eloss_ad = 0.0;
    return kFALSE;
  }
  nPhotons_ad += nPhotonsInStep_ad;

  if (fMC->IsTrackExiting() || fMC->IsTrackStop() || fMC->IsTrackDisappeared()) {
    Int_t trackID = fMC->GetStack()->GetCurrentTrackNumber();

    Float_t time = fMC->TrackTime() * 1.0e9; //time from seconds to ns
    TVector3 vPos(x, y, z);

    addHit(trackID, ADsector, vPos, time, eloss_ad, nPhotons_ad);
    return kTRUE;
  }

  return kFALSE;
}

//_____________________________________________________________________________
Hit* Detector::addHit(int trackID, unsigned short detID, const TVector3& Pos, double Time, double eLoss, int nPhot)
{
  //LOG(INFO) << "FDD hit "<<trackID<<" "<<detID<<" "<<Time<<" "<<eLoss<<" "<<nPhot;
  mHits->emplace_back(trackID, detID, Pos, Time, eLoss, nPhot);
  auto stack = (o2::data::Stack*)fMC->GetStack();
  stack->addHit(GetDetId());

  return &(mHits->back());
}

//_____________________________________________________________________________
void Detector::CreateMaterials()
{

  Float_t density, as[11], zs[11], ws[11];
  Double_t radLength, absLength, a_ad, z_ad;
  Int_t id;

  // PVC (C2H3Cl)n
  Float_t aPVC[3] = { 12.0107, 1.00794, 35.4527 };
  Float_t zPVC[3] = { 6., 1., 35. };
  Float_t wPVC[3] = { 2., 3., 1. };
  Float_t dPVC = 1.3;
  o2::base::Detector::Mixture(47, "PVC", aPVC, zPVC, dPVC, -3, wPVC);

  // Air
  Float_t aAir[4] = { 12.0107, 14.0067, 15.9994, 39.948 };
  Float_t zAir[4] = { 6., 7., 8., 18. };
  Float_t wAir[4] = { 0.000124, 0.755267, 0.231781, 0.012827 };
  Float_t dAir1 = 1.20479E-11;
  // Steel
  Float_t asteel[4] = { 55.847, 51.9961, 58.6934, 28.0855 };
  Float_t zsteel[4] = { 26., 24., 28., 14. };
  Float_t wsteel[4] = { .715, .18, .1, .005 };
  // Cast iron
  Float_t acasti[4] = { 55.847, 12.011, 28.085, 54.938 };
  Float_t zcasti[4] = { 26., 6., 14., 25. };
  Float_t wcasti[4] = { 0.929, 0.035, 0.031, 0.005 };

  o2::base::Detector::Material(9, "ALU", 26.98, 13., 2.7, 8.9, 37.2);
  o2::base::Detector::Material(10, "IRON", 55.85, 26., 7.87, 1.76, 17.1);
  o2::base::Detector::Material(11, "COPPER", 63.55, 29., 8.96, 1.43, 15.1);
  o2::base::Detector::Mixture(16, "VACUUM", aAir, zAir, dAir1, 4, wAir);
  o2::base::Detector::Mixture(19, "STAINLESS_STEEL", asteel, zsteel, 7.88, 4, wsteel);
  o2::base::Detector::Material(13, "LEAD", 207.19, 82., 11.35, .56, 18.5);
  o2::base::Detector::Mixture(18, "CAST_IRON", acasti, zcasti, 7.2, 4, wcasti);

  // ****************
  // Tracking media parameters.
  Int_t fieldType = 3;     // Field type
  Double_t maxField = 5.0; // Field max.
  Float_t epsil, stmin, tmaxfd, deemax, stemax;
  epsil = 0.001; // Tracking precision,
  stemax = -1.;  // Maximum displacement for multiple scat
  tmaxfd = -20.; // Maximum angle due to field deflection
  deemax = -.3;  // Maximum fractional energy loss, DLS
  stmin = -.8;
  // ***************

  o2::base::Detector::Medium(47, "PVC", 47, 0, fieldType, maxField, tmaxfd, stemax, deemax, epsil, stmin);
  o2::base::Detector::Medium(9, "ALU", 9, 0, fieldType, maxField, tmaxfd, stemax, deemax, epsil, stmin);
  o2::base::Detector::Medium(10, "FE", 10, 0, fieldType, maxField, tmaxfd, stemax, deemax, epsil, stmin);
  o2::base::Detector::Medium(11, "Cu", 11, 0, fieldType, maxField, tmaxfd, stemax, deemax, epsil, stmin);
  o2::base::Detector::Medium(16, "VA", 16, 0, fieldType, maxField, tmaxfd, stemax, deemax, epsil, stmin);
  o2::base::Detector::Medium(13, "PB", 13, 0, fieldType, maxField, tmaxfd, stemax, deemax, epsil, stmin);
  o2::base::Detector::Medium(19, "ST", 19, 0, fieldType, maxField, tmaxfd, stemax, deemax, epsil, stmin);

  // Parameters  for AD scintillator: BC420
  // NE-102, has the following properties :
  //    Density : ca. 1.032 g/cm3
  //    Electrons/cm3: 3.37 x 10^23
  //    H atoms/cm3: 5.21 x 10^22
  //    C atoms/cm3: 4.74 x 10^22
  //    Ratio of H to C : 1.100
  //    wavelength of emission : 408 nm.
  //    Decay time : 1.8 ns.
  //    Photons/MeV: 0.64 of antracene which is 20k photons/MeV
  // H                // C
  as[0] = 1.00794;
  as[1] = 12.011;
  zs[0] = 1.;
  zs[1] = 6.;
  ws[0] = 5.21;
  ws[1] = 4.74;
  density = 1.032;
  id = 1;
  o2::base::Detector::Mixture(id, "BC420", as, zs, density, -2, ws);
  o2::base::Detector::Medium(id, "BC420", id, 1, fieldType, maxField, tmaxfd, stemax, deemax, epsil, stmin);

  // Parameters for lightGuide:
  // Should be Poly(methyl methacrylate) (PMMA) acrylic
  // (C5O2H8)n
  // Density  1.18 g/cm3
  // Mixture PMMA    Aeff=12.3994 Zeff=6.23653 rho=1.18 radlen=34.0677 intlen=63.3073
  // Element #0 : C  Z=  6.00 A= 12.01 w= 0.600 natoms=5
  // Element #1 : H  Z=  1.00 A=  1.01 w= 0.081 natoms=8
  // Element #2 : O  Z=  8.00 A= 16.00 w= 0.320 natoms=2

  // Carbon          Hydrogen          Oxygen
  as[0] = 12.0107;
  as[1] = 1.00794;
  as[2] = 15.9994;
  zs[0] = 6.;
  zs[1] = 1.;
  zs[2] = 8.;
  ws[0] = 0.60;
  ws[1] = 0.081;
  ws[2] = 0.32;
  density = 1.18;
  id = 2;
  o2::base::Detector::Mixture(id, "PMMA", as, zs, density, 3, ws);
  o2::base::Detector::Medium(id, "PMMA", id, 1, fieldType, maxField, tmaxfd, stemax, deemax, epsil, stmin);

  // mu-metal
  // Niquel          Iron              Molybdenum        Manganese
  as[0] = 58.6934;
  as[1] = 55.845;
  as[2] = 95.94;
  as[3] = 54.9380;
  zs[0] = 28.;
  zs[1] = 26.;
  zs[2] = 42.;
  zs[3] = 25.;
  ws[0] = 0.802;
  ws[1] = 0.14079;
  ws[2] = 0.0485;
  ws[3] = 0.005;
  // Silicon         Chromium          Cobalt            Aluminium
  as[4] = 28.0855;
  as[5] = 51.9961;
  as[6] = 58.9332;
  as[7] = 26.981539;
  zs[4] = 14.;
  zs[5] = 24.;
  zs[6] = 27.;
  zs[7] = 13.;
  ws[4] = 0.003;
  ws[5] = 0.0002;
  ws[6] = 0.0002;
  ws[7] = 0.0001;
  // Carbon          Phosphorus        Sulfur
  as[8] = 12.0107;
  as[9] = 30.97376;
  as[10] = 32.066;
  zs[8] = 6.;
  zs[9] = 15.;
  zs[10] = 16.;
  ws[8] = 0.00015;
  ws[9] = 0.00005;
  ws[10] = 0.00001;
  density = 8.25;
  id = 3;
  o2::base::Detector::Mixture(id, "MuMetal", as, zs, density, 11, ws);
  o2::base::Detector::Medium(id, "MuMetal", id, 1, fieldType, maxField, tmaxfd, stemax, deemax, epsil, stmin);

  // Parameters for FDCPMA: Aluminium
  a_ad = 26.98;
  z_ad = 13.00;
  density = 2.7;
  radLength = 8.9;
  absLength = 37.2;
  id = 4;
  o2::base::Detector::Material(id, "Alum", a_ad, z_ad, density, radLength, absLength);
  o2::base::Detector::Medium(id, "Alum", id, 1, fieldType, maxField, tmaxfd, stemax, deemax, epsil, stmin);

  // Parameters for FDCPMG: Glass for the simulation Aluminium
  a_ad = 26.98;
  z_ad = 13.00;
  density = 2.7;
  radLength = 8.9;
  absLength = 37.2;
  id = 5;
  o2::base::Detector::Material(id, "Glass", a_ad, z_ad, density, radLength, absLength);
  o2::base::Detector::Medium(id, "Glass", id, 1, fieldType, maxField, tmaxfd, stemax, deemax, epsil, stmin);
}
//_____________________________________________________________________________
void Detector::ConstructGeometry()
{
  CreateMaterials();
  mGeometry = new Geometry(Geometry::eOnlySensitive);
}
//_____________________________________________________________________________
void Detector::Register()
{
  // This will create a branch in the output tree called Hit, setting the last
  // parameter to kFALSE means that this collection will not be written to the file,
  // it will exist only during the simulation

  if (FairRootManager::Instance()) {
    FairRootManager::Instance()->RegisterAny(addNameTo("Hit").data(), mHits, kTRUE);
  }
}

//_____________________________________________________________________________
void Detector::Reset()
{
  if (!o2::utils::ShmManager::Instance().isOperational()) {
    mHits->clear();
  }
}
