// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TPCSimulation/Detector.h"
#include "TPCSimulation/Point.h"
#include "TPCBase/ParameterGas.h"

#include "SimulationDataFormat/Stack.h"
#include "SimulationDataFormat/TrackReference.h"

#include "FairVolume.h" // for FairVolume

#include "TVirtualMC.h" // for TVirtualMC, gMC

#include <cstddef> // for NULL

#include "FairGeoVolume.h"
#include "FairGeoNode.h"
#include "FairGeoLoader.h"
#include "FairGeoInterface.h"
#include "FairRun.h"
#include "FairRuntimeDb.h"
#include "FairLogger.h"
#include "FairRootManager.h"

#include "TSystem.h"
#include "TVirtualMC.h"

#include "TFile.h"

// geo stuff
#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoPcon.h"
#include "TGeoTube.h"
#include "TGeoCone.h"
#include "TGeoPgon.h"
#include "TGeoTrd1.h"
#include "TGeoCompositeShape.h"
#include "TGeoPara.h"
#include "TGeoPhysicalNode.h"
#include "TGeoHalfSpace.h"
#include "TGeoArb8.h"
#include "TGeoMatrix.h"

#include <iostream>
#include <cmath>

using std::cout;
using std::endl;
using std::ifstream;
using std::ios_base;
using namespace o2::tpc;

Detector::Detector(Bool_t active) : o2::base::DetImpl<Detector>("TPC", active), mGeoFileName()
{
  for (int i = 0; i < Sector::MAXSECTOR; ++i) {
    mHitsPerSectorCollection[i] = o2::utils::createSimVector<o2::tpc::HitGroup>(); //new std::vector<o2::tpc::HitGroup>;
  }
}

// forward default constructor
Detector::Detector() : o2::tpc::Detector(kTRUE) {}

Detector::Detector(const Detector& rhs)
  : o2::base::DetImpl<Detector>(rhs),
    mHitCounter(0),
    mElectronCounter(0),
    mStepCounter(0),
    mGeoFileName(rhs.mGeoFileName)
{
  for (int i = 0; i < Sector::MAXSECTOR; ++i) {
    mHitsPerSectorCollection[i] = o2::utils::createSimVector<o2::tpc::HitGroup>(); //new std::vector<o2::tpc::HitGroup>;new std::vector<o2::tpc::HitGroup>;
  }
}

Detector::~Detector()
{
  for (int i = 0; i < Sector::MAXSECTOR; ++i) {
    // mHitsPerSectorCollection[i]->clear();
    o2::utils::freeSimVector(mHitsPerSectorCollection[i]);
  }
  std::cout << "Produced hits " << mHitCounter << "\n";
  std::cout << "Produced electrons " << mElectronCounter << "\n";
  std::cout << "Stepping called " << mStepCounter << "\n";
}

void Detector::InitializeO2Detector()
{
  // Define the list of sensitive volumes
  defineSensitiveVolumes();
}

void Detector::SetSpecialPhysicsCuts()
{
  using namespace o2::base; // to have enum values of EProc and ECut available

  FairRun* fRun = FairRun::Instance();
  if (strcmp(fRun->GetName(), "TGeant3") == 0) {
    const Float_t cut1 = 1e-5;
    const Float_t cutTofmax = 1e10;

    // Some cuts implemented in AliRoot
    // \note
    //    Cuts in TPC were set globally before and hence affected the default settings of all other media,
    //    also outside of TPC. Also, settings were just done for G3.
    //    Now cuts are set in both cases, for G3 and G4; Further cuts are only set for TPC medium DriftGas2.
    // \todo discussion needed!!
    // cut settings for DriftGas2
    SpecialCuts(kDriftGas2, {{ECut::kCUTGAM, cut1},
                             {ECut::kCUTELE, cut1},
                             {ECut::kCUTNEU, cut1},
                             {ECut::kCUTHAD, cut1},
                             {ECut::kCUTMUO, cut1},
                             {ECut::kBCUTE, cut1},
                             {ECut::kBCUTM, cut1},
                             {ECut::kDCUTE, cut1},
                             {ECut::kDCUTM, cut1},
                             {ECut::kPPCUTM, cut1},
                             {ECut::kTOFMAX, cutTofmax}});
    // process settings for DriftGas2
    SpecialProcesses(kDriftGas2, {{EProc::kPAIR, 1},
                                  {EProc::kCOMP, 1},
                                  {EProc::kPHOT, 1},
                                  {EProc::kPFIS, 0},
                                  {EProc::kDRAY, 1},
                                  {EProc::kANNI, 1},
                                  {EProc::kBREM, 1},
                                  {EProc::kHADR, 1},
                                  {EProc::kMUNU, 1},
                                  {EProc::kDCAY, 1},
                                  {EProc::kLOSS, 1},
                                  {EProc::kMULS, 1}});
  }
}

Bool_t Detector::ProcessHits(FairVolume* vol)
{
  mStepCounter++;
  auto& gasParam = ParameterGas::Instance();

  /* This method is called from the MC stepping for the sensitive volume only */
  //   LOG(INFO) << "tpc::ProcessHits";
  const double trackCharge = fMC->TrackCharge();
  if (static_cast<int>(trackCharge) == 0) {

    // set a very large step size for neutral particles
    fMC->SetMaxStep(1.e10);
    return kFALSE; // take only charged particles
  }

  // ===| SET THE LENGTH OF THE NEXT ENERGY LOSS STEP |=========================
  // (We do this first so we can skip out of the method in the following
  // Eloss->Ionization part.)
  //
  // In all cases we have multiple collisions and we use 2mm (+ some
  // random shift to avoid binning effects), which was tuned for GEANT4, see
  // https://indico.cern.ch/event/316891/contributions/732168/

  const double rnd = fMC->GetRandom()->Rndm();
  fMC->SetMaxStep(0.2 + (2. * rnd - 1.) * 0.05); // 2 mm +- rndm*0.5mm step

  // ===| check active sector |=================================================
  //
  // Get the sector ID and check if the sector is active
  static thread_local TLorentzVector position;
  fMC->TrackPosition(position);
  // for processing reasons in the digitizer, the sectors are shifted by -10deg, so sector 0 will be
  //   from -10 - +10 deg instead of 0-20 and so on
  const int sectorID = static_cast<int>(Sector::ToShiftedSector(position.X(), position.Y(), position.Z()));
  // const int sectorID = static_cast<int>(Sector::ToSector(position.X(), position.Y(), position.Z()));
  // TODO: Temporary hack to process only one sector
  // if (sectorID != 0) return kFALSE;

  // ---| momentum and beta gamma |---
  static TLorentzVector momentum; // static to make avoid creation/deletion of this expensive object
  fMC->TrackMomentum(momentum);

  const float time = fMC->TrackTime() * 1.0e9;
  const int trackID = fMC->GetStack()->GetCurrentTrackNumber();
  const int detID = vol->getMCid();
  o2::data::Stack* stack = (o2::data::Stack*)fMC->GetStack();
  if (fMC->IsTrackEntering() || fMC->IsTrackExiting()) {
    stack->addTrackReference(o2::TrackReference(position.X(), position.Y(), position.Z(), momentum.X(), momentum.Y(),
                                                momentum.Z(), fMC->TrackLength(), time, trackID, GetDetId()));
  }

  // ===| CONVERT THE ENERGY LOSS TO IONIZATION ELECTRONS |=====================
  //
  // The energy loss is implemented directly below and taken GEANT3,
  // ILOSS model 5 (in gfluct.F), which gives
  // the energy loss in a single collision (NA49 model).
  // TODO: Add discussion about drawback

  Int_t numberOfElectrons = 0;
  // I.H. - the type expected in addHit is short

  // ---| Stepsize in cm |---
  const double stepSize = fMC->TrackStep();

  double betaGamma = momentum.P() / fMC->TrackMass();
  betaGamma = TMath::Max(betaGamma, 7.e-3); // protection against too small bg

  // ---| number of primary ionisations per cm |---
  const double primaryElectronsPerCM =
    gasParam.Nprim * BetheBlochAleph(static_cast<float>(betaGamma), gasParam.BetheBlochParam[0],
                                     gasParam.BetheBlochParam[1], gasParam.BetheBlochParam[2],
                                     gasParam.BetheBlochParam[3], gasParam.BetheBlochParam[4]);

  // ---| mean number of collisions and random for this event |---
  const double meanNcoll = stepSize * trackCharge * trackCharge * primaryElectronsPerCM;
  const int nColl = static_cast<int>(fMC->GetRandom()->Poisson(meanNcoll));

  // Variables needed to generate random powerlaw distributed energy loss
  const double alpha_p1 = 1. - gasParam.Exp; // NA49/G3 value
  const double oneOverAlpha_p1 = 1. / alpha_p1;
  const double eMin = gasParam.Ipot;
  const double eMax = gasParam.Eend;
  const double kMin = TMath::Power(eMin, alpha_p1);
  const double kMax = TMath::Power(eMax, alpha_p1);
  const double wIon = gasParam.Wion;

  for (Int_t n = 0; n < nColl; n++) {
    // Use GEANT3 / NA49 expression:
    // P(eDep) ~ k * edep^-gasParam.getExp()
    // eMin(~I) < eDep < eMax(300 electrons)
    // k fixed so that Int_Emin^EMax P(Edep) = 1.
    const double rndm = fMC->GetRandom()->Rndm();
    const double eDep = TMath::Power((kMax - kMin) * rndm + kMin, oneOverAlpha_p1);
    int nel_step = static_cast<int>(((eDep - eMin) / wIon) + 1);
    nel_step = TMath::Min(nel_step, 300); // 300 electrons corresponds to 10 keV
    numberOfElectrons += nel_step;
  }

  // LOG(INFO) << "tpc::AddHit" << FairLogger::endl << "Eloss: "
  //<< fMC->Edep() << ", Nelectrons: "
  //<< numberOfElectrons;

  if (numberOfElectrons <= 0) // Could maybe be smaller than 0 due to the Gamma function
    return kFALSE;

  // ADD HIT
  static thread_local int oldTrackId = trackID;
  static thread_local int oldDetId = detID;
  static thread_local int groupCounter = 0;
  static thread_local int oldSectorId = sectorID;

  //  a new group is starting -> put it into the container
  static thread_local HitGroup* currentgroup = nullptr;
  if (groupCounter == 0) {
    mHitsPerSectorCollection[sectorID]->emplace_back(trackID);
    currentgroup = &(mHitsPerSectorCollection[sectorID]->back());
  }
  if (trackID == oldTrackId && oldSectorId == sectorID) {
    groupCounter++;
    mHitCounter++;
    mElectronCounter += numberOfElectrons;
    currentgroup->addHit(position.X(), position.Y(), position.Z(), time, numberOfElectrons);
  }
  // finish group
  else {
    oldTrackId = trackID;
    oldSectorId = sectorID;
    groupCounter = 0;
  }

  // LOG(INFO) << "tpc::AddHit" << FairLogger::endl
  //<< "   -- " << trackNumberID <<","  << volumeID << " " << vol->GetName()
  //<< ", Pos: (" << position.X() << ", "  << position.Y() <<", "<<  position.Z()<< ", " << r << ") "
  //<< ", Mom: (" << momentum.Px() << ", " << momentum.Py() << ", "  <<  momentum.Pz() << ") "
  //<< " Time: "<<  time <<", Len: " << length << ", Nelectrons: " <<
  // numberOfElectrons;
  // I.H. - the code above does not compile if uncommented

  // Increment number of Detector det points in TParticle
  stack->addHit(GetDetId());

  return kTRUE;
}

void Detector::EndOfEvent()
{
  if (!o2::utils::ShmManager::Instance().isOperational()) {
    for (int i = 0; i < Sector::MAXSECTOR; ++i) {
      mHitsPerSectorCollection[i]->clear();
    }
  }
}

void Detector::Register()
{
  /** This will create a branch in the output tree called
      DetectorPoint, setting the last parameter to kFALSE means:
      this collection will not be written to the file, it will exist
      only during the simulation.
  */

  auto* mgr = FairRootManager::Instance();
  for (int i = 0; i < Sector::MAXSECTOR; ++i) {
    mgr->RegisterAny(getHitBranchNames(i).c_str(), mHitsPerSectorCollection[i], kTRUE);
  }
}

void Detector::Reset()
{
  for (int i = 0; i < Sector::MAXSECTOR; ++i) {
    // cout << "CLEARED (reset) hitsCollection " << i << " " << mHitsPerSectorCollection[i] << endl;
    mHitsPerSectorCollection[i]->clear();
  }
}

void Detector::ConstructGeometry()
{
  // Create the detector materials
  CreateMaterials();

  // Load geometry
  //   LoadGeometryFromFile();
  ConstructTPCGeometry();
}

void Detector::CreateMaterials()
{
  //-----------------------------------------------
  // Create Materials for for TPC simulations
  //-----------------------------------------------

  //-----------------------------------------------------------------
  // Origin: Marek Kowalski  IFJ, Krakow, Marek.Kowalski@ifj.edu.pl
  //-----------------------------------------------------------------

  Int_t iSXFLD = 2;
  Float_t sXMGMX = 10.0;
  // init the field tracking params
  o2::base::Detector::initFieldTrackingParams(iSXFLD, sXMGMX);

  Float_t amat[7]; // atomic numbers
  Float_t zmat[7]; // z
  Float_t wmat[7]; // proportions

  Float_t density;

  //***************** Gases *************************

  //--------------------------------------------------------------
  // gases - air and CO2
  //--------------------------------------------------------------

  // CO2

  amat[0] = 12.011;
  amat[1] = 15.9994;

  zmat[0] = 6.;
  zmat[1] = 8.;

  wmat[0] = 0.2729;
  wmat[1] = 0.7271;

  density = 1.842e-3;

  o2::base::Detector::Mixture(10, "CO2", amat, zmat, density, 2, wmat);
  //
  // Air
  //
  amat[0] = 15.9994;
  amat[1] = 14.007;
  //
  zmat[0] = 8.;
  zmat[1] = 7.;
  //
  wmat[0] = 0.233;
  wmat[1] = 0.767;
  //
  density = 0.001205;

  o2::base::Detector::Mixture(11, "Air", amat, zmat, density, 2, wmat);

  //----------------------------------------------------------------
  // drift gases 5 mixtures, 5 materials
  //----------------------------------------------------------------
  //
  // Drift gases 1 - nonsensitive, 2 - sensitive, 3 - for Kr
  //  Composition by % of volume, values at 20deg and 1 atm.
  //
  //  get the geometry title - defined in Config.C
  //
  //--------------------------------------------------------------
  //  predefined gases, composition taken from param file
  //--------------------------------------------------------------
  TString names[6] = {"Ne", "Ar", "CO2", "N", "CF4", "CH4"};
  TString gname;

  /// @todo: Gas mixture is hard coded here, this should be moved to some kind of parameter
  //       container in the future
  Float_t comp[6] = {90. / 105., 0., 10. / 105., 5. / 105., 0., 0.};
  // indices:
  // 0-Ne, 1-Ar, 2-CO2, 3-N, 4-CF4, 5-CH4
  //
  // elements' masses
  //
  amat[0] = 20.18;   // Ne
  amat[1] = 39.95;   // Ar
  amat[2] = 12.011;  // C
  amat[3] = 15.9994; // O
  amat[4] = 14.007;  // N
  amat[5] = 18.998;  // F
  amat[6] = 1.;      // H
  //
  // elements' atomic numbers
  //
  //
  zmat[0] = 10.; // Ne
  zmat[1] = 18.; // Ar
  zmat[2] = 6.;  // C
  zmat[3] = 8.;  // O
  zmat[4] = 7.;  // N
  zmat[5] = 9.;  // F
  zmat[6] = 1.;  // H
  //
  // Mol masses
  //
  Float_t wmol[6];
  wmol[0] = 20.18;        // Ne
  wmol[1] = 39.948;       // Ar
  wmol[2] = 44.0098;      // CO2
  wmol[3] = 2. * 14.0067; // N2
  wmol[4] = 88.0046;      // CF4
  wmol[5] = 16.011;       // CH4
  //
  Float_t wtot = 0.; // total mass of the mixture
  for (Int_t i = 0; i < 6; i++) {
    wtot += *(comp + i) * wmol[i];
  }
  wmat[0] = comp[0] * amat[0] / wtot;                                           // Ne
  wmat[1] = comp[1] * amat[1] / wtot;                                           // Ar
  wmat[2] = (comp[2] * amat[2] + comp[4] * amat[2] + comp[5] * amat[2]) / wtot; // C
  wmat[3] = comp[2] * amat[3] * 2. / wtot;                                      // O
  wmat[4] = comp[3] * amat[4] * 2. / wtot;                                      // N
  wmat[5] = comp[4] * amat[5] * 4. / wtot;                                      // F
  wmat[6] = comp[5] * amat[6] * 4. / wtot;                                      // H
  //
  // densities (NTP)
  //
  Float_t dens[6] = {0.839e-3, 1.661e-3, 1.842e-3, 1.251e-3, 3.466e-3, 0.668e-3};
  //
  density = 0.;
  for (Int_t i = 0; i < 6; i++) {
    density += comp[i] * dens[i];
  }
  //
  // names
  //
  Int_t cnt = 0;
  for (Int_t i = 0; i < 6; i++) {
    if (comp[i]) {
      if (cnt)
        gname += "-";
      gname += names[i];
      cnt++;
    }
  }
  TString gname1, gname2, gname3;
  gname1 = gname + "-1";
  gname2 = gname + "-2";
  gname3 = gname + "-3";
  //
  // take only elements with nonzero weights
  //
  Float_t amat1[6], zmat1[6], wmat1[6];
  cnt = 0;
  for (Int_t i = 0; i < 7; i++) {
    if (wmat[i]) {
      zmat1[cnt] = zmat[i];
      amat1[cnt] = amat[i];
      wmat1[cnt] = wmat[i];
      cnt++;
    }
  }

  //
  o2::base::Detector::Mixture(12, gname1.Data(), amat1, zmat1, density, cnt, wmat1); // nonsensitive
  o2::base::Detector::Mixture(13, gname2.Data(), amat1, zmat1, density, cnt, wmat1); // sensitive
  o2::base::Detector::Mixture(40, gname3.Data(), amat1, zmat1, density, cnt, wmat1); // sensitive Kr

  //----------------------------------------------------------------------
  //               solid materials
  //----------------------------------------------------------------------

  // Kevlar C14H22O2N2

  amat[0] = 12.011;
  amat[1] = 1.;
  amat[2] = 15.999;
  amat[3] = 14.006;

  zmat[0] = 6.;
  zmat[1] = 1.;
  zmat[2] = 8.;
  zmat[3] = 7.;

  wmat[0] = 14.;
  wmat[1] = 22.;
  wmat[2] = 2.;
  wmat[3] = 2.;

  density = 1.45;

  o2::base::Detector::Mixture(14, "Kevlar", amat, zmat, density, -4, wmat);

  // NOMEX

  amat[0] = 12.011;
  amat[1] = 1.;
  amat[2] = 15.999;
  amat[3] = 14.006;

  zmat[0] = 6.;
  zmat[1] = 1.;
  zmat[2] = 8.;
  zmat[3] = 7.;

  wmat[0] = 14.;
  wmat[1] = 22.;
  wmat[2] = 2.;
  wmat[3] = 2.;

  density = 0.029;

  o2::base::Detector::Mixture(15, "NOMEX", amat, zmat, density, -4, wmat);

  // Makrolon C16H18O3

  amat[0] = 12.011;
  amat[1] = 1.;
  amat[2] = 15.999;

  zmat[0] = 6.;
  zmat[1] = 1.;
  zmat[2] = 8.;

  wmat[0] = 16.;
  wmat[1] = 18.;
  wmat[2] = 3.;

  density = 1.2;

  o2::base::Detector::Mixture(16, "Makrolon", amat, zmat, density, -3, wmat);

  // Tedlar C2H3F

  amat[0] = 12.011;
  amat[1] = 1.;
  amat[2] = 18.998;

  zmat[0] = 6.;
  zmat[1] = 1.;
  zmat[2] = 9.;

  wmat[0] = 2.;
  wmat[1] = 3.;
  wmat[2] = 1.;

  density = 1.71;

  o2::base::Detector::Mixture(17, "Tedlar", amat, zmat, density, -3, wmat);

  // Mylar C5H4O2

  amat[0] = 12.011;
  amat[1] = 1.;
  amat[2] = 15.9994;

  zmat[0] = 6.;
  zmat[1] = 1.;
  zmat[2] = 8.;

  wmat[0] = 5.;
  wmat[1] = 4.;
  wmat[2] = 2.;

  density = 1.39;

  o2::base::Detector::Mixture(18, "Mylar", amat, zmat, density, -3, wmat);
  // material for "prepregs"
  // Epoxy - C14 H20 O3
  // Quartz SiO2
  // Carbon C
  // prepreg1 60% C-fiber, 40% epoxy (vol)
  amat[0] = 12.011;
  amat[1] = 1.;
  amat[2] = 15.994;

  zmat[0] = 6.;
  zmat[1] = 1.;
  zmat[2] = 8.;

  wmat[0] = 0.923;
  wmat[1] = 0.023;
  wmat[2] = 0.054;

  density = 1.859;

  o2::base::Detector::Mixture(19, "Prepreg1", amat, zmat, density, 3, wmat);

  // prepreg2 60% glass-fiber, 40% epoxy

  amat[0] = 12.01;
  amat[1] = 1.;
  amat[2] = 15.994;
  amat[3] = 28.086;

  zmat[0] = 6.;
  zmat[1] = 1.;
  zmat[2] = 8.;
  zmat[3] = 14.;

  wmat[0] = 0.194;
  wmat[1] = 0.023;
  wmat[2] = 0.443;
  wmat[3] = 0.34;

  density = 1.82;

  o2::base::Detector::Mixture(20, "Prepreg2", amat, zmat, density, 4, wmat);

  // prepreg3 50% glass-fiber, 50% epoxy

  amat[0] = 12.01;
  amat[1] = 1.;
  amat[2] = 15.994;
  amat[3] = 28.086;

  zmat[0] = 6.;
  zmat[1] = 1.;
  zmat[2] = 8.;
  zmat[3] = 14.;

  wmat[0] = 0.257;
  wmat[1] = 0.03;
  wmat[2] = 0.412;
  wmat[3] = 0.3;

  density = 1.725;

  o2::base::Detector::Mixture(21, "Prepreg3", amat, zmat, density, 4, wmat);

  // G10 60% SiO2 40% epoxy

  amat[0] = 12.01;
  amat[1] = 1.;
  amat[2] = 15.994;
  amat[3] = 28.086;

  zmat[0] = 6.;
  zmat[1] = 1.;
  zmat[2] = 8.;
  zmat[3] = 14.;

  wmat[0] = 0.194;
  wmat[1] = 0.023;
  wmat[2] = 0.443;
  wmat[3] = 0.340;

  density = 1.7;

  o2::base::Detector::Mixture(22, "G10", amat, zmat, density, 4, wmat);

  // Al

  amat[0] = 26.98;
  zmat[0] = 13.;

  density = 2.7;

  o2::base::Detector::Material(23, "Al", amat[0], zmat[0], density, 999., 999.);

  // Si (for electronics

  amat[0] = 28.086;
  zmat[0] = 14.;

  density = 2.33;

  o2::base::Detector::Material(24, "Si", amat[0], zmat[0], density, 999., 999.);

  // Cu

  amat[0] = 63.546;
  zmat[0] = 29.;

  density = 8.96;

  o2::base::Detector::Material(25, "Cu", amat[0], zmat[0], density, 999., 999.);

  // brass

  amat[0] = 63.546;
  zmat[0] = 29.;
  //
  amat[1] = 65.409;
  zmat[1] = 30.;
  //
  wmat[0] = 0.6;
  wmat[1] = 0.4;

  //
  density = 8.23;

  //
  o2::base::Detector::Mixture(33, "Brass", amat, zmat, density, 2, wmat);

  // Epoxy - C14 H20 O3

  amat[0] = 12.011;
  amat[1] = 1.;
  amat[2] = 15.9994;

  zmat[0] = 6.;
  zmat[1] = 1.;
  zmat[2] = 8.;

  wmat[0] = 14.;
  wmat[1] = 20.;
  wmat[2] = 3.;

  density = 1.25;

  o2::base::Detector::Mixture(26, "Epoxy", amat, zmat, density, -3, wmat);

  // Epoxy - C14 H20 O3 for glue

  amat[0] = 12.011;
  amat[1] = 1.;
  amat[2] = 15.9994;

  zmat[0] = 6.;
  zmat[1] = 1.;
  zmat[2] = 8.;

  wmat[0] = 14.;
  wmat[1] = 20.;
  wmat[2] = 3.;

  density = 1.25;

  density *= 1.25;

  o2::base::Detector::Mixture(35, "Epoxy1", amat, zmat, density, -3, wmat);
  //
  // epoxy film - 90% epoxy, 10% glass fiber
  //
  amat[0] = 12.01;
  amat[1] = 1.;
  amat[2] = 15.994;
  amat[3] = 28.086;

  zmat[0] = 6.;
  zmat[1] = 1.;
  zmat[2] = 8.;
  zmat[3] = 14.;

  wmat[0] = 0.596;
  wmat[1] = 0.071;
  wmat[2] = 0.257;
  wmat[3] = 0.076;

  density = 1.345;

  o2::base::Detector::Mixture(34, "Epoxy-film", amat, zmat, density, 4, wmat);

  // Plexiglas  C5H8O2

  amat[0] = 12.011;
  amat[1] = 1.;
  amat[2] = 15.9994;

  zmat[0] = 6.;
  zmat[1] = 1.;
  zmat[2] = 8.;

  wmat[0] = 5.;
  wmat[1] = 8.;
  wmat[2] = 2.;

  density = 1.18;

  o2::base::Detector::Mixture(27, "Plexiglas", amat, zmat, density, -3, wmat);

  // Carbon

  amat[0] = 12.011;
  zmat[0] = 6.;
  density = 2.265;

  o2::base::Detector::Material(28, "C", amat[0], zmat[0], density, 999., 999.);

  // Fe (steel for the inner heat screen)

  amat[0] = 55.845;

  zmat[0] = 26.;

  density = 7.87;

  o2::base::Detector::Material(29, "Fe", amat[0], zmat[0], density, 999., 999.);
  //
  // Peek - (C6H4-O-OC6H4-O-C6H4-CO)n
  amat[0] = 12.011;
  amat[1] = 1.;
  amat[2] = 15.9994;

  zmat[0] = 6.;
  zmat[1] = 1.;
  zmat[2] = 8.;

  wmat[0] = 19.;
  wmat[1] = 12.;
  wmat[2] = 3.;
  //
  density = 1.3;
  //
  o2::base::Detector::Mixture(30, "Peek", amat, zmat, density, -3, wmat);
  //
  //  Ceramics - Al2O3
  //
  amat[0] = 26.98;
  amat[1] = 15.9994;

  zmat[0] = 13.;
  zmat[1] = 8.;

  wmat[0] = 2.;
  wmat[1] = 3.;

  density = 3.97;

  o2::base::Detector::Mixture(31, "Alumina", amat, zmat, density, -2, wmat);
  //
  // Ceramics for resistors
  //
  amat[0] = 26.98;
  amat[1] = 15.9994;

  zmat[0] = 13.;
  zmat[1] = 8.;

  wmat[0] = 2.;
  wmat[1] = 3.;

  density = 3.97;
  //
  density *= 1.25;

  o2::base::Detector::Mixture(36, "Alumina1", amat, zmat, density, -2, wmat);
  //
  // liquids
  //

  // water

  amat[0] = 1.;
  amat[1] = 15.9994;

  zmat[0] = 1.;
  zmat[1] = 8.;

  wmat[0] = 2.;
  wmat[1] = 1.;

  density = 1.;

  o2::base::Detector::Mixture(32, "Water", amat, zmat, density, -2, wmat);

  //----------------------------------------------------------
  // tracking media for gases
  //----------------------------------------------------------

  o2::base::Detector::Medium(kAir, "Air", 11, 0, iSXFLD, sXMGMX, 10., 999., .1, .01, .1);
  o2::base::Detector::Medium(kDriftGas1, "DriftGas1", 12, 0, iSXFLD, sXMGMX, 10., 999., .1, .001, .001);
  o2::base::Detector::Medium(kDriftGas2, "DriftGas2", 13, 1, iSXFLD, sXMGMX, 10., 999., .1, .001, .001);
  o2::base::Detector::Medium(kCO2, "CO2", 10, 0, iSXFLD, sXMGMX, 10., 999., .1, .001, .001);
  o2::base::Detector::Medium(kDriftGas3, "DriftGas3", 40, 1, iSXFLD, sXMGMX, 10., 999., .1, .001, .001);
  //-----------------------------------------------------------
  // tracking media for solids
  //-----------------------------------------------------------

  o2::base::Detector::Medium(kAl, "Al", 23, 0, iSXFLD, sXMGMX, 10., 999., .1, .0005, .001);
  o2::base::Detector::Medium(kKevlar, "Kevlar", 14, 0, iSXFLD, sXMGMX, 10., 999., .1, .0005, .001);
  o2::base::Detector::Medium(kNomex, "Nomex", 15, 0, iSXFLD, sXMGMX, 10., 999., .1, .001, .001);
  o2::base::Detector::Medium(kMakrolon, "Makrolon", 16, 0, iSXFLD, sXMGMX, 10., 999., .1, .001, .001);
  o2::base::Detector::Medium(kMylar, "Mylar", 18, 0, iSXFLD, sXMGMX, 10., 999., .1, .0005, .001);
  o2::base::Detector::Medium(kTedlar, "Tedlar", 17, 0, iSXFLD, sXMGMX, 10., 999., .1, .0005, .001);
  //
  o2::base::Detector::Medium(kPrepreg1, "Prepreg1", 19, 0, iSXFLD, sXMGMX, 10., 999., .1, .0005, .001);
  o2::base::Detector::Medium(kPrepreg2, "Prepreg2", 20, 0, iSXFLD, sXMGMX, 10., 999., .1, .0005, .001);
  o2::base::Detector::Medium(kPrepreg3, "Prepreg3", 21, 0, iSXFLD, sXMGMX, 10., 999., .1, .0005, .001);
  o2::base::Detector::Medium(kEpoxy, "Epoxy", 26, 0, iSXFLD, sXMGMX, 10., 999., .1, .0005, .001);

  o2::base::Detector::Medium(kCu, "Cu", 25, 0, iSXFLD, sXMGMX, 10., 999., .1, .001, .001);
  o2::base::Detector::Medium(kSi, "Si", 24, 0, iSXFLD, sXMGMX, 10., 999., .1, .001, .001);
  o2::base::Detector::Medium(kG10, "G10", 22, 0, iSXFLD, sXMGMX, 10., 999., .1, .001, .001);
  o2::base::Detector::Medium(kPlexiglas, "Plexiglas", 27, 0, iSXFLD, sXMGMX, 10., 999., .1, .001, .001);
  o2::base::Detector::Medium(kSteel, "Steel", 29, 0, iSXFLD, sXMGMX, 10., 999., .1, .001, .001);
  o2::base::Detector::Medium(kPeek, "Peek", 30, 0, iSXFLD, sXMGMX, 10., 999., .1, .001, .001);
  o2::base::Detector::Medium(kAlumina, "Alumina", 31, 0, iSXFLD, sXMGMX, 10., 999., .1, .001, .001);
  o2::base::Detector::Medium(kWater, "Water", 32, 0, iSXFLD, sXMGMX, 10., 999., .1, .001, .001);
  o2::base::Detector::Medium(kBrass, "Brass", 33, 0, iSXFLD, sXMGMX, 10., 999., .1, .001, .001);
  o2::base::Detector::Medium(kEpoxyfm, "Epoxyfm", 34, 0, iSXFLD, sXMGMX, 10., 999., .1, .0005, .001);
  o2::base::Detector::Medium(kEpoxy1, "Epoxy1", 35, 0, iSXFLD, sXMGMX, 10., 999., .1, .0005, .001);
  o2::base::Detector::Medium(kAlumina1, "Alumina1", 36, 0, iSXFLD, sXMGMX, 10., 999., .1, .001, .001);
}

void Detector::ConstructTPCGeometry()
{
  //
  // Create the geometry of Time Projection Chamber version 2
  //
  // Begin_Html
  /*
   *    <img src="picts/AliTPC.gif">
   */
  // End_Html
  // Begin_Html
  /*
   *    <img src="picts/AliTPCv2Tree.gif">
   */
  // End_Html

  //----------------------------------------------------------
  // This geometry is written using TGeo class
  // Firstly the shapes are defined, and only then the volumes
  // What is recognized by the MC are volumes
  //----------------------------------------------------------
  //
  //  tpc - this will be the mother volume
  //

  //   if (!mParam) {
  //     LOG(ERROR) << "TPC Parameters not available, cannot create Geometry";
  //     return;
  //   }

  //
  // here I define a volume TPC
  // retrive the medium name with "TPC_" as a leading string
  //
  auto* tpc = new TGeoPcon(0., 360., 30); // 30 sections
  //
  tpc->DefineSection(0, -289.6, 77., 278.);
  tpc->DefineSection(1, -262.1, 77., 278.);
  //
  tpc->DefineSection(2, -262.1, 83.1, 278.);
  tpc->DefineSection(3, -260., 83.1, 278.);
  //
  tpc->DefineSection(4, -260., 70., 278.);
  tpc->DefineSection(5, -259.6, 70., 278.);
  //
  tpc->DefineSection(6, -259.6, 68.1, 278.);
  tpc->DefineSection(7, -253.6, 68.1, 278.);
  //
  tpc->DefineSection(8, -253.6, 67.88, 278.); // hs
  tpc->DefineSection(9, -74.0, 60.68, 278.);  // hs
  //
  tpc->DefineSection(10, -74.0, 60.1, 278.);
  tpc->DefineSection(11, -73.3, 60.1, 278.);
  //
  tpc->DefineSection(12, -73.3, 56.9, 278.);
  tpc->DefineSection(13, -68.5, 56.9, 278.);
  //
  tpc->DefineSection(14, -68.5, 60., 278.);
  tpc->DefineSection(15, -64.7, 60., 278.);
  //
  tpc->DefineSection(16, -64.7, 56.9, 278.);
  tpc->DefineSection(17, 73.3, 56.9, 278.);
  //
  tpc->DefineSection(18, 73.3, 60.1, 278.);
  tpc->DefineSection(19, 74.0, 60.1, 278.);
  //
  tpc->DefineSection(20, 74.0, 60.68, 278.);  // hs
  tpc->DefineSection(21, 253.6, 65.38, 278.); // hs
  //
  tpc->DefineSection(22, 253.6, 65.6, 278.);
  tpc->DefineSection(23, 259.6, 65.6, 278.);
  //
  tpc->DefineSection(24, 259.6, 70.0, 278.);
  tpc->DefineSection(25, 260., 70.0, 278.);
  //
  tpc->DefineSection(26, 260., 83.1, 278.);
  tpc->DefineSection(27, 262.1, 83.1, 278.);
  //
  tpc->DefineSection(28, 262.1, 77., 278);
  tpc->DefineSection(29, 289.6, 77., 278.);
  //
  TGeoMedium* m1 = gGeoManager->GetMedium("TPC_Air");
  auto* v1 = new TGeoVolume("TPC_M", tpc, m1);
  //
  // drift volume - sensitive volume, extended beyond the
  // endcaps, because of the alignment
  //
  auto* dvol = new TGeoPcon(0., 360., 6);
  dvol->DefineSection(0, -260., 74.5, 264.4);
  dvol->DefineSection(1, -253.6, 74.5, 264.4);
  //
  dvol->DefineSection(2, -253.6, 76.6774, 258.);
  dvol->DefineSection(3, 253.6, 76.6774, 258.);
  //
  dvol->DefineSection(4, 253.6, 74.5, 264.4);
  dvol->DefineSection(5, 260., 74.5, 264.4);
  //
  TGeoMedium* m5 = gGeoManager->GetMedium("TPC_DriftGas2");
  auto* v9 = new TGeoVolume("TPC_Drift", dvol, m5);
  //
  v1->AddNode(v9, 1);
  //
  // outer insulator
  //
  auto* tpco = new TGeoPcon(0., 360., 6); // insulator
  //
  tpco->DefineSection(0, -256.6, 264.8, 278.);
  tpco->DefineSection(1, -253.6, 264.8, 278.);
  //
  tpco->DefineSection(2, -253.6, 258., 278.);
  tpco->DefineSection(3, 250.6, 258., 278.);
  //
  tpco->DefineSection(4, 250.6, 258., 275.5);
  tpco->DefineSection(5, 253.6, 258., 275.5);
  //
  TGeoMedium* m2 = gGeoManager->GetMedium("TPC_CO2");
  auto* v2 = new TGeoVolume("TPC_OI", tpco, m2);
  //
  TGeoRotation* segrot; // segment rotations
  //
  // outer containment vessel
  //
  auto* tocv = new TGeoPcon(0., 360., 6); // containment vessel
  //
  tocv->DefineSection(0, -256.6, 264.8, 278.);
  tocv->DefineSection(1, -253.6, 264.8, 278.);
  //
  tocv->DefineSection(2, -253.6, 274.8124, 278.);
  tocv->DefineSection(3, 247.6, 274.8124, 278.);
  //
  tocv->DefineSection(4, 247.6, 270.4, 278.);
  tocv->DefineSection(5, 250.6, 270.4, 278.);
  //
  TGeoMedium* m3 = gGeoManager->GetMedium("TPC_Al");
  auto* v3 = new TGeoVolume("TPC_OCV", tocv, m3);
  //
  auto* to1 = new TGeoTubeSeg(274.8174, 277.995, 252.1, 0., 59.9);   // epoxy
  auto* to2 = new TGeoTubeSeg(274.8274, 277.985, 252.1, 0., 59.9);   // tedlar
  auto* to3 = new TGeoTubeSeg(274.8312, 277.9812, 252.1, 0., 59.9);  // prepreg2
  auto* to4 = new TGeoTubeSeg(274.9062, 277.9062, 252.1, 0., 59.9);  // nomex
  auto* tog5 = new TGeoTubeSeg(274.8174, 277.995, 252.1, 59.9, 60.); // epoxy
  //
  TGeoMedium* sm1 = gGeoManager->GetMedium("TPC_Epoxy");
  TGeoMedium* sm2 = gGeoManager->GetMedium("TPC_Tedlar");
  TGeoMedium* sm3 = gGeoManager->GetMedium("TPC_Prepreg2");
  TGeoMedium* sm4 = gGeoManager->GetMedium("TPC_Nomex");
  //
  TGeoMedium* smep = gGeoManager->GetMedium("TPC_Epoxy1");
  //
  auto* tov1 = new TGeoVolume("TPC_OCV1", to1, sm1);
  auto* tov2 = new TGeoVolume("TPC_OCV2", to2, sm2);
  auto* tov3 = new TGeoVolume("TPC_OCV3", to3, sm3);
  auto* tov4 = new TGeoVolume("TPC_OCV4", to4, sm4);
  auto* togv5 = new TGeoVolume("TPC_OCVG5", tog5, sm1);
  //
  TGeoMedium* mhs = gGeoManager->GetMedium("TPC_Steel");
  TGeoMedium* m12 = gGeoManager->GetMedium("TPC_Water");
  //-------------------------------------------------------
  //  Tpc Outer Field Cage
  //  daughters - composite (sandwich)
  //-------------------------------------------------------

  auto* tofc = new TGeoPcon(0., 360., 6);
  //
  tofc->DefineSection(0, -253.6, 258., 269.6);
  tofc->DefineSection(1, -250.6, 258., 269.6);
  //
  tofc->DefineSection(2, -250.6, 258., 260.0676);
  tofc->DefineSection(3, 250.6, 258., 260.0676);
  //
  tofc->DefineSection(4, 250.6, 258., 275.5);
  tofc->DefineSection(5, 253.6, 258., 275.5);
  //
  auto* v4 = new TGeoVolume("TPC_TOFC", tofc, m3);
  // sandwich
  auto* tf1 = new TGeoTubeSeg(258.0, 260.0676, 252.1, 0., 59.9);    // tedlar
  auto* tf2 = new TGeoTubeSeg(258.0038, 260.0638, 252.1, 0., 59.9); // prepreg3
  auto* tf3 = new TGeoTubeSeg(258.0338, 260.0338, 252.1, 0., 59.9); // nomex
  auto* tfg4 = new TGeoTubeSeg(258.0, 260.0676, 252.1, 59.9, 60.);  // epoxy glue
  //
  TGeoMedium* sm5 = gGeoManager->GetMedium("TPC_Prepreg3");
  //
  auto* tf1v = new TGeoVolume("TPC_OFC1", tf1, sm2);
  auto* tf2v = new TGeoVolume("TPC_OFC2", tf2, sm5);
  auto* tf3v = new TGeoVolume("TPC_OFC3", tf3, sm4);
  auto* tfg4v = new TGeoVolume("TPC_OFCG4", tfg4, smep);
  //
  // outer part - positioning
  //
  tov1->AddNode(tov2, 1);
  tov2->AddNode(tov3, 1);
  tov3->AddNode(tov4, 1); // ocv
  //
  tf1v->AddNode(tf2v, 1);
  tf2v->AddNode(tf3v, 1); // ofc
  //
  auto* t200 = new TGeoVolumeAssembly("TPC_OCVSEG");
  auto* t300 = new TGeoVolumeAssembly("TPC_OFCSEG");
  //
  // assembly OCV and OFC
  //
  // 1st - no rotation
  t200->AddNode(tov1, 1);
  t200->AddNode(togv5, 1);
  t300->AddNode(tf1v, 1);
  t300->AddNode(tfg4v, 1);
  // 2nd - rotation 60 deg
  segrot = new TGeoRotation();
  segrot->RotateZ(60.);
  t200->AddNode(tov1, 2, segrot);
  t200->AddNode(togv5, 2, segrot);
  t300->AddNode(tf1v, 2, segrot);
  t300->AddNode(tfg4v, 2, segrot);
  // 3rd rotation 120 deg
  segrot = new TGeoRotation();
  segrot->RotateZ(120.);
  t200->AddNode(tov1, 3, segrot);
  t200->AddNode(togv5, 3, segrot);
  t300->AddNode(tf1v, 3, segrot);
  t300->AddNode(tfg4v, 3, segrot);
  // 4th rotation 180 deg
  segrot = new TGeoRotation();
  segrot->RotateZ(180.);
  t200->AddNode(tov1, 4, segrot);
  t200->AddNode(togv5, 4, segrot);
  t300->AddNode(tf1v, 4, segrot);
  t300->AddNode(tfg4v, 4, segrot);
  // 5th rotation 240 deg
  segrot = new TGeoRotation();
  segrot->RotateZ(240.);
  t200->AddNode(tov1, 5, segrot);
  t200->AddNode(togv5, 5, segrot);
  t300->AddNode(tf1v, 5, segrot);
  t300->AddNode(tfg4v, 5, segrot);
  // 6th rotation 300 deg
  segrot = new TGeoRotation();
  segrot->RotateZ(300.);
  t200->AddNode(tov1, 6, segrot);
  t200->AddNode(togv5, 6, segrot);
  t300->AddNode(tf1v, 6, segrot);
  t300->AddNode(tfg4v, 6, segrot);
  //
  v3->AddNode(t200, 1, new TGeoTranslation(0., 0., -1.5));
  v4->AddNode(t300, 1);
  //
  v2->AddNode(v3, 1);
  v2->AddNode(v4, 1);
  //
  v1->AddNode(v2, 1);
  //--------------------------------------------------------------------
  // Tpc Inner INsulator (CO2)
  // the cones, the central drum and the inner f.c. sandwich with a piece
  // of the flane will be placed in the TPC
  //--------------------------------------------------------------------
  auto* tpci = new TGeoPcon(0., 360., 4);
  //
  tpci->DefineSection(0, -253.6, 68.4, 76.6774);
  tpci->DefineSection(1, -74.0, 61.2, 76.6774);
  //
  tpci->DefineSection(2, 74.0, 61.2, 76.6774);
  //
  tpci->DefineSection(3, 253.6, 65.9, 76.6774);
  //
  auto* v5 = new TGeoVolume("TPC_INI", tpci, m2);
  //
  // now the inner field cage - only part of flanges (2 copies)
  //
  auto* tif1 = new TGeoTube(69.9, 76.6774, 1.5);
  auto* v6 = new TGeoVolume("TPC_IFC1", tif1, m3);
  //
  //---------------------------------------------------------
  // Tpc Inner Containment vessel - Muon side
  //---------------------------------------------------------
  auto* tcms = new TGeoPcon(0., 360., 10);
  //
  tcms->DefineSection(0, -259.1, 68.1, 74.2);
  tcms->DefineSection(1, -253.6, 68.1, 74.2);
  //
  tcms->DefineSection(2, -253.6, 68.1, 68.4);
  tcms->DefineSection(3, -74.0, 60.9, 61.2);
  //
  tcms->DefineSection(4, -74.0, 60.1, 61.2);
  tcms->DefineSection(5, -73.3, 60.1, 61.2);
  //
  tcms->DefineSection(6, -73.3, 56.9, 61.2);
  tcms->DefineSection(7, -73.0, 56.9, 61.2);
  //
  tcms->DefineSection(8, -73.0, 56.9, 58.8);
  tcms->DefineSection(9, -71.3, 56.9, 58.8);
  //
  auto* v7 = new TGeoVolume("TPC_ICVM", tcms, m3);
  //------------------------------------------------
  //  Heat screen muon side
  //------------------------------------------------

  auto* thsm = new TGeoCone(89.8, 67.88, 68.1, 60.68, 60.9);
  auto* thsmw = new TGeoCone(89.8, 67.94, 68.04, 60.74, 60.84);
  auto* hvsm = new TGeoVolume("TPC_HSM", thsm, mhs);    // steel
  auto* hvsmw = new TGeoVolume("TPC_HSMW", thsmw, m12); // water
  // assembly heat screen muon
  hvsm->AddNode(hvsmw, 1);
  //-----------------------------------------------
  // inner containment vessel - shaft side
  //-----------------------------------------------
  auto* tcss = new TGeoPcon(0., 360., 10);
  //
  tcss->DefineSection(0, 71.3, 56.9, 58.8);
  tcss->DefineSection(1, 73.0, 56.9, 58.8);
  //
  tcss->DefineSection(2, 73.0, 56.9, 61.2);
  tcss->DefineSection(3, 73.3, 56.9, 61.2);
  //
  tcss->DefineSection(4, 73.3, 60.1, 61.2);
  tcss->DefineSection(5, 74.0, 60.1, 61.2);
  //
  tcss->DefineSection(6, 74.0, 60.9, 61.2);
  tcss->DefineSection(7, 253.6, 65.6, 65.9);
  //
  tcss->DefineSection(8, 253.6, 65.6, 74.2);
  tcss->DefineSection(9, 258.1, 65.6, 74.2);
  //
  auto* v8 = new TGeoVolume("TPC_ICVS", tcss, m3);
  //-------------------------------------------------
  //  Heat screen shaft side
  //--------------------------------------------------
  auto* thss = new TGeoCone(89.8, 60.68, 60.9, 65.38, 65.6);
  auto* thssw = new TGeoCone(89.8, 60.74, 60.84, 65.44, 65.54);
  auto* hvss = new TGeoVolume("TPC_HSS", thss, mhs);    // steel
  auto* hvssw = new TGeoVolume("TPC_HSSW", thssw, m12); // water
  // assembly heat screen shaft
  hvss->AddNode(hvssw, 1);
  //-----------------------------------------------
  //  Inner field cage
  //  define 4 parts and make an assembly
  //-----------------------------------------------
  // part1 - Al - 2 copies
  auto* t1 = new TGeoTube(76.6774, 78.845, 0.75);
  auto* tv1 = new TGeoVolume("TPC_IFC2", t1, m3);
  // sandwich - outermost parts - 2 copies
  //
  // segment outermost
  //
  auto* t2 = new TGeoTubeSeg(76.6774, 78.845, 74.175, 350., 109.4);     // tedlar 38 microns
  auto* t3 = new TGeoTubeSeg(76.6812, 78.8412, 74.175, 350., 109.4);    // prepreg2 500 microns
  auto* t4 = new TGeoTubeSeg(76.7312, 78.7912, 74.175, 350., 109.4);    // prepreg3 300 microns
  auto* t5 = new TGeoTubeSeg(76.7612, 78.7612, 74.175, 350., 109.4);    // nomex 2 cm
  auto* tepox1 = new TGeoTubeSeg(76.6774, 78.845, 74.175, 109.4, 110.); // epoxy
  auto* tpr1 = new TGeoTubeSeg(78.845, 78.885, 74.175, 109., 111.);

  // volumes for the outer part
  auto* tv2 = new TGeoVolume("TPC_IFC3", t2, sm2);
  auto* tv3 = new TGeoVolume("TPC_IFC4", t3, sm3);
  auto* tv4 = new TGeoVolume("TPC_IFC5", t4, sm5);
  auto* tv5 = new TGeoVolume("TPC_IFC6", t5, sm4);
  auto* tvep1 = new TGeoVolume("TPC_IFEPOX1", tepox1, smep);
  auto* tvpr1 = new TGeoVolume("TPC_PRSTR1", tpr1, sm2);
  //
  // middle parts - 2 copies
  //
  // segment middle
  //
  auto* t6 = new TGeoTubeSeg(76.6774, 78.795, 5., 350., 109.4);     // tedlar 38 microns
  auto* t7 = new TGeoTubeSeg(76.6812, 78.7912, 5., 350., 109.4);    // prepreg2 250 microns
  auto* t8 = new TGeoTubeSeg(76.7062, 78.7662, 5., 350., 109.4);    // prepreg3 300 microns
  auto* t9 = new TGeoTubeSeg(76.7362, 78.7362, 5., 350., 109.4);    // nomex 2 cm
  auto* tepox2 = new TGeoTubeSeg(76.6774, 78.795, 5., 109.4, 110.); // epoxy
  auto* tpr2 = new TGeoTubeSeg(78.795, 78.835, 5., 109., 111.);
  // volumes for the middle part
  auto* tv6 = new TGeoVolume("TPC_IFC7", t6, sm2);
  auto* tv7 = new TGeoVolume("TPC_IFC8", t7, sm3);
  auto* tv8 = new TGeoVolume("TPC_IFC9", t8, sm5);
  auto* tv9 = new TGeoVolume("TPC_IFC10", t9, sm4);
  auto* tvep2 = new TGeoVolume("TPC_IFEPOX2", tepox2, smep);
  auto* tvpr2 = new TGeoVolume("TPC_PRSTR2", tpr2, sm2);
  // central part - 1 copy
  //
  // segment central part
  //
  auto* t10 = new TGeoTubeSeg(76.6774, 78.785, 93.75, 350., 109.4);    // tedlar 38 microns
  auto* t11 = new TGeoTubeSeg(76.6812, 78.7812, 93.75, 350., 109.4);   // prepreg3 500 microns
  auto* t12 = new TGeoTubeSeg(76.7312, 78.7312, 93.75, 350., 109.4);   // nomex 2 cm
  auto* tepox3 = new TGeoTubeSeg(76.6774, 78.785, 93.75, 109.4, 110.); // epoxy
  auto* tpr3 = new TGeoTubeSeg(78.785, 78.825, 93.75, 109., 111.);
  // volumes for the central part
  auto* tv10 = new TGeoVolume("TPC_IFC11", t10, sm2);
  auto* tv11 = new TGeoVolume("TPC_IFC12", t11, sm5);
  auto* tv12 = new TGeoVolume("TPC_IFC13", t12, sm4);
  auto* tvep3 = new TGeoVolume("TPC_IFEPOX3", tepox3, smep);
  auto* tvpr3 = new TGeoVolume("TPC_PRSTR3", tpr3, sm2);
  //
  // creating a sandwich for the outer par,t tv2 is the mother
  //
  tv2->AddNode(tv3, 1);
  tv3->AddNode(tv4, 1);
  tv4->AddNode(tv5, 1);
  //
  // creating a sandwich for the middle part, tv6 is the mother
  //
  tv6->AddNode(tv7, 1);
  tv7->AddNode(tv8, 1);
  tv8->AddNode(tv9, 1);
  //
  // creating a sandwich for the central part, tv10 is the mother
  //
  tv10->AddNode(tv11, 1);
  tv11->AddNode(tv12, 1);
  //
  auto* tv100 = new TGeoVolumeAssembly("TPC_IFC"); // ifc itself - 3 segments

  //
  // first segment - no rotation
  //
  // central
  tv100->AddNode(tv10, 1);  // sandwich
  tv100->AddNode(tvep3, 1); // epoxy
  tv100->AddNode(tvpr3, 1); // prepreg strip
  // middle
  tv100->AddNode(tv6, 1, new TGeoTranslation(0., 0., -98.75));   // sandwich1
  tv100->AddNode(tv6, 2, new TGeoTranslation(0., 0., 98.75));    // sandwich2
  tv100->AddNode(tvep2, 1, new TGeoTranslation(0., 0., -98.75)); // epoxy
  tv100->AddNode(tvep2, 2, new TGeoTranslation(0., 0., 98.75));  // epoxy
  tv100->AddNode(tvpr2, 1, new TGeoTranslation(0., 0., -98.75)); // prepreg strip
  tv100->AddNode(tvpr2, 2, new TGeoTranslation(0., 0., 98.75));
  // outer
  tv100->AddNode(tv2, 1, new TGeoTranslation(0., 0., -177.925)); // sandwich
  tv100->AddNode(tv2, 2, new TGeoTranslation(0., 0., 177.925));
  tv100->AddNode(tvep1, 1, new TGeoTranslation(0., 0., -177.925)); // epoxy
  tv100->AddNode(tvep1, 2, new TGeoTranslation(0., 0., 177.925));
  tv100->AddNode(tvpr1, 1, new TGeoTranslation(0., 0., -177.925)); // prepreg strip
  tv100->AddNode(tvpr1, 2, new TGeoTranslation(0., 0., -177.925));
  //
  // second segment - rotation 120 deg.
  //
  segrot = new TGeoRotation();
  segrot->RotateZ(120.);
  //
  // central
  tv100->AddNode(tv10, 2, segrot);  // sandwich
  tv100->AddNode(tvep3, 2, segrot); // epoxy
  tv100->AddNode(tvpr3, 2, segrot); // prepreg strip
  // middle
  tv100->AddNode(tv6, 3, new TGeoCombiTrans(0., 0., -98.75, segrot));   // sandwich1
  tv100->AddNode(tv6, 4, new TGeoCombiTrans(0., 0., 98.75, segrot));    // sandwich2
  tv100->AddNode(tvep2, 3, new TGeoCombiTrans(0., 0., -98.75, segrot)); // epoxy
  tv100->AddNode(tvep2, 4, new TGeoCombiTrans(0., 0., 98.75, segrot));  // epoxy
  tv100->AddNode(tvpr2, 3, new TGeoCombiTrans(0., 0., -98.75, segrot)); // prepreg strip
  tv100->AddNode(tvpr2, 4, new TGeoCombiTrans(0., 0., 98.75, segrot));
  // outer
  tv100->AddNode(tv2, 3, new TGeoCombiTrans(0., 0., -177.925, segrot)); // sandwich
  tv100->AddNode(tv2, 4, new TGeoCombiTrans(0., 0., 177.925, segrot));
  tv100->AddNode(tvep1, 3, new TGeoCombiTrans(0., 0., -177.925, segrot)); // epoxy
  tv100->AddNode(tvep1, 4, new TGeoCombiTrans(0., 0., 177.925, segrot));
  tv100->AddNode(tvpr1, 3, new TGeoCombiTrans(0., 0., -177.925, segrot)); // prepreg strip
  tv100->AddNode(tvpr1, 4, new TGeoCombiTrans(0., 0., 177.925, segrot));
  //
  //  third segment - rotation 240 deg.
  //
  segrot = new TGeoRotation();
  segrot->RotateZ(240.);
  //
  // central
  tv100->AddNode(tv10, 3, segrot);  // sandwich
  tv100->AddNode(tvep3, 3, segrot); // epoxy
  tv100->AddNode(tvpr3, 3, segrot); // prepreg strip
  // middle
  tv100->AddNode(tv6, 5, new TGeoCombiTrans(0., 0., -98.75, segrot));   // sandwich1
  tv100->AddNode(tv6, 6, new TGeoCombiTrans(0., 0., 98.75, segrot));    // sandwich2
  tv100->AddNode(tvep2, 5, new TGeoCombiTrans(0., 0., -98.75, segrot)); // epoxy
  tv100->AddNode(tvep2, 6, new TGeoCombiTrans(0., 0., 98.75, segrot));  // epoxy
  tv100->AddNode(tvpr2, 5, new TGeoCombiTrans(0., 0., -98.75, segrot)); // prepreg strip
  tv100->AddNode(tvpr2, 6, new TGeoCombiTrans(0., 0., 98.75, segrot));
  // outer
  tv100->AddNode(tv2, 5, new TGeoCombiTrans(0., 0., -177.925, segrot)); // sandwich
  tv100->AddNode(tv2, 6, new TGeoCombiTrans(0., 0., 177.925, segrot));
  tv100->AddNode(tvep1, 5, new TGeoCombiTrans(0., 0., -177.925, segrot)); // epoxy
  tv100->AddNode(tvep1, 6, new TGeoCombiTrans(0., 0., 177.925, segrot));
  tv100->AddNode(tvpr1, 5, new TGeoCombiTrans(0., 0., -177.925, segrot)); // prepreg strip
  tv100->AddNode(tvpr1, 6, new TGeoCombiTrans(0., 0., 177.925, segrot));
  // Al parts - rings
  tv100->AddNode(tv1, 1, new TGeoTranslation(0., 0., -252.85));
  tv100->AddNode(tv1, 2, new TGeoTranslation(0., 0., 252.85));
  //
  v5->AddNode(v6, 1, new TGeoTranslation(0., 0., -252.1));
  v5->AddNode(v6, 2, new TGeoTranslation(0., 0., 252.1));
  v1->AddNode(v5, 1);
  v1->AddNode(v7, 1);
  v1->AddNode(v8, 1);
  v1->AddNode(hvsm, 1, new TGeoTranslation(0., 0., -163.8));
  v1->AddNode(hvss, 1, new TGeoTranslation(0., 0., 163.8));
  v9->AddNode(tv100, 1);
  //
  // central drum
  //
  // flange + sandwich
  //
  auto* cfl = new TGeoPcon(0., 360., 6);
  cfl->DefineSection(0, -71.1, 59.7, 61.2);
  cfl->DefineSection(1, -68.6, 59.7, 61.2);
  //
  cfl->DefineSection(2, -68.6, 60.6124, 61.2);
  cfl->DefineSection(3, 68.6, 60.6124, 61.2);
  //
  cfl->DefineSection(4, 68.6, 59.7, 61.2);
  cfl->DefineSection(5, 71.1, 59.7, 61.2);
  //
  auto* cflv = new TGeoVolume("TPC_CDR", cfl, m3);
  // sandwich
  auto* cd1 = new TGeoTubeSeg(60.6224, 61.19, 71.1, 0.2, 119.2);
  auto* cd2 = new TGeoTubeSeg(60.6262, 61.1862, 71.1, 0.2, 119.2);
  auto* cd3 = new TGeoTubeSeg(60.6462, 61.1662, 71.1, 0.2, 119.2);
  auto* cd4 = new TGeoTubeSeg(60.6562, 61.1562, 71.1, 0.2, 119.2);
  auto* tepox4 = new TGeoTubeSeg(60.6224, 61.19, 71.1, 359.8, 0.8);
  //
  TGeoMedium* sm6 = gGeoManager->GetMedium("TPC_Prepreg1");
  TGeoMedium* sm8 = gGeoManager->GetMedium("TPC_Epoxyfm");
  auto* cd1v = new TGeoVolume("TPC_CDR1", cd1, sm2); // tedlar
  auto* cd2v = new TGeoVolume("TPC_CDR2", cd2, sm6); // prepreg1
  auto* cd3v = new TGeoVolume("TPC_CDR3", cd3, sm8); // epoxy film
  auto* cd4v = new TGeoVolume("TPC_CDR4", cd4, sm4); // nomex
  auto* tvep4 = new TGeoVolume("TPC_IFEPOX4", tepox4, smep);

  //
  // seals for central drum 2 copies
  //
  auto* cs = new TGeoTube(56.9, 61.2, 0.1);
  TGeoMedium* sm7 = gGeoManager->GetMedium("TPC_Mylar");
  auto* csv = new TGeoVolume("TPC_CDRS", cs, sm7);
  v1->AddNode(csv, 1, new TGeoTranslation(0., 0., -71.2));
  v1->AddNode(csv, 2, new TGeoTranslation(0., 0., 71.2));
  //
  // seal collars
  auto* se = new TGeoPcon(0., 360., 6);
  se->DefineSection(0, -72.8, 59.7, 61.2);
  se->DefineSection(1, -72.3, 59.7, 61.2);
  //
  se->DefineSection(2, -72.3, 58.85, 61.2);
  se->DefineSection(3, -71.6, 58.85, 61.2);
  //
  se->DefineSection(4, -71.6, 59.7, 61.2);
  se->DefineSection(5, -71.3, 59.7, 61.2);
  //
  auto* sev = new TGeoVolume("TPC_CDCE", se, m3);
  //
  auto* si = new TGeoTube(56.9, 58.8, 1.);
  auto* siv = new TGeoVolume("TPC_CDCI", si, m3);
  //
  // define reflection matrix
  //
  auto* ref = new TGeoRotation("ref", 90., 0., 90., 90., 180., 0.);
  //
  cd1v->AddNode(cd2v, 1);
  cd2v->AddNode(cd3v, 1);
  cd3v->AddNode(cd4v, 1); // sandwich
  // first segment
  cflv->AddNode(cd1v, 1);
  cflv->AddNode(tvep4, 1);
  // second segment
  segrot = new TGeoRotation();
  segrot->RotateZ(120.);
  cflv->AddNode(cd1v, 2, segrot);
  cflv->AddNode(tvep4, 2, segrot);
  // third segment
  segrot = new TGeoRotation();
  segrot->RotateZ(240.);
  cflv->AddNode(cd1v, 3, segrot);
  cflv->AddNode(tvep4, 3, segrot);
  //
  v1->AddNode(siv, 1, new TGeoTranslation(0., 0., -69.9));
  v1->AddNode(siv, 2, new TGeoTranslation(0., 0., 69.9));
  v1->AddNode(sev, 1);
  v1->AddNode(sev, 2, ref);
  v1->AddNode(cflv, 1);
  //
  // central membrane - 2 rings and a mylar membrane - assembly
  //
  auto* ih = new TGeoTube(81.05, 84.05, 0.3);
  auto* oh = new TGeoTube(250., 256., 0.5);
  auto* mem = new TGeoTube(84.05, 250., 0.00115);

  //
  TGeoMedium* m4 = gGeoManager->GetMedium("TPC_G10");
  //
  auto* ihv = new TGeoVolume("TPC_IHVH", ih, m3);
  auto* ohv = new TGeoVolume("TPC_OHVH", oh, m3);

  auto* memv = new TGeoVolume("TPC_HV", mem, sm7);
  //
  auto* cm = new TGeoVolumeAssembly("TPC_HVMEM");
  cm->AddNode(ihv, 1);
  cm->AddNode(ohv, 1);
  cm->AddNode(memv, 1);

  v9->AddNode(cm, 1);
  //
  // end caps - they are make as an assembly of single segments
  // containing both readout chambers
  //
  Double_t openingAngle = 10. * TMath::DegToRad();
  Double_t thick = 1.5; // rib
  Double_t shift = thick / TMath::Sin(openingAngle);
  //
  Double_t lowEdge = 86.3; // hole in the wheel
  Double_t upEdge = 240.4; // hole in the wheel
  //
  new TGeoTubeSeg("sec", 74.5, 264.4, 3., 0., 20.);
  //
  auto* hole = new TGeoPgon("hole", 0., 20., 1, 4);
  //
  hole->DefineSection(0, -3.5, lowEdge - shift, upEdge - shift);
  hole->DefineSection(1, -1.5, lowEdge - shift, upEdge - shift);
  //
  hole->DefineSection(2, -1.5, lowEdge - shift, upEdge + 3. - shift);
  hole->DefineSection(3, 3.5, lowEdge - shift, upEdge + 3. - shift);
  //
  Double_t ys = shift * TMath::Sin(openingAngle);
  Double_t xs = shift * TMath::Cos(openingAngle);
  auto* tr = new TGeoTranslation("tr", xs, ys, 0.);
  tr->RegisterYourself();
  auto* chamber = new TGeoCompositeShape("sec-hole:tr");
  auto* sv = new TGeoVolume("TPC_WSEG", chamber, m3);
  auto* bar = new TGeoPgon("bar", 0., 20., 1, 2);
  bar->DefineSection(0, -3., 131.5 - shift, 136.5 - shift);
  bar->DefineSection(1, 1.5, 131.5 - shift, 136.5 - shift);
  auto* barv = new TGeoVolume("TPC_WBAR", bar, m3);
  auto* ch = new TGeoVolumeAssembly("TPC_WCH"); // empty segment
  //
  ch->AddNode(sv, 1);
  ch->AddNode(barv, 1, tr);
  //
  // readout chambers
  //
  // IROC first
  //
  TGeoMedium* m6 = gGeoManager->GetMedium("TPC_Makrolon");
  //
  auto* ibody = new TGeoTrd1(13.8742, 21.3328, 4.29, 21.15);
  auto* ibdv = new TGeoVolume("TPC_IROCB", ibody, m3);
  // empty space
  auto* emp = new TGeoTrd1(12.3742, 19.8328, 4.05, 19.65);
  auto* empv = new TGeoVolume("TPC_IROCE", emp, m1);
  ibdv->AddNode(empv, 1, new TGeoTranslation(0., -0.24, 0.));
  // bars
  Double_t tga = (19.8328 - 12.3742) / 39.3;
  Double_t xmin, xmax;
  // 1
  xmin = 9.65 * tga + 12.3742;
  xmax = 10.05 * tga + 12.3742;
  //
  auto* ib1 = new TGeoTrd1(xmin, xmax, 2.06, 0.2);
  auto* ib1v = new TGeoVolume("TPC_IRB1", ib1, m3);
  empv->AddNode(ib1v, 1, new TGeoTranslation("tt1", 0., 1.99, -9.8));
  // 2
  xmin = 19.5 * tga + 12.3742;
  xmax = 19.9 * tga + 12.3742;
  //
  auto* ib2 = new TGeoTrd1(xmin, xmax, 2.06, 0.2);
  auto* ib2v = new TGeoVolume("TPC_TRB2", ib2, m3);
  empv->AddNode(ib2v, 1, new TGeoTranslation(0., 1.99, 0.05));
  // 3
  xmin = 29.35 * tga + 12.3742;
  xmax = 29.75 * tga + 12.3742;
  //
  auto* ib3 = new TGeoTrd1(xmin, xmax, 2.06, 0.2);
  auto* ib3v = new TGeoVolume("TPC_IRB3", ib3, m3);
  empv->AddNode(ib3v, 1, new TGeoTranslation(0., 1.99, 9.9));
  //
  // connectors alu body & strong back
  //
  auto* conn = new TGeoBBox(0.4, 0.24, 1.937);         //connectors alu body
  auto* connv = new TGeoVolume("TPC_RCCON", conn, m6); //makrolon
  //
  auto* connb = new TGeoBBox(0.5, 0.25, 2.375); //connectors strong back
  auto* connbv = new TGeoVolume("TPC_RCCONB", connb, m6);
  //
  // strong back
  //
  auto* icsb = new TGeoTrd1(14.5974, 23.3521, 0.25, 24.825);
  auto* icsbv = new TGeoVolume("TPC_ISB", icsb, m4);
  //
  // file with positions of connectors
  //
  TString fileName(gSystem->Getenv("VMCWORKDIR"));
  fileName += "/Detectors/Geometry/TPC/conn_iroc.dat";
  ifstream in;
  in.open(fileName.Data(), ios_base::in); // asci file
  if (!in.is_open()) {
    LOG(FATAL) << "Cannot open input file : " << fileName.Data();
  }
  for (Int_t i = 0; i < 132; i++) {
    Double_t x, z, ang;
    in >> ang >> x >> z;
    //
    ang = -ang;
    z -= 1102.; // changing the reference frame from the beam to the volume
    z *= 0.1;
    x *= 0.1;
    //
    auto* rrr = new TGeoRotation();
    rrr->RotateY(ang);
    //
    ibdv->AddNode(connv, i + 1, new TGeoCombiTrans(x, 4.05, z, rrr));         //connectors alu body
    icsbv->AddNode(connbv, i + 1, new TGeoCombiTrans(x, 0., z + 1.725, rrr)); //connectors strong back
  }
  in.close();
  //
  // "cap"
  new TGeoTrd1("icap", 14.5974, 23.3521, 1.19, 24.825);
  // "hole"
  new TGeoTrd1("ihole", 13.8742, 21.3328, 1.2, 21.15);
  auto* tr1 = new TGeoTranslation("tr1", 0., 0., 1.725);
  tr1->RegisterYourself();
  auto* ic = new TGeoCompositeShape("icap-ihole:tr1");
  auto* icv = new TGeoVolume("TPC_IRCAP", ic, m3);
  //
  // pad plane G10 3.2 mm thick
  //
  auto* icpp = new TGeoTrd1(14.5974, 23.3521, 0.16, 24.825);
  auto* icppv = new TGeoVolume("TPC_IPP", icpp, m4); //pad plane
  //
  // gem
  //
  new TGeoTrd1("igem", 14.5974, 23.3521, 0.1, 24.825);
  new TGeoTrd1("igemh", 14.5974 - .5, 23.3521 - .5, 0.11, 24.825 - .5);
  auto* icgem = new TGeoCompositeShape("igem-igemh");
  auto* icgemv = new TGeoVolume("TPC_ICGEM", icgem, m4);
  //
  // assembly of the iroc
  //
  auto* iroc = new TGeoVolumeAssembly("TPC_IROC");
  //
  iroc->AddNode(ibdv, 1);                                          //main body
  iroc->AddNode(icv, 1, new TGeoTranslation(0., 3.1, -1.725));     //cap
  iroc->AddNode(icsbv, 1, new TGeoTranslation(0., 4.54, -1.725));  //strong back
  iroc->AddNode(icppv, 1, new TGeoTranslation(0., 4.95, -1.725));  //pad plane
  iroc->AddNode(icgemv, 1, new TGeoTranslation(0., 5.21, -1.725)); //gem
  //
  // OROC
  //
  auto* obody = new TGeoTrd1(22.2938, 40.5084, 4.29, 51.65);
  auto* obdv = new TGeoVolume("TPC_OROCB", obody, m3);
  auto* oemp = new TGeoTrd1(20.7938, 39.0084, 3.89, 50.15);
  auto* oempv = new TGeoVolume("TPC_OROCE", oemp, m1);
  obdv->AddNode(oempv, 1, new TGeoTranslation(0., -0.4, 0.));
  //
  // horizontal bars
  //
  tga = (39.0084 - 20.7938) / 100.3;
  xmin = tga * 14.2 + 20.7938;
  xmax = tga * 14.6 + 20.7938;
  auto* ob1 = new TGeoTrd1(xmin, xmax, 2.94, 0.2);
  auto* ob1v = new TGeoVolume("TPC_ORB1", ob1, m3);
  //
  xmin = 30.4 * tga + 20.7938;
  xmax = 32.1 * tga + 20.7938;
  auto* ob2 = new TGeoTrd1(xmin, xmax, 2.94, 1.05);
  auto* ob2v = new TGeoVolume("TPC_ORB2", ob2, m3);
  //
  xmin = 51.5 * tga + 20.7938;
  xmax = 51.9 * tga + 20.7938;
  auto* ob3 = new TGeoTrd1(xmin, xmax, 2.94, 0.2);
  auto* ob3v = new TGeoVolume("TPC_ORB3", ob3, m3);
  //
  xmin = 68.5 * tga + 20.7938;
  xmax = 70.6 * tga + 20.7938;
  auto* ob4 = new TGeoTrd1(xmin, xmax, 2.94, 1.05);
  auto* ob4v = new TGeoVolume("TPC_ORB4", ob4, m3);
  //
  xmin = 89.9 * tga + 20.7938;
  xmax = 90.3 * tga + 20.7938;
  auto* ob5 = new TGeoTrd1(xmin, xmax, 2.94, 0.2);
  auto* ob5v = new TGeoVolume("TPC_ORB5", ob5, m3);
  //
  oempv->AddNode(ob1v, 1, new TGeoTranslation(0., 0.59, -35.75));
  oempv->AddNode(ob2v, 1, new TGeoTranslation(0., 0.59, -18.7));
  oempv->AddNode(ob3v, 1, new TGeoTranslation(0., 0.59, 1.55));
  oempv->AddNode(ob4v, 1, new TGeoTranslation(0., 0.59, 19.4));
  oempv->AddNode(ob5v, 1, new TGeoTranslation(0., 0.59, 39.95));
  //
  // connectors, identical as for iroc, but I prefer to have separate volumes for better control
  //
  auto* conno = new TGeoBBox(0.4, 0.4, 1.937);            //alu body
  auto* connov = new TGeoVolume("TPC_RCCONO", conno, m6); //makrolon
  //
  auto* connob = new TGeoBBox(0.5, 0.25, 2.375);
  auto* connobv = new TGeoVolume("TPC_RCCONOB", connob, m6); //strong back
  //
  // cap
  //
  new TGeoTrd1("ocap", 23.3875, 43.524, 1.19, 57.1);
  new TGeoTrd1("ohole", 22.2938, 40.5084, 1.19, 51.65);
  auto* tr5 = new TGeoTranslation("tr5", 0., 0., -2.15);
  tr5->RegisterYourself();
  auto* oc = new TGeoCompositeShape("ocap-ohole:tr5");
  auto* ocv = new TGeoVolume("TPC_ORCAP", oc, m3);
  //
  // stron back 5 mm
  //
  auto* osb = new TGeoTrd1(23.3874, 43.524, 0.25, 57.1);
  auto* osbv = new TGeoVolume("TPC_OSB", osb, m4);
  //
  // pad plane 3.2 mm
  //
  auto* opp = new TGeoTrd1(23.3874, 43.524, 0.16, 57.1);
  auto* oppv = new TGeoVolume("TPC_OPP", opp, m4);
  //
  // gem
  //
  new TGeoTrd1("ogem", 23.3874, 43.524, 0.1, 57.1);
  //
  // ogemh1 - first "hole" ends at z = 36.25, ogemh2 - second "hole" starts at z = 74.15
  //
  new TGeoTrd1("ogemh1", 22.548, 28.579, 0.1, 17.625); //ogemh1 - lower hole
  new TGeoTrd1("ogemh2", 28.949, 35.297, 0.1, 18.45);  //ogemh2 - middle hole
  new TGeoTrd1("ogemh3", 35.667, 42.332, 0.1, 19.425); //ogemh3 - upper hole
  //
  auto* tr2 = new TGeoTranslation("tr2", 0., 0., 18.125 - 57.1);
  auto* tr3 = new TGeoTranslation("tr3", 0., 0., 55.2 - 57.1);
  auto* tr4 = new TGeoTranslation("tr4", 0., 0., 94.175 - 57.1);
  tr2->RegisterYourself();
  tr3->RegisterYourself();
  tr4->RegisterYourself();
  auto* ocgem = new TGeoCompositeShape("ogem-ogemh1:tr2-ogemh2:tr3-ogemh3:tr4");
  auto* ocgemv = new TGeoVolume("TPC_OCGEM", ocgem, m4);
  //
  // holes for connectors
  //
  fileName = gSystem->Getenv("VMCWORKDIR");
  fileName += "/Detectors/Geometry/TPC/conn_oroc.dat";
  in.open(fileName.Data(), ios_base::in); // asci file
  if (!in.is_open()) {
    LOG(FATAL) << "Cannot open input file : " << fileName.Data();
  }
  for (Int_t i = 0; i < 232; i++) {
    Double_t x, z, ang;
    in >> ang >> x >> z;
    ang = -ang;
    z -= 1884.5;
    z *= 0.1;
    x *= 0.1;
    //
    auto* rrr = new TGeoRotation();
    rrr->RotateY(ang);
    obdv->AddNode(connov, i + 1, new TGeoCombiTrans(x, 3.89, z, rrr));
    osbv->AddNode(connobv, i + 1, new TGeoCombiTrans(x, 0., z - 2.15, rrr));
  }
  in.close();
  //
  auto* oroc = new TGeoVolumeAssembly("TPC_OROC");
  //
  oroc->AddNode(obdv, 1);
  oroc->AddNode(ocv, 1, new TGeoTranslation(0., 3.1, 2.15));
  oroc->AddNode(osbv, 1, new TGeoTranslation(0., 4.54, 2.15));
  oroc->AddNode(oppv, 1, new TGeoTranslation(0., 4.95, 2.15));
  oroc->AddNode(ocgemv, 1, new TGeoTranslation(0., 5.21, 2.15));
  //
  // now iroc and oroc are placed into a sector...
  //
  auto* secta = new TGeoVolumeAssembly("TPC_SECT"); // a-side
  auto* sectc = new TGeoVolumeAssembly("TPC_SECT"); // c-side
  TGeoRotation rot1("rot1", 90., 90., 0.);
  TGeoRotation rot2("rot2");
  rot2.RotateY(10.);
  auto* rot = new TGeoRotation("rot");
  *rot = rot1 * rot2;
  //
  Double_t x0, y0;
  //
  // a and c sides separately because of possible different z-coordinate of first gem
  //
  x0 = 110.2 * TMath::Cos(openingAngle);
  y0 = 110.2 * TMath::Sin(openingAngle);
  auto* combi1a = new TGeoCombiTrans("combi1", x0, y0, 1.09, rot); // a-side
  auto* combi1c = new TGeoCombiTrans("combi1", x0, y0, 1.09, rot); // c-side
  x0 = 188.45 * TMath::Cos(openingAngle);
  y0 = 188.45 * TMath::Sin(openingAngle);
  auto* combi2a = new TGeoCombiTrans("combi2", x0, y0, 1.09, rot); // a-side
  auto* combi2c = new TGeoCombiTrans("combi2", x0, y0, 1.09, rot); // c-side
  //
  //
  // A-side
  //
  secta->AddNode(ch, 1);
  secta->AddNode(iroc, 1, combi1a);
  secta->AddNode(oroc, 1, combi2a);
  //
  // C-side
  //
  sectc->AddNode(ch, 1);
  sectc->AddNode(iroc, 1, combi1c);
  sectc->AddNode(oroc, 1, combi2c);
  //
  // now I try to make  wheels...
  //
  auto* wheela = new TGeoVolumeAssembly("TPC_ENDCAP");
  auto* wheelc = new TGeoVolumeAssembly("TPC_ENDCAP");
  //
  for (Int_t i = 0; i < 18; i++) {
    Double_t phi = (20. * i);
    auto* rwh = new TGeoRotation();
    rwh->RotateZ(phi);
    wheela->AddNode(secta, i + 1, rwh);
    wheelc->AddNode(sectc, i + 1, rwh);
  }
  // wheels in the drift volume!

  auto* combi3 = new TGeoCombiTrans("combi3", 0., 0., 256.6, ref);
  v9->AddNode(wheela, 1, combi3);
  v9->AddNode(wheelc, 2, new TGeoTranslation(0., 0., -256.6));
  //_____________________________________________________________
  // service support wheel
  //_____________________________________________________________
  auto* sw = new TGeoPgon(0., 20., 1, 2);
  sw->DefineSection(0, -4., 80.5, 251.75);
  sw->DefineSection(1, 4., 80.5, 251.75);
  auto* swv = new TGeoVolume("TPC_SWSEG", sw, m3); // Al
  //
  thick = 1.;
  shift = thick / TMath::Sin(openingAngle);
  auto* sh = new TGeoPgon(0., 20., 1, 2);
  sh->DefineSection(0, -4., 81.5 - shift, 250.75 - shift);
  sh->DefineSection(1, 4., 81.5 - shift, 250.75 - shift);
  auto* shv = new TGeoVolume("TPC_SWS1", sh, m1); // Air
  //
  TGeoMedium* m9 = gGeoManager->GetMedium("TPC_Si");
  auto* el = new TGeoPgon(0., 20., 1, 2);
  el->DefineSection(0, -1.872, 81.5 - shift, 250.75 - shift);
  el->DefineSection(1, 1.872, 81.5 - shift, 250.75 - shift);
  auto* elv = new TGeoVolume("TPC_ELEC", el, m9); // Si
  //
  shv->AddNode(elv, 1);
  //
  //
  ys = shift * TMath::Sin(openingAngle);
  xs = shift * TMath::Cos(openingAngle);
  swv->AddNode(shv, 1, new TGeoTranslation(xs, ys, 0.));
  // cover
  auto* co = new TGeoPgon(0., 20., 1, 2);
  co->DefineSection(0, -0.5, 77., 255.25);
  co->DefineSection(1, 0.5, 77., 255.25);
  auto* cov = new TGeoVolume("TPC_SWC1", co, m3); // Al
  // hole in a cover
  auto* coh = new TGeoPgon(0., 20., 1, 2);
  shift = 4. / TMath::Sin(openingAngle);
  coh->DefineSection(0, -0.5, 85. - shift, 247.25 - shift);
  coh->DefineSection(1, 0.5, 85. - shift, 247.25 - shift);
  //
  auto* cohv = new TGeoVolume("TPC_SWC2", coh, m1);
  //
  ys = shift * TMath::Sin(openingAngle);
  xs = shift * TMath::Cos(openingAngle);
  cov->AddNode(cohv, 1, new TGeoTranslation(xs, ys, 0.));
  //
  // Sector as an Assembly
  //
  auto* swhs = new TGeoVolumeAssembly("TPC_SSWSEC");
  swhs->AddNode(swv, 1);
  swhs->AddNode(cov, 1, new TGeoTranslation(0., 0., -4.5));
  swhs->AddNode(cov, 2, new TGeoTranslation(0., 0., 4.5));
  //
  // SSW as an Assembly of sectors
  //
  TGeoRotation* rsw[18];
  auto* swheel = new TGeoVolumeAssembly("TPC_SSWHEEL");
  for (Int_t i = 0; i < 18; i++) {
    Double_t phi = (20. * i);
    rsw[i] = new TGeoRotation();
    rsw[i]->RotateZ(phi);
    swheel->AddNode(swhs, i + 1, rsw[i]);
  }
  v1->AddNode(swheel, 1, new TGeoTranslation(0., 0., -284.6));
  v1->AddNode(swheel, 2, new TGeoTranslation(0., 0., 284.6));

  // sensitive strips - strip "0" is always set
  // conditional
  /// @todo: Hard coded numbers. Will need to be changed!
  Int_t totrows = 159;
  //   totrows = mParam->GetNRowLow() + mParam->GetNRowUp();
  Double_t* upar;
  upar = nullptr;
  gGeoManager->Volume("TPC_Strip", "PGON", m5->GetId(), upar);
  upar = new Double_t[10];
  upar[0] = 0.;
  upar[1] = 360.;
  upar[2] = 18.;
  upar[3] = 2.;
  //
  upar[4] = -124.8;
  upar[7] = 124.8;

  /// @todo: hard coded value
  //   Double_t rlow=mParam->GetPadRowRadiiLow(0);
  Double_t rlow = 85.225; // cm

  upar[5] = rlow;
  upar[6] = rlow + .01;
  upar[8] = upar[5];
  upar[9] = upar[6];
  //
  gGeoManager->Node("TPC_Strip", 1, "TPC_Drift", 0., 0., 124.82, 0, kTRUE, upar, 10);
  gGeoManager->Node("TPC_Strip", totrows + 1, "TPC_Drift", 0., 0., -124.82, 0, kTRUE, upar, 10);
  //
  // now, strips optionally
  //
  //   if(mSens){
  //     //lower sectors
  //     for(Int_t i=2;i<mParam->GetNRowLow()+1;i++){
  //       rlow=mParam->GetPadRowRadiiLow(i-1);
  //       upar[5]=rlow;
  //       upar[6]=rlow+.01;
  //       upar[8]=upar[5];
  //       upar[9]=upar[6];
  //       gGeoManager->Node("TPC_Strip",i,
  //                         "TPC_Drift",0.,0.,124.82,0,kTRUE,upar,10);
  //       gGeoManager->Node("TPC_Strip",totrows+i,
  //                         "TPC_Drift",0.,0.,-124.82,0,kTRUE,upar,10);
  //     }
  //     //upper sectors
  //     for(Int_t i=1;i<mParam->GetNRowUp()+1;i++){
  //       rlow=mParam->GetPadRowRadiiUp(i-1);
  //       upar[5]=rlow;
  //       upar[6]=rlow+.01;
  //       upar[8]=upar[5];
  //       upar[9]=upar[6];
  //       gGeoManager->Node("TPC_Strip",i+mParam->GetNRowLow(),
  //                         "TPC_Drift",0.,0.,124.82,0,kTRUE,upar,10);
  //       gGeoManager->Node("TPC_Strip",totrows+i+mParam->GetNRowLow(),
  //                         "TPC_Drift",0.,0.,-124.82,0,kTRUE,upar,10);
  //     }
  //   }//strips
  //----------------------------------------------------------
  // TPC Support Rods - MAKROLON
  //----------------------------------------------------------
  //
  TGeoMedium* m7 = gGeoManager->GetMedium("TPC_Cu");
  TGeoMedium* m10 = gGeoManager->GetMedium("TPC_Alumina");
  TGeoMedium* m11 = gGeoManager->GetMedium("TPC_Peek");
  TGeoMedium* m13 = gGeoManager->GetMedium("TPC_Brass");
  TGeoMedium* m14 = gGeoManager->GetMedium("TPC_Alumina1");
  //
  // tpc rod is an assembly of 10 long parts and 2 short parts
  // connected with alu rings and plagged on both sides.
  //
  //
  // tpc rod long
  //
  auto* rod = new TGeoPcon("rod", 0., 360., 6);
  rod->DefineSection(0, -10.43, 1.92, 2.08);
  rod->DefineSection(1, -9.75, 1.92, 2.08);

  rod->DefineSection(2, -9.75, 1.8, 2.2);
  rod->DefineSection(3, 9.75, 1.8, 2.2);

  rod->DefineSection(4, 9.75, 1.92, 2.08);
  rod->DefineSection(5, 10.43, 1.92, 2.08);
  //
  auto* mrodl = new TGeoVolume("TPC_mrodl", rod, m6);
  //
  // tpc rod short
  //
  auto* rod1 = new TGeoPcon("rod1", 0., 360., 6);
  rod1->DefineSection(0, -8.93, 1.92, 2.08);
  rod1->DefineSection(1, -8.25, 1.92, 2.08);

  rod1->DefineSection(2, -8.25, 1.8, 2.2);
  rod1->DefineSection(3, 8.25, 1.8, 2.2);

  rod1->DefineSection(4, 8.25, 1.92, 2.08);
  rod1->DefineSection(5, 8.93, 1.92, 2.08);
  //
  auto* mrods = new TGeoVolume("TPC_mrods", rod1, m6);
  //
  // below is for the resistor rod
  //
  // hole for the brass connectors
  //

  new TGeoTube("hhole", 0., 0.3, 0.3);
  //
  // transformations for holes - initialy they
  // are placed at x=0 and negative y
  //
  auto* rhole = new TGeoRotation();
  rhole->RotateX(90.);
  TGeoCombiTrans* transf[13];
  Char_t name[30];
  for (Int_t i = 0; i < 13; i++) {
    snprintf(name, 30, "transf%d", i);
    transf[i] = new TGeoCombiTrans(name, 0., -2., -9. + i * 1.5, rhole);
    transf[i]->RegisterYourself();
  }
  // union expression for holes
  TString operl("hhole:transf0");
  for (Int_t i = 1; i < 13; i++) {
    snprintf(name, 30, "+hhole:transf%d", i);
    operl.Append(name);
  }
  //
  TString opers("hhole:transf1");
  for (Int_t i = 2; i < 12; i++) {
    snprintf(name, 30, "+hhole:transf%d", i);
    opers.Append(name);
  }
  // union of holes
  new TGeoCompositeShape("hlv", operl.Data());
  new TGeoCompositeShape("hsv", opers.Data());
  //
  auto* rodl = new TGeoCompositeShape("rodl", "rod-hlv");
  auto* rods = new TGeoCompositeShape("rods", "rod1-hsv");
  // rods - volumes - makrolon rods with holes
  auto* rodlv = new TGeoVolume("TPC_rodl", rodl, m6);
  auto* rodsv = new TGeoVolume("TPC_rods", rods, m6);
  // brass connectors
  // connectors
  auto* bcon = new TGeoTube(0., 0.3, 0.3); // connectors
  auto* bconv = new TGeoVolume("TPC_bcon", bcon, m13);
  //
  // hooks holding strips
  //
  new TGeoBBox("hk1", 0.625, 0.015, 0.75);
  new TGeoBBox("hk2", 0.625, 0.015, 0.15);
  auto* tr21 = new TGeoTranslation("tr21", 0., -0.03, -0.6);
  auto* tr12 = new TGeoTranslation("tr12", 0., -0.03, 0.6);
  tr21->RegisterYourself();
  tr12->RegisterYourself();

  auto* hook = new TGeoCompositeShape("hook", "hk1+hk2:tr21+hk2:tr12");
  auto* hookv = new TGeoVolume("TPC_hook", hook, m13);
  //
  // assembly of the short rod with connectors and hooks
  //
  //
  // short rod
  //
  auto* spart = new TGeoVolumeAssembly("TPC_spart");
  //
  spart->AddNode(rodsv, 1);
  for (Int_t i = 1; i < 12; i++) {
    spart->AddNode(bconv, i, transf[i]);
  }
  for (Int_t i = 0; i < 11; i++) {
    spart->AddNode(hookv, i + 1, new TGeoTranslation(0., -2.315, -7.5 + i * 1.5));
  }
  //
  // long rod
  //
  auto* lpart = new TGeoVolumeAssembly("TPC_lpart");
  //
  lpart->AddNode(rodlv, 1);
  for (Int_t i = 0; i < 13; i++) {
    lpart->AddNode(bconv, i + 12, transf[i]);
  }
  for (Int_t i = 0; i < 13; i++) {
    lpart->AddNode(hookv, i + 12, new TGeoTranslation(0., -2.315, -9. + i * 1.5));
  }
  //
  // alu ring
  //
  new TGeoTube("ring1", 2.1075, 2.235, 0.53);
  new TGeoTube("ring2", 1.7925, 1.89, 0.43);
  new TGeoTube("ring3", 1.89, 2.1075, 0.05);
  auto* ring = new TGeoCompositeShape("ring", "ring1+ring2+ring3");
  auto* ringv = new TGeoVolume("TPC_ring", ring, m3);
  //
  // rod assembly
  //
  auto* tpcrrod = new TGeoVolumeAssembly("TPC_rrod"); // rrod
  auto* tpcmrod = new TGeoVolumeAssembly("TPC_mrod"); // makrolon rod
  // long pieces
  for (Int_t i = 0; i < 11; i++) {
    tpcrrod->AddNode(ringv, i + 1, new TGeoTranslation(0., 0., -105. + i * 21));
    tpcmrod->AddNode(ringv, i + 12, new TGeoTranslation(0., 0., -105. + i * 21));
  }
  for (Int_t i = 0; i < 10; i++) {
    tpcrrod->AddNode(lpart, i + 1, new TGeoTranslation(0., 0., -94.5 + i * 21)); // resistor rod
    tpcmrod->AddNode(mrodl, i + 1, new TGeoTranslation(0., 0., -94.5 + i * 21)); // makrolon rod
  }
  //
  // right plug - identical for all rods
  //
  auto* tpcrp = new TGeoPcon(0., 360., 6);
  //
  tpcrp->DefineSection(0, 123.05, 1.89, 2.1075);
  tpcrp->DefineSection(1, 123.59, 1.89, 2.1075);
  //
  tpcrp->DefineSection(2, 123.59, 1.8, 2.2);
  tpcrp->DefineSection(3, 127., 1.8, 2.2);
  //
  tpcrp->DefineSection(4, 127., 0., 2.2);
  tpcrp->DefineSection(5, 127.5, 0., 2.2);
  //
  auto* tpcrpv = new TGeoVolume("TPC_RP", tpcrp, m6);
  //
  // adding short pieces and right plug
  //
  tpcrrod->AddNode(spart, 1, new TGeoTranslation(0., 0., -114.));
  tpcrrod->AddNode(spart, 2, new TGeoTranslation(0., 0., 114.));
  tpcrrod->AddNode(ringv, 23, new TGeoTranslation(0., 0., -123.));
  tpcrrod->AddNode(ringv, 24, new TGeoTranslation(0., 0., 123.));
  tpcrrod->AddNode(tpcrpv, 1);
  //
  tpcmrod->AddNode(mrods, 1, new TGeoTranslation(0., 0., -114.));
  tpcmrod->AddNode(mrods, 2, new TGeoTranslation(0., 0., 114.));
  tpcmrod->AddNode(ringv, 25, new TGeoTranslation(0., 0., -123.));
  tpcmrod->AddNode(ringv, 26, new TGeoTranslation(0., 0., 123.));
  tpcmrod->AddNode(tpcrpv, 2);
  //
  // from the ringv position to the CM is 3.0 cm!
  //----------------------------------------
  //
  //
  // HV rods - makrolon + 0.58cm (diameter) Cu ->check the length
  auto* hvr = new TGeoTube(0., 1.465, 123.);
  auto* hvc = new TGeoTube(0., 0.29, 123.);
  //
  auto* hvrv = new TGeoVolume("TPC_HV_Rod", hvr, m6);
  auto* hvcv = new TGeoVolume("TPC_HV_Cable", hvc, m7);
  hvrv->AddNode(hvcv, 1);
  //
  // resistor rod
  //
  auto* cr = new TGeoTube(0., 0.45, 123.);
  auto* cw = new TGeoTube(0., 0.15, 123.);
  auto* crv = new TGeoVolume("TPC_CR", cr, m10);
  auto* cwv = new TGeoVolume("TPC_W", cw, m12);
  //
  // ceramic rod with water
  //
  crv->AddNode(cwv, 1);
  //
  // peek rod
  //
  auto* pr = new TGeoTube(0.2, 0.35, 123.);
  auto* prv = new TGeoVolume("TPC_PR", pr, m11);
  //
  // copper plates with connectors
  //
  new TGeoTube("tub", 0., 1.7, 0.025);
  //
  // half space - points on the plane and a normal vector
  //
  Double_t n[3], p[3];
  Double_t slope = TMath::Tan(22. * TMath::DegToRad());
  Double_t intp = 1.245;
  //
  Double_t b = slope * slope + 1.;
  p[0] = intp * slope / b;
  p[1] = -intp / b;
  p[2] = 0.;
  //
  n[0] = -p[0];
  n[1] = -p[1];
  n[2] = 0.;
  Double_t norm;
  norm = TMath::Sqrt(n[0] * n[0] + n[1] * n[1]);
  n[0] /= norm;
  n[1] /= norm;
  //
  new TGeoHalfSpace("sp1", p, n);
  //
  slope = -slope;
  //
  p[0] = intp * slope / b;
  p[1] = -intp / b;
  //
  n[0] = -p[0];
  n[1] = -p[1];
  norm = TMath::Sqrt(n[0] * n[0] + n[1] * n[1]);
  n[0] /= norm;
  n[1] /= norm;
  //
  new TGeoHalfSpace("sp2", p, n);
  // holes for rods
  // holes
  new TGeoTube("h1", 0., 0.5, 0.025);
  new TGeoTube("h2", 0., 0.35, 0.025);
  // translations:
  auto* ttr11 = new TGeoTranslation("ttr11", -0.866, 0.5, 0.);
  auto* ttr22 = new TGeoTranslation("ttr22", 0.866, 0.5, 0.);
  ttr11->RegisterYourself();
  ttr22->RegisterYourself();
  // elastic connector
  new TGeoBBox("elcon", 0.72, 0.005, 0.3);
  auto* crr1 = new TGeoRotation();
  crr1->RotateZ(-22.);
  auto* ctr1 = new TGeoCombiTrans("ctr1", -0.36011, -1.09951, -0.325, crr1);
  ctr1->RegisterYourself();
  auto* cs1 = new TGeoCompositeShape("cs1", "(((((tub-h1:ttr11)-h1:ttr22)-sp1)-sp2)-h2)+elcon:ctr1");
  //
  auto* csvv = new TGeoVolume("TPC_RR_CU", cs1, m7);
  //
  // resistor rod assembly 2 ceramic rods, peak rod, Cu plates
  // and resistors
  //
  auto* rrod = new TGeoVolumeAssembly("TPC_RRIN");
  // rods
  rrod->AddNode(crv, 1, ttr11);
  rrod->AddNode(crv, 2, ttr22);
  rrod->AddNode(prv, 1);
  // Cu plates
  for (Int_t i = 0; i < 165; i++) {
    rrod->AddNode(csvv, i + 1, new TGeoTranslation(0., 0., -122.675 + i * 1.5));
  }
  // resistors
  auto* res = new TGeoTube(0., 0.15, 0.5);
  auto* resv = new TGeoVolume("TPC_RES", res, m14);
  auto* ress = new TGeoVolumeAssembly("TPC_RES_CH");
  ress->AddNode(resv, 1, new TGeoTranslation(0.2, 0., 0.));
  ress->AddNode(resv, 2, new TGeoTranslation(-0.2, 0., 0.));
  //
  auto* crr2 = new TGeoRotation();
  crr2->RotateY(30.);
  auto* crr3 = new TGeoRotation();
  crr3->RotateY(-30.);
  //
  for (Int_t i = 0; i < 164; i += 2) {
    rrod->AddNode(ress, i + 1, new TGeoCombiTrans(0., 1.2, -121.925 + i * 1.5, crr2));
    rrod->AddNode(ress, i + 2, new TGeoCombiTrans(0., 1.2, -121.925 + (i + 1) * 1.5, crr3));
  }

  tpcrrod->AddNode(rrod, 1, new TGeoCombiTrans(0., 0., 0.5, crr1));
  //
  // rod left head with holders - inner
  //
  // first element - support for inner holder  TPC_IHS
  Double_t shift1[3] = {0.0, -0.175, 0.0};

  new TGeoBBox("tpcihs1", 4.7, 0.66, 2.35);
  new TGeoBBox("tpcihs2", 4.7, 0.485, 1.0, shift1);
  new TGeoBBox("tpcihs3", 1.5, 0.485, 2.35, shift1);
  new TGeoTube("tpcihs4", 0.0, 2.38, 0.1);
  //
  Double_t pointstrap[16];
  pointstrap[0] = 0.0;
  pointstrap[1] = 0.0;
  pointstrap[2] = 0.0;
  pointstrap[3] = 1.08;
  pointstrap[4] = 2.3;
  pointstrap[5] = 1.08;
  pointstrap[6] = 3.38;
  pointstrap[7] = 0.0;
  pointstrap[8] = 0.0;
  pointstrap[9] = 0.0;
  pointstrap[10] = 0.0;
  pointstrap[11] = 1.08;
  pointstrap[12] = 2.3;
  pointstrap[13] = 1.08;
  pointstrap[14] = 3.38;
  pointstrap[15] = 0.0;
  //
  auto* tpcihs5 = new TGeoArb8("tpcihs5", 0.6, pointstrap);
  //
  //  half space - cutting "legs"
  //
  p[0] = 0.0;
  p[1] = 0.105;
  p[2] = 0.0;
  //
  n[0] = 0.0;
  n[1] = 1.0;
  n[2] = 0.0;

  new TGeoHalfSpace("cutil1", p, n);

  //
  // transformations
  //
  auto* trans2 = new TGeoTranslation("trans2", 0.0, 2.84, 2.25);
  trans2->RegisterYourself();
  auto* trans3 = new TGeoTranslation("trans3", 0.0, 2.84, -2.25);
  trans3->RegisterYourself();
  // support - composite volume
  //
  auto* tpcihs6 =
    new TGeoCompositeShape("tpcihs6", "tpcihs1-(tpcihs2+tpcihs3)-(tpcihs4:trans2)-(tpcihs4:trans3)-cutil1");
  //
  // volumes - all makrolon
  //
  auto* tpcihss = new TGeoVolume("TPC_IHSS", tpcihs6, m6);  // support
  auto* tpcihst = new TGeoVolume("TPC_IHSTR", tpcihs5, m6); // trapesoid
  // now assembly
  auto* rot111 = new TGeoRotation();
  rot111->RotateY(180.0);
  //
  auto* tpcihs = new TGeoVolumeAssembly("TPC_IHS"); // assembly of the support
  tpcihs->AddNode(tpcihss, 1);
  tpcihs->AddNode(tpcihst, 1, new TGeoTranslation(-4.7, 0.66, 0.0));
  tpcihs->AddNode(tpcihst, 2, new TGeoCombiTrans(4.7, 0.66, 0.0, rot111));
  //
  // two rod holders (TPC_IRH) assembled with the support
  //
  new TGeoBBox("tpcirh1", 4.7, 1.33, 0.5);
  shift1[0] = -3.65;
  shift1[1] = 0.53;
  shift1[2] = 0.;
  new TGeoBBox("tpcirh2", 1.05, 0.8, 0.5, shift1);
  shift1[0] = 3.65;
  shift1[1] = 0.53;
  shift1[2] = 0.;
  new TGeoBBox("tpcirh3", 1.05, 0.8, 0.5, shift1);
  shift1[0] = 0.0;
  shift1[1] = 1.08;
  shift1[2] = 0.;
  new TGeoBBox("tpcirh4", 1.9, 0.25, 0.5, shift1);
  new TGeoTube("tpcirh5", 0, 1.9, 5);
  //
  auto* trans4 = new TGeoTranslation("trans4", 0, 0.83, 0.0);
  trans4->RegisterYourself();
  //
  auto* tpcirh6 = new TGeoCompositeShape("tpcirh6", "tpcirh1-tpcirh2-tpcirh3-(tpcirh5:trans4)-tpcirh4");
  //
  // now volume
  //
  auto* tpcirh = new TGeoVolume("TPC_IRH", tpcirh6, m6);
  //
  // and all together...
  //
  TGeoVolume* tpciclamp = new TGeoVolumeAssembly("TPC_ICLP");
  tpciclamp->AddNode(tpcihs, 1);
  tpciclamp->AddNode(tpcirh, 1, new TGeoTranslation(0, 1.99, 1.1));
  tpciclamp->AddNode(tpcirh, 2, new TGeoTranslation(0, 1.99, -1.1));
  //
  // and now left inner "head"
  //
  auto* inplug = new TGeoPcon("inplug", 0.0, 360.0, 13);

  inplug->DefineSection(0, 0.3, 0.0, 2.2);
  inplug->DefineSection(1, 0.6, 0.0, 2.2);

  inplug->DefineSection(2, 0.6, 0.0, 1.75);
  inplug->DefineSection(3, 0.7, 0.0, 1.75);

  inplug->DefineSection(4, 0.7, 1.55, 1.75);
  inplug->DefineSection(5, 1.6, 1.55, 1.75);

  inplug->DefineSection(6, 1.6, 1.55, 2.2);
  inplug->DefineSection(7, 1.875, 1.55, 2.2);

  inplug->DefineSection(8, 2.47, 1.75, 2.2);

  inplug->DefineSection(9, 2.47, 1.75, 2.08);
  inplug->DefineSection(10, 2.57, 1.8, 2.08);

  inplug->DefineSection(11, 2.57, 1.92, 2.08);
  inplug->DefineSection(12, 2.95, 1.92, 2.08);
  //
  shift1[0] = 0.0;
  shift1[1] = -2.09;
  shift1[2] = 1.075;
  //
  new TGeoBBox("pcuti", 1.5, 0.11, 1.075, shift1);
  //
  auto* inplleft = new TGeoCompositeShape("inplleft", "inplug-pcuti");
  auto* tpcinlplug = new TGeoVolume("TPC_INPLL", inplleft, m6);
  //
  //  holder + plugs
  //
  TGeoVolume* tpcihpl = new TGeoVolumeAssembly("TPC_IHPL"); // holder+2 plugs (reflected)
  tpcihpl->AddNode(tpcinlplug, 1);
  tpcihpl->AddNode(tpcinlplug, 2, ref);
  tpcihpl->AddNode(tpciclamp, 1, new TGeoTranslation(0.0, -2.765, 0.0));
  //
  // outer holders and clamps
  //

  // outer membrane holder (between rods)
  pointstrap[0] = 0.0;
  pointstrap[1] = 0.0;
  pointstrap[2] = 0.0;
  pointstrap[3] = 2.8;
  pointstrap[4] = 3.1;
  pointstrap[5] = 2.8 - 3.1 * TMath::Tan(15. * TMath::DegToRad());
  pointstrap[6] = 3.1;
  pointstrap[7] = 0.0;
  pointstrap[8] = 0.0;
  pointstrap[9] = 0.0;
  pointstrap[10] = 0.0;
  pointstrap[11] = 2.8;
  pointstrap[12] = 3.1;
  pointstrap[13] = 2.8 - 3.1 * TMath::Tan(15. * TMath::DegToRad());
  pointstrap[14] = 3.1;
  pointstrap[15] = 0.0;
  //
  auto* tpcomh1 = new TGeoArb8("tpcomh1", 1.05, pointstrap);
  auto* tpcomh2 = new TGeoBBox("tpcomh2", 0.8, 1.4, 6);
  //
  auto* tpcomh1v = new TGeoVolume("TPC_OMH1", tpcomh1, m7);
  auto* tpcomh2v = new TGeoVolume("TPC_OMH2", tpcomh2, m7);
  //
  TGeoVolume* tpcomh3v = new TGeoVolumeAssembly("TPC_OMH3"); // assembly1
  tpcomh3v->AddNode(tpcomh1v, 1, new TGeoTranslation(0.8, -1.4, 4.95));
  tpcomh3v->AddNode(tpcomh1v, 2, new TGeoTranslation(0.8, -1.4, -4.95));
  tpcomh3v->AddNode(tpcomh2v, 1);
  //
  shift1[0] = 0.9;
  shift1[1] = -1.85;
  shift1[2] = 0.0;
  //
  new TGeoBBox("tpcomh3", 1.65, 1.15, 3.4);
  auto* tpcomh4 = new TGeoBBox("tpcomh4", 0.75, 0.7, 3.4, shift1);
  //
  // halfspace 1
  //
  p[0] = 0.0;
  p[1] = -1.05;
  p[2] = -3.4;
  //
  n[0] = 0.0;
  n[1] = -1.0 * TMath::Tan(30. * TMath::DegToRad());
  n[2] = 1.0;
  //
  new TGeoHalfSpace("cutomh1", p, n);
  //
  // halfspace 2
  //
  p[0] = 0.0;
  p[1] = -1.05;
  p[2] = 3.4;
  //
  n[0] = 0.0;
  n[1] = -1.0 * TMath::Tan(30. * TMath::DegToRad());
  n[2] = -1.0;
  //
  new TGeoHalfSpace("cutomh2", p, n);
  //
  // halfspace 3
  //
  p[0] = -1.65;
  p[1] = 0.0;
  p[2] = -0.9;
  //
  n[0] = 1.0 * TMath::Tan(75. * TMath::DegToRad());
  n[1] = 0.0;
  n[2] = 1.0;
  //
  new TGeoHalfSpace("cutomh3", p, n);
  //
  // halfspace 4
  //
  p[0] = -1.65;
  p[1] = 0.0;
  p[2] = 0.9;
  //
  n[0] = 1.0 * TMath::Tan(75 * TMath::DegToRad());
  n[1] = 0.0;
  n[2] = -1.0;
  //
  new TGeoHalfSpace("cutomh4", p, n);
  //
  // halsfspace 5
  //
  p[0] = 1.65;
  p[1] = -1.05;
  p[2] = 0.0;
  //
  n[0] = -1.0;
  n[1] = -1.0 * TMath::Tan(20. * TMath::DegToRad());
  n[2] = 0.0;
  //
  new TGeoHalfSpace("cutomh5", p, n);
  //
  auto* tpcomh5 = new TGeoCompositeShape("tpcomh5", "tpcomh3-cutomh1-cutomh2-cutomh3-cutomh4-cutomh5");
  //
  auto* tpcomh5v = new TGeoVolume("TPC_OMH5", tpcomh5, m6);
  auto* tpcomh4v = new TGeoVolume("TPC_OMH6", tpcomh4, m6);
  //
  auto* tpcomh7v = new TGeoVolumeAssembly("TPC_OMH7");
  tpcomh7v->AddNode(tpcomh5v, 1);
  tpcomh7v->AddNode(tpcomh4v, 1);
  //
  // full membrane holder - tpcomh3v + tpcomh7v
  //
  auto* tpcomh = new TGeoVolumeAssembly("TPC_OMH");
  tpcomh->AddNode(tpcomh3v, 1, new TGeoTranslation(1.5, 0., 0.));
  tpcomh->AddNode(tpcomh3v, 2, new TGeoCombiTrans(-1.5, 0., 0., rot111));
  tpcomh->AddNode(tpcomh7v, 1, new TGeoTranslation(0.65 + 1.5, 2.55, 0.0));
  tpcomh->AddNode(tpcomh7v, 2, new TGeoCombiTrans(-0.65 - 1.5, 2.55, 0.0, rot111));
  //
  //  outer rod holder support
  //
  new TGeoBBox("tpcohs1", 3.8, 0.675, 2.35);
  //
  shift1[0] = 0.0;
  shift1[1] = 0.175;
  shift1[2] = 0.0;
  //
  new TGeoBBox("tpcohs2", 1.5, 0.5, 2.35, shift1);
  new TGeoBBox("tpcohs3", 3.8, 0.5, 0.85, shift1);
  //
  shift1[0] = 0.0;
  shift1[1] = -1.175;
  shift1[2] = 0.0;
  //
  auto* tpcohs4 = new TGeoBBox("tpsohs4", 3.1, 0.5, 0.7, shift1);
  //
  auto* tpcohs4v = new TGeoVolume("TPC_OHS4", tpcohs4, m6);
  //
  p[0] = 0.0;
  p[1] = -0.186;
  p[2] = 0.0;
  //
  n[0] = 0.0;
  n[1] = -1.0;
  n[2] = 0.0;
  //
  new TGeoHalfSpace("cutohs1", p, n);
  //
  auto* tpcohs5 = new TGeoCompositeShape("tpcohs5", "tpcohs1-tpcohs2-tpcohs3-cutohs1");
  auto* tpcohs5v = new TGeoVolume("TPC_OHS5", tpcohs5, m6);
  //
  auto* tpcohs = new TGeoVolumeAssembly("TPC_OHS");
  tpcohs->AddNode(tpcohs5v, 1);
  tpcohs->AddNode(tpcohs4v, 1);
  //
  // outer rod holder itself
  //
  shift1[0] = 0.0;
  shift1[1] = 1.325;
  shift1[2] = 0.0;
  new TGeoBBox("tpcorh1", 3.1, 1.825, 0.55); // from this box we cut pieces...
  //
  shift1[0] = -3.1;
  shift1[1] = -0.5;
  shift1[2] = 0.0;
  //
  new TGeoBBox("tpcorh2", 0.5, 2.75, 1.1, shift1);
  //
  shift1[0] = 3.1;
  shift1[1] = -0.5;
  shift1[2] = 0.0;
  //
  new TGeoBBox("tpcorh3", 0.5, 2.75, 1.1, shift1);
  //
  shift1[0] = 0.0;
  shift1[1] = -0.5;
  shift1[2] = -0.95;
  //
  new TGeoBBox("tpcorh4", 3.9, 2.75, 0.5, shift1);
  //
  shift1[0] = 0.0;
  shift1[1] = -0.5;
  shift1[2] = 0.0;
  //
  new TGeoBBox("tpcorh5", 1.95, 0.5, 1.1, shift1);
  //
  shift1[0] = 0.0;
  shift1[1] = -0.5;
  shift1[2] = 0.55;
  //
  new TGeoBBox("tpcorh6", 2.4, 0.5, 0.6, shift1);
  //
  new TGeoTube("tpcorh7", 0, 1.95, 0.85);
  new TGeoTube("tpcorh8", 0, 2.4, 0.6);
  //
  auto* trans33 = new TGeoTranslation("trans33", 0.0, 0.0, 0.55);
  trans33->RegisterYourself();
  //
  auto* tpcorh9 =
    new TGeoCompositeShape("tpcorh9", "tpcorh1-tpcorh2-tpcorh3-tpcorh4-tpcorh5-tpcorh6-(tpcorh8:trans33)-tpcorh7");
  //
  auto* tpcorh9v = new TGeoVolume("TPC_ORH", tpcorh9, m6); // outer rod holder
  //
  // now 2 holders together
  //
  auto* tpcorh = new TGeoVolumeAssembly("TPC_ORH2");
  //
  tpcorh->AddNode(tpcorh9v, 1, new TGeoTranslation(0.0, 0.0, 1.25));
  tpcorh->AddNode(tpcorh9v, 2, new TGeoCombiTrans(0.0, 0.0, -1.25, rot111));
  //
  // outer rod plug left
  //
  auto* outplug = new TGeoPcon("outplug", 0.0, 360.0, 13);

  outplug->DefineSection(0, 0.5, 0.0, 2.2);
  outplug->DefineSection(1, 0.7, 0.0, 2.2);
  outplug->DefineSection(2, 0.7, 1.55, 2.2);
  outplug->DefineSection(3, 0.8, 1.55, 2.2);
  outplug->DefineSection(4, 0.8, 1.55, 1.75);
  outplug->DefineSection(5, 1.2, 1.55, 1.75);
  outplug->DefineSection(6, 1.2, 1.55, 2.2);
  outplug->DefineSection(7, 1.875, 1.55, 2.2);
  outplug->DefineSection(8, 2.47, 1.75, 2.2);
  outplug->DefineSection(9, 2.47, 1.75, 2.08);
  outplug->DefineSection(10, 2.57, 1.8, 2.08);
  outplug->DefineSection(11, 2.57, 1.92, 2.08);
  outplug->DefineSection(12, 2.95, 1.92, 2.08);
  //
  shift1[0] = 0.0;
  shift1[1] = 2.09;
  shift1[2] = 1.01;

  new TGeoBBox("cutout", 2.5, 0.11, 1.01, shift1);
  //

  auto* outplleft = new TGeoCompositeShape("outplleft", "outplug-cutout");
  auto* outplleftv = new TGeoVolume("TPC_OPLL", outplleft, m6);
  //
  //  support + holder + plug
  //

  auto* tpcohpl = new TGeoVolumeAssembly("TPC_OHPL");
  //
  tpcohpl->AddNode(outplleftv, 1);                                 // plug
  tpcohpl->AddNode(outplleftv, 2, ref);                            // plug reflected
  tpcohpl->AddNode(tpcorh, 1);                                     // rod holder
  tpcohpl->AddNode(tpcohs, 1, new TGeoTranslation(0.0, 3.925, 0)); // support
  //

  //
  // main membrane holder
  //
  pointstrap[0] = 0.0;
  pointstrap[1] = 0.0;
  pointstrap[2] = 0.0;
  pointstrap[3] = 2.8;
  pointstrap[4] = 3.1;
  pointstrap[5] = 1.96;
  pointstrap[6] = 3.1;
  pointstrap[7] = 0.0;
  pointstrap[8] = 0.0;
  pointstrap[9] = 0.0;
  pointstrap[10] = 0.0;
  pointstrap[11] = 2.8;
  pointstrap[12] = 3.1;
  pointstrap[13] = 1.96;
  pointstrap[14] = 3.1;
  pointstrap[15] = 0.0;
  //
  auto* tpcmmh1 = new TGeoArb8("tpcmmh1", 1.75, pointstrap);
  auto* tpcmmh2 = new TGeoBBox("tpcmmh2", 0.8, 1.4, 12.5);
  //
  auto* tpcmmh1v = new TGeoVolume("TPC_MMH1", tpcmmh1, m6);
  auto* tpcmmh2v = new TGeoVolume("TPC_MMH2", tpcmmh2, m6);
  //
  auto* tpcmmhs = new TGeoVolumeAssembly("TPC_MMHS");
  tpcmmhs->AddNode(tpcmmh1v, 1, new TGeoTranslation(0.8, -1.4, 10.75));
  tpcmmhs->AddNode(tpcmmh1v, 2, new TGeoTranslation(0.8, -1.4, -10.75));
  tpcmmhs->AddNode(tpcmmh2v, 1);
  //
  // main membrahe holder clamp
  //
  shift1[0] = -0.75;
  shift1[1] = -1.15;
  shift1[2] = 0.0;
  //
  new TGeoBBox("tpcmmhc1", 1.65, 1.85, 8.9);
  new TGeoBBox("tpcmmhc2", 0.9, 0.7, 8.9, shift1);
  //
  // half spaces  - cuts
  //
  p[0] = -1.65;
  p[1] = 0.0;
  p[2] = -0.9;
  //
  n[0] = 8.0;
  n[1] = 0.0;
  n[2] = 8.0 * TMath::Tan(13. * TMath::DegToRad());
  //
  new TGeoHalfSpace("cutmmh1", p, n);
  //
  p[0] = -1.65;
  p[1] = 0.0;
  p[2] = 0.9;
  //
  n[0] = 8.0;
  n[1] = 0.0;
  n[2] = -8.0 * TMath::Tan(13. * TMath::DegToRad());
  //
  new TGeoHalfSpace("cutmmh2", p, n);
  //
  p[0] = 0.0;
  p[1] = 1.85;
  p[2] = -2.8;
  //
  n[0] = 0.0;
  n[1] = -6.1;
  n[2] = 6.1 * TMath::Tan(20. * TMath::DegToRad());
  //
  new TGeoHalfSpace("cutmmh3", p, n);
  //
  p[0] = 0.0;
  p[1] = 1.85;
  p[2] = 2.8;
  //
  n[0] = 0.0;
  n[1] = -6.1;
  n[2] = -6.1 * TMath::Tan(20 * TMath::DegToRad());
  //
  new TGeoHalfSpace("cutmmh4", p, n);
  //
  p[0] = 0.75;
  p[1] = 0.0;
  p[2] = -8.9;
  //
  n[0] = 2.4 * TMath::Tan(30 * TMath::DegToRad());
  n[1] = 0.0;
  n[2] = 2.4;
  //
  new TGeoHalfSpace("cutmmh5", p, n);
  //
  p[0] = 0.75;
  p[1] = 0.0;
  p[2] = 8.9;
  //
  n[0] = 2.4 * TMath::Tan(30 * TMath::DegToRad());
  n[1] = 0.0;
  n[2] = -2.4;
  //
  new TGeoHalfSpace("cutmmh6", p, n);

  auto* tpcmmhc =
    new TGeoCompositeShape("TPC_MMHC", "tpcmmhc1-tpcmmhc2-cutmmh1-cutmmh2-cutmmh3-cutmmh4-cutmmh5-cutmmh6");

  auto* tpcmmhcv = new TGeoVolume("TPC_MMHC", tpcmmhc, m6);
  //
  TGeoVolume* tpcmmh = new TGeoVolumeAssembly("TPC_MMH");
  //
  tpcmmh->AddNode(tpcmmhcv, 1, new TGeoTranslation(0.65 + 1.5, 1.85, 0.0));
  tpcmmh->AddNode(tpcmmhcv, 2, new TGeoCombiTrans(-0.65 - 1.5, 1.85, 0.0, rot111));
  tpcmmh->AddNode(tpcmmhs, 1, new TGeoTranslation(1.5, 0.0, 0.0));
  tpcmmh->AddNode(tpcmmhs, 2, new TGeoCombiTrans(-1.5, 0.0, 0.0, rot111));
  //

  //

  //--------------------------------------------
  //
  // guard ring resistor chain
  //

  auto* gres1 = new TGeoTube(0., 0.375, 125.); // inside ifc
  //
  auto* vgres1 = new TGeoVolume("TPC_GRES1", gres1, m14);

  //
  Double_t xrc, yrc;
  //
  xrc = 79.3 * TMath::Cos(350. * TMath::DegToRad());
  yrc = 79.3 * TMath::Sin(350. * TMath::DegToRad());
  //
  v9->AddNode(vgres1, 1, new TGeoTranslation(xrc, yrc, 126.9));
  v9->AddNode(vgres1, 2, new TGeoTranslation(xrc, yrc, -126.9));
  //
  xrc = 79.3 * TMath::Cos(190. * TMath::DegToRad());
  yrc = 79.3 * TMath::Sin(190. * TMath::DegToRad());
  //
  v9->AddNode(vgres1, 3, new TGeoTranslation(xrc, yrc, 126.9));
  v9->AddNode(vgres1, 4, new TGeoTranslation(xrc, yrc, -126.9));
  //------------------------------------------------------------------
  TGeoRotation refl("refl", 90., 0., 90., 90., 180., 0.);
  TGeoRotation rotrod("rotrod");
  //
  TGeoRotation* rotpos[2];
  //
  TGeoRotation* rotrod1[2];
  //
  // clamps holding rods
  //
  auto* clampi1 = new TGeoBBox("clampi1", 0.2, 3.1, 0.8);
  auto* clampi1v = new TGeoVolume("TPC_clampi1v", clampi1, m6);
  //
  pointstrap[0] = 0.49;
  pointstrap[1] = 0.375;
  //
  pointstrap[2] = 0.49;
  pointstrap[3] = -0.375;
  //
  pointstrap[4] = -0.49;
  pointstrap[5] = -0.375;
  //
  pointstrap[6] = -0.49;
  pointstrap[7] = 1.225;
  //
  pointstrap[8] = 0.49;
  pointstrap[9] = 0.375;
  //
  pointstrap[10] = 0.49;
  pointstrap[11] = -0.375;
  //
  pointstrap[12] = -0.49;
  pointstrap[13] = -0.375;
  //
  pointstrap[14] = -0.49;
  pointstrap[15] = 1.225;
  //
  auto* clitrap = new TGeoArb8("clitrap", 0.25, pointstrap);
  auto* clitrapv = new TGeoVolume("TPC_clitrapv", clitrap, m6);
  //
  auto* clamprot = new TGeoRotation();
  clamprot->RotateX(180.);
  //
  new TGeoBBox("clibox", 1.125, 3.1, .1);
  new TGeoTube("clitub", 0., 2.2, 0.1);
  //
  // copmisite shape for the clamp holder
  //
  auto* clitr1 = new TGeoTranslation("clitr1", 1.125, 0., 0.);
  clitr1->RegisterYourself();
  auto* clihold = new TGeoCompositeShape("clihold", "clibox-clitub:clitr1");
  auto* cliholdv = new TGeoVolume("TPC_cliholdv", clihold, m6);
  //
  // now assembly the whole inner clamp
  //
  TGeoVolume* iclamp = new TGeoVolumeAssembly("TPC_iclamp");
  //
  iclamp->AddNode(clampi1v, 1);                                          // main box
  iclamp->AddNode(clitrapv, 1, new TGeoTranslation(0.69, -2.725, 0.35)); // trapezoids
  iclamp->AddNode(clitrapv, 2, new TGeoTranslation(0.69, -2.725, -0.35));
  iclamp->AddNode(clitrapv, 3, new TGeoCombiTrans(0.69, 2.725, 0.35, clamprot));
  iclamp->AddNode(clitrapv, 4, new TGeoCombiTrans(0.69, 2.725, -0.35, clamprot));
  iclamp->AddNode(cliholdv, 1, new TGeoTranslation(1.325, 0., 0.)); // holder
  //
  //  outer clamps
  //
  auto* clampo1 = new TGeoBBox("clampo1", 0.25, 3.1, 1.);
  auto* clampo2 = new TGeoBBox("clampo2", 0.4, 0.85, 1.);
  //
  auto* clampo1v = new TGeoVolume("TPC_clampo1v", clampo1, m6);
  auto* clampo2v = new TGeoVolume("TPC_clampo2v", clampo2, m6);
  //
  auto* oclamp = new TGeoVolumeAssembly("TPC_oclamp");
  //
  oclamp->AddNode(clampo1v, 1);
  //
  oclamp->AddNode(clampo2v, 1, new TGeoTranslation(0.65, -2.25, 0));
  oclamp->AddNode(clampo2v, 2, new TGeoTranslation(0.65, 2.25, 0));

  //
  pointstrap[0] = 0.375;
  pointstrap[1] = 0.75;
  pointstrap[2] = 0.375;
  pointstrap[3] = -0.35;
  pointstrap[5] = -0.375;
  pointstrap[4] = -0.35;
  pointstrap[6] = -0.375;
  pointstrap[7] = 0.35;
  //
  pointstrap[8] = 0.375;
  pointstrap[9] = 0.75;
  pointstrap[10] = 0.375;
  pointstrap[11] = -0.35;
  pointstrap[12] = -0.375;
  pointstrap[13] = -0.35;
  pointstrap[14] = -0.375;
  pointstrap[15] = 0.35;
  //
  auto* clotrap = new TGeoArb8("clotrap", 0.25, pointstrap);
  auto* clotrapv = new TGeoVolume("TPC_clotrapv", clotrap, m6);
  //
  oclamp->AddNode(clotrapv, 1, new TGeoTranslation(-0.625, -2.75, 0.35));
  oclamp->AddNode(clotrapv, 2, new TGeoTranslation(-0.625, -2.75, -0.35));
  oclamp->AddNode(clotrapv, 3, new TGeoCombiTrans(-0.625, 2.75, 0.35, clamprot));
  oclamp->AddNode(clotrapv, 4, new TGeoCombiTrans(-0.625, 2.75, -0.35, clamprot));
  //
  auto* clampo3 = new TGeoBBox("clampo3", 1.6, 0.45, .1);
  auto* clampo3v = new TGeoVolume("TPC_clampo3v", clampo3, m6);
  //
  oclamp->AddNode(clampo3v, 1, new TGeoTranslation(-1.85, 2.625, 0.));
  oclamp->AddNode(clampo3v, 2, new TGeoTranslation(-1.85, -2.625, 0));
  //
  auto* clampo4 = new TGeoTubeSeg("clampo4", 2.2, 3.1, 0.1, 90., 270.);
  auto* clampo4v = new TGeoVolume("TPC_clampo4v", clampo4, m6);
  //
  oclamp->AddNode(clampo4v, 1, new TGeoTranslation(-3.45, 0., 0.));

  // v9 - drift gas

  TGeoRotation rot102("rot102");
  rot102.RotateY(-90.);

  for (Int_t i = 0; i < 18; i++) {
    Double_t angle, x, y;
    Double_t z, r;
    angle = TMath::DegToRad() * 20. * (Double_t)i;
    // inner rods
    r = 81.5;
    x = r * TMath::Cos(angle);
    y = r * TMath::Sin(angle);
    z = 126.;
    auto* rot12 = new TGeoRotation();
    rot12->RotateZ(-90.0 + i * 20.);
    v9->AddNode(tpcihpl, i + 1, new TGeoCombiTrans(x, y, 0., rot12));
    //
    if (i == 11) { // resistor rod inner
      rotrod.RotateZ(-90. + i * 20.);
      rotrod1[0] = new TGeoRotation();
      rotpos[0] = new TGeoRotation();
      //
      rotrod1[0]->RotateZ(90. + i * 20.);
      *rotpos[0] = refl * rotrod;                                       // rotation+reflection
      v9->AddNode(tpcrrod, 1, new TGeoCombiTrans(x, y, z, rotrod1[0])); // A
      v9->AddNode(tpcrrod, 2, new TGeoCombiTrans(x, y, -z, rotpos[0])); // C
    } else {
      v9->AddNode(tpcmrod, i + 1, new TGeoTranslation(x, y, z));       // shaft
      v9->AddNode(tpcmrod, i + 19, new TGeoCombiTrans(x, y, -z, ref)); // muon
    }
    //
    // inner clamps positioning
    //
    r = 79.05;
    x = r * TMath::Cos(angle);
    y = r * TMath::Sin(angle);
    rot12 = new TGeoRotation();
    rot12->RotateZ(i * 20.);
    //
    // A-side
    v9->AddNode(iclamp, 7 * i + 1, new TGeoCombiTrans(x, y, 5.25, rot12));
    v9->AddNode(iclamp, 7 * i + 2, new TGeoCombiTrans(x, y, 38.25, rot12));
    v9->AddNode(iclamp, 7 * i + 3, new TGeoCombiTrans(x, y, 80.25, rot12));
    v9->AddNode(iclamp, 7 * i + 4, new TGeoCombiTrans(x, y, 122.25, rot12));
    v9->AddNode(iclamp, 7 * i + 5, new TGeoCombiTrans(x, y, 164.25, rot12));
    v9->AddNode(iclamp, 7 * i + 6, new TGeoCombiTrans(x, y, 206.25, rot12));
    v9->AddNode(iclamp, 7 * i + 7, new TGeoCombiTrans(x, y, 246.75, rot12));
    // C-side
    v9->AddNode(iclamp, 7 * i + 127, new TGeoCombiTrans(x, y, -5.25, rot12));
    v9->AddNode(iclamp, 7 * i + 128, new TGeoCombiTrans(x, y, -38.25, rot12));
    v9->AddNode(iclamp, 7 * i + 129, new TGeoCombiTrans(x, y, -80.25, rot12));
    v9->AddNode(iclamp, 7 * i + 130, new TGeoCombiTrans(x, y, -122.25, rot12));
    v9->AddNode(iclamp, 7 * i + 131, new TGeoCombiTrans(x, y, -164.25, rot12));
    v9->AddNode(iclamp, 7 * i + 132, new TGeoCombiTrans(x, y, -206.25, rot12));
    v9->AddNode(iclamp, 7 * i + 133, new TGeoCombiTrans(x, y, -246.75, rot12));
    //
    //--------------------------
    // outer rods
    r = 254.25;
    x = r * TMath::Cos(angle);
    y = r * TMath::Sin(angle);
    z = 126.;
    //
    // outer rod holder + outer left plug
    //

    auto* rot33 = new TGeoRotation();
    rot33->RotateZ(-90 + i * 20.);
    //
    v9->AddNode(tpcohpl, i + 1, new TGeoCombiTrans(x, y, 0., rot33));
    //
    Double_t xxx = 256.297 * TMath::Cos((i * 20. + 10.) * TMath::DegToRad());
    Double_t yyy = 256.297 * TMath::Sin((i * 20. + 10.) * TMath::DegToRad());
    //
    TGeoRotation rot101("rot101");
    rot101.RotateZ(90. + i * 20. + 10.);
    auto* rot103 = new TGeoRotation("rot103");
    *rot103 = rot101 * rot102;
    //
    auto* trh100 = new TGeoCombiTrans(xxx, yyy, 0., rot103);
    //
    if (i == 2) {
      // main membrane holder
      v9->AddNode(tpcmmh, 1, trh100);
    } else {
      // "normal" membrane holder
      v9->AddNode(tpcomh, i + 1, trh100);
    }

    //
    if (i == 3) { // resistor rod outer
      rotrod.RotateZ(90. + i * 20.);
      rotrod1[1] = new TGeoRotation();
      rotpos[1] = new TGeoRotation();
      rotrod1[1]->RotateZ(90. + i * 20.);
      *rotpos[1] = refl * rotrod;                                       // rotation+reflection
      v9->AddNode(tpcrrod, 3, new TGeoCombiTrans(x, y, z, rotrod1[1])); // A
      v9->AddNode(tpcrrod, 4, new TGeoCombiTrans(x, y, -z, rotpos[1])); // C
    } else {
      v9->AddNode(tpcmrod, i + 37, new TGeoTranslation(x, y, z));      // shaft
      v9->AddNode(tpcmrod, i + 55, new TGeoCombiTrans(x, y, -z, ref)); // muon
    }
    if (i == 15) {
      v9->AddNode(hvrv, 1, new TGeoTranslation(x, y, z + 0.7)); // hv->A-side only
    }
    //
    // outer clamps
    //
    r = 256.9;
    x = r * TMath::Cos(angle);
    y = r * TMath::Sin(angle);
    rot12 = new TGeoRotation();
    rot12->RotateZ(i * 20.);
    //
    // A-side
    v9->AddNode(oclamp, 7 * i + 1, new TGeoCombiTrans(x, y, 5.25, rot12));
    v9->AddNode(oclamp, 7 * i + 2, new TGeoCombiTrans(x, y, 38.25, rot12));
    v9->AddNode(oclamp, 7 * i + 3, new TGeoCombiTrans(x, y, 80.25, rot12));
    v9->AddNode(oclamp, 7 * i + 4, new TGeoCombiTrans(x, y, 122.25, rot12));
    v9->AddNode(oclamp, 7 * i + 5, new TGeoCombiTrans(x, y, 164.25, rot12));
    v9->AddNode(oclamp, 7 * i + 6, new TGeoCombiTrans(x, y, 206.25, rot12));
    v9->AddNode(oclamp, 7 * i + 7, new TGeoCombiTrans(x, y, 246.75, rot12));
    // C-side
    v9->AddNode(oclamp, 7 * i + 127, new TGeoCombiTrans(x, y, -5.25, rot12));
    v9->AddNode(oclamp, 7 * i + 128, new TGeoCombiTrans(x, y, -38.25, rot12));
    v9->AddNode(oclamp, 7 * i + 129, new TGeoCombiTrans(x, y, -80.25, rot12));
    v9->AddNode(oclamp, 7 * i + 130, new TGeoCombiTrans(x, y, -122.25, rot12));
    v9->AddNode(oclamp, 7 * i + 131, new TGeoCombiTrans(x, y, -164.25, rot12));
    v9->AddNode(oclamp, 7 * i + 132, new TGeoCombiTrans(x, y, -206.25, rot12));
    v9->AddNode(oclamp, 7 * i + 133, new TGeoCombiTrans(x, y, -246.75, rot12));

  } // end of rods positioning

  TGeoVolume* alice = gGeoManager->GetVolume("cave");
  alice->AddNode(v1, 1);

} // end of function

void Detector::LoadGeometryFromFile()
{
  // ===| Read the TPC geometry from file |=====================================
  if (mGeoFileName.IsNull()) {
    LOG(FATAL) << "TPC geometry file name not set";
    return;
  }

  TFile* fGeoFile = TFile::Open(mGeoFileName);
  if (!fGeoFile || !fGeoFile->IsOpen() || fGeoFile->IsZombie()) {
    LOG(FATAL) << "Could not open TPC geometry file '" << mGeoFileName << "'";
    return;
  }

  TGeoVolume* tpcVolume = dynamic_cast<TGeoVolume*>(fGeoFile->Get("TPC_M"));
  if (!tpcVolume) {
    LOG(FATAL) << "Could not retrieve TPC geometry from file '" << mGeoFileName << "'";
    return;
  }

  LOG(INFO) << "Loaded TPC geometry from file '" << mGeoFileName << "'";
  TGeoVolume* alice = gGeoManager->GetVolume("cave");
  alice->AddNode(tpcVolume, 1);
}

void Detector::defineSensitiveVolumes()
{
  TGeoManager* geoManager = gGeoManager;
  TGeoVolume* v = nullptr;

  // const Int_t nSensitive=2;
  // const char* volumeNames[nSensitive]={"TPC_Drift","TPC_Strip"};
  const Int_t nSensitive = 1;
  const char* volumeNames[nSensitive] = {"TPC_Drift"};

  // The names of the ITS sensitive volumes have the format: ITSUSensor(0...mNumberLayers-1)
  for (Int_t ivol = 0; ivol < nSensitive; ++ivol) {
    TString volumeName = volumeNames[ivol];
    v = geoManager->GetVolume(volumeName.Data());
    if (!v) {
      LOG(ERROR) << "Could not find volume '" << volumeName << "'";
      continue;
    }

    // set volume sentive
    AddSensitiveVolume(v);
  }
}

Double_t Detector::Gamma(Double_t k)
{
  static thread_local Double_t n = 0;
  static thread_local Double_t c1 = 0;
  static thread_local Double_t c2 = 0;
  static thread_local Double_t b1 = 0;
  static thread_local Double_t b2 = 0;
  if (k > 0) {
    if (k < 0.4)
      n = 1. / k;
    else if (k >= 0.4 && k < 4)
      n = 1. / k + (k - 0.4) / k / 3.6;
    else if (k >= 4.)
      n = 1. / TMath::Sqrt(k);
    b1 = k - 1. / n;
    b2 = k + 1. / n;
    c1 = (k < 0.4) ? 0 : b1 * (TMath::Log(b1) - 1.) / 2.;
    c2 = b2 * (TMath::Log(b2) - 1.) / 2.;
  }
  Double_t x;
  Double_t y = -1.;
  while (1) {
    Double_t nu1 = gRandom->Rndm();
    Double_t nu2 = gRandom->Rndm();
    Double_t w1 = c1 + TMath::Log(nu1);
    Double_t w2 = c2 + TMath::Log(nu2);
    y = n * (b1 * w2 - b2 * w1);
    if (y < 0)
      continue;
    x = n * (w2 - w1);
    if (TMath::Log(y) >= x)
      break;
  }
  return TMath::Exp(x);
}

std::string Detector::getHitBranchNames(int probe) const
{
  if (probe >= 0 && probe < Sector::MAXSECTOR) {
    TString name;
    name.Form("%sHitsShiftedSector%d", GetName(), probe);
    return std::string(name.Data());
  }
  return std::string();
}

ClassImp(o2::tpc::Detector);
