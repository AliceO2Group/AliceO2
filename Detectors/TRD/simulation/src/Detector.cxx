// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TRDSimulation/Detector.h"
#include <TGeoManager.h>
#include <TVirtualMC.h>
#include <vector>
#include "FairVolume.h"
#include "FairRootManager.h"
#include "TRDBase/TRDCommonParam.h"
#include "TRDBase/TRDGeometry.h"
#include "SimulationDataFormat/Stack.h"
#include "CommonUtils/ShmAllocator.h"
#include <stdexcept>

using namespace o2::trd;

Detector::Detector(Bool_t active)
  : o2::base::DetImpl<Detector>("TRD", active)
{
  mHits = o2::utils::createSimVector<HitType>();
  if (TRDCommonParam::Instance()->IsXenon()) {
    mWion = 23.53; // Ionization energy XeCO2 (85/15)
  } else if (TRDCommonParam::Instance()->IsArgon()) {
    mWion = 27.21; // Ionization energy ArCO2 (82/18)
  } else {
    LOG(FATAL) << "Wrong gas mixture";
  }
  // Switch on TR simulation as default
  mTRon = true;
  if (!mTRon) {
    LOG(INFO) << "TR simulation off";
  } else {
    mTR = new TRsim();
  }
}

Detector::Detector(const Detector& rhs)
  : o2::base::DetImpl<Detector>(rhs),
    mHits(o2::utils::createSimVector<HitType>()),
    mFoilDensity(rhs.mFoilDensity),
    mGasNobleFraction(rhs.mGasNobleFraction),
    mGasDensity(rhs.mGasDensity),
    mGeom(rhs.mGeom)
{
  if (TRDCommonParam::Instance()->IsXenon()) {
    mWion = 23.53; // Ionization energy XeCO2 (85/15)
  } else if (TRDCommonParam::Instance()->IsArgon()) {
    mWion = 27.21; // Ionization energy ArCO2 (82/18)
  } else {
    LOG(FATAL) << "Wrong gas mixture";
    // add hard exit here!
  }
  // Switch on TR simulation as default
  mTRon = true;
  if (!mTRon) {
    LOG(INFO) << "TR simulation off";
  } else {
    mTR = new TRsim();
  }
}

Detector::~Detector()
{
  o2::utils::freeSimVector(mHits);
}

void Detector::InitializeO2Detector()
{
  // register the sensitive volumes with FairRoot
  defineSensitiveVolumes();
}

bool Detector::ProcessHits(FairVolume* v)
{
  // If not charged track or already stopped or disappeared, just return.
  if ((!fMC->TrackCharge()) || fMC->IsTrackDisappeared()) {
    return false;
  }

  // Inside sensitive volume ?
  bool drRegion = false;
  bool amRegion = false;
  const TString cIdSensDr = "J";
  const TString cIdSensAm = "K";
  const TString cIdCurrent = fMC->CurrentVolName();
  if (cIdCurrent[1] == cIdSensDr) {
    drRegion = true;
  }
  if (cIdCurrent[1] == cIdSensAm) {
    amRegion = true;
  }
  if (!drRegion && !amRegion) {
    return false;
  }

  // Determine the dectector number
  int sector, det;
  // The plane number and chamber number
  char cIdChamber[3];
  cIdChamber[0] = cIdCurrent[2];
  cIdChamber[1] = cIdCurrent[3];
  cIdChamber[2] = 0;
  // The det-sec number (0 – 29)
  const int idChamber = mGeom->getDetectorSec(atoi(cIdChamber));
  // The sector number (0 - 17), according to the standard coordinate system
  TString cIdPath = fMC->CurrentVolPath();
  char cIdSector[3];
  cIdSector[0] = cIdPath[21];
  cIdSector[1] = cIdPath[22];
  cIdSector[2] = 0;
  sector = atoi(cIdSector);
  // The detector number (0 – 539)
  det = mGeom->getDetector(mGeom->getLayer(idChamber), mGeom->getStack(idChamber), sector);

  // 0: InFlight 1: Entering 2: Exiting
  int trkStat = 0;

  o2::data::Stack* stack = (o2::data::Stack*)fMC->GetStack();
  float xp, yp, zp;
  float px, py, pz, etot;
  float trackLength = fMC->TrackLength(); // Return the length of the current track from its origin (in cm)
  float tof = fMC->TrackTime();           // Return the current time of flight of the track being transported (in s).

  // Special hits if track is entering
  if (drRegion && fMC->IsTrackEntering()) {
    // Create a track reference at the entrance of each
    // chamber that contains the momentum components of the particle
    fMC->TrackMomentum(px, py, pz, etot);
    fMC->TrackPosition(xp, yp, zp);
    stack->addTrackReference(o2::TrackReference(xp, yp, zp, px, py, pz,
                                                trackLength,
                                                tof,
                                                stack->GetCurrentTrackNumber(),
                                                GetDetId()));
    // Update track status
    trkStat = 1;
    // Create the hits from TR photons if electron/positron is entering the drift volume
    const bool ele = (TMath::Abs(fMC->TrackPid()) == 11); // electron PDG code.
    if (mTRon && ele) {
      createTRhit(det);
    }
  } else if (amRegion && fMC->IsTrackExiting()) {
    // Create a track reference at the exit of each
    // chamber that contains the momentum components of the particle
    fMC->TrackMomentum(px, py, pz, etot);
    fMC->TrackPosition(xp, yp, zp);
    stack->addTrackReference(o2::TrackReference(xp, yp, zp, px, py, pz,
                                                trackLength,
                                                tof,
                                                stack->GetCurrentTrackNumber(),
                                                GetDetId()));
    // Update track status
    trkStat = 2;
  }

  // Calculate the charge according to GEANT Edep
  // Create a new dEdx hit
  const float enDep = TMath::Max(fMC->Edep(), 0.0) * 1e9; // Energy in eV
  const int totalChargeDep = (int)(enDep / mWion);        // Total charge

  // Store those hits with enDep bigger than the ionization potential of the gas mixture for in-flight tracks
  // or store hits of tracks that are entering or exiting
  if (totalChargeDep || trkStat) {
    fMC->TrackPosition(xp, yp, zp);
    tof = tof * 1e6; // The time of flight in micro-seconds
    const int trackID = stack->GetCurrentTrackNumber();
    addHit(xp, yp, zp, tof, totalChargeDep, trackID, det, drRegion);
    stack->addHit(GetDetId());
    return true;
  }
  return false;
}

void Detector::createTRhit(int det)
{
  //
  // Creates an electron cluster from a TR photon.
  // The photon is assumed to be created a the end of the radiator. The
  // distance after which it deposits its energy takes into account the
  // absorbtion of the entrance window and of the gas mixture in drift
  // volume.
  //

  // Maximum number of TR photons per track
  constexpr int mMaxNumberOfTRPhotons = 50; // Make this a class member?

  float px, py, pz, etot;
  fMC->TrackMomentum(px, py, pz, etot);
  float pTot = TMath::Sqrt(px * px + py * py + pz * pz);
  std::vector<float> photonEnergyContainer;            // energy in keV
  mTR->createPhotons(11, pTot, photonEnergyContainer); // Create TR photons
  if (photonEnergyContainer.size() > mMaxNumberOfTRPhotons) {
    LOG(ERROR) << "Boundary error: nTR = " << photonEnergyContainer.size() << ", mMaxNumberOfTRPhotons = " << mMaxNumberOfTRPhotons;
  }

  // Loop through the TR photons
  for (const float& photonEnergy : photonEnergyContainer) {
    const double energyMeV = photonEnergy * 1e-3;
    const double energyeV = photonEnergy * 1e3;
    double absLength = 0.0;
    double sigma = 0.0;
    // Take the absorbtion in the entrance window into account
    double muMy = mTR->getMuMy(energyMeV);
    sigma = muMy * mFoilDensity;
    if (sigma > 0.0) {
      absLength = gRandom->Exp(1.0 / sigma);
      if (absLength < TRDGeometry::myThick()) {
        continue;
      }
    } else {
      continue;
    }
    // The absorbtion cross sections in the drift gas
    // Gas-mixture (Xe/CO2)
    double muNo = 0.0;
    if (TRDCommonParam::Instance()->IsXenon()) {
      muNo = mTR->getMuXe(energyMeV);
    } else if (TRDCommonParam::Instance()->IsArgon()) {
      muNo = mTR->getMuAr(energyMeV);
    }
    double muCO = mTR->getMuCO(energyMeV);
    double fGasNobleFraction = 1;
    double fGasDensity = 1;
    sigma = (fGasNobleFraction * muNo + (1.0 - fGasNobleFraction) * muCO) * fGasDensity * mTR->getTemp();

    // The distance after which the energy of the TR photon
    // is deposited.
    if (sigma > 0.0) {
      absLength = gRandom->Exp(1.0 / sigma);
      if (absLength > (TRDGeometry::drThick() + TRDGeometry::amThick())) {
        continue;
      }
    } else {
      continue;
    }

    // The position of the absorbtion
    float xp, yp, zp;
    fMC->TrackPosition(xp, yp, zp);
    float invpTot = 1. / pTot;
    float x = xp + px * invpTot * absLength;
    float y = yp + py * invpTot * absLength;
    float z = zp + pz * invpTot * absLength;

    // Add the hit to the array. TR photon hits are marked by negative energy (and not by charge)
    float tof = fMC->TrackTime() * 1e6; // The time of flight in micro-seconds
    o2::data::Stack* stack = (o2::data::Stack*)fMC->GetStack();
    const int trackID = stack->GetCurrentTrackNumber();
    const int totalChargeDep = -1 * (int)(energyeV / mWion); // Negative charge for tagging TR photon hits
    addHit(x, y, z, tof, totalChargeDep, trackID, det, true); // All TR hits are in drift region
    stack->addHit(GetDetId());
  }
}

void Detector::Register()
{
  FairRootManager::Instance()->RegisterAny(addNameTo("Hit").data(), mHits, true);
}

void Detector::FinishEvent()
{
  // Sort hit vector by detector number before the End of the Event
  std::sort(mHits->begin(), mHits->end(),
            [](const HitType& a, const HitType& b) {
              return a.GetDetectorID() < b.GetDetectorID();
            });
}

// this is very problematic; we should do round robin or the clear needs
// to be done by the HitMerger
void Detector::Reset()
{
  if (!o2::utils::ShmManager::Instance().isOperational()) {
    mHits->clear();
  }
}

void Detector::EndOfEvent() { Reset(); }

//_____________________________________________________________________________
void Detector::createMaterials()
{
  //
  // Create the materials for the TRD
  //
  int isxfld = 2;
  float sxmgmx = 10.;
  o2::base::Detector::initFieldTrackingParams(isxfld, sxmgmx);

  //////////////////////////////////////////////////////////////////////////
  //     Define Materials
  //////////////////////////////////////////////////////////////////////////

  // Aluminum
  Material(1, "Al", 26.98, 13.0, 2.7, 8.9, 37.2);
  // Copper
  Material(2, "Cu", 63.54, 29.0, 8.96, 1.43, 14.8);
  // Carbon
  Material(3, "C", 12.01, 6.0, 2.265, 18.8, 74.4);
  // Carbon for fiber mats
  Material(4, "C2", 12.01, 6.0, 1.75, 18.8, 74.4);
  // Zinc
  Material(5, "Sn", 118.71, 50.0, 7.31, 1.21, 14.8);
  // Silicon
  Material(6, "Si", 28.09, 14.0, 2.33, 9.36, 37.2);
  // Iron
  Material(7, "Fe", 55.85, 26.0, 7.87, 1.76, 14.8);

  // Air
  float aAir[4] = { 12.011, 14.0, 15.9994, 36.0 };
  float zAir[4] = { 6.0, 7.0, 8.0, 18.0 };
  float wAir[4] = { 0.000124, 0.755267, 0.231781, 0.012827 };
  float dAir = 1.20479e-03;
  Mixture(51, "Air", aAir, zAir, dAir, 4, wAir);
  // Polyethilene (CH2)
  float ape[2] = { 12.011, 1.0079 };
  float zpe[2] = { 6.0, 1.0 };
  float wpe[2] = { 1.0, 2.0 };
  float dpe = 0.95;
  Mixture(52, "Polyethilene", ape, zpe, dpe, -2, wpe);
  // Gas mixtures
  // Xe/CO2-gas-mixture (85% / 15%)
  float aXeCO2[3] = { 131.29, 12.0107, 15.9994 };
  float zXeCO2[3] = { 54.0, 6.0, 8.0 };
  float wXeCO2[3] = { 8.5, 1.5, 3.0 };
  float fxc = 0.85;
  float dxe = 0.00549; // at 20C
  float dco = 0.00186; // at 20C
  float dgmXe = fxc * dxe + (1.0 - fxc) * dco;
  // Ar/CO2-gas-mixture
  float aArCO2[3] = { 39.948, 12.0107, 15.9994 };
  float zArCO2[3] = { 18.0, 6.0, 8.0 };
  float wArCO2[3] = { 8.2, 1.8, 3.6 };
  float fac = 0.82;
  float dar = 0.00166; // at 20C
  float dgmAr = fac * dar + (1.0 - fac) * dco;
  if (TRDCommonParam::Instance()->IsXenon()) {
    Mixture(53, "XeCO2", aXeCO2, zXeCO2, dgmXe, -3, wXeCO2);
  } else if (TRDCommonParam::Instance()->IsArgon()) {
    LOG(INFO) << "Gas mixture: Ar C02 (80/20)";
    Mixture(53, "ArCO2", aArCO2, zArCO2, dgmAr, -3, wArCO2);
  } else {
    LOG(FATAL) << "Wrong gas mixture";
    exit(1);
  }
  // G10
  float aG10[4] = { 1.0079, 12.011, 15.9994, 28.086 };
  float zG10[4] = { 1.0, 6.0, 8.0, 14.0 };
  float wG10[4] = { 0.023, 0.194, 0.443, 0.340 };
  float dG10 = 2.0;
  Mixture(54, "G10", aG10, zG10, dG10, 4, wG10);
  // Water
  float awa[2] = { 1.0079, 15.9994 };
  float zwa[2] = { 1.0, 8.0 };
  float wwa[2] = { 2.0, 1.0 };
  float dwa = 1.0;
  Mixture(55, "Water", awa, zwa, dwa, -2, wwa);
  // Rohacell (C5H8O2), X0 = 535.005cm
  float arh[3] = { 12.011, 1.0079, 15.9994 };
  float zrh[3] = { 6.0, 1.0, 8.0 };
  float wrh[3] = { 5.0, 8.0, 2.0 };
  float drh = 0.075;
  Mixture(56, "Rohacell", arh, zrh, drh, -3, wrh);
  // Epoxy (C18H19O3)
  float aEpoxy[3] = { 15.9994, 1.0079, 12.011 };
  float zEpoxy[3] = { 8.0, 1.0, 6.0 };
  float wEpoxy[3] = { 3.0, 19.0, 18.0 };
  float dEpoxy = 1.8;
  Mixture(57, "Epoxy", aEpoxy, zEpoxy, dEpoxy, -3, wEpoxy);
  // Araldite, low density epoxy (C18H19O3)
  float aAral[3] = { 15.9994, 1.0079, 12.011 };
  float zAral[3] = { 8.0, 1.0, 6.0 };
  float wAral[3] = { 3.0, 19.0, 18.0 };
  float dAral = 1.12; // Hardener: 1.15, epoxy: 1.1, mixture: 1/2
  Mixture(58, "Araldite", aAral, zAral, dAral, -3, wAral);
  // Mylar
  float aMy[3] = { 12.011, 1.0, 15.9994 };
  float zMy[3] = { 6.0, 1.0, 8.0 };
  float wMy[3] = { 5.0, 4.0, 2.0 };
  float dMy = 1.39;
  Mixture(59, "Mylar", aMy, zMy, dMy, -3, wMy);
  // Polypropylene (C3H6) for radiator fibers
  float app[2] = { 12.011, 1.0079 };
  float zpp[2] = { 6.0, 1.0 };
  float wpp[2] = { 3.0, 6.0 };
  float dpp = 0.068;
  Mixture(60, "Polypropylene", app, zpp, dpp, -2, wpp);
  // Aramide for honeycomb
  float aAra[4] = { 1.0079, 12.011, 15.9994, 14.0067 };
  float zAra[4] = { 1.0, 6.0, 8.0, 7.0 };
  float wAra[4] = { 3.0, 1.0, 1.0, 1.0 };
  float dAra = 0.032;
  Mixture(61, "Aramide", aAra, zAra, dAra, -4, wAra);
  // GFK for Wacosit (Epoxy + Si)
  float aGFK[4] = { 1.0079, 12.011, 15.9994, 28.086 };
  float zGFK[4] = { 1.0, 6.0, 8.0, 14.0 };
  float wGFK[4] = { 0.0445, 0.5031, 0.1118, 0.340 };
  float dGFK = 2.0;
  Mixture(62, "GFK", aGFK, zGFK, dGFK, 4, wGFK);

  //////////////////////////////////////////////////////////////////////////
  //     Tracking Media Parameters
  //////////////////////////////////////////////////////////////////////////

  // General tracking parameter
  float tmaxfd = -10.0;
  float stemax = -1.0e10;
  float deemax = -0.1;
  float epsil = 1.0e-4;
  float stmin = -0.001;

  // Al Frame
  Medium(1, "Al Frame", 1, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  // Air
  Medium(2, "Air", 51, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  // Wires
  Medium(3, "Wires", 2, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  // All other ROB materials (caps, etc.)
  Medium(4, "ROB Other", 2, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  // Cu pads
  Medium(5, "Padplane", 2, 1, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  // Fee + cables
  Medium(6, "Readout", 2, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  // C frame (Wacosit)
  Medium(7, "Wacosit", 62, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  // INOX of cooling bus bars
  Medium(8, "Cooling bus", 7, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  // Gas-mixture (Xe/CO2)
  Medium(9, "Gas-mix", 53, 1, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  // Honeycomb
  Medium(10, "Honeycomb", 61, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  // Araldite glue
  Medium(11, "Glue", 58, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  // G10-plates
  Medium(13, "G10-plates", 54, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  // Cooling water
  Medium(14, "Water", 55, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  // Rohacell for the radiator
  Medium(15, "Rohacell", 56, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  // Al layer in MCMs
  Medium(16, "MCM-Al", 1, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  // Sn layer in MCMs
  Medium(17, "MCM-Sn", 5, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  // Cu layer in MCMs
  Medium(18, "MCM-Cu", 2, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  // G10 layer in MCMs
  Medium(19, "MCM-G10", 54, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  // Si in readout chips
  Medium(20, "Chip-Si", 6, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  // Epoxy in readout chips
  Medium(21, "Chip-Ep", 57, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  // PE in connectors
  Medium(22, "Conn-PE", 52, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  // Cu in connectors
  Medium(23, "Chip-Cu", 2, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  // Al of cooling pipes
  Medium(24, "Cooling", 1, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  // Cu in services
  Medium(25, "Serv-Cu", 2, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  // Carbon fiber mat
  Medium(26, "Carbon", 4, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  // Mylar foil
  Medium(27, "Mylar", 59, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  // Polypropylene fibers
  Medium(28, "Fiber", 60, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

  // Save the density values for the TRD absorbtion
  float dmy = 1.39;
  mFoilDensity = dmy;
  if (TRDCommonParam::Instance()->IsXenon()) {
    mGasDensity = dgmXe;
    mGasNobleFraction = fxc;
  } else if (TRDCommonParam::Instance()->IsArgon()) {
    mGasDensity = dgmAr;
    mGasNobleFraction = fac;
  }
}

// setting up the geometry
void Detector::ConstructGeometry()
{
  createMaterials();

  std::vector<int> medmapping;
  // now query the medium mapping and fill a vector to be passed along
  // to the geometry creation
  getMediumIDMappingAsVector(medmapping);

  mGeom = new TRDGeometry();
  mGeom->createGeometry(medmapping);
}

void Detector::defineSensitiveVolumes()
{
  auto vols = mGeom->getSensitiveTRDVolumes();
  for (auto& name : vols) {
    auto tgeovol = gGeoManager->GetVolume(name.c_str());
    if (tgeovol != nullptr) {
      AddSensitiveVolume(tgeovol);
    } else {
      LOG(ERROR) << "No TGeo volume for TRD vol name " << name << " found\n";
    }
  }
}

void Detector::addAlignableVolumes() const
{
  mGeom->addAlignableVolumes();
}

ClassImp(Detector);
