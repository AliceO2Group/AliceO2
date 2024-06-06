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

#include <TParticle.h>
#include <TVirtualMC.h>

#include <FairVolume.h>

#include "FOCALSimulation/Detector.h"

using namespace o2::focal;

Detector::Detector(const Detector& rhs)
{
}

Detector::~Detector()
{
  o2::utils::freeSimVector(mHits);
}

void Detector::InitializeO2Detector()
{
}

Bool_t Detector::ProcessHits(FairVolume* v)
{
  int track = fMC->GetStack()->GetCurrentTrackNumber(),
      directparent = fMC->GetStack()->GetCurrentParentTrackNumber();
  // Like other calorimeters FOCAL will create a huge amount of shower particles during tracking
  // Instead, the hits should be assigned to the incoming particle in FOCAL.
  // Implementation of the incoming particle search taken from implementation in EMCAL.
  if (track != mCurrentTrack) {
    LOG(debug4) << "Doing new track " << track << " current (" << mCurrentTrack << "), direct parent (" << directparent << ")" << std::endl;
    // new current track - check parentage
    auto hasSuperParent = mSuperParentsIndices.find(directparent);
    if (hasSuperParent != mSuperParentsIndices.end()) {
      // same superparent as direct parent
      mCurrentParentID = hasSuperParent->second;
      mSuperParentsIndices[track] = hasSuperParent->second;
      auto superparent = mSuperParents.find(mCurrentParentID);
      if (superparent != mSuperParents.end()) {
        mCurrentSuperparent = &(superparent->second);
      } else {
        LOG(error) << "Attention: No superparent object found (parent " << mCurrentParentID << ")";
        mCurrentSuperparent = nullptr;
      }
      LOG(debug4) << "Found superparent " << mCurrentParentID << std::endl;
    } else {
      // start of new chain
      // for new incoming tracks the super parent index is equal to the track ID (for recursion)
      mSuperParentsIndices[track] = track;
      mCurrentSuperparent = AddSuperparent(track, fMC->TrackPid(), fMC->Etot());
      mCurrentParentID = track;
    }
    mCurrentTrack = track;
  }
  return true;
}

Parent* Detector::AddSuperparent(int trackID, int pdg, double energy)
{
  LOG(debug3) << "Adding superparent for track " << trackID << " with PID " << pdg << " and energy " << energy;
  auto entry = mSuperParents.insert({trackID, {pdg, energy, false}});
  return &(entry.first->second);
}

void Detector::EndOfEvent() { Reset(); }

void Detector::Register()
{
  FairRootManager::Instance()->RegisterAny(addNameTo("Hit").data(), mHits, kTRUE);
}

void Detector::Reset()
{
  LOG(debug) << "Cleaning FOCAL hits ...";
  if (!o2::utils::ShmManager::Instance().isOperational()) {
    mHits->clear();
  }

  mSuperParentsIndices.clear();
  mSuperParents.clear();
  mCurrentTrack = -1;
  mCurrentParentID = -1;
}

void Detector::CreateMaterials()
{

  // --- Define the various materials for GEANT ---

  /// Silicon
  float aSi = 28.09;
  float zSi = 14.0;
  float dSi = 2.33;
  float x0Si = 9.36;
  Material(1, "Si $", aSi, zSi, dSi, x0Si, 18.5);

  //// W Tungsten
  float aW = 183.84;
  float zW = 74.0;
  float dW = 19.3;
  float x0W = 0.35;
  Material(0, "W $", aW, zW, dW, x0W, 17.1);

  // Cu
  Material(3, "Cu   $", 63.54, 29., 8.96, 1.43, 15.);

  //// Pb
  Material(10, "Pb    $", 207.19, 82., 11.35, .56, 18.5);

  //// Scintillator (copied from EMCal)
  // --- The polysterene scintillator (CH) ---
  float aP[2] = {12.011, 1.00794};
  float zP[2] = {6.0, 1.0};
  float wP[2] = {1.0, 1.0};
  float dP = 1.032;
  Mixture(11, "Polystyrene$", aP, zP, dP, -2, wP);

  // G10

  float aG10[4] = {1., 12.011, 15.9994, 28.086};
  float zG10[4] = {1., 6., 8., 14.};
  // PH  float wG10[4]={0.148648649,0.104054054,0.483499056,0.241666667};
  float wG10[4] = {0.15201, 0.10641, 0.49444, 0.24714};
  Mixture(2, "G10  $", aG10, zG10, 1.7, 4, wG10);

  //// 94W-4Ni-2Cu
  float aAlloy[3] = {183.84, 58.6934, 63.54};
  float zAlloy[3] = {74.0, 28, 29};
  float wAlloy[3] = {0.94, 0.04, 0.02};
  float dAlloy = wAlloy[0] * 19.3 + wAlloy[1] * 8.908 + wAlloy[2] * 8.96;
  Mixture(5, "Alloy $", aAlloy, zAlloy, dAlloy, 3, wAlloy);

  // Steel
  float aSteel[4] = {55.847, 51.9961, 58.6934, 28.0855};
  float zSteel[4] = {26., 24., 28., 14.};
  float wSteel[4] = {.715, .18, .1, .005};
  float dSteel = 7.88;
  Mixture(4, "STAINLESS STEEL$", aSteel, zSteel, dSteel, 4, wSteel);

  // Air
  float aAir[4] = {12.0107, 14.0067, 15.9994, 39.948};
  float zAir[4] = {6., 7., 8., 18.};
  float wAir[4] = {0.000124, 0.755268, 0.231781, 0.012827};
  float dAir1 = 1.20479E-10;
  float dAir = 1.20479E-3;
  Mixture(98, "Vacum$", aAir, zAir, dAir1, 4, wAir);
  Mixture(99, "Air  $", aAir, zAir, dAir, 4, wAir);

  // Ceramic
  //  Ceramic   97.2% Al2O3 , 2.8% SiO2
  //  float wcer[2]={0.972,0.028};  // Not used
  float aal2o3[2] = {26.981539, 15.9994};
  float zal2o3[2] = {13., 8.};
  float wal2o3[2] = {2., 3.};
  float denscer = 3.6;
  // SiO2
  float aglass[2] = {28.0855, 15.9994};
  float zglass[2] = {14., 8.};
  float wglass[2] = {1., 2.};
  float dglass = 2.65;
  Mixture(6, "Al2O3   $", aal2o3, zal2o3, denscer, -2, wal2o3);
  Mixture(7, "glass   $", aglass, zglass, dglass, -2, wglass);

  // Ceramic is a mixtur of glass and Al2O3 ?
  //   Not clear how to do this with AliMixture
  //   Not needed; so skip for now

  /*
float acer[2],zcer[2];
char namate[21]="";
float a,z,d,radl,absl,buf[1];
Int_t nbuf;
gMC->Gfmate((*fIdmate)[6], namate, a, z, d, radl, absl, buf, nbuf);
acer[0]=a;
zcer[0]=z;
gMC->Gfmate((*fIdmate)[7], namate, a, z, d, radl, absl, buf, nbuf);
acer[1]=a;
zcer[1]=z;

AliMixture( 8, "Ceramic    $", acer, zcer, denscer, 2, wcer);
*/

  // Use Al2O3 instead:

  Mixture(8, "Ceramic    $", aal2o3, zal2o3, denscer, -2, wal2o3);

  // Define tracking media
  // format

  float tmaxfdSi = 10.0; // 0.1; // .10000E+01; // Degree
  float stemaxSi = 0.1;  //  .10000E+01; // cm
  float deemaxSi = 0.1;  // 0.30000E-02; // Fraction of particle's energy 0<deemax<=1
  // float epsilSi  = 1.e-3;//1e-3;//1.0E-4;// .10000E+01;
  float epsilSi = 1.e-3; // 1.0E-4;// .10000E+01; // This drives the step size ? 1e-4 makes multiple steps even in pixels?
  float stminSi = 0.001; // cm "Default value used"

  float epsil = 0.001;
  // MvL: need to look up itdmed dynamically?
  // or move to TGeo: uses pointers for medium

  Int_t isxfld = 2;
  Float_t sxmgmx = 10.0;
  o2::base::Detector::initFieldTrackingParams(isxfld, sxmgmx);

  /// W plate -> idtmed[3599];
  Medium(ID_TUNGSTEN, "W conv.$", 0, 0,
         isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.1, nullptr, 0);
  /// Si plate  -> idtmed[3600];
  Medium(ID_SILICON, "Si sens$", 1, 0,
         isxfld, sxmgmx, tmaxfdSi, stemaxSi, deemaxSi, epsilSi, stminSi, nullptr, 0);

  //// G10 plate -> idtmed[3601];
  Medium(ID_G10, "G10 plate$", 2, 0,
         isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.01, nullptr, 0);

  //// Cu plate --> idtmed[3602];
  Medium(ID_COPPER, "Cu      $", 3, 0,
         isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.0001, nullptr, 0);

  //// S steel -->  idtmed[3603];
  Medium(ID_STEEL, "S  steel$", 4, 0,
         isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.0001, nullptr, 0);

  //// Alloy --> idtmed[3604];
  Medium(ID_ALLOY, "Alloy  conv.$", 5, 0,
         isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.1, nullptr, 0);

  //// Ceramic --> idtmed[3607]
  Medium(ID_CERAMIC, "Ceramic$", 8, 0,
         isxfld, sxmgmx, 10.0, 0.01, 0.1, 0.003, 0.003, nullptr, 0);

  // HCAL materials   // Need to double-check  tracking pars for this
  /// Pb plate --> idtmed[3608]
  Medium(ID_PB, "Pb    $", 10, 0,
         isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.1, nullptr, 0);
  /// Scintillator --> idtmed[3609]
  Medium(ID_SC, "Scint $", 11, 0,
         isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.001, 0.001, nullptr, 0);

  /// Si plate  -> idtmed[3610];
  Medium(ID_SIINSENS, "Si insens$", 1, 0,
         10.0, 0.1, 0.1, epsil, 0.001, 0, 0);

  /// idtmed[3697]
  Medium(ID_VAC, "Vacuum  $", 98, 0,
         isxfld, sxmgmx, 10.0, 1.0, 0.1, 0.1, 1.0, nullptr, 0);

  /// idtmed[3698]
  Medium(ID_AIR, "Air gaps$", 99, 0,
         isxfld, sxmgmx, 10.0, 1.0, 0.1, epsil, 0.001, nullptr, 0);
}

void Detector::ConstructGeometry()
{
}

void Detector::BeginPrimary()
{
  mCurrentPrimaryID = fMC->GetStack()->GetCurrentTrackNumber();
  LOG(debug) << "Starting primary " << mCurrentPrimaryID << " with energy " << fMC->GetStack()->GetCurrentTrack()->Energy();
}

void Detector::FinishPrimary()
{
  LOG(debug) << "Finishing primary " << mCurrentPrimaryID << std::endl;
  // Resetting primary and parent ID
  mCurrentPrimaryID = -1;
}