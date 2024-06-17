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
#include <TGeoVolume.h>
#include <TGeoManager.h>
#include <TGeoBBox.h>
#include <TGeoCompositeShape.h>

#include <FairVolume.h>

#include "DetectorsBase/Stack.h"
#include "FOCALSimulation/Detector.h"

using namespace o2::focal;

Detector::Detector(bool active, std::string geofilename)
  : o2::base::DetImpl<Detector>("FOC", active),
    mHits(o2::utils::createSimVector<Hit>()),
    mGeometry(nullptr),
    mMedSensHCal(-1),
    mGeoCompositions(0x0),
    mSuperParentsIndices(),
    mSuperParents(),
    mCurrentSuperparent(nullptr),
    mCurrentTrack(-1),
    mCurrentPrimaryID(-1),
    mCurrentParentID(-1),
    mVolumeIDScintillator(-1)
{
  mGeometry = getGeometry(geofilename);
}

Detector::Detector(const Detector& rhs)
  : o2::base::DetImpl<Detector>(rhs)
{
  mGeometry = rhs.mGeometry;
  mMedSensHCal = rhs.mMedSensHCal;
  mSensitiveHCAL = rhs.mSensitiveHCAL;
  mVolumeIDScintillator = rhs.mVolumeIDScintillator;
}

Detector::~Detector()
{
  o2::utils::freeSimVector(mHits);
}

Geometry* Detector::getGeometry(std::string name)
{
  if (!mGeometry) {
    mGeometry = Geometry::getInstance(name);
  }
  if (!mGeometry) {
    LOG(error) << "Failure accessing geometry";
  }
  return mGeometry;
}

void Detector::InitializeO2Detector()
{

  LOG(info) << "Intializing FOCAL detector";

  // All FOCAL volumes must be declared as sensitive, otherwise
  // the decay chains are broken by volumes not processed in ProceeHits
  for (const auto& child : mSensitiveHCAL) {
    LOG(debug1) << "Adding sensitive volume " << child;
    auto svolID = registerSensitiveVolumeAndGetVolID(child);
    if (child == "ScintFiber" || child == "HScint") {
      LOG(debug1) << "Adding ScintFiber/HScint volume as sensitive volume with ID " << svolID;
      mVolumeIDScintillator = svolID;
    }
  }

  mMedSensHCal = getMediumID(ID_SC);
}

Bool_t Detector::ProcessHits(FairVolume* v)
{
  int track = gMC->GetStack()->GetCurrentTrackNumber(),
      directparent = gMC->GetStack()->GetCurrentParentTrackNumber();
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
      mCurrentSuperparent = AddSuperparent(track, gMC->TrackPid(), gMC->Etot());
      mCurrentParentID = track;
    }
    mCurrentTrack = track;
  }

  // Processing HCAL hits
  if (gMC->CurrentMedium() == mMedSensHCal) {
    LOG(debug) << "We are in sensitive volume " << v->GetName() << ": " << gMC->CurrentVolPath() << std::endl;

    double eloss = gMC->Edep() * 1e9; // energy in eV  (GeV->eV)
    if (eloss < DBL_EPSILON) {
      return false; // only process hits which actually deposit some energy in the FOCAL
    }

    // In case of new parent track create new track reference
    auto o2stack = static_cast<o2::data::Stack*>(gMC->GetStack());
    if (!mCurrentSuperparent->mHasTrackReference) {
      float x, y, z, px, py, pz, e;
      gMC->TrackPosition(x, y, z);
      gMC->TrackMomentum(px, py, pz, e);
      o2::TrackReference trackref(x, y, z, px, py, pz, gMC->TrackLength(), gMC->TrackTime(), mCurrentParentID, GetDetId());
      o2stack->addTrackReference(trackref);
      mCurrentSuperparent->mHasTrackReference = true;
    }

    float posX, posY, posZ;
    gMC->TrackPosition(posX, posY, posZ);

    auto [indetector, col, row, layer, segment] = mGeometry->getVirtualInfo(posX, posY, posZ);

    if (!indetector) {
      // particle outside the detector
      return true;
    }

    auto currenthit = FindHit(mCurrentParentID, col, row, layer);
    if (!currenthit) {
      // Condition for new hit:
      // - Processing different partent track (parent track must be produced outside FOCAL)
      // - Inside different cell
      // - First track of the event
      Double_t time = gMC->TrackTime() * 1e9; // time in ns
      LOG(debug3) << "Adding new hit for parent " << mCurrentParentID << " and cell Col: " << col << " Row: " << row << " segment: " << segment << std::endl;

      /// check handling of primary particles
      AddHit(mCurrentParentID, mCurrentPrimaryID, mCurrentSuperparent->mEnergy, row * col + col, o2::focal::Hit::Subsystem_t::HCAL, math_utils::Point3D<float>(posX, posY, posZ), time, eloss);
      o2stack->addHit(GetDetId());
    } else {
      LOG(debug3) << "Adding energy to the current hit" << std::endl;
      currenthit->SetEnergyLoss(currenthit->GetEnergyLoss() + eloss);
    }
  }

  return true;
}

Hit* Detector::AddHit(int trackID, int primary, double initialEnergy, int detID, o2::focal::Hit::Subsystem_t subsystem,
                      const math_utils::Point3D<float>& pos, double time, double eLoss)
{
  LOG(debug3) << "Adding hit for track " << trackID << " with position (" << pos.X() << ", "
              << pos.Y() << ", " << pos.Z() << ")  with energy " << initialEnergy << " loosing " << eLoss;
  mHits->emplace_back(primary, trackID, detID, subsystem, initialEnergy, pos, time, eLoss);
  return &(mHits->back());
}

Hit* Detector::FindHit(int parentID, int col, int row, int layer)
{

  auto HitComparison = [&](const Hit& hit) {
    auto information = mGeometry->getVirtualInfo(hit.GetX(), hit.GetY(), hit.GetZ());
    // FIXME Should we compare segments instead of layers ???
    return hit.GetTrackID() == parentID && col == std::get<1>(information) && row == std::get<2>(information) && layer == std::get<3>(information);
  };

  auto result = std::find_if(mHits->begin(), mHits->end(), HitComparison);
  if (result == mHits->end()) {
    return nullptr;
  }
  return &(*result);
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

  // Al
  Material(9, "Al$", 26.98, 13.0, 2.7, 8.9, 37.2);

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

  int isxfld = 2;
  float sxmgmx = 10.0;
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
  Medium(ID_COPPER, "Cu$", 3, 0,
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
  Medium(ID_PB, "Pb // The Scintillator must be first in order in vector for Rin to be set$", 10, 0,
         isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.1, nullptr, 0);
  /// Scintillator --> idtmed[3609]
  Medium(ID_SC, "Scint$", 11, 0,
         isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.001, 0.001, nullptr, 0);

  /// Si plate  -> idtmed[3610];
  Medium(ID_SIINSENS, "Si insens$", 1, 0,
         10.0, 0.1, 0.1, epsil, 0.001, 0, 0);

  // Al for the cold plates
  Medium(ID_ALUMINIUM, "Aluminium$", 9, 0,
         isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.1, 0, 0);

  /// idtmed[3697]
  Medium(ID_VAC, "Vacuum  $", 98, 0,
         isxfld, sxmgmx, 10.0, 1.0, 0.1, 0.1, 1.0, nullptr, 0);

  /// idtmed[3698]
  Medium(ID_AIR, "Air gaps$", 99, 0,
         isxfld, sxmgmx, 10.0, 1.0, 0.1, epsil, 0.001, nullptr, 0);
}

//____________________________________________________________________________
void Detector::addAlignableVolumes() const
{
  // Create entries for alignable volumes associating the symbolic volume
  // name with the corresponding volume path. Needs to be syncronized with
  // eventual changes in the geometry
  // Alignable volumes are:

  // AddAlignableVolumesECAL();
  addAlignableVolumesHCAL();
}

//____________________________________________________________________________
void Detector::addAlignableVolumesHCAL() const
{
  TString vpsector = "/cave_1/barrel_1/FOCAL_1/HCAL_1";
  TString snsector = "FOCAL/HCAL";

  if (!gGeoManager->SetAlignableEntry(snsector.Data(), vpsector.Data())) {
    LOG(fatal) << (Form("Alignable entry %s not created. Volume path %s not valid", snsector.Data(), vpsector.Data()));
  }
}

void Detector::ConstructGeometry()
{

  LOG(debug) << "Creating FOCAL geometry\n";

  CreateMaterials();

  /// -1 means get all the material object
  mGeoCompositions = mGeometry->getFOCALMicroModule(-1);
  if (!mGeoCompositions.size()) {
    LOG(error) << "FOCAL compositions not found!!";
    return;
  }

  float pars[4];
  pars[0] = (mGeometry->getFOCALSizeX() + 2 * mGeometry->getMiddleTowerOffset()) / 2;
  pars[1] = mGeometry->getFOCALSizeY() / 2;
  pars[2] = mGeometry->getFOCALSizeZ() / 2;
  // Add space to place 2 SiPad layers in front of ECAL
  // The global position of ECAL and HCAL remains the same, but the FOCAL box needs to be slightly larger to accomodate
  //   the 2 SiPad layers which will sit at z=698 and 699cm (2 and 1 cm in front of ECAL)
  if (mGeometry->getInsertFrontPadLayers()) {
    pars[2] += 1.0;
  }
  if (mGeometry->getInsertHCalReadoutMaterial()) {
    pars[2] += (1.0 + 0.5);  // place Aluminium 1cm thick box (0.5 means half) at 2cm behind HCal to simulate SiPM readout material
    pars[1] += (10.0 + 1.0); // place Aluminium 1cm thick box at 10 cm below FOCAL
  }
  pars[3] = 0;

  LOG(info) << "Creating FOCAL with dimensions X: " << (mGeometry->getFOCALSizeX() + 2 * mGeometry->getMiddleTowerOffset()) << ", Y: "
            << mGeometry->getFOCALSizeY() << ", Z: " << mGeometry->getFOCALSizeZ() + (mGeometry->getInsertFrontPadLayers() ? 2.0 : 0.0) + (mGeometry->getInsertHCalReadoutMaterial() ? 3.0 : 0.0) << std::endl;

  gMC->Gsvolu("FOCAL", "BOX", getMediumID(ID_AIR), pars, 4);
  mSensitiveHCAL.push_back("FOCAL");

  if (mGeometry->getUseHCALSandwich()) {
    CreateHCALSandwich();
  } else {
    CreateHCALSpaghetti();
  }

  gMC->Gspos("FOCAL", 1, "barrel", 0, 0, mGeometry->getFOCALZ0() - (mGeometry->getInsertFrontPadLayers() ? 2.0 : 0.0) + (mGeometry->getInsertHCalReadoutMaterial() ? 1.5 : 0.0), 0, "ONLY");
}

void Detector::CreateHCALSpaghetti()
{
  TGeoVolumeAssembly* volHCAL = new TGeoVolumeAssembly("HCAL");

  TGeoVolumeAssembly* HcalTube = gGeoManager->MakeVolumeAssembly("ScintCuTubes");

  TGeoVolume* volCuTube;
  TGeoVolume* volSciFi;

  float RScint = 0.;
  float Rin = 0.;
  float Rout = 0.;
  float Length = 0.;

  for (auto& icomp : mGeoCompositions) {
    Length = icomp->sizeZ() / 2;

    if (icomp->material() == "Pb") {
      Rout = icomp->sizeX() / 2;
      TGeoMedium* medium = gGeoManager->GetMedium(getMediumID(ID_PB));
      volCuTube = gGeoManager->MakeTube("Tube", medium, Rin, Rout, Length); // The Scintillator must be first in order in vector for Rin to be set
      volCuTube->SetLineWidth(2);
      volCuTube->SetLineColor(kRed);
      mSensitiveHCAL.push_back(volCuTube->GetName());
      HcalTube->AddNode(volCuTube, 1, 0x0);
    }
    if (icomp->material() == "Scint") {
      RScint = icomp->sizeX() / 2;
      Rin = RScint + 0.005;
      TGeoMedium* medium = gGeoManager->GetMedium(getMediumID(ID_SC));
      volSciFi = gGeoManager->MakeTube("ScintFiber", medium, 0., RScint, Length);
      volSciFi->SetLineWidth(2);
      volSciFi->SetLineColor(kBlue);
      mSensitiveHCAL.push_back(volSciFi->GetName());
      HcalTube->AddNode(volSciFi, 1, 0x0);
    }
    if (icomp->material() == "CuHCAL") {
      Rout = icomp->sizeX() / 2;
      TGeoMedium* medium = gGeoManager->GetMedium(getMediumID(ID_COPPER));
      volCuTube = gGeoManager->MakeTube("Tube", medium, Rin, Rout, Length); // The Scintillator must be first in order in vector for Rin to be set
      volCuTube->SetLineWidth(2);
      volCuTube->SetLineColor(kRed);
      mSensitiveHCAL.push_back(volCuTube->GetName());
      HcalTube->AddNode(volCuTube, 1, 0x0);
    }
  }

  double TowerSize = mGeometry->getHCALTowerSize();
  double CuBoxThickness = 0.3; // Thickness of the Cu box carrying capillary tubes

  TGeoBBox* ODBox = new TGeoBBox("TowerOD", TowerSize / 2, TowerSize / 2, Length);
  TGeoBBox* IDBox = new TGeoBBox("TowerID", (TowerSize - CuBoxThickness) / 2, (TowerSize - CuBoxThickness) / 2, Length + 0.01);
  TGeoCompositeShape* TowerHCAL = new TGeoCompositeShape("TowerHCAL", "TowerOD - TowerID");
  TGeoVolume* volTower = new TGeoVolume("volTower", TowerHCAL, gGeoManager->GetMedium(getMediumID(ID_COPPER)));
  volTower->SetLineWidth(2);
  volTower->SetLineColor(42);
  mSensitiveHCAL.push_back(volTower->GetName());

  TGeoVolumeAssembly* volTowerHCAL = new TGeoVolumeAssembly("volTowerHCAL");
  volTowerHCAL->AddNode(volTower, 1, 0x0);

  int Rows = 0;
  float RowPos = 0.;
  int Columns = 0;
  int NumTubes = 1;

  // Packing circles in Hexagonal shape
  while (RowPos + CuBoxThickness / 2 + Rout + 2 * Rout < TowerSize) {
    Columns = 0;
    float ColumnPos = (Rows % 2 == 0) ? 0. : Rout;
    while (ColumnPos + CuBoxThickness / 2 + Rout + 2 * Rout < TowerSize) {

      TGeoTranslation* trans = new TGeoTranslation(ColumnPos - TowerSize / 2 + CuBoxThickness / 2 + Rout, RowPos - TowerSize / 2 + CuBoxThickness / 2 + Rout, 0.);

      trans->SetName(Form("trans_Num_%d", NumTubes));
      trans->RegisterYourself();

      volTowerHCAL->AddNode(HcalTube, NumTubes, trans);
      // volTowerHCAL->AddNode(volCuTube, NumTubes, trans);
      // volTowerHCAL->AddNode(volSciFi, NumTubes, trans);

      Columns++;
      ColumnPos = Columns * 2 * Rout + ((Rows % 2 == 0) ? 0. : Rout);
      NumTubes++;
    }

    Rows++;
    RowPos = Rows * 2 * Rout * TMath::Sin(TMath::Pi() / 3);
  }

  // Define the distance from the beam pipe in which towers will ommitted
  Double_t BeamPipeRadius = 3.0;                             // in cm To be changed later
  Double_t TowerHalfDiag = TMath::Sqrt2() * 0.5 * TowerSize; // tower half diagonal
  Double_t MinRadius = BeamPipeRadius + TowerSize / 2;

  float SizeXHCAL = mGeometry->getHCALTowersInX() * TowerSize;
  float SizeYHCAL = mGeometry->getHCALTowersInY() * TowerSize;

  int nTowersX = mGeometry->getHCALTowersInX();
  int nTowersY = mGeometry->getHCALTowersInY();

  Rows = 0;
  Columns = 0;
  RowPos = 0.;
  Int_t NumTowers = 1;
  for (Rows = 0; Rows < nTowersY; Rows++) {

    float ColumnPos = 0.;
    RowPos = Rows * TowerSize;
    for (Columns = 0; Columns < nTowersX; Columns++) {
      ColumnPos = Columns * TowerSize;
      TGeoTranslation* trans = new TGeoTranslation(ColumnPos - SizeXHCAL / 2 + TowerSize / 2, RowPos - SizeYHCAL / 2 + TowerSize / 2, 0.);

      // Remove the Towers that overlaps with the beam pipe
      Double_t RadialDistance = TMath::Power(trans->GetTranslation()[0], 2) + TMath::Power(trans->GetTranslation()[1], 2);

      if (RadialDistance < MinRadius * MinRadius || TMath::Abs(trans->GetTranslation()[0]) > SizeXHCAL / 2) {
        continue;
      }

      // Adding the Tower to the HCAL
      volHCAL->AddNode(volTowerHCAL, NumTowers, trans);

      NumTowers++;
    }
  }

  std::cout << "Number of Towers is: " << (NumTowers - 1) << std::endl;
  std::cout << "Number of tubes is: " << (NumTubes - 1) * (NumTowers - 1) << std::endl;

  // Create Aluminium plate at the back of HCal to simulate the electronics readout material
  // Hardcoded thickness of 1cm and placement at 2 cm behind HCAL
  TGeoBBox* alHcalBox = new TGeoBBox("AlHCalBox", SizeXHCAL / 2.0, SizeYHCAL / 2.0, 0.5 / 2.0);
  TGeoVolume* volumeAlHcalBox = new TGeoVolume("volAlHcalBox", alHcalBox, gGeoManager->GetMedium(getMediumID(ID_ALUMINIUM)));
  volumeAlHcalBox->SetLineColor(kOrange);
  if (mGeometry->getInsertHCalReadoutMaterial()) {
    gMC->Gspos("volAlHcalBox", 9999, "FOCAL", 0.0, 0.0, +1.0 * mGeometry->getFOCALSizeZ() / 2.0 + 1.0, 0, "ONLY");
    mSensitiveHCAL.push_back("volAlHcalBox");
  }
  TGeoBBox* alUnderBox = new TGeoBBox("AlUnderBox", SizeXHCAL / 2.0, 0.5, mGeometry->getFOCALSizeZ() / 2.0 + 1.5);
  TGeoVolume* volumeAlUnderBox = new TGeoVolume("volAlUnderBox", alUnderBox, gGeoManager->GetMedium(getMediumID(ID_ALUMINIUM)));
  volumeAlUnderBox->SetLineColor(kOrange);
  if (mGeometry->getInsertHCalReadoutMaterial()) {
    gMC->Gspos("volAlUnderBox", 9999, "FOCAL", 0.0, -1.0 * mGeometry->getFOCALSizeY() / 2 - 10.5, 0.0, 0, "ONLY");
    mSensitiveHCAL.push_back("volAlUnderBox");
  }

  gMC->Gspos("HCAL", 1, "FOCAL", 0, 0, mGeometry->getHCALCenterZ() - mGeometry->getFOCALSizeZ() / 2 + 0.01 + (mGeometry->getInsertFrontPadLayers() ? 2.0 : 0.0) - (mGeometry->getInsertHCalReadoutMaterial() ? 1.5 : 0.0), 0, "ONLY");
}

//_____________________________________________________________________________
void Detector::CreateHCALSandwich()
{
  TGeoVolumeAssembly* volHCAL = new TGeoVolumeAssembly("HCAL");

  /// make big volume containing all the longitudinal layers
  Float_t pars[4]; // this is HMSC Assembly
  pars[0] = mGeometry->getHCALTowerSize() / 2;
  pars[1] = mGeometry->getHCALTowerSize() / 2;
  pars[2] = mGeometry->getECALSizeZ() + mGeometry->getHCALSizeZ() / 2; // ECAL sizeZ is already added to the HCAL materials CenterZ, so it is also treated as offset
  pars[3] = 0;

  float offset = pars[2];

  TGeoVolumeAssembly* volTower = new TGeoVolumeAssembly("Tower");

  int iCu(0), iScint(0);
  for (auto& icomp : mGeoCompositions) {

    pars[0] = icomp->sizeX() / 2;
    pars[1] = icomp->sizeY() / 2;
    pars[2] = icomp->sizeZ() / 2;
    pars[3] = 0;

    // HCal materials

    if (icomp->material() == "Pb") {
      iCu++;
      const TGeoMedium* medium = gGeoManager->GetMedium(getMediumID(ID_PB));
      const TGeoBBox* HPadBox = new TGeoBBox("HPadBox", pars[0], pars[1], pars[2]);
      TGeoVolume* HPad = new TGeoVolume("HPad", HPadBox, medium);
      HPad->SetLineColor(kGray);
      mSensitiveHCAL.push_back(HPad->GetName());
      TGeoTranslation* trans = new TGeoTranslation(icomp->centerX(), icomp->centerY(), icomp->centerZ() - offset);
      volTower->AddNode(HPad, iCu, trans);
    }
    if (icomp->material() == "Scint") {
      iScint++;
      const TGeoMedium* medium = gGeoManager->GetMedium(getMediumID(ID_SC));
      const TGeoBBox* HScintBox = new TGeoBBox("HScintBox", pars[0], pars[1], pars[2]);
      TGeoVolume* HScint = new TGeoVolume("HScint", HScintBox, medium);
      HScint->SetLineColor(kBlue);
      mSensitiveHCAL.push_back(HScint->GetName());
      TGeoTranslation* trans = new TGeoTranslation(icomp->centerX(), icomp->centerY(), icomp->centerZ() - offset);
      volTower->AddNode(HScint, iScint, trans);
    }
    if (icomp->material() == "CuHCAL") {
      iCu++;
      const TGeoMedium* medium = gGeoManager->GetMedium(getMediumID(ID_COPPER));
      const TGeoBBox* HPadBox = new TGeoBBox("HPadBox", pars[0], pars[1], pars[2]);
      TGeoVolume* HPad = new TGeoVolume("HPad", HPadBox, medium);
      HPad->SetLineColor(kRed);
      mSensitiveHCAL.push_back(HPad->GetName());
      TGeoTranslation* trans = new TGeoTranslation(icomp->centerX(), icomp->centerY(), icomp->centerZ() - offset);
      volTower->AddNode(HPad, iCu, trans);
    }
  }
  double TowerSize = mGeometry->getHCALTowerSize();

  // Define the distance from the beam pipe in which towers will ommitted
  double BeamPipeRadius = 3.6;                             // in cm
  double TowerHalfDiag = TMath::Sqrt2() * 0.5 * TowerSize; // tower half diagonal
  double MinRadius = BeamPipeRadius + TowerHalfDiag;

  float SizeXHCAL = mGeometry->getHCALTowersInX() * TowerSize;
  float SizeYHCAL = mGeometry->getHCALTowersInY() * TowerSize;

  int nTowersX = mGeometry->getHCALTowersInX();
  int nTowersY = mGeometry->getHCALTowersInY();

  int Rows = 0;
  int Columns = 0;
  double RowPos = 0.;
  int NumTowers = 1;

  // Arranging towers
  for (Rows = 0; Rows < nTowersY; Rows++) {
    Columns = 0;
    float ColumnPos = 0.;
    RowPos = Rows * TowerSize;
    for (Columns = 0; Columns < nTowersX; Columns++) {
      ColumnPos = Columns * TowerSize;

      TGeoTranslation* trans = new TGeoTranslation(ColumnPos - SizeXHCAL / 2 + TowerSize / 2, RowPos - SizeYHCAL / 2 + TowerSize / 2, 0.);

      // Remove the Towers that overlaps with the beam pipe
      double RadialDistance = TMath::Power(ColumnPos - SizeXHCAL / 2 + TowerSize / 2, 2) + TMath::Power(RowPos - SizeYHCAL / 2 + TowerSize / 2, 2);

      if (RadialDistance < MinRadius * MinRadius) {
        continue;
      }

      // Adding the Tower to the HCAL
      volHCAL->AddNode(volTower, NumTowers, trans);

      NumTowers++;
    }
  }
  std::cout << "Number of Towers is: " << (NumTowers - 1) << std::endl;

  // Create an Aluminium plate at the back of HCal to simulate the electronics readout material
  // Hardcoded thickness of 1cm and placement at 2 cm behind HCAL
  TGeoBBox* alHcalBox = new TGeoBBox("AlHCalBox", SizeXHCAL / 2.0, SizeYHCAL / 2.0, 0.5 / 2.0);
  TGeoVolume* volumeAlHcalBox = new TGeoVolume("volAlHcalBox", alHcalBox, gGeoManager->GetMedium(getMediumID(ID_ALUMINIUM)));
  volumeAlHcalBox->SetLineColor(kOrange);
  if (mGeometry->getInsertHCalReadoutMaterial()) {
    gMC->Gspos("volAlHcalBox", 9999, "FOCAL", 0.0, 0.0, +1.0 * mGeometry->getFOCALSizeZ() / 2.0 + 1.0, 0, "ONLY");
  }
  TGeoBBox* alUnderBox = new TGeoBBox("AlUnderBox", SizeXHCAL / 2.0, 0.5, mGeometry->getFOCALSizeZ() / 2.0 + 1.5);
  TGeoVolume* volumeAlUnderBox = new TGeoVolume("volAlUnderBox", alUnderBox, gGeoManager->GetMedium(getMediumID(ID_ALUMINIUM)));
  volumeAlUnderBox->SetLineColor(kOrange);
  if (mGeometry->getInsertHCalReadoutMaterial()) {
    gMC->Gspos("volAlUnderBox", 9999, "FOCAL", 0.0, -1.0 * mGeometry->getFOCALSizeY() / 2 - 10.5, 0.0, 0, "ONLY");
  }

  TGeoVolume* volFOCAL = gGeoManager->GetVolume("FOCAL");
  volFOCAL->AddNode(volHCAL, 1, new TGeoTranslation(0, 0, mGeometry->getHCALCenterZ() - mGeometry->getFOCALSizeZ() / 2 + 0.01 + (mGeometry->getInsertFrontPadLayers() ? 2.0 : 0.0) - (mGeometry->getInsertHCalReadoutMaterial() ? 1.5 : 0.0))); // 0.01 to avoid overlap with ECAL
}

void Detector::BeginPrimary()
{
  mCurrentPrimaryID = gMC->GetStack()->GetCurrentTrackNumber();
  LOG(debug) << "Starting primary " << mCurrentPrimaryID << " with energy " << gMC->GetStack()->GetCurrentTrack()->Energy();
}

void Detector::FinishPrimary()
{
  LOG(debug) << "Finishing primary " << mCurrentPrimaryID << std::endl;
  // Resetting primary and parent ID
  mCurrentPrimaryID = -1;
}