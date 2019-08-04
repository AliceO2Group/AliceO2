// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <algorithm>
#include <iomanip>
#include <map>

#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TVirtualMC.h"

#include "FairGeoNode.h"
#include "FairRootManager.h"
#include "FairVolume.h"

#include "PHOSBase/Geometry.h"
#include "PHOSBase/Hit.h"
#include "PHOSSimulation/Detector.h"
#include "PHOSSimulation/GeometryParams.h"

#include "SimulationDataFormat/Stack.h"

#include <boost/algorithm/string/predicate.hpp>
#include <boost/range/irange.hpp>

using namespace o2::phos;

ClassImp(Detector);

Detector::Detector(Bool_t active)
  : o2::base::DetImpl<Detector>("PHS", active),
    mHits(o2::utils::createSimVector<o2::phos::Hit>()),
    mCurrentTrackID(-1),
    mCurrentCellID(-1),
    mCurentSuperParent(-1),
    mCurrentHit(nullptr)
{
}

Detector::Detector(const Detector& rhs)
  : o2::base::DetImpl<Detector>(rhs),
    mHits(o2::utils::createSimVector<o2::phos::Hit>()),
    mCurrentTrackID(-1),
    mCurrentCellID(-1),
    mCurentSuperParent(-1),
    mCurrentHit(nullptr)
{
}

Detector::~Detector()
{
  o2::utils::freeSimVector(mHits);
}

void Detector::InitializeO2Detector()
{
  Reset();

  // Define sensitive volumes
  defineSensitiveVolumes();
}

void Detector::EndOfEvent()
{
  Reset();
}
void Detector::FinishEvent()
{
  // Sort Hits
  // Add duplicates if any and remove them
  // TODO: Apply Poisson smearing of light production
  if (!mHits || mHits->size() == 0) {
    return;
  }

  auto first = mHits->begin();
  auto last = mHits->end();

  std::sort(first, last);

  first = mHits->begin();
  last = mHits->end();

  // this is copy of std::unique() method with addition: adding identical Hits
  auto itr = first;
  while (++first != last) {
    if (*itr == *first) {
      *itr += *first;
    } else {
      *(++itr) = *first;
    }
  }
  ++itr;

  mHits->erase(itr, mHits->end());

  /*
        std::ostream stream(nullptr);
        stream.rdbuf(std::cout.rdbuf()); // uses cout's buffer
  //      stream.rdbuf(LOG(DEBUG2));
      for (int i = 0; i < mHits->size(); i++) {
         mHits->at(i).PrintStream(stream);
        }
  */
}
void Detector::Reset()
{
  mSuperParents.clear();
  if (!o2::utils::ShmManager::Instance().isOperational()) {
    mHits->clear();
  }
  mCurrentTrackID = -1;
  mCurrentCellID = -1;
  mCurentSuperParent = -1;
  mCurrentHit = nullptr;
}

void Detector::Register() { FairRootManager::Instance()->RegisterAny(addNameTo("Hit").data(), mHits, kTRUE); }

Bool_t Detector::ProcessHits(FairVolume* v)
{

  // 1. Remember all particles first entered PHOS (active medium)
  // 2. Collect all energy depositions in Cell by all secondaries from particle first entered PHOS

  // Check if this is first entered PHOS particle ("SuperParent")
  TVirtualMCStack* stack = fMC->GetStack();
  const Int_t partID = stack->GetCurrentTrackNumber();
  Int_t superParent = -1;
  Bool_t isNewPartile = false;     // Create Hit even if zero energy deposition
  if (partID != mCurrentTrackID) { // not same track as before, check: same SuperParent or new one?
    auto itTr = mSuperParents.find(partID);
    if (itTr == mSuperParents.end()) {
      // Search parent
      Int_t parentID = stack->GetCurrentTrack()->GetMother(0);
      itTr = mSuperParents.find(parentID);
      if (itTr == mSuperParents.end()) { // Neither track or its parent found: new SuperParent
        mSuperParents[partID] = partID;
        superParent = partID;
        isNewPartile = true;
      } else { // parent found, this track - not
        superParent = itTr->second;
        mSuperParents[partID] = superParent;
        mCurrentTrackID = partID;
      }
    } else {
      superParent = itTr->second;
      mCurrentTrackID = partID;
    }
  } else {
    superParent = mCurentSuperParent;
  }

  Double_t lostenergy = fMC->Edep();
  if (lostenergy < DBL_EPSILON && !isNewPartile)
    return false; // do not create hits with zero energy deposition

  if (!mGeom)
    mGeom = Geometry::GetInstance();

  //  if(strcmp(mc->CurrentVolName(),"PXTL")!=0) //Non need to check, alwais there...
  //    return false ; //  We are not inside a PBWO crystal

  Int_t moduleNumber;
  fMC->CurrentVolOffID(
    11, moduleNumber); // 11: number of geom. levels between PXTL and PHOS module: get the PHOS module number ;
  Int_t strip;
  fMC->CurrentVolOffID(3, strip); // 3: number of geom levels between PXTL and strip: get strip number in PHOS module
  Int_t cell;
  fMC->CurrentVolOffID(2, cell); // 2: number of geom levels between PXTL and cell: get sell in strip number.
  Int_t detID = mGeom->RelToAbsId(moduleNumber, strip, cell);

  if (superParent == mCurentSuperParent && detID == mCurrentCellID && mCurrentHit) {
    // continue with current hit
    mCurrentHit->AddEnergyLoss(lostenergy);
    return true;
  }

  // try to find existing Hit
  if (!isNewPartile) {
    for (Int_t itr = mHits->size() - 1; itr >= 0; itr--) {
      Hit* h = &(mHits->at(itr));
      if (h->GetTrackID() != superParent) // switched to another SuperParent, do not search further
        break;
      if (h->GetDetectorID() == detID) { // found correct hit
        h->AddEnergyLoss(lostenergy);
        mCurentSuperParent = superParent;
        mCurrentTrackID = partID;
        mCurrentCellID = detID;
        mCurrentHit = h;
        return true;
      }
    }
  }
  // Create new Hit
  Float_t posX = 0., posY = 0., posZ = 0., momX = 0, momY = 0., momZ = 0., energy = 0.;
  fMC->TrackPosition(posX, posY, posZ);
  fMC->TrackMomentum(momX, momY, momZ, energy);
  Double_t estart = fMC->Etot();
  Double_t time = fMC->TrackTime() * 1.e+9; // time in ns?? To be consistent with EMCAL

  mCurrentHit = AddHit(superParent, detID, Point3D<float>(posX, posY, posZ), Vector3D<float>(momX, momY, momZ), estart,
                       time, lostenergy);
  mCurentSuperParent = superParent;
  mCurrentTrackID = partID;
  mCurrentCellID = detID;

  return true;
}

Hit* Detector::AddHit(Int_t trackID, Int_t detID, const Point3D<float>& pos, const Vector3D<float>& mom, Double_t totE,
                      Double_t time, Double_t eLoss)
{
  LOG(DEBUG4) << "Adding hit for track " << trackID << " with position (" << pos.X() << ", " << pos.Y() << ", "
              << pos.Z() << ") and momentum (" << mom.X() << ", " << mom.Y() << ", " << mom.Z() << ")  with energy "
              << totE << " loosing " << eLoss << std::endl;
  mHits->emplace_back(trackID, detID, pos, mom, totE, time, eLoss);
  return &(mHits->back());
}

void Detector::ConstructGeometry()
{
  // Create geometry description of PHOS depector for Geant simulations.

  using boost::algorithm::contains;
  LOG(DEBUG) << "Creating PHOS geometry\n";

  phos::GeometryParams* geom = phos::GeometryParams::GetInstance("Run2");

  if (!geom) {
    LOG(ERROR) << "ConstructGeometry: PHOS Geometry class has not been set up.\n";
  }

  if (!fMC) {
    fMC = TVirtualMC::GetMC();
  }

  // Configure geometry So far we have only one: Run2
  {
    mCreateHalfMod = kTRUE;
    mActiveModule[0] = kFALSE;
    mActiveModule[1] = kTRUE;
    mActiveModule[2] = kTRUE;
    mActiveModule[3] = kTRUE;
    mActiveModule[4] = kTRUE;
    mActiveModule[5] = kFALSE;
  }

  // First create necessary materials
  CreateMaterials();

  // Create a PHOS modules-containers which will be filled with the stuff later.
  // Depending on configuration we should prepare containers for normal PHOS module "PHOS"
  // and half-module "PHOH"
  // This is still air tight box around PHOS
  fMC->Gsvolu("PHOS", "TRD1", getMediumID(ID_FE), geom->GetPHOSParams(), 4);
  if (mCreateHalfMod) {
    fMC->Gsvolu("PHOH", "TRD1", getMediumID(ID_FE), geom->GetPHOSParams(), 4);
  }

  // Fill prepared containers PHOS,PHOH,PHOC
  ConstructEMCGeometry();

  ConstructSupportGeometry();

  // --- Position  PHOS modules in ALICE setup ---
  Int_t idrotm[5];
  Int_t iXYZ, iAngle;
  char im[5];
  for (Int_t iModule = 0; iModule < 5; iModule++) {
    if (!mActiveModule[iModule + 1]) {
      continue;
    }
    Float_t angle[3][2] = {0};
    geom->GetModuleAngle(iModule, angle);
    Matrix(idrotm[iModule], angle[0][0], angle[0][1], angle[1][0], angle[1][1], angle[2][0], angle[2][1]);

    Float_t pos[3] = {0};
    geom->GetModuleCenter(iModule, pos);

    if (iModule == 3) { // special 1/2 module
      fMC->Gspos("PHOH", iModule + 1, "cave", pos[0], pos[1], pos[2], idrotm[iModule], "ONLY");
    } else {
      fMC->Gspos("PHOS", iModule + 1, "cave", pos[0], pos[1], pos[2], idrotm[iModule], "ONLY");
    }
  }

  gGeoManager->CheckGeometry();
}
//-----------------------------------------
void Detector::CreateMaterials()
{
  // Definitions of materials to build PHOS and associated tracking media.

  // --- The PbWO4 crystals ---
  Float_t aX[3] = {207.19, 183.85, 16.0};
  Float_t zX[3] = {82.0, 74.0, 8.0};
  Float_t wX[3] = {1.0, 1.0, 4.0};
  Float_t dX = 8.28;

  Mixture(ID_PWO, "PbWO4", aX, zX, dX, -3, wX);

  // --- Aluminium ---
  Material(ID_AL, "Al", 26.98, 13., 2.7, 8.9, 999., nullptr, 0);
  // ---          Absorption length is ignored ^

  // --- Tyvek (CnH2n) ---
  Float_t aT[2] = {12.011, 1.00794};
  Float_t zT[2] = {6.0, 1.0};
  Float_t wT[2] = {1.0, 2.0};
  Float_t dT = 0.331;

  Mixture(ID_TYVEK, "Tyvek", aT, zT, dT, -2, wT);

  // --- Polystyrene foam ---
  Float_t aF[2] = {12.011, 1.00794};
  Float_t zF[2] = {6.0, 1.0};
  Float_t wF[2] = {1.0, 1.0};
  Float_t dF = 0.12;

  Mixture(ID_POLYFOAM, "Foam", aF, zF, dF, -2, wF);

  // --- Titanium ---
  Float_t aTIT[3] = {47.88, 26.98, 54.94};
  Float_t zTIT[3] = {22.0, 13.0, 25.0};
  Float_t wTIT[3] = {69.0, 6.0, 1.0};
  Float_t dTIT = 4.5;

  // --- Silicon ---
  Material(ID_APD, "Si", 28.0855, 14., 2.33, 9.36, 42.3, nullptr, 0);

  // --- Foam thermo insulation ---
  Float_t aTI[2] = {12.011, 1.00794};
  Float_t zTI[2] = {6.0, 1.0};
  Float_t wTI[2] = {1.0, 1.0};
  Float_t dTI = 0.04;

  Mixture(ID_THERMOINS, "Thermo Insul.", aTI, zTI, dTI, -2, wTI);

  // --- Textolith ---
  Float_t aTX[4] = {16.0, 28.09, 12.011, 1.00794};
  Float_t zTX[4] = {8.0, 14.0, 6.0, 1.0};
  Float_t wTX[4] = {292.0, 68.0, 462.0, 736.0};
  Float_t dTX = 1.75;

  Mixture(ID_TEXTOLIT, "Textolit", aTX, zTX, dTX, -4, wTX);

  // --- G10 : Printed Circuit Materiall ---
  Float_t aG10[4] = {12., 1., 16., 28.};
  Float_t zG10[4] = {6., 1., 8., 14.};
  Float_t wG10[4] = {.259, .288, .248, .205};
  Float_t dG10 = 1.7;

  Mixture(ID_PRINTCIRC, "G10", aG10, zG10, dG10, -4, wG10);

  // --- Stainless steel (let it be pure iron) ---
  Material(ID_FE, "Steel", 55.845, 26, 7.87, 1.76, 0., nullptr, 0);

  // --- Fiberglass ---
  Float_t aFG[4] = {16.0, 28.09, 12.011, 1.00794};
  Float_t zFG[4] = {8.0, 14.0, 6.0, 1.0};
  Float_t wFG[4] = {292.0, 68.0, 462.0, 736.0};
  Float_t dFG = 1.9;

  Mixture(ID_FIBERGLASS, "Fiberglas", aFG, zFG, dFG, -4, wFG);

  // --- Cables in Air box  ---
  // SERVICES

  Float_t aCA[4] = {1., 12., 55.8, 63.5};
  Float_t zCA[4] = {1., 6., 26., 29.};
  Float_t wCA[4] = {.014, .086, .42, .48};
  Float_t dCA = 0.8; // this density is raw estimation, if you know better - correct

  Mixture(ID_CABLES, "Cables", aCA, zCA, dCA, -4, wCA);

  // --- Air ---
  Float_t aAir[4] = {12.0107, 14.0067, 15.9994, 39.948};
  Float_t zAir[4] = {6., 7., 8., 18.};
  Float_t wAir[4] = {0.000124, 0.755267, 0.231781, 0.012827};
  Float_t dAir = 1.20479E-3;

  Mixture(ID_AIR, "Air", aAir, zAir, dAir, 4, wAir);

  // DEFINITION OF THE TRACKING MEDIA

  // for PHOS: idtmed[699->798] equivalent to fIdtmed[0->100]
  //  Int_t   isxfld = ((AliMagF*)TGeoGlobalMagField::Instance()->GetField())->Integ() ;
  //  Float_t sxmgmx = ((AliMagF*)TGeoGlobalMagField::Instance()->GetField())->Max() ;
  Int_t isxfld = 2;
  Float_t sxmgmx = 10.0;
  o2::base::Detector::initFieldTrackingParams(isxfld, sxmgmx);

  // The scintillator of the calorimeter made of PBW04                              -> idtmed[699]
  if (fActive) {
    Medium(ID_PWO, "Crystal", ID_PWO, 1, isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.1, nullptr, 0);
  } else {
    Medium(ID_PWO, "Crystal", ID_PWO, 0, isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.1, nullptr, 0);
  }

  // Various Aluminium parts made of Al                                             -> idtmed[701]
  Medium(ID_AL, "Alparts", ID_AL, 0, isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.001, 0.001, nullptr, 0);

  // The Tywek which wraps the calorimeter crystals                                 -> idtmed[702]
  Medium(ID_TYVEK, "Tyvek", ID_TYVEK, 0, isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.001, 0.001, nullptr, 0);

  // The Polystyrene foam around the calorimeter module                             -> idtmed[703]
  Medium(ID_POLYFOAM, "Polyst.foam", ID_POLYFOAM, 0, isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.1, nullptr, 0);

  // The Silicon of the APD diode to read out the calorimeter crystal               -> idtmed[705]
  Medium(ID_APD, "SiAPD", ID_APD, 0, isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.01, 0.01, nullptr, 0);

  // The thermo insulating material of the box which contains the calorimeter module -> getMediumID(ID_THERMOINS)
  Medium(ID_THERMOINS, "ThermoInsul.", ID_THERMOINS, 0, isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.1, nullptr, 0);

  // The Textolit which makes up the box which contains the calorimeter module      -> idtmed[707]
  Medium(ID_TEXTOLIT, "Textolit", ID_TEXTOLIT, 0, isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.1, nullptr, 0);

  // G10: Printed Circuit material                                                  -> idtmed[711]
  Medium(ID_PRINTCIRC, "G10", ID_PRINTCIRC, 0, isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.01, nullptr, 0);

  // Stainless steel                                                                -> idtmed[716]
  Medium(ID_FE, "Steel", ID_FE, 0, isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.0001, nullptr, 0);

  // Fibergalss                                                                     -> getMediumID(ID_FIBERGLASS)
  Medium(ID_FIBERGLASS, "Fiberglass", ID_FIBERGLASS, 0, isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.1, nullptr, 0);

  // Cables in air                                                                  -> idtmed[718]
  Medium(ID_CABLES, "Cables", ID_CABLES, 0, isxfld, sxmgmx, 10.0, 0.1, 0.1, 0.1, 0.1, nullptr, 0);

  // Air                                                                            -> idtmed[798]
  Medium(ID_AIR, "Air", ID_AIR, 0, isxfld, sxmgmx, 10.0, 1.0, 0.1, 0.1, 10.0, nullptr, 0);
}
//-----------------------------------------
void Detector::ConstructEMCGeometry()
{
  // Create the PHOS-EMC geometry for GEANT
  // Author: Dmitri Peressounko August 2001
  // Adopted for O2 project 2017
  // The used coordinate system:
  //   1. in Module: X along longer side, Y out of beam, Z along shorter side (along beam)
  //   2. In Strip the same: X along longer side, Y out of beam, Z along shorter side (along beam)

  phos::GeometryParams* geom = phos::GeometryParams::GetInstance();

  Float_t par[4] = {0};
  Int_t ipar;

  // ======= Define the strip ===============
  for (ipar = 0; ipar < 3; ipar++) {
    par[ipar] = *(geom->GetStripHalfSize() + ipar);
  }
  // --- define steel volume (cell of the strip unit)
  fMC->Gsvolu("PSTR", "BOX ", getMediumID(ID_FE), par, 3); // Made of steel

  // --- define air cell in the steel strip
  for (ipar = 0; ipar < 3; ipar++) {
    par[ipar] = *(geom->GetAirCellHalfSize() + ipar);
  }
  fMC->Gsvolu("PCEL", "BOX ", getMediumID(ID_AIR), par, 3);

  // --- define wrapped crystal and put it into steel cell
  for (ipar = 0; ipar < 3; ipar++) {
    par[ipar] = *(geom->GetWrappedHalfSize() + ipar);
  }
  fMC->Gsvolu("PWRA", "BOX ", getMediumID(ID_TYVEK), par, 3);
  const Float_t* pin = geom->GetAPDHalfSize();
  const Float_t* preamp = geom->GetPreampHalfSize();
  Float_t y = (geom->GetAirGapLed() - 2 * pin[1] - 2 * preamp[1]) / 2;
  fMC->Gspos("PWRA", 1, "PCEL", 0.0, y, 0.0, 0, "ONLY");

  // --- Define crystal and put it into wrapped crystall ---
  for (ipar = 0; ipar < 3; ipar++) {
    par[ipar] = *(geom->GetCrystalHalfSize() + ipar);
  }

  fMC->Gsvolu("PXTL", "BOX ", getMediumID(ID_PWO), par, 3);
  //  fMC->Gsvolu("PXTL", "BOX ", 1, par, 3) ;
  fMC->Gspos("PXTL", 1, "PWRA", 0.0, 0.0, 0.0, 0, "ONLY");

  // --- define APD/PIN preamp and put it into AirCell
  for (ipar = 0; ipar < 3; ipar++) {
    par[ipar] = *(geom->GetAPDHalfSize() + ipar);
  }
  fMC->Gsvolu("PPIN", "BOX ", getMediumID(ID_APD), par, 3);
  const Float_t* crystal = geom->GetCrystalHalfSize();
  y = crystal[1] + geom->GetAirGapLed() / 2 - preamp[1];
  fMC->Gspos("PPIN", 1, "PCEL", 0.0, y, 0.0, 0, "ONLY");
  for (ipar = 0; ipar < 3; ipar++) {
    par[ipar] = *(geom->GetPreampHalfSize() + ipar);
  }
  fMC->Gsvolu("PREA", "BOX ", getMediumID(ID_PRINTCIRC), par, 3); // Here I assumed preamp as a printed Circuit
  y = crystal[1] + geom->GetAirGapLed() / 2 + pin[1];             // May it should be changed
  fMC->Gspos("PREA", 1, "PCEL", 0.0, y, 0.0, 0, "ONLY");          // to ceramics?

  // --- Fill strip with wrapped cristals in cells

  const Float_t* splate = geom->GetSupportPlateHalfSize();
  y = -splate[1];
  const Float_t* acel = geom->GetAirCellHalfSize();

  for (Int_t lev = 2, icel = 1; icel <= geom->GetNCellsXInStrip() * geom->GetNCellsZInStrip(); icel += 2, lev += 2) {
    Float_t x = (2 * (lev / 2) - 1 - geom->GetNCellsXInStrip()) * acel[0];
    Float_t z = acel[2];

    fMC->Gspos("PCEL", icel, "PSTR", x, y, +z, 0, "ONLY");
    fMC->Gspos("PCEL", icel + 1, "PSTR", x, y, -z, 0, "ONLY");
  }

  // --- define the support plate, hole in it and position it in strip ----
  for (ipar = 0; ipar < 3; ipar++) {
    par[ipar] = *(geom->GetSupportPlateHalfSize() + ipar);
  }
  fMC->Gsvolu("PSUP", "BOX ", getMediumID(ID_AL), par, 3);

  for (ipar = 0; ipar < 3; ipar++) {
    par[ipar] = *(geom->GetSupportPlateInHalfSize() + ipar);
  }
  fMC->Gsvolu("PSHO", "BOX ", getMediumID(ID_AIR), par, 3);
  Float_t z = geom->GetSupportPlateThickness() / 2;
  fMC->Gspos("PSHO", 1, "PSUP", 0.0, 0.0, z, 0, "ONLY");

  y = acel[1];
  fMC->Gspos("PSUP", 1, "PSTR", 0.0, y, 0.0, 0, "ONLY");

  // ========== Fill module with strips and put them into inner thermoinsulation=============
  for (ipar = 0; ipar < 3; ipar++) {
    par[ipar] = *(geom->GetInnerThermoHalfSize() + ipar);
  }
  fMC->Gsvolu("PTII", "BOX ", getMediumID(ID_THERMOINS), par, 3);

  if (mCreateHalfMod) {
    fMC->Gsvolu("PTIH", "BOX ", getMediumID(ID_THERMOINS), par, 3);
  }

  const Float_t* inthermo = geom->GetInnerThermoHalfSize();
  const Float_t* strip = geom->GetStripHalfSize();
  y = inthermo[1] - strip[1];
  Int_t irow;
  Int_t nr = 1;
  Int_t icol;

  for (irow = 0; irow < geom->GetNStripX(); irow++) {
    Float_t x = (2 * irow + 1 - geom->GetNStripX()) * strip[0];
    for (icol = 0; icol < geom->GetNStripZ(); icol++) {
      z = (2 * icol + 1 - geom->GetNStripZ()) * strip[2];
      fMC->Gspos("PSTR", nr, "PTII", x, y, z, 0, "ONLY");
      nr++;
    }
  }
  if (mCreateHalfMod) {
    nr = 1;
    for (irow = 0; irow < geom->GetNStripX(); irow++) {
      Float_t x = (2 * irow + 1 - geom->GetNStripX()) * strip[0];
      for (icol = 0; icol < geom->GetNStripZ(); icol++) {
        z = (2 * icol + 1 - geom->GetNStripZ()) * strip[2];
        if (irow >= geom->GetNStripX() / 2)
          fMC->Gspos("PSTR", nr, "PTIH", x, y, z, 0, "ONLY");
        nr++;
      }
    }
  }

  // ------- define the air gap between thermoinsulation and cooler
  for (ipar = 0; ipar < 3; ipar++) {
    par[ipar] = *(geom->GetAirGapHalfSize() + ipar);
  }
  fMC->Gsvolu("PAGA", "BOX ", getMediumID(ID_AIR), par, 3);
  if (mCreateHalfMod) {
    fMC->Gsvolu("PAGH", "BOX ", getMediumID(ID_AIR), par, 3);
  }
  const Float_t* agap = geom->GetAirGapHalfSize();
  y = agap[1] - inthermo[1];

  fMC->Gspos("PTII", 1, "PAGA", 0.0, y, 0.0, 0, "ONLY");
  if (mCreateHalfMod) {
    fMC->Gspos("PTIH", 1, "PAGH", 0.0, y, 0.0, 0, "ONLY");
  }

  // ------- define the Al passive cooler
  for (ipar = 0; ipar < 3; ipar++) {
    par[ipar] = *(geom->GetCoolerHalfSize() + ipar);
  }
  fMC->Gsvolu("PCOR", "BOX ", getMediumID(ID_AL), par, 3);
  if (mCreateHalfMod) {
    fMC->Gsvolu("PCOH", "BOX ", getMediumID(ID_AL), par, 3);
  }

  const Float_t* cooler = geom->GetCoolerHalfSize();
  y = cooler[1] - agap[1];

  fMC->Gspos("PAGA", 1, "PCOR", 0.0, y, 0.0, 0, "ONLY");
  if (mCreateHalfMod) {
    fMC->Gspos("PAGH", 1, "PCOH", 0.0, y, 0.0, 0, "ONLY");
  }

  // ------- define the outer thermoinsulating cover
  for (ipar = 0; ipar < 4; ipar++) {
    par[ipar] = *(geom->GetOuterThermoParams() + ipar);
  }
  fMC->Gsvolu("PTIO", "TRD1", getMediumID(ID_THERMOINS), par, 4);
  if (mCreateHalfMod) {
    fMC->Gsvolu("PIOH", "TRD1", getMediumID(ID_THERMOINS), par, 4);
  }
  const Float_t* outparams = geom->GetOuterThermoParams();

  Int_t idrotm = -1;
  Matrix(idrotm, 90.0, 0.0, 0.0, 0.0, 90.0, 270.0);
  // Frame in outer thermoinsulation and so on: z out of beam, y along beam, x across beam

  z = outparams[3] - cooler[1];
  fMC->Gspos("PCOR", 1, "PTIO", 0., 0.0, z, idrotm, "ONLY");
  if (mCreateHalfMod) {
    fMC->Gspos("PCOH", 1, "PIOH", 0., 0.0, z, idrotm, "ONLY");
  }

  // -------- Define the outer Aluminium cover -----
  for (ipar = 0; ipar < 4; ipar++) {
    par[ipar] = *(geom->GetAlCoverParams() + ipar);
  }
  fMC->Gsvolu("PCOL", "TRD1", getMediumID(ID_AL), par, 4);
  if (mCreateHalfMod) {
    fMC->Gsvolu("PCLH", "TRD1", getMediumID(ID_AL), par, 4);
  }

  const Float_t* covparams = geom->GetAlCoverParams();
  z = covparams[3] - outparams[3];
  fMC->Gspos("PTIO", 1, "PCOL", 0., 0.0, z, 0, "ONLY");
  if (mCreateHalfMod) {
    fMC->Gspos("PIOH", 1, "PCLH", 0., 0.0, z, 0, "ONLY");
  }

  // --------- Define front fiberglass cover -----------
  for (ipar = 0; ipar < 3; ipar++) {
    par[ipar] = *(geom->GetFiberGlassHalfSize() + ipar);
  }
  fMC->Gsvolu("PFGC", "BOX ", getMediumID(ID_FIBERGLASS), par, 3);
  z = -outparams[3];
  fMC->Gspos("PFGC", 1, "PCOL", 0., 0.0, z, 0, "ONLY");
  if (mCreateHalfMod) {
    fMC->Gspos("PFGC", 1, "PCLH", 0., 0.0, z, 0, "ONLY");
  }

  //=============This is all with cold section==============

  //------ Warm Section --------------
  for (ipar = 0; ipar < 3; ipar++) {
    par[ipar] = *(geom->GetWarmAlCoverHalfSize() + ipar);
  }
  fMC->Gsvolu("PWAR", "BOX ", getMediumID(ID_AL), par, 3);
  const Float_t* warmcov = geom->GetWarmAlCoverHalfSize();

  // --- Define the outer thermoinsulation ---
  for (ipar = 0; ipar < 3; ipar++) {
    par[ipar] = *(geom->GetWarmThermoHalfSize() + ipar);
  }
  fMC->Gsvolu("PWTI", "BOX ", getMediumID(ID_THERMOINS), par, 3);
  const Float_t* warmthermo = geom->GetWarmThermoHalfSize();
  z = -warmcov[2] + warmthermo[2];

  fMC->Gspos("PWTI", 1, "PWAR", 0., 0.0, z, 0, "ONLY");

  // --- Define cables area and put in it T-supports ----
  for (ipar = 0; ipar < 3; ipar++) {
    par[ipar] = *(geom->GetTCables1HalfSize() + ipar);
  }
  fMC->Gsvolu("PCA1", "BOX ", getMediumID(ID_CABLES), par, 3);
  const Float_t* cbox = geom->GetTCables1HalfSize();

  for (ipar = 0; ipar < 3; ipar++) {
    par[ipar] = *(geom->GetTSupport1HalfSize() + ipar);
  }
  fMC->Gsvolu("PBE1", "BOX ", getMediumID(ID_AL), par, 3);
  const Float_t* beams = geom->GetTSupport1HalfSize();
  Int_t isup;
  for (isup = 0; isup < geom->GetNTSuppots(); isup++) {
    Float_t x = -cbox[0] + beams[0] + (2 * beams[0] + geom->GetTSupportDist()) * isup;
    fMC->Gspos("PBE1", isup, "PCA1", x, 0.0, 0.0, 0, "ONLY");
  }

  z = -warmthermo[2] + cbox[2];
  fMC->Gspos("PCA1", 1, "PWTI", 0.0, 0.0, z, 0, "ONLY");

  for (ipar = 0; ipar < 3; ipar++) {
    par[ipar] = *(geom->GetTCables2HalfSize() + ipar);
  }
  fMC->Gsvolu("PCA2", "BOX ", getMediumID(ID_CABLES), par, 3);
  const Float_t* cbox2 = geom->GetTCables2HalfSize();

  for (ipar = 0; ipar < 3; ipar++) {
    par[ipar] = *(geom->GetTSupport2HalfSize() + ipar);
  }
  fMC->Gsvolu("PBE2", "BOX ", getMediumID(ID_AL), par, 3);
  for (isup = 0; isup < geom->GetNTSuppots(); isup++) {
    Float_t x = -cbox[0] + beams[0] + (2 * beams[0] + geom->GetTSupportDist()) * isup;
    fMC->Gspos("PBE2", isup, "PCA2", x, 0.0, 0.0, 0, "ONLY");
  }

  z = -warmthermo[2] + 2 * cbox[2] + cbox2[2];
  fMC->Gspos("PCA2", 1, "PWTI", 0.0, 0.0, z, 0, "ONLY");

  // --- Define frame ---
  for (ipar = 0; ipar < 3; ipar++) {
    par[ipar] = *(geom->GetFrameXHalfSize() + ipar);
  }
  fMC->Gsvolu("PFRX", "BOX ", getMediumID(ID_FE), par, 3);
  const Float_t* posit1 = geom->GetFrameXPosition();
  fMC->Gspos("PFRX", 1, "PWTI", posit1[0], posit1[1], posit1[2], 0, "ONLY");
  fMC->Gspos("PFRX", 2, "PWTI", posit1[0], -posit1[1], posit1[2], 0, "ONLY");

  for (ipar = 0; ipar < 3; ipar++) {
    par[ipar] = *(geom->GetFrameZHalfSize() + ipar);
  }
  fMC->Gsvolu("PFRZ", "BOX ", getMediumID(ID_FE), par, 3);
  const Float_t* posit2 = geom->GetFrameZPosition();
  fMC->Gspos("PFRZ", 1, "PWTI", posit2[0], posit2[1], posit2[2], 0, "ONLY");
  fMC->Gspos("PFRZ", 2, "PWTI", -posit2[0], posit2[1], posit2[2], 0, "ONLY");

  // --- Define Fiber Glass support ---
  for (ipar = 0; ipar < 3; ipar++) {
    par[ipar] = *(geom->GetFGupXHalfSize() + ipar);
  }
  fMC->Gsvolu("PFG1", "BOX ", getMediumID(ID_FIBERGLASS), par, 3);
  const Float_t* posit3 = geom->GetFGupXPosition();
  fMC->Gspos("PFG1", 1, "PWTI", posit3[0], posit3[1], posit3[2], 0, "ONLY");
  fMC->Gspos("PFG1", 2, "PWTI", posit3[0], -posit3[1], posit3[2], 0, "ONLY");

  for (ipar = 0; ipar < 3; ipar++) {
    par[ipar] = *(geom->GetFGupZHalfSize() + ipar);
  }
  fMC->Gsvolu("PFG2", "BOX ", getMediumID(ID_FIBERGLASS), par, 3);
  const Float_t* posit4 = geom->GetFGupZPosition();
  fMC->Gspos("PFG2", 1, "PWTI", posit4[0], posit4[1], posit4[2], 0, "ONLY");
  fMC->Gspos("PFG2", 2, "PWTI", -posit4[0], posit4[1], posit4[2], 0, "ONLY");
  for (ipar = 0; ipar < 3; ipar++) {
    par[ipar] = *(geom->GetFGlowXHalfSize() + ipar);
  }
  fMC->Gsvolu("PFG3", "BOX ", getMediumID(ID_FIBERGLASS), par, 3);
  const Float_t* posit5 = geom->GetFGlowXPosition();
  fMC->Gspos("PFG3", 1, "PWTI", posit5[0], posit5[1], posit5[2], 0, "ONLY");
  fMC->Gspos("PFG3", 2, "PWTI", posit5[0], -posit5[1], posit5[2], 0, "ONLY");

  for (ipar = 0; ipar < 3; ipar++) {
    par[ipar] = *(geom->GetFGlowZHalfSize() + ipar);
  }
  fMC->Gsvolu("PFG4", "BOX ", getMediumID(ID_FIBERGLASS), par, 3);
  const Float_t* posit6 = geom->GetFGlowZPosition();
  fMC->Gspos("PFG4", 1, "PWTI", posit6[0], posit6[1], posit6[2], 0, "ONLY");
  fMC->Gspos("PFG4", 2, "PWTI", -posit6[0], posit6[1], posit6[2], 0, "ONLY");

  // --- Define Air Gap for FEE electronics -----
  for (ipar = 0; ipar < 3; ipar++) {
    par[ipar] = *(geom->GetFEEAirHalfSize() + ipar);
  }
  fMC->Gsvolu("PAFE", "BOX ", getMediumID(ID_AIR), par, 3);
  const Float_t* posit7 = geom->GetFEEAirPosition();
  fMC->Gspos("PAFE", 1, "PWTI", posit7[0], posit7[1], posit7[2], 0, "ONLY");

  // Define the EMC module volume and combine Cool and Warm sections
  for (ipar = 0; ipar < 4; ipar++) {
    par[ipar] = *(geom->GetEMCParams() + ipar);
  }
  fMC->Gsvolu("PEMC", "TRD1", getMediumID(ID_AIR), par, 4);
  if (mCreateHalfMod) {
    fMC->Gsvolu("PEMH", "TRD1", getMediumID(ID_AIR), par, 4);
  }
  z = -warmcov[2];
  fMC->Gspos("PCOL", 1, "PEMC", 0., 0., z, 0, "ONLY");
  if (mCreateHalfMod) {
    fMC->Gspos("PCLH", 1, "PEMH", 0., 0., z, 0, "ONLY");
  }
  z = covparams[3];
  fMC->Gspos("PWAR", 1, "PEMC", 0., 0., z, 0, "ONLY");
  if (mCreateHalfMod) {
    fMC->Gspos("PWAR", 1, "PEMH", 0., 0., z, 0, "ONLY");
  }

  // Put created EMC geometry into PHOS volume

  z = geom->GetDistATBtoModule() - geom->GetPHOSATBParams()[3] + geom->GetEMCParams()[3];

  // PHOS AirTightBox
  fMC->Gsvolu("PATB", "TRD1", getMediumID(ID_AIR), geom->GetPHOSATBParams(), 4);
  if (mCreateHalfMod) { // half of PHOS module
    fMC->Gsvolu("PATH", "TRD1", getMediumID(ID_AIR), geom->GetPHOSATBParams(), 4);
  }

  fMC->Gspos("PEMC", 1, "PATB", 0., 0., z, 0, "ONLY");
  if (mCreateHalfMod) { // half of PHOS module
    fMC->Gspos("PEMH", 1, "PATH", 0., 0., z, 0, "ONLY");
  }

  fMC->Gspos("PATB", 1, "PHOS", 0., 0., 0, 0, "ONLY");
  if (mCreateHalfMod) { // half of PHOS module
    fMC->Gspos("PATH", 1, "PHOH", 0., 0., 0, 0, "ONLY");
  }
}

//-----------------------------------------
void Detector::ConstructSupportGeometry()
{
  // Create the PHOS support geometry for GEANT
  phos::GeometryParams* geom = phos::GeometryParams::GetInstance();

  Float_t par[5] = {0}, x0 = 0., y0 = 0., z0 = 0.;
  Int_t i, j, copy;

  // --- Dummy box containing two rails on which PHOS support moves
  // --- Put these rails to the bottom of the L3 magnet

  par[0] = geom->GetRailRoadSize(0) / 2.0;
  par[1] = geom->GetRailRoadSize(1) / 2.0;
  par[2] = geom->GetRailRoadSize(2) / 2.0;
  fMC->Gsvolu("PRRD", "BOX ", getMediumID(ID_AIR), par, 3);

  y0 = -(geom->GetRailsDistanceFromIP() - geom->GetRailRoadSize(1) / 2.0);
  fMC->Gspos("PRRD", 1, "cave", 0.0, y0, 0.0, 0, "ONLY");

  // --- Dummy box containing one rail

  par[0] = geom->GetRailOuterSize(0) / 2.0;
  par[1] = geom->GetRailOuterSize(1) / 2.0;
  par[2] = geom->GetRailOuterSize(2) / 2.0;
  fMC->Gsvolu("PRAI", "BOX ", getMediumID(ID_AIR), par, 3);

  for (i = 0; i < 2; i++) {
    x0 = (2 * i - 1) * geom->GetDistanceBetwRails() / 2.0;
    fMC->Gspos("PRAI", i, "PRRD", x0, 0.0, 0.0, 0, "ONLY");
  }

  // --- Upper and bottom steel parts of the rail

  par[0] = geom->GetRailPart1(0) / 2.0;
  par[1] = geom->GetRailPart1(1) / 2.0;
  par[2] = geom->GetRailPart1(2) / 2.0;
  fMC->Gsvolu("PRP1", "BOX ", getMediumID(ID_FE), par, 3);

  y0 = -(geom->GetRailOuterSize(1) - geom->GetRailPart1(1)) / 2.0;
  fMC->Gspos("PRP1", 1, "PRAI", 0.0, y0, 0.0, 0, "ONLY");
  y0 = (geom->GetRailOuterSize(1) - geom->GetRailPart1(1)) / 2.0 - geom->GetRailPart3(1);
  fMC->Gspos("PRP1", 2, "PRAI", 0.0, y0, 0.0, 0, "ONLY");

  // --- The middle vertical steel parts of the rail

  par[0] = geom->GetRailPart2(0) / 2.0;
  par[1] = geom->GetRailPart2(1) / 2.0;
  par[2] = geom->GetRailPart2(2) / 2.0;
  fMC->Gsvolu("PRP2", "BOX ", getMediumID(ID_FE), par, 3);

  y0 = -geom->GetRailPart3(1) / 2.0;
  fMC->Gspos("PRP2", 1, "PRAI", 0.0, y0, 0.0, 0, "ONLY");

  // --- The most upper steel parts of the rail

  par[0] = geom->GetRailPart3(0) / 2.0;
  par[1] = geom->GetRailPart3(1) / 2.0;
  par[2] = geom->GetRailPart3(2) / 2.0;
  fMC->Gsvolu("PRP3", "BOX ", getMediumID(ID_FE), par, 3);

  y0 = (geom->GetRailOuterSize(1) - geom->GetRailPart3(1)) / 2.0;
  fMC->Gspos("PRP3", 1, "PRAI", 0.0, y0, 0.0, 0, "ONLY");

  // --- The wall of the cradle
  // --- The wall is empty: steel thin walls and air inside

  par[1] = TMath::Sqrt(TMath::Power((geom->GetIPtoOuterCoverDistance() + geom->GetOuterBoxSize(3)), 2) +
                       TMath::Power((geom->GetOuterBoxSize(1) / 2), 2)) +
           10.;
  par[0] = par[1] - geom->GetCradleWall(1);
  par[2] = geom->GetCradleWall(2) / 2.0;
  par[3] = geom->GetCradleWall(3);
  par[4] = geom->GetCradleWall(4);
  fMC->Gsvolu("PCRA", "TUBS", getMediumID(ID_FE), par, 5);

  par[0] += geom->GetCradleWallThickness();
  par[1] -= geom->GetCradleWallThickness();
  par[2] -= geom->GetCradleWallThickness();
  fMC->Gsvolu("PCRE", "TUBS", getMediumID(ID_FE), par, 5);
  fMC->Gspos("PCRE", 1, "PCRA", 0.0, 0.0, 0.0, 0, "ONLY");

  for (i = 0; i < 2; i++) {
    z0 = (2 * i - 1) * (geom->GetOuterBoxSize(2) + geom->GetCradleWall(2)) / 2.0;
    fMC->Gspos("PCRA", i, "cave", 0.0, 0.0, z0, 0, "ONLY");
  }

  // --- The "wheels" of the cradle

  par[0] = geom->GetCradleWheel(0) / 2;
  par[1] = geom->GetCradleWheel(1) / 2;
  par[2] = geom->GetCradleWheel(2) / 2;
  fMC->Gsvolu("PWHE", "BOX ", getMediumID(ID_FE), par, 3);

  y0 = -(geom->GetRailsDistanceFromIP() - geom->GetRailRoadSize(1) - geom->GetCradleWheel(1) / 2);
  for (i = 0; i < 2; i++) {
    z0 = (2 * i - 1) * ((geom->GetOuterBoxSize(2) + geom->GetCradleWheel(2)) / 2.0 + geom->GetCradleWall(2));
    for (j = 0; j < 2; j++) {
      copy = 2 * i + j;
      x0 = (2 * j - 1) * geom->GetDistanceBetwRails() / 2.0;
      fMC->Gspos("PWHE", copy, "cave", x0, y0, z0, 0, "ONLY");
    }
  }
}

//-----------------------------------------
void Detector::defineSensitiveVolumes()
{
  if (fActive) {
    TGeoVolume* vsense = gGeoManager->GetVolume("PXTL");
    if (vsense) {
      AddSensitiveVolume(vsense);
    } else {
      LOG(ERROR) << "PHOS Sensitive volume PXTL not found ... No hit creation!\n";
    }
  }
}
