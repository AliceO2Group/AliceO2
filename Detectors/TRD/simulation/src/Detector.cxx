// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TRDSimulation/Detector.h"
#include <TGeoManager.h>
#include <TVirtualMC.h>
#include <vector>
#include "FairRootManager.h"
#include "FairVolume.h"
#include "TClonesArray.h"
#include "TRDBase/TRDCommonParam.h"
#include "TRDBase/TRDGeometry.h"

using namespace o2::trd;

Detector::Detector(const char* Name, bool Active)
  : o2::Base::Detector(Name, Active), mHitCollection(new TClonesArray("o2::BasicXYZEHit<float>"))
{
}

void Detector::Initialize() { o2::Base::Detector::Initialize(); }
bool Detector::ProcessHits(FairVolume* v)
{
  // very rudimentatary hit creation
  // TODO: needs upgrade to the level of AliROOT

  // TODO: reference to vmc --> put this as member of detector
  static auto vmc = TVirtualMC::GetMC();

  // If not charged track or already stopped or disappeared, just return.
  if ((!vmc->TrackCharge()) || vmc->IsTrackDisappeared()) {
    return false;
  }

  // just record position and basic quantities for the moment
  // TODO: needs to be interpreted properly
  double x, y, z;
  vmc->TrackPosition(x, y, z);

  float enDep = vmc->Edep();
  float time = vmc->TrackTime() * 1.0e09;
  auto trackID = vmc->GetStack()->GetCurrentTrackNumber();
  auto detID = v->getMCid();
  addHit((float)x, (float)y, (float)z, time, enDep, trackID, detID);

  return true;
}

void Detector::Register() { FairRootManager::Instance()->Register("TRDHit", "TRD", mHitCollection, true); }
TClonesArray* Detector::GetCollection(int iColl) const
{
  if (iColl == 0) {
    return mHitCollection;
  }
  return nullptr;
}

void Detector::Reset() {}
//_____________________________________________________________________________
void Detector::createMaterials()
{
  //
  // Create the materials for the TRD
  //
  int isxfld = 2;
  float sxmgmx = 10.;
  o2::Base::Detector::initFieldTrackingParams(isxfld, sxmgmx);

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
    // AliInfo("Gas mixture: Ar C02 (80/20)");
    LOG(INFO) << "Gas mixture: Ar C02 (80/20)\n";
    Mixture(53, "ArCO2", aArCO2, zArCO2, dgmAr, -3, wArCO2);
  } else {
    // AliFatal("Wrong gas mixture");
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
  mGeom->CreateGeometry(medmapping);

  // register the sensitive volumes with FairRoot
  defineSensitiveVolumes();
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

ClassImp(Detector);
