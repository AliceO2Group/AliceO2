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

#include <FairVolume.h>

#include <TVirtualMC.h>
#include <TVirtualMCStack.h>
#include <TGeoVolume.h>
#include <TGeoTube.h>
#include <TGeoMatrix.h>
#include <cmath>
#include <limits>

#include "DetectorsBase/Stack.h"
#include "ITSMFTSimulation/Hit.h"
#include "RICHSimulation/Detector.h"
#include "RICHBase/RICHBaseParam.h"

using o2::itsmft::Hit;

namespace o2
{
namespace rich
{
namespace // quadrant equation solver
{
double quadrantDeltaPhiEquation(double x, int nTilesPhi, double rMin, double totalBoundaryWidth)
{
  const double argument = totalBoundaryWidth * TMath::Cos(x / 2.0) / (2.0 * rMin);
  if (TMath::Abs(argument) >= 1.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double rhs = 2.0 * TMath::Pi() / static_cast<double>(nTilesPhi) - (8.0 / static_cast<double>(nTilesPhi)) * TMath::ASin(argument);
  return rhs - x;
}

double solveQuadrantDeltaPhi(int nTilesPhi, double rMin, double totalBoundaryWidth)
{
  double lower = 0.0;
  double upper = 1.1 * 2.0 * TMath::Pi() / static_cast<double>(nTilesPhi);
  double fLower = quadrantDeltaPhiEquation(lower, nTilesPhi, rMin, totalBoundaryWidth);
  double fUpper = quadrantDeltaPhiEquation(upper, nTilesPhi, rMin, totalBoundaryWidth);
  if (!std::isfinite(fLower) || !std::isfinite(fUpper) || fLower * fUpper > 0.0) {
    return -1.0;
  }
  constexpr double tolerance = 1.0e-12;
  constexpr int maxIterations = 200;
  for (int iteration = 0; iteration < maxIterations; iteration++) {
    const double middle = 0.5 * (lower + upper);
    const double fMiddle = quadrantDeltaPhiEquation(middle, nTilesPhi, rMin, totalBoundaryWidth);
    if (!std::isfinite(fMiddle)) {
      return -1.0;
    }
    if (TMath::Abs(fMiddle) < tolerance || 0.5 * (upper - lower) < tolerance) {
      return middle;
    }
    if (fLower * fMiddle < 0.0) {
      upper = middle;
      fUpper = fMiddle;
    } else {
      lower = middle;
      fLower = fMiddle;
    }
  }
  return 0.5 * (lower + upper);
}
} // namespace

Detector::Detector()
  : o2::base::DetImpl<Detector>("RCH", true),
    mTrackData(),
    mHits(o2::utils::createSimVector<o2::itsmft::Hit>())
{
}

Detector::Detector(bool active)
  : o2::base::DetImpl<Detector>("RCH", true),
    mTrackData(),
    mHits(o2::utils::createSimVector<o2::itsmft::Hit>())
{
  auto& richPars = RICHBaseParam::Instance();
  mRings.resize(richPars.nRings);
  mNTiles = richPars.nTiles;
  LOGP(info, "Summary of RICH configuration:\n\tNumber of rings: {}\n\tNumber of tiles per ring: {}", mRings.size(), mNTiles);
}

Detector::~Detector()
{
  if (mHits) {
    o2::utils::freeSimVector(mHits);
  }
}

void Detector::ConstructGeometry()
{
  createMaterials();
  createGeometry();
}

void Detector::createMaterials()
{
  auto& richPars = RICHBaseParam::Instance();
  const double nGasEffective = richPars.nGasEffective;
  const double nAerogelEffective = richPars.nAerogelEffective;

  int ifield = 2;      // ?
  float fieldm = 10.0; // ?
  o2::base::Detector::initFieldTrackingParams(ifield, fieldm);

  float tmaxfdSi = 0.1;    // .10000E+01; // Degree
  float stemaxSi = 0.0075; //  .10000E+01; // cm
  float deemaxSi = 0.1;    // 0.30000E-02; // Fraction of particle's energy 0<deemax<=1
  float epsilSi = 1.0E-4;  // .10000E+01;
  float stminSi = 0.0;     // cm "Default value used"

  float tmaxfdAir = 0.1;        // .10000E+01; // Degree
  float stemaxAir = .10000E+01; // cm
  float deemaxAir = 0.1;        // 0.30000E-02; // Fraction of particle's energy 0<deemax<=1
  float epsilAir = 1.0E-4;      // .10000E+01;
  float stminAir = 0.0;         // cm "Default value used"

  float tmaxfdCer = 0.1;        // .10000E+01; // Degree
  float stemaxCer = .10000E+01; // cm
  float deemaxCer = 0.1;        // 0.30000E-02; // Fraction of particle's energy 0<deemax<=1
  float epsilCer = 1.0E-4;      // .10000E+01;
  float stminCer = 0.0;         // cm "Default value used"

  float tmaxfdAerogel = 0.1;        // .10000E+01; // Degree
  float stemaxAerogel = .10000E+01; // cm
  float deemaxAerogel = 0.1;        // 0.30000E-02; // Fraction of particle's energy 0<deemax<=1
  float epsilAerogel = 1.0E-4;      // .10000E+01;
  float stminAerogel = 0.0;         // cm "Default value used"

  float tmaxfdCO2 = 0.1;        // .10000E+01; // Degree
  float stemaxCO2 = .10000E+01; // cm
  float deemaxCO2 = 0.1;        // 0.30000E-02; // Fraction of particle's energy 0<deemax<=1
  float epsilCO2 = 1.0E-4;      // .10000E+01;
  float stminCO2 = 0.0;         // cm "Default value used"

  float tmaxfdFR4 = 0.1; // degree
  float stemaxFR4 = 0.1; // cm (1 mm is a reasonable choice for PCB)
  float deemaxFR4 = 0.1;
  float epsilFR4 = 1.0E-4;
  float stminFR4 = 0.0;

  float tmaxfdPEEK = 0.1; // degree
  float stemaxPEEK = 0.1; // cm (1 mm is a reasonable choice for PCB)
  float deemaxPEEK = 0.1;
  float epsilPEEK = 1.0E-4;
  float stminPEEK = 0.0;

  float tmaxfdAl = 0.1; // degree
  float stemaxAl = 0.1; // cm
  float deemaxAl = 0.1;
  float epsilAl = 1.0E-4;
  float stminAl = 0.0;

  float tmaxfdArgon = 0.1;        // .10000E+01; // Degree
  float stemaxArgon = .10000E+01; // cm
  float deemaxArgon = 0.1;        // 0.30000E-02; // Fraction of particle's energy 0<deemax<=1
  float epsilArgon = 1.0E-4;      // .10000E+01;
  float stminArgon = 0.0;         // cm "Default value used"

  float tmaxfdSiO2 = 0.1; // degree
  float stemaxSiO2 = 0.1; // cm
  float deemaxSiO2 = 0.1;
  float epsilSiO2 = 1.0E-4;
  float stminSiO2 = 0.0;

  float tmaxfdSilicone = 0.1; // degree
  float stemaxSilicone = 0.1; // cm
  float deemaxSilicone = 0.1;
  float epsilSilicone = 1.0E-4;
  float stminSilicone = 0.0;

  float tmaxfdSiAbsorber = tmaxfdSi;
  float stemaxSiAbsorber = stemaxSi;
  float deemaxSiAbsorber = deemaxSi;
  float epsilSiAbsorber = epsilSi;
  float stminSiAbsorber = stminSi;

  // ArmaFlex elastomeric insulation
  float tmaxfdArmaFlex = 0.1;
  float stemaxArmaFlex = 0.1; // cm
  float deemaxArmaFlex = 0.1;
  float epsilArmaFlex = 1.0E-4;
  float stminArmaFlex = 0.0;

  // ArmaGel aerogel blanket
  float tmaxfdArmaGel = 0.1;
  float stemaxArmaGel = 0.1; // cm
  float deemaxArmaGel = 0.1;
  float epsilArmaGel = 1.0E-4;
  float stminArmaGel = 0.0;

  // High-temperature cofired ceramic
  float tmaxfdHTCC = 0.1;
  float stemaxHTCC = 0.1; // cm
  float deemaxHTCC = 0.1;
  float epsilHTCC = 1.0E-4;
  float stminHTCC = 0.0;

  // AIR
  float aAir[4] = {12.0107, 14.0067, 15.9994, 39.948};
  float zAir[4] = {6., 7., 8., 18.};
  float wAir[4] = {0.000124, 0.755267, 0.231781, 0.012827};
  float dAir = 1.20479E-3;

  // Carbon fiber
  float aCf[2] = {12.0107, 1.00794};
  float zCf[2] = {6., 1.};

  // Silica aerogel https://pdg.lbl.gov/2023/AtomicNuclearProperties/HTML/silica_aerogel.html
  float aAerogel[3] = {15.9990, 28.0855, 1.00794};
  float zAerogel[3] = {8., 14., 1.};
  float wAerogel[3] = {0.543192, 0.453451, 0.003357};
  float dAerogel = 0.200; // g/cm3

  // CO2 https://pdg.lbl.gov/2023/AtomicNuclearProperties/HTML/carbon_dioxide.html
  float aCO2[2] = {12.0107, 15.9994};
  float zCO2[2] = {6., 8.};
  float wCO2[2] = {0.2729, 0.7271};
  float dCO2 = 1.842E-3; // g/cm3

  // FR4 for PCBs (approximate composition with H, C, O, Si, Br)
  float aFR4[5] = {1.00794, 12.0107, 15.9994, 28.0855, 79.904};
  float zFR4[5] = {1., 6., 8., 14., 35.};
  float wFR4[5] = {0.068, 0.278, 0.405, 0.180, 0.069};
  float dFR4 = 1.86; // g/cm3

  // PEEK for insulation shielding
  float aPEEK[3] = {1.00794, 12.0107, 15.9994};
  float zPEEK[3] = {1., 6., 8.};
  float wPEEK[3] = {0.041954, 0.791557, 0.166489};
  float dPEEK = 1.30; // g/cm3

  // Aluminum cooling plate
  float aAl = 26.9815385;
  float zAl = 13.;
  float dAl = 2.70;                 // g/cm3
  float radLengthAl = 8.897;        // cm
  float interactionLengthAl = 39.4; // cm

  // Argon
  float aArgon = 39.948;
  float zArgon = 18.;
  float dArgon = 1.782E-3;               // g/cm3
  float radLengthAr = 1.09708E4;         // cm
  float interactionLengthAr = 6.71717E4; // cm

  // Fused silica, SiO2
  float aSiO2[2] = {28.0855, 15.9994};
  float zSiO2[2] = {14., 8.};
  float wSiO2[2] = {0.467435, 0.532565};
  float dSiO2 = 2.20; // g/cm3

  // Silicone resin approximated as PDMS: (C2 H6 O Si)n
  float aSilicone[4] = {12.0107, 1.00794, 15.9994, 28.0855};
  float zSilicone[4] = {6., 1., 8., 14.};
  float wSilicone[4] = {0.323940, 0.081555, 0.215759, 0.378746};
  float dSilicone = 1.05; // g/cm3

  // ArmaFlex approximation of an NBR/PVC closed-cell elastomeric foam (C, H, N, Cl)
  float aArmaFlex[4] = {12.0107, 1.00794, 14.0067, 35.453};
  float zArmaFlex[4] = {6., 1., 7., 17.};
  float wArmaFlex[4] = {0.6053, 0.0720, 0.0391, 0.2836};
  // Manufacturer range is approximately 0.048--0.096 g/cm3.
  // Use the midpoint as an effective foam density.
  float dArmaFlex = 0.072; // g/cm3

  // ArmaGel HT approximated as a silica-aerogel blanket (same as aerogel but with ArmaGel nominal blanket density)
  float aArmaGel[3] = {15.9990, 28.0855, 1.00794};
  float zArmaGel[3] = {8., 14., 1.};
  float wArmaGel[3] = {0.543192, 0.453451, 0.003357};
  float dArmaGel = 0.180; // g/cm3

  // High-temperature cofired ceramic based on aluminium nitride
  float aHTCC[2] = {26.9815385, 14.0067};
  float zHTCC[2] = {13., 7.};
  float wHTCC[2] = {0.658275, 0.341725};
  float dHTCC = 3.30; // g/cm3

  o2::base::Detector::Mixture(1, "AIR$", aAir, zAir, dAir, 4, wAir);
  o2::base::Detector::Medium(1, "AIR$", 1, 0, ifield, fieldm, tmaxfdAir, stemaxAir, deemaxAir, epsilAir, stminAir);

  o2::base::Detector::Material(3, "SILICON$", 0.28086E+02, 0.14000E+02, 0.23300E+01, 0.93600E+01, 0.99900E+03);
  o2::base::Detector::Medium(3, "SILICON$", 3, 0, ifield, fieldm, tmaxfdSi, stemaxSi, deemaxSi, epsilSi, stminSi);

  o2::base::Detector::Mixture(2, "AEROGEL$", aAerogel, zAerogel, dAerogel, 3, wAerogel);
  o2::base::Detector::Medium(2, "AEROGEL$", 2, 0, ifield, fieldm, tmaxfdAerogel, stemaxAerogel, deemaxAerogel, epsilAerogel, stminAerogel);

  o2::base::Detector::Material(4, "ARGON$", aArgon, zArgon, dArgon, radLengthAr, interactionLengthAr);
  o2::base::Detector::Medium(4, "ARGON$", 4, 0, ifield, fieldm, tmaxfdArgon, stemaxArgon, deemaxArgon, epsilArgon, stminArgon);

  o2::base::Detector::Mixture(5, "CO2$", aCO2, zCO2, dCO2, 2, wCO2);
  o2::base::Detector::Medium(5, "CO2$", 5, 0, ifield, fieldm, tmaxfdCO2, stemaxCO2, deemaxCO2, epsilCO2, stminCO2);

  o2::base::Detector::Mixture(6, "FR4$", aFR4, zFR4, dFR4, 5, wFR4);
  o2::base::Detector::Medium(6, "FR4$", 6, 0, ifield, fieldm, tmaxfdFR4, stemaxFR4, deemaxFR4, epsilFR4, stminFR4);

  o2::base::Detector::Mixture(7, "PEEK$", aPEEK, zPEEK, dPEEK, 3, wPEEK);
  o2::base::Detector::Medium(7, "PEEK$", 7, 0, ifield, fieldm, tmaxfdPEEK, stemaxPEEK, deemaxPEEK, epsilPEEK, stminPEEK);

  o2::base::Detector::Material(8, "ALUMINUM$", aAl, zAl, dAl, radLengthAl, interactionLengthAl);
  o2::base::Detector::Medium(8, "ALUMINUM$", 8, 0, ifield, fieldm, tmaxfdAl, stemaxAl, deemaxAl, epsilAl, stminAl);

  o2::base::Detector::Mixture(9, "SIO2$", aSiO2, zSiO2, dSiO2, 2, wSiO2);
  o2::base::Detector::Medium(9, "SIO2$", 9, 0, ifield, fieldm, tmaxfdSiO2, stemaxSiO2, deemaxSiO2, epsilSiO2, stminSiO2);

  o2::base::Detector::Mixture(10, "SILICONE$", aSilicone, zSilicone, dSilicone, 4, wSilicone);
  o2::base::Detector::Medium(10, "SILICONE$", 10, 0, ifield, fieldm, tmaxfdSilicone, stemaxSilicone, deemaxSilicone, epsilSilicone, stminSilicone);

  o2::base::Detector::Material(11, "SILICON_ABSORBER$", 0.28086E+02, 0.14000E+02, 0.23300E+01, 0.93600E+01, 0.99900E+03);
  o2::base::Detector::Medium(11, "SILICON_ABSORBER$", 11, 0, ifield, fieldm, tmaxfdSiAbsorber, stemaxSiAbsorber, deemaxSiAbsorber, epsilSiAbsorber, stminSiAbsorber);

  o2::base::Detector::Mixture(12, "ARMAFLEX$", aArmaFlex, zArmaFlex, dArmaFlex, 4, wArmaFlex);
  o2::base::Detector::Medium(12, "ARMAFLEX$", 12, 0, ifield, fieldm, tmaxfdArmaFlex, stemaxArmaFlex, deemaxArmaFlex, epsilArmaFlex, stminArmaFlex);

  o2::base::Detector::Mixture(13, "ARMAGEL$", aArmaGel, zArmaGel, dArmaGel, 3, wArmaGel);
  o2::base::Detector::Medium(13, "ARMAGEL$", 13, 0, ifield, fieldm, tmaxfdArmaGel, stemaxArmaGel, deemaxArmaGel, epsilArmaGel, stminArmaGel);

  o2::base::Detector::Mixture(14, "HTCC$", aHTCC, zHTCC, dHTCC, 2, wHTCC);
  o2::base::Detector::Medium(14, "HTCC$", 14, 0, ifield, fieldm, tmaxfdHTCC, stemaxHTCC, deemaxHTCC, epsilHTCC, stminHTCC);

  // Optical properties
  auto* mc = TVirtualMC::GetMC();
  if (!mc) {
    LOGP(fatal,
         "RICH: TVirtualMC instance is not available while "
         "defining optical properties");
  }

  constexpr double eVInGeV = 1.0e-9;

  auto globalMediumID = [&](int localMediumID, const char* mediumName) {
    const int id = getMediumID(localMediumID);
    if (id < 0) {
      LOGP(fatal, "RICH: no global medium ID found for {} local medium {}", mediumName, localMediumID);
    }
    return id;
  };

  /// AEROGEL
  constexpr int nAerogelRindex = 20;
  double aerogelRindexEnergyGeV[nAerogelRindex] = {1.00 * eVInGeV, 1.06 * eVInGeV, 1.12 * eVInGeV, 1.18 * eVInGeV, 1.23984 * eVInGeV, 1.3051 * eVInGeV, 1.3776 * eVInGeV, 1.45864 * eVInGeV, 1.5498 * eVInGeV, 1.65312 * eVInGeV, 1.7712 * eVInGeV, 1.90745 * eVInGeV, 2.0664 * eVInGeV, 2.25426 * eVInGeV, 2.47968 * eVInGeV, 2.7552 * eVInGeV, 3.0996 * eVInGeV, 3.54241 * eVInGeV, 4.13281 * eVInGeV, 4.5 * eVInGeV};
  double aerogelRindex[nAerogelRindex] = {1.030402, 1.030410, 1.030418, 1.030426, 1.03044, 1.03045, 1.03046, 1.03047, 1.03049, 1.03051, 1.03054, 1.03057, 1.03061, 1.03066, 1.03073, 1.03082, 1.03095, 1.03114, 1.03144, 1.031695};
  const double scale = nAerogelEffective / 1.03095; // <-- Original table has n(400 nm = 3.0996 eV) = 1.03095.
  for (int i = 0; i < nAerogelRindex; ++i) {
    aerogelRindex[i] *= scale;
  }
  // SetCerenkov() requires absorption and efficiency arrays on the same grid as n
  double aerogelAbsorptionLengthCm = 1.0e5f; // 1 km
  double aerogelAbsorptionOnRindexGrid[nAerogelRindex];
  double aerogelDetectionEfficiency[nAerogelRindex] = {};
  for (int i = 0; i < nAerogelRindex; ++i) {
    aerogelAbsorptionOnRindexGrid[i] = aerogelAbsorptionLengthCm;
  }
  mc->SetCerenkov(globalMediumID(2, "AEROGEL"), nAerogelRindex, aerogelRindexEnergyGeV, aerogelAbsorptionOnRindexGrid, aerogelDetectionEfficiency, aerogelRindex);
  //
  // constexpr int nAerogelAbsorption = 2;
  // double aerogelAbsorptionEnergyGeV[nAerogelAbsorption] = {1.0 * eVInGeV, 8.26561 * eVInGeV};
  // double aerogelAbsorptionLengthCm[nAerogelAbsorption] = {aerogelAbsorptionLengthCm, aerogelAbsorptionLengthCm};
  // mc->SetMaterialProperty(globalMediumID(2, "AEROGEL"), "ABSLENGTH", nAerogelAbsorption, aerogelAbsorptionEnergyGeV, aerogelAbsorptionLengthCm);
  //
  constexpr int nAerogelRayleigh = 22;
  double aerogelRayleighEnergyGeV[nAerogelRayleigh] = {1.00 * eVInGeV, 1.06 * eVInGeV, 1.12 * eVInGeV, 1.18 * eVInGeV, 1.23984 * eVInGeV, 1.3051 * eVInGeV, 1.3776 * eVInGeV, 1.45864 * eVInGeV, 1.5498 * eVInGeV, 1.65312 * eVInGeV, 1.7712 * eVInGeV, 1.90745 * eVInGeV, 2.0664 * eVInGeV, 2.25426 * eVInGeV, 2.47968 * eVInGeV, 2.7552 * eVInGeV, 3.0996 * eVInGeV, 3.54241 * eVInGeV, 4.13281 * eVInGeV, 4.95937 * eVInGeV, 6.19921 * eVInGeV, 8.26561 * eVInGeV};
  double aerogelRayleighLengthCm[nAerogelRayleigh] = {543.253684, 430.307801, 345.247537, 280.204207, 229.885, 187.243, 150.828, 120.001, 94.1609, 72.7371, 55.1954, 41.0359, 29.7931, 21.0359, 14.3678, 9.42672, 5.88506, 3.44971, 1.86207, 0.897989, 0.367816, 0.116379};
  mc->SetMaterialProperty(globalMediumID(2, "AEROGEL"), "RAYLEIGH", nAerogelRayleigh, aerogelRayleighEnergyGeV, aerogelRayleighLengthCm);

  /// GAS
  constexpr int nCO2Optical = 2;
  double co2EnergyGeV[nCO2Optical] = {1.0 * eVInGeV, 8.26561 * eVInGeV};
  double co2Rindex[nCO2Optical] = {nGasEffective, nGasEffective}; // <- Target gas index for dielectrons
  double co2AbsorptionLengthCm[nCO2Optical] = {1.0e5, 1.0e5};
  double co2DetectionEfficiency[nCO2Optical] = {0.0, 0.0};
  mc->SetCerenkov(globalMediumID(5, "CO2"), nCO2Optical, co2EnergyGeV, co2AbsorptionLengthCm, co2DetectionEfficiency, co2Rindex);

  /// SiO2
  constexpr int nSiO2Optical = 2;
  double sio2EnergyGeV[nSiO2Optical] = {1.0 * eVInGeV, 8.26561 * eVInGeV};
  double sio2Rindex[nSiO2Optical] = {1.47, 1.47};
  double sio2AbsorptionLengthCm[nSiO2Optical] = {1.0e5, 1.0e5};
  double sio2DetectionEfficiency[nSiO2Optical] = {0.0, 0.0};
  mc->SetCerenkov(globalMediumID(9, "SIO2"), nSiO2Optical, sio2EnergyGeV, sio2AbsorptionLengthCm, sio2DetectionEfficiency, sio2Rindex);

  /// Silicone resin
  constexpr int nSiliconeOptical = 2;
  double siliconeEnergyGeV[nSiliconeOptical] = {1.0 * eVInGeV, 8.26561 * eVInGeV};
  double siliconeRindex[nSiliconeOptical] = {1.41, 1.41};
  double siliconeAbsorptionLengthCm[nSiliconeOptical] = {1.0e5, 1.0e5};
  double siliconeDetectionEfficiency[nSiliconeOptical] = {0.0, 0.0};
  mc->SetCerenkov(globalMediumID(10, "SILICONE"), nSiliconeOptical, siliconeEnergyGeV, siliconeAbsorptionLengthCm, siliconeDetectionEfficiency, siliconeRindex);

  /// Si (assuming same index as silicone resin as reflection losses are already included in PDE)
  constexpr int nSiliconOptical = 2;
  double siliconEnergyGeV[nSiliconOptical] = {1.0 * eVInGeV, 8.26561 * eVInGeV};
  double siliconRindex[nSiliconOptical] = {1.41, 1.41};
  double siliconAbsorptionLengthCm[nSiliconOptical] = {1.0e5, 1.0e5};
  double siliconDetectionEfficiency[nSiliconOptical] = {0.0, 0.0};
  mc->SetCerenkov(globalMediumID(3, "SILICON"), nSiliconOptical, siliconEnergyGeV, siliconAbsorptionLengthCm, siliconDetectionEfficiency, siliconRindex);

  // Si: outer layer just for photon absorption
  constexpr int nSiliconAbsorberOptical = 2;
  double siliconAbsorberEnergyGeV[nSiliconAbsorberOptical] = {1.0 * eVInGeV, 8.26561 * eVInGeV};
  double siliconAbsorberAbsorptionLengthCm[nSiliconAbsorberOptical] = {1.0e-7, 1.0e-7}; // 1 nm
  mc->SetMaterialProperty(globalMediumID(11, "SILICON_ABSORBER"), "ABSLENGTH", nSiliconAbsorberOptical, siliconAbsorberEnergyGeV, siliconAbsorberAbsorptionLengthCm);
}

void Detector::createGeometry()
{
  TGeoManager* geoManager = gGeoManager;
  TGeoVolume* vALIC = geoManager->GetVolume("barrel");
  if (!vALIC) {
    LOGP(fatal, "Could not find barrel volume while constructing RICH geometry");
  }
  new TGeoVolumeAssembly(GeometryTGeo::getRICHVolPattern());
  TGeoVolume* vRICH = geoManager->GetVolume(GeometryTGeo::getRICHVolPattern());
  vALIC->AddNode(vRICH, 2, new TGeoTranslation(0, 30., 0));

  char vstrng[100] = "RICHV";
  vRICH->SetTitle(vstrng);
  auto& richPars = RICHBaseParam::Instance();

  // Quadrant parameters
  const bool flagUseQuadrants = richPars.flagUseQuadrants;
  const double vesselPhiGap = richPars.vesselPhiGap;
  const double vesselThicknessShieldingLateral = richPars.vesselThicknessShieldingLateral;

  // shielding parameters
  double shieldRMin = richPars.shieldRMin;
  double shieldRMax = richPars.shieldRMax;
  double innerWallThickness = richPars.innerWallThickness;
  double outerWallThickness = richPars.outerWallThickness;
  double shieldLengthZ = richPars.shieldLengthZ;
  double endCapThicknessZ = richPars.endCapThicknessZ;

  if (innerWallThickness <= 0.0 || outerWallThickness <= 0.0 || endCapThicknessZ <= 0.0 || shieldLengthZ <= 0.0) {
    LOGP(fatal, "RICH shielding dimensions must be positive");
  }

  if (shieldRMin + innerWallThickness >= shieldRMax - outerWallThickness) {
    LOGP(fatal,
         "RICH shielding walls overlap: inner outer radius = {}, outer inner radius = {}",
         shieldRMin + innerWallThickness,
         shieldRMax - outerWallThickness);
  }

  if (flagUseQuadrants) {
    if (richPars.nTiles <= 0 || richPars.nTiles % 4 != 0) {
      LOGP(fatal, "RICH quadrant geometry requires nTiles to be positive and divisible by four; received {}", richPars.nTiles);
    }
    if (vesselPhiGap < 0.0 || vesselThicknessShieldingLateral <= 0.0) {
      LOGP(fatal, "RICH quadrant gap must be non-negative and lateral shielding thickness must be positive");
    }
    const double totalBoundaryWidth = 2.0 * vesselThicknessShieldingLateral + vesselPhiGap;
    if (totalBoundaryWidth >= 2.0 * richPars.rMin || vesselPhiGap >= 2.0 * shieldRMin) {
      LOGP(fatal, "RICH quadrant boundary dimensions are incompatible with rMin={} cm and shieldRMin={} cm", richPars.rMin, shieldRMin);
    }
    const double quadrantDeltaPhi = solveQuadrantDeltaPhi(richPars.nTiles, richPars.rMin, totalBoundaryWidth);
    if (!(quadrantDeltaPhi > 0.0)) {
      LOGP(fatal, "RICH could not solve the quadrant module angular pitch");
    }
  }

  // Name of the gas mother volume. This name will also be passed
  // to each Ring so that the ring components become its daughters.
  const char* richGasMotherName = "RICH_GAS_MOTHER";

  TGeoMedium* medCO2 = gGeoManager->GetMedium("RCH_CO2$");
  if (!medCO2) {
    LOGP(fatal, "RICH: CO2 medium not found");
  }

  TGeoMedium* medPeek = gGeoManager->GetMedium("RCH_PEEK$");
  if (!medPeek) {
    LOGP(fatal, "RICH: PEEK medium not found");
  }

  TGeoMedium* medArmaFlex = gGeoManager->GetMedium("RCH_ARMAFLEX$");
  if (!medArmaFlex) {
    LOGP(fatal, "RICH: ArmaFlex medium not found");
  }

  TGeoMedium* medArmaGel = gGeoManager->GetMedium("RCH_ARMAGEL$");
  if (!medArmaGel) {
    LOGP(fatal, "RICH: ArmaGel medium not found");
  }

  prepareLayout(); // Preparing the positions of the rings and tiles

  // The gas mother includes the side-wall region and both end caps. ( as vessel )
  const double gasEnvelopeLengthZ = shieldLengthZ + 2.0 * endCapThicknessZ;
  auto* gasEnvelopeShape = new TGeoTube(shieldRMin, shieldRMax, gasEnvelopeLengthZ / 2.0);
  auto* gasEnvelopeVolume = new TGeoVolume(richGasMotherName, gasEnvelopeShape, medCO2);

  gasEnvelopeVolume->SetLineColor(kBlue - 9);
  gasEnvelopeVolume->SetTransparency(90);

  // The gas envelope is a daughter of the general RICH volume.
  vRICH->AddNode(gasEnvelopeVolume, 1, new TGeoTranslation(0.0, 0.0, 0.0));

  if (!flagUseQuadrants) {
    // ============================================================
    // Inner cylindrical insulating wall
    //
    // Radial interval:
    //   shieldRMin --> shieldRMin + innerWallThickness
    //
    // Longitudinal interval:
    //   -shieldLengthZ/2 --> +shieldLengthZ/2
    // ============================================================
    auto* innerWallShape = new TGeoTube(shieldRMin, shieldRMin + innerWallThickness, shieldLengthZ / 2.0);
    auto* innerWallVolume = new TGeoVolume("RICH_SHIELD_INNER_WALL", innerWallShape, medArmaGel);

    innerWallVolume->SetLineColor(kOrange - 8); // kGray
    innerWallVolume->SetTransparency(0);        // 80
    gasEnvelopeVolume->AddNode(innerWallVolume, 1, new TGeoTranslation(0.0, 0.0, 0.0));

    // ============================================================
    // Outer cylindrical insulating wall
    //
    // Radial interval:
    //   shieldRMax - outerWallThickness --> shieldRMax
    //
    // Longitudinal interval:
    //   -shieldLengthZ/2 --> +shieldLengthZ/2
    // ============================================================

    auto* outerWallShape = new TGeoTube(shieldRMax - outerWallThickness, shieldRMax, shieldLengthZ / 2.0);
    auto* outerWallVolume = new TGeoVolume("RICH_SHIELD_OUTER_WALL", outerWallShape, medArmaGel);

    outerWallVolume->SetLineColor(kOrange - 8); // kGray
    outerWallVolume->SetTransparency(0);        // 80
    gasEnvelopeVolume->AddNode(outerWallVolume, 1, new TGeoTranslation(0.0, 0.0, 0.0));

    // ============================================================
    // Insulating end caps
    //
    // Each end cap covers:
    //   shieldRMin --> shieldRMax
    //
    // Each has full thickness:
    //   endCapThicknessZ
    // ============================================================

    auto* endCapShape = new TGeoTube(shieldRMin, shieldRMax, endCapThicknessZ / 2.0);
    auto* endCapPlusVolume = new TGeoVolume("RICH_SHIELD_ENDCAP_PLUS", endCapShape, medArmaGel);
    auto* endCapMinusVolume = new TGeoVolume("RICH_SHIELD_ENDCAP_MINUS", endCapShape, medArmaGel);

    endCapPlusVolume->SetLineColor(kOrange - 8); // kGray
    endCapPlusVolume->SetTransparency(0);        // 80

    endCapMinusVolume->SetLineColor(kOrange - 8); // kGray
    endCapMinusVolume->SetTransparency(0);        // 80

    const double endCapCenterZ = shieldLengthZ / 2.0 + endCapThicknessZ / 2.0;

    gasEnvelopeVolume->AddNode(endCapPlusVolume, 1, new TGeoTranslation(0.0, 0.0, endCapCenterZ));
    gasEnvelopeVolume->AddNode(endCapMinusVolume, 1, new TGeoTranslation(0.0, 0.0, -endCapCenterZ));
  } else {
    // ============================================================
    // Four independent insulating vessel quadrants
    // ============================================================
    const double totalBoundaryWidth = 2.0 * vesselThicknessShieldingLateral + vesselPhiGap;
    const double moduleDeltaPhi = solveQuadrantDeltaPhi(richPars.nTiles, richPars.rMin, totalBoundaryWidth);
    const double moduleExtraPhi = TMath::ASin(totalBoundaryWidth / (2.0 * richPars.rMin));
    const double vesselGapHalfPhi = TMath::ASin(vesselPhiGap / (2.0 * shieldRMin));
    const double quadrantSpanPhi = TMath::Pi() / 2.0 - 2.0 * vesselGapHalfPhi;
    const int modulesPerQuadrant = richPars.nTiles / 4;

    // Remaining angular space between the last module of a quadrant and the following vessel gap.
    const double endModuleExtraPhi = TMath::Pi() / 2.0 - moduleExtraPhi - static_cast<double>(modulesPerQuadrant) * moduleDeltaPhi;
    const double lateralStartWallSpanPhi = moduleExtraPhi - vesselGapHalfPhi;
    const double lateralEndWallSpanPhi = endModuleExtraPhi - vesselGapHalfPhi;

    if (quadrantSpanPhi <= 0.0 || lateralStartWallSpanPhi <= 0.0 || lateralEndWallSpanPhi <= 0.0 || lateralStartWallSpanPhi + lateralEndWallSpanPhi >= quadrantSpanPhi) {
      LOGP(fatal, "RICH invalid quadrant angular dimensions: vessel span={}, start wall span={}, end wall span={}", quadrantSpanPhi, lateralStartWallSpanPhi, lateralEndWallSpanPhi);
    }

    const double radToDeg = 180.0 / TMath::Pi();
    const double quadrantSpanDeg = quadrantSpanPhi * radToDeg;
    const double lateralStartWallSpanDeg = lateralStartWallSpanPhi * radToDeg;
    const double lateralEndWallSpanDeg = lateralEndWallSpanPhi * radToDeg;
    const double vesselGapHalfDeg = vesselGapHalfPhi * radToDeg;
    const double innerGasRadius = shieldRMin + innerWallThickness;
    const double outerGasRadius = shieldRMax - outerWallThickness;

    // Inner cylindrical shielding, divided into four sectors.
    auto* innerWallQuadrantShape = new TGeoTubeSeg("RICH_SHIELD_INNER_WALL_QUADRANT_SHAPE", shieldRMin, innerGasRadius, shieldLengthZ / 2.0, 0.0, quadrantSpanDeg);

    // Outer cylindrical shielding, divided into four sectors.
    auto* outerWallQuadrantShape = new TGeoTubeSeg("RICH_SHIELD_OUTER_WALL_QUADRANT_SHAPE", outerGasRadius, shieldRMax, shieldLengthZ / 2.0, 0.0, quadrantSpanDeg);

    // End caps divided into four sectors.
    auto* endCapQuadrantShape = new TGeoTubeSeg("RICH_SHIELD_ENDCAP_QUADRANT_SHAPE", shieldRMin, shieldRMax, endCapThicknessZ / 2.0, 0.0, quadrantSpanDeg);

    // Lateral wall at the beginning of each quadrant.
    auto* lateralStartWallShape = new TGeoTubeSeg("RICH_SHIELD_LATERAL_START_WALL_SHAPE", innerGasRadius, outerGasRadius, shieldLengthZ / 2.0, 0.0, lateralStartWallSpanDeg);

    // Lateral wall at the end of each quadrant.
    auto* lateralEndWallShape = new TGeoTubeSeg("RICH_SHIELD_LATERAL_END_WALL_SHAPE", innerGasRadius, outerGasRadius, shieldLengthZ / 2.0, 0.0, lateralEndWallSpanDeg);

    auto* innerWallQuadrantVolume = new TGeoVolume("RICH_SHIELD_INNER_WALL_QUADRANT", innerWallQuadrantShape, medArmaGel);
    auto* outerWallQuadrantVolume = new TGeoVolume("RICH_SHIELD_OUTER_WALL_QUADRANT", outerWallQuadrantShape, medArmaGel);
    auto* endCapPlusQuadrantVolume = new TGeoVolume("RICH_SHIELD_ENDCAP_PLUS_QUADRANT", endCapQuadrantShape, medArmaGel);
    auto* endCapMinusQuadrantVolume = new TGeoVolume("RICH_SHIELD_ENDCAP_MINUS_QUADRANT", endCapQuadrantShape, medArmaGel);
    auto* lateralStartWallVolume = new TGeoVolume("RICH_SHIELD_LATERAL_START_WALL", lateralStartWallShape, medArmaGel);
    auto* lateralEndWallVolume = new TGeoVolume("RICH_SHIELD_LATERAL_END_WALL", lateralEndWallShape, medArmaGel);

    innerWallQuadrantVolume->SetLineColor(kOrange - 8);   // kGray
    outerWallQuadrantVolume->SetLineColor(kOrange - 8);   // kGray
    endCapPlusQuadrantVolume->SetLineColor(kOrange - 8);  // kGray
    endCapMinusQuadrantVolume->SetLineColor(kOrange - 8); // kGray
    lateralStartWallVolume->SetLineColor(kOrange - 8);    // kGray
    lateralEndWallVolume->SetLineColor(kOrange - 8);      // kGray

    innerWallQuadrantVolume->SetTransparency(0);   // 80
    outerWallQuadrantVolume->SetTransparency(0);   // 80
    endCapPlusQuadrantVolume->SetTransparency(0);  // 80
    endCapMinusQuadrantVolume->SetTransparency(0); // 80
    lateralStartWallVolume->SetTransparency(0);    // 80
    lateralEndWallVolume->SetTransparency(0);      // 80

    const double endCapCenterZ = shieldLengthZ / 2.0 + endCapThicknessZ / 2.0;

    for (int quadrant = 0; quadrant < 4; quadrant++) {

      const double quadrantStartDeg = -45.0 + static_cast<double>(quadrant) * 90.0 + vesselGapHalfDeg;
      const double quadrantEndDeg = quadrantStartDeg + quadrantSpanDeg;

      auto makeRotation = [&](const char* prefix, double angleDeg) {
        auto* rotation = new TGeoRotation(Form("%s_%d", prefix, quadrant));
        rotation->RotateZ(angleDeg);
        return rotation;
      };

      // Inner cylindrical wall sector.
      gasEnvelopeVolume->AddNode(innerWallQuadrantVolume, quadrant + 1, new TGeoCombiTrans(0.0, 0.0, 0.0, makeRotation("RICHInnerQuadrantRotation", quadrantStartDeg)));
      // Outer cylindrical wall sector.
      gasEnvelopeVolume->AddNode(outerWallQuadrantVolume, quadrant + 1, new TGeoCombiTrans(0.0, 0.0, 0.0, makeRotation("RICHOuterQuadrantRotation", quadrantStartDeg)));
      // Positive-z end cap sector.
      gasEnvelopeVolume->AddNode(endCapPlusQuadrantVolume, quadrant + 1, new TGeoCombiTrans(0.0, 0.0, endCapCenterZ, makeRotation("RICHEndCapPlusQuadrantRotation", quadrantStartDeg)));
      // Negative-z end cap sector.
      gasEnvelopeVolume->AddNode(endCapMinusQuadrantVolume, quadrant + 1, new TGeoCombiTrans(0.0, 0.0, -endCapCenterZ, makeRotation("RICHEndCapMinusQuadrantRotation", quadrantStartDeg)));
      // Start-side lateral wall.
      gasEnvelopeVolume->AddNode(lateralStartWallVolume, quadrant + 1, new TGeoCombiTrans(0.0, 0.0, 0.0, makeRotation("RICHLateralStartRotation", quadrantStartDeg)));
      // End-side lateral wall.
      gasEnvelopeVolume->AddNode(lateralEndWallVolume, quadrant + 1, new TGeoCombiTrans(0.0, 0.0, 0.0, makeRotation("RICHLateralEndRotation", quadrantEndDeg - lateralEndWallSpanDeg)));
    }

    LOGP(info, "RICH quadrant geometry: module pitch={} deg, module boundary half-gap={} deg, vessel half-gap={} deg", moduleDeltaPhi * radToDeg, moduleExtraPhi * radToDeg, vesselGapHalfDeg);
  }

  // ============================================================ modules
  for (int iRing{0}; iRing < richPars.nRings; ++iRing) {
    if (!richPars.oddGeom && iRing == (richPars.nRings / 2)) {
      continue;
    }
    mRings[iRing] = o2::rich::Ring{iRing,
                                   richPars.nTiles,
                                   richPars.rMin,
                                   richPars.rMax,
                                   richPars.radiatorThickness,
                                   (double)mVTile1[iRing],
                                   (double)mVTile2[iRing],
                                   (double)mLAerogelZ[iRing],
                                   richPars.detectorThickness,
                                   (double)mVMirror1[iRing],
                                   (double)mVMirror2[iRing],
                                   richPars.zBaseSize,
                                   (double)mR0Radiator[iRing],
                                   (double)mR0PhotoDet[iRing],
                                   (double)mTRplusG[iRing],
                                   (double)mThetaBi[iRing],
                                   richGasMotherName}; // GeometryTGeo::getRICHVolPattern()
  }

  if (richPars.enableFWDRich) {
    mFWDRich.createFWDRich(vRICH);
  }
  if (richPars.enableBWDRich) {
    mBWDRich.createBWDRich(vRICH);
  }
}

void Detector::InitializeO2Detector()
{
  LOG(info) << "Initialize RICH O2Detector";
  mGeometryTGeo = GeometryTGeo::Instance();
  defineSensitiveVolumes();
}

void Detector::defineSensitiveVolumes()
{
  TGeoManager* geoManager = gGeoManager;
  TGeoVolume* v;

  TString volumeName;
  LOGP(info, "Adding RICH Sensitive Volumes");

  // The names of the RICH sensitive volumes have the format: Ring(0...mRings.size()-1)
  for (auto ring : mRings) {
    for (int j = 0; j < ring.getNTiles(); j++) {
      volumeName = Form("%s_%d_%d", GeometryTGeo::getRICHSensorPattern(), ring.getPosId(), j);
      LOGP(info, "Trying {}", volumeName.Data());
      v = geoManager->GetVolume(volumeName.Data());
      if (!v) {
        LOG(error) << "Geometry does not contain volume " << volumeName.Data();
        geoManager->GetListOfVolumes()->Print();
        LOG(fatal) << "Could not find volume " << volumeName.Data() << " in the geometry";
      }
      LOGP(info, "Adding RICH Sensitive Volume {}", v->GetName());
      AddSensitiveVolume(v);
    }
  }
}

void Detector::EndOfEvent() { Reset(); }

void Detector::Register()
{
  // This will create a branch in the output tree called Hit, setting the last
  // parameter to kFALSE means that this collection will not be written to the file,
  // it will exist only during the simulation

  if (FairRootManager::Instance()) {
    FairRootManager::Instance()->RegisterAny(addNameTo("Hit").data(), mHits, true);
  }
}

void Detector::Reset()
{
  if (!o2::utils::ShmManager::Instance().isOperational()) {
    mHits->clear();
  }
}

bool Detector::ProcessHits(FairVolume* vol)
{
  // This method is called from the MC stepping

  constexpr int kOpticalPhotonPDG = 50000050;
  const bool isOpticalPhoton = (fMC->TrackPid() == kOpticalPhotonPDG);
  const bool isChargedParticle = (TMath::Abs(fMC->TrackCharge()) > 0.0);
  // Reject neutral particles other than optical photons.
  if (!isChargedParticle && !isOpticalPhoton) {
    return false;
  }

  int lay = vol->getVolumeId();
  int volID = vol->getMCid();

  // Is it needed to keep a track reference when the outer ITS volume is encountered?
  auto stack = (o2::data::Stack*)fMC->GetStack();

  // Only the active silicon volumes are registered as sensitive in
  // defineSensitiveVolumes(). The explicit volume-name check is kept
  // as a safety guard in case additional sensitive volumes are added.
  if (isOpticalPhoton) {
    const char* currentVolumeName = fMC->CurrentVolName();
    const bool isActiveSiliconVolume = currentVolumeName && TString(currentVolumeName).BeginsWith(GeometryTGeo::getRICHSensorPattern());
    // Create only one hit when entering the active silicon.
    if (!isActiveSiliconVolume || !fMC->IsTrackEntering()) {
      return false;
    }
    TLorentzVector photonPosition;
    TLorentzVector photonMomentum;
    fMC->TrackPosition(photonPosition);
    fMC->TrackMomentum(photonMomentum);
    constexpr unsigned char photonStatus = Hit::kTrackEntering;
    addHit(
      stack->GetCurrentTrackNumber(),
      lay,
      photonPosition.Vect(),
      photonPosition.Vect(),
      photonMomentum.Vect(),
      photonMomentum.E(),
      photonPosition.T(),
      0.0,
      photonStatus,
      photonStatus);

    stack->addHit(GetDetId());

    return true;
  }

  if (fMC->IsTrackExiting() && (lay == 0 || lay == mRings.size() - 1)) {
    // Keep the track refs for the innermost and outermost rings only
    o2::TrackReference tr(*fMC, GetDetId());
    tr.setTrackID(stack->GetCurrentTrackNumber());
    tr.setUserId(lay);
    stack->addTrackReference(tr);
  }
  bool startHit = false, stopHit = false;
  unsigned char status = 0;
  if (fMC->IsTrackEntering()) {
    status |= Hit::kTrackEntering;
  }
  if (fMC->IsTrackInside()) {
    status |= Hit::kTrackInside;
  }
  if (fMC->IsTrackExiting()) {
    status |= Hit::kTrackExiting;
  }
  if (fMC->IsTrackOut()) {
    status |= Hit::kTrackOut;
  }
  if (fMC->IsTrackStop()) {
    status |= Hit::kTrackStopped;
  }
  if (fMC->IsTrackAlive()) {
    status |= Hit::kTrackAlive;
  }

  // track is entering or created in the volume
  if ((status & Hit::kTrackEntering) || (status & Hit::kTrackInside && !mTrackData.mHitStarted)) {
    startHit = true;
  } else if ((status & (Hit::kTrackExiting | Hit::kTrackOut | Hit::kTrackStopped))) {
    stopHit = true;
  }

  // increment energy loss at all steps except entrance
  if (!startHit) {
    mTrackData.mEnergyLoss += fMC->Edep();
  }
  if (!(startHit | stopHit)) {
    return false; // do noting
  }

  if (startHit) {
    mTrackData.mEnergyLoss = 0.;
    fMC->TrackMomentum(mTrackData.mMomentumStart);
    fMC->TrackPosition(mTrackData.mPositionStart);
    mTrackData.mTrkStatusStart = status;
    mTrackData.mHitStarted = true;
  }
  if (stopHit) {
    TLorentzVector positionStop;
    fMC->TrackPosition(positionStop);
    // Retrieve the indices with the volume path
    int stave(0), halfstave(0), chipinmodule(0), module;
    fMC->CurrentVolOffID(1, chipinmodule);
    fMC->CurrentVolOffID(2, module);
    fMC->CurrentVolOffID(3, halfstave);
    fMC->CurrentVolOffID(4, stave);

    Hit* p = addHit(stack->GetCurrentTrackNumber(), lay, mTrackData.mPositionStart.Vect(), positionStop.Vect(),
                    mTrackData.mMomentumStart.Vect(), mTrackData.mMomentumStart.E(), positionStop.T(),
                    mTrackData.mEnergyLoss, mTrackData.mTrkStatusStart, status);
    // p->SetTotalEnergy(vmc->Etot());

    // RS: not sure this is needed
    // Increment number of Detector det points in TParticle
    stack->addHit(GetDetId());
  }

  return true;
}

o2::itsmft::Hit* Detector::addHit(int trackID, int detID, const TVector3& startPos, const TVector3& endPos,
                                  const TVector3& startMom, double startE, double endTime, double eLoss, unsigned char startStatus,
                                  unsigned char endStatus)
{
  mHits->emplace_back(trackID, detID, startPos, endPos, startMom, startE, endTime, eLoss, startStatus, endStatus);
  return &(mHits->back());
}

void Detector::prepareLayout()
{ // Mere translation of Nicola's code
  auto& richPars = RICHBaseParam::Instance();
  LOGP(info, "Setting up {} layout for bRICH", richPars.oddGeom ? "odd" : "even");

  bool isOdd = richPars.oddGeom;
  mThetaBi.resize(richPars.nRings);
  mR0Tilt.resize(richPars.nRings);
  mZ0Tilt.resize(richPars.nRings);
  mLAerogelZ.resize(richPars.nRings);
  mTRplusG.resize(richPars.nRings);
  mMinRadialMirror.resize(richPars.nRings);
  mMaxRadialMirror.resize(richPars.nRings);
  mMaxRadialRadiator.resize(richPars.nRings);
  mVMirror1.resize(richPars.nRings);
  mVMirror2.resize(richPars.nRings);
  mVTile1.resize(richPars.nRings);
  mVTile2.resize(richPars.nRings);
  mR0Radiator.resize(richPars.nRings);
  mR0PhotoDet.resize(richPars.nRings);

  // Start from middle one
  double mVal = TMath::Tan(0.0);
  mThetaBi[richPars.nRings / 2] = TMath::ATan(mVal);
  mR0Tilt[richPars.nRings / 2] = richPars.rMax;
  mZ0Tilt[richPars.nRings / 2] = mR0Tilt[richPars.nRings / 2] * TMath::Tan(mThetaBi[richPars.nRings / 2]);
  mLAerogelZ[richPars.nRings / 2] = isOdd ? TMath::Sqrt(1.0 + mVal * mVal) * richPars.rMin * richPars.zBaseSize / (TMath::Sqrt(1.0 + mVal * mVal) * richPars.rMax - mVal * richPars.zBaseSize) : 0.f;
  mTRplusG[richPars.nRings / 2] = richPars.rMax - richPars.rMin;
  double t = isOdd ? TMath::Tan(TMath::ATan(mVal) + TMath::ATan(richPars.zBaseSize / (2.0 * richPars.rMax * TMath::Sqrt(1.0 + mVal * mVal) - richPars.zBaseSize * mVal))) : 0.f;
  mMinRadialMirror[richPars.nRings / 2] = richPars.rMax;
  mMaxRadialRadiator[richPars.nRings / 2] = richPars.rMin;

  // Configure rest of the rings
  for (int iRing{richPars.nRings / 2 + 1}; iRing < richPars.nRings; ++iRing) {
    double parA = t;
    double parB = 2.0 * richPars.rMax / richPars.zBaseSize;
    mVal = (TMath::Sqrt(parA * parA * parB * parB + parB * parB - 1.0) + parA * parB * parB) / (parB * parB - 1.0);
    t = TMath::Tan(TMath::ATan(mVal) + TMath::ATan(richPars.zBaseSize / (2.0 * richPars.rMax * TMath::Sqrt(1.0 + mVal * mVal) - richPars.zBaseSize * mVal)));
    // forward rings
    mThetaBi[iRing] = TMath::ATan(mVal);
    mR0Tilt[iRing] = richPars.rMax - richPars.zBaseSize / 2.0 * sin(TMath::ATan(mVal));
    mZ0Tilt[iRing] = mR0Tilt[iRing] * TMath::Tan(mThetaBi[iRing]);
    mLAerogelZ[iRing] = TMath::Sqrt(1.0 + mVal * mVal) * richPars.rMin * richPars.zBaseSize / (TMath::Sqrt(1.0 + mVal * mVal) * richPars.rMax - mVal * richPars.zBaseSize);
    mTRplusG[iRing] = TMath::Sqrt(1.0 + mVal * mVal) * (richPars.rMax - richPars.rMin) - mVal / 2.0 * (richPars.zBaseSize + mLAerogelZ[iRing]);
    mMinRadialMirror[iRing] = mR0Tilt[iRing] - richPars.zBaseSize / 2.0 * sin(TMath::ATan(mVal));
    mMaxRadialRadiator[iRing] = richPars.rMin + 2.0 * mLAerogelZ[iRing] / 2.0 * sin(TMath::ATan(mVal));
    // backward rings
    mThetaBi[2 * (richPars.nRings / 2) - iRing] = -TMath::ATan(mVal);
    mR0Tilt[2 * (richPars.nRings / 2) - iRing] = richPars.rMax - richPars.zBaseSize / 2.0 * sin(TMath::ATan(mVal));
    mZ0Tilt[2 * (richPars.nRings / 2) - iRing] = -mR0Tilt[iRing] * TMath::Tan(mThetaBi[iRing]);
    mLAerogelZ[2 * (richPars.nRings / 2) - iRing] = TMath::Sqrt(1.0 + mVal * mVal) * richPars.rMin * richPars.zBaseSize / (TMath::Sqrt(1.0 + mVal * mVal) * richPars.rMax - mVal * richPars.zBaseSize);
    mTRplusG[2 * (richPars.nRings / 2) - iRing] = TMath::Sqrt(1.0 + mVal * mVal) * (richPars.rMax - richPars.rMin) - mVal / 2.0 * (richPars.zBaseSize + mLAerogelZ[iRing]);
    mMinRadialMirror[2 * (richPars.nRings / 2) - iRing] = mR0Tilt[iRing] - richPars.zBaseSize / 2.0 * sin(TMath::ATan(mVal));
    mMaxRadialRadiator[2 * (richPars.nRings / 2) - iRing] = richPars.rMin + 2.0 * mLAerogelZ[iRing] / 2.0 * sin(TMath::ATan(mVal));
  }

  // Dimensioning tiles
  if (!richPars.flagUseQuadrants) {
    double percentage = 0.999;
    for (int iRing = 0; iRing < richPars.nRings; iRing++) {
      if (iRing == richPars.nRings / 2) {
        mVMirror1[iRing] = percentage * 2.0 * richPars.rMax * TMath::Tan(TMath::Pi() / double(richPars.nTiles));
        mVMirror2[iRing] = percentage * 2.0 * richPars.rMax * TMath::Tan(TMath::Pi() / double(richPars.nTiles));
        mVTile1[iRing] = percentage * 2.0 * richPars.rMin * TMath::Tan(TMath::Pi() / double(richPars.nTiles));
        mVTile2[iRing] = percentage * 2.0 * richPars.rMin * TMath::Tan(TMath::Pi() / double(richPars.nTiles));
      } else if (iRing > richPars.nRings / 2) {
        mVMirror1[iRing] = percentage * 2.0 * richPars.rMax * TMath::Tan(TMath::Pi() / double(richPars.nTiles));
        mVMirror2[iRing] = percentage * 2.0 * mMinRadialMirror[iRing] * TMath::Tan(TMath::Pi() / double(richPars.nTiles));
        mVTile1[iRing] = percentage * 2.0 * mMaxRadialRadiator[iRing] * TMath::Tan(TMath::Pi() / double(richPars.nTiles));
        mVTile2[iRing] = percentage * 2.0 * richPars.rMin * TMath::Tan(TMath::Pi() / double(richPars.nTiles));
      } else {
        mVMirror2[iRing] = percentage * 2.0 * richPars.rMax * TMath::Tan(TMath::Pi() / double(richPars.nTiles));
        mVMirror1[iRing] = percentage * 2.0 * mMinRadialMirror[iRing] * TMath::Tan(TMath::Pi() / double(richPars.nTiles));
        mVTile2[iRing] = percentage * 2.0 * mMaxRadialRadiator[iRing] * TMath::Tan(TMath::Pi() / double(richPars.nTiles));
        mVTile1[iRing] = percentage * 2.0 * richPars.rMin * TMath::Tan(TMath::Pi() / double(richPars.nTiles));
      }
    }

  } else {

    const double totalBoundaryWidth = 2.0 * richPars.vesselThicknessShieldingLateral + richPars.vesselPhiGap;
    const double quadrantDeltaPhi = solveQuadrantDeltaPhi(richPars.nTiles, richPars.rMin, totalBoundaryWidth);
    if (!(quadrantDeltaPhi > 0.0)) {
      LOGP(fatal, "RICH could not solve the quadrant module angular pitch");
    }
    const double halfWidthFactor = TMath::Tan(quadrantDeltaPhi / 2.0);
    double percentage = 0.999;
    for (int iRing = 0; iRing < richPars.nRings; iRing++) {
      if (iRing == richPars.nRings / 2) {
        mVMirror1[iRing] = percentage * 2.0 * richPars.rMax * halfWidthFactor;
        mVMirror2[iRing] = percentage * 2.0 * richPars.rMax * halfWidthFactor;
        mVTile1[iRing] = percentage * 2.0 * richPars.rMin * halfWidthFactor;
        mVTile2[iRing] = percentage * 2.0 * richPars.rMin * halfWidthFactor;
      } else if (iRing > richPars.nRings / 2) {
        mVMirror1[iRing] = percentage * 2.0 * richPars.rMax * halfWidthFactor;
        mVMirror2[iRing] = percentage * 2.0 * mMinRadialMirror[iRing] * halfWidthFactor;
        mVTile1[iRing] = percentage * 2.0 * mMaxRadialRadiator[iRing] * halfWidthFactor;
        mVTile2[iRing] = percentage * 2.0 * richPars.rMin * halfWidthFactor;

      } else {
        mVMirror2[iRing] = percentage * 2.0 * richPars.rMax * halfWidthFactor;
        mVMirror1[iRing] = percentage * 2.0 * mMinRadialMirror[iRing] * halfWidthFactor;
        mVTile2[iRing] = percentage * 2.0 * mMaxRadialRadiator[iRing] * halfWidthFactor;
        mVTile1[iRing] = percentage * 2.0 * richPars.rMin * halfWidthFactor;
      }
    }
  }

  // ============================================================
  // Cylindrical aerogel geometry
  // ============================================================
  //
  // In this mode the photosensors remain projective, but all
  // aerogel tiles:
  //
  //   - have identical dimensions;
  //   - are parallel to the beam axis;
  //   - lie at the same cylindrical radius;
  //   - are uniformly distributed along Z.
  //
  if (richPars.useCylindricalAerogel) {

    // In the even geometry the central projective ring is skipped
    // in createGeometry(), so the number of actual aerogel rows is
    // nRings - 1.
    const int nAerogelRows = richPars.oddGeom ? richPars.nRings : richPars.nRings - 1;

    if (nAerogelRows <= 0) {
      LOGP(fatal, "Invalid number of cylindrical aerogel rows: {}", nAerogelRows);
    }

    if (richPars.nTiles <= 0) {
      LOGP(fatal, "Invalid number of aerogel tiles in phi: {}", richPars.nTiles);
    }

    const double thetaRef = 2.0 * TMath::ATan(TMath::Exp(-richPars.cylindricalAerogelEtaRef));
    const double cylindricalAerogelTileSizeZ = (2.0 * richPars.rMin / TMath::Tan(thetaRef)) / static_cast<double>(nAerogelRows);
    // const double cylindricalAerogelTileSizeRPhi = 2.0 * richPars.rMin * TMath::Tan(TMath::Pi() / static_cast<double>(richPars.nTiles));
    double cylindricalAerogelTileSizeRPhi = 0.0;

    if (!richPars.flagUseQuadrants) {
      // Original uniform-phi geometry.
      cylindricalAerogelTileSizeRPhi = 2.0 * richPars.rMin * TMath::Tan(TMath::Pi() / static_cast<double>(richPars.nTiles));
    } else {
      const double totalBoundaryWidth = 2.0 * richPars.vesselThicknessShieldingLateral + richPars.vesselPhiGap;
      const double quadrantDeltaPhi = solveQuadrantDeltaPhi(richPars.nTiles, richPars.rMin, totalBoundaryWidth);
      cylindricalAerogelTileSizeRPhi = 2.0 * richPars.rMin * TMath::Tan(quadrantDeltaPhi / 2.0);
    }

    LOGP(info, "Cylindrical aerogel: rows={}, etaRef={}, tileSizeZ={} cm, tileSizeRPhi={} cm", nAerogelRows, richPars.cylindricalAerogelEtaRef, cylindricalAerogelTileSizeZ, cylindricalAerogelTileSizeRPhi);

    for (int iRing = 0; iRing < richPars.nRings; iRing++) {
      mLAerogelZ[iRing] = cylindricalAerogelTileSizeZ;

      // Equal values make the TGeoArb8 a rectangle instead of the projective trapezoid.
      mVTile1[iRing] = cylindricalAerogelTileSizeRPhi;
      mVTile2[iRing] = cylindricalAerogelTileSizeRPhi;
    }
  }

  // Translation parameters
  for (size_t iRing{0}; iRing < richPars.nRings; ++iRing) {

    if (richPars.useCylindricalAerogel) {
      mR0Radiator[iRing] = richPars.rMin + richPars.radiatorThickness / 2.0;
    } else {
      // Original projective aerogel position.
      mR0Radiator[iRing] = mR0Tilt[iRing] - (mTRplusG[iRing] - richPars.radiatorThickness / 2.0) * TMath::Cos(mThetaBi[iRing]);
    }

    // Photosensors remain projective for both configurations.
    mR0PhotoDet[iRing] = mR0Tilt[iRing] - richPars.detectorThickness / 2.0 * TMath::Cos(mThetaBi[iRing]);
  }

  // FWD and BWD RICH
  if (richPars.enableFWDRich) {
    LOGP(info, "Setting up FWD RICH layout");
    mFWDRich = FWDRich(GeometryTGeo::getRICHSensorFWDPattern(),
                       richPars.rFWDMin,
                       richPars.rFWDMax,
                       richPars.zAerogelMin,
                       richPars.zAerogelMax - richPars.zAerogelMin,
                       richPars.zArgonMin,
                       richPars.zArgonMax - richPars.zArgonMin,
                       richPars.zSiliconMin,
                       richPars.zSiliconMax - richPars.zSiliconMin);
  }
  if (richPars.enableBWDRich) {
    LOGP(info, "Setting up BWD RICH layout");
    mBWDRich = BWDRich(GeometryTGeo::getRICHSensorBWDPattern(),
                       richPars.rFWDMin,
                       richPars.rFWDMax,
                       richPars.zAerogelMin,
                       richPars.zAerogelMax - richPars.zAerogelMin,
                       richPars.zArgonMin,
                       richPars.zArgonMax - richPars.zArgonMin,
                       richPars.zSiliconMin,
                       richPars.zSiliconMax - richPars.zSiliconMin);
  }
}
} // namespace rich
} // namespace o2

ClassImp(o2::rich::Detector);