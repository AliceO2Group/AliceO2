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

#define BOOST_TEST_MODULE Test ITSMFTTracking ITSMFTDetectorDefinitions
#include <boost/test/unit_test.hpp>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ios>
#include <string>
#include <vector>

#include "ITSMFTTracking/ITSMFTDetectorDefinitions.h"
#include "ITSMFTTracking/Constants.h"
#include "ITStracking/Configuration.h"
#include "MFTTracking/Constants.h"

using namespace o2::itsmft::tracking;

namespace
{
uint32_t bitsOf(float value)
{
  uint32_t bits{};
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float parseToken(const char* token)
{
  char* endptr = nullptr;
  const float value = std::strtof(token, &endptr);
  BOOST_REQUIRE_MESSAGE(endptr != nullptr && *endptr == '\0', "strtof left unparsed characters in \"" << token << "\"");
  return value;
}

struct ExpectedSurface {
  uint16_t index;
  const char* referenceCoordinateToken; // exact production tracking default
  uint8_t detectorId;
  SurfaceKind kind;
};

// Exact ITS radii used by production tracking before descriptor consolidation.
const std::vector<ExpectedSurface> kExpectedITS{
  {0, "2.33959", 0, SurfaceKind::Cylinder},
  {1, "3.14076", 0, SurfaceKind::Cylinder},
  {2, "3.91924", 0, SurfaceKind::Cylinder},
  {3, "19.6213", 0, SurfaceKind::Cylinder},
  {4, "24.5597", 0, SurfaceKind::Cylinder},
  {5, "34.388", 0, SurfaceKind::Cylinder},
  {6, "39.3329", 0, SurfaceKind::Cylinder},
};

// Tokens copied verbatim from
// O2-validation-artifacts/itsmft/gate4-b1-slice1-nominal-geometry-validation/
// pp-20ev-run303000-seed20260716-daily20260717/acceptance-cleanup-c1-lossless-json/
// mft-report.json (geometry SHA-256
// 2a428746b3a0b57179d5ffe631afc9c4afb4ca41cc9baa948ff670099b9204e4; full
// provenance in doc/decisions/0004-its-mft-static-surface-spec-tables.md).
const std::vector<ExpectedSurface> kExpectedMFT{
  {0, "-45.2889", 8, SurfaceKind::Disk},
  {1, "-46.7111", 8, SurfaceKind::Disk},
  {2, "-48.5889", 8, SurfaceKind::Disk},
  {3, "-50.0111", 8, SurfaceKind::Disk},
  {4, "-52.3889", 8, SurfaceKind::Disk},
  {5, "-53.8111", 8, SurfaceKind::Disk},
  {6, "-67.6889", 8, SurfaceKind::Disk},
  {7, "-69.1111", 8, SurfaceKind::Disk},
  {8, "-76.0889", 8, SurfaceKind::Disk},
  {9, "-77.5111", 8, SurfaceKind::Disk},
};

template <std::size_t N>
void checkAuthoredLiteralsMatchExpectedTokens(const std::array<SurfaceDescriptor, N>& surfaces, const std::vector<ExpectedSurface>& expected)
{
  BOOST_REQUIRE_EQUAL(surfaces.size(), expected.size());
  for (const auto& row : expected) {
    const auto& authored = surfaces[row.index];
    const float fromToken = parseToken(row.referenceCoordinateToken);
    BOOST_CHECK_MESSAGE(bitsOf(authored.referenceCoordinate) == bitsOf(fromToken),
                        "surface " << row.index << ": authored literal (bits 0x" << std::hex
                                   << bitsOf(authored.referenceCoordinate) << ") does not bit-match expected token \""
                                   << row.referenceCoordinateToken << "\" (bits 0x" << bitsOf(fromToken) << ")" << std::dec);
  }
}

template <std::size_t N>
void checkIdentityAndKind(const std::array<SurfaceDescriptor, N>& surfaces, const std::vector<ExpectedSurface>& expected)
{
  for (const auto& row : expected) {
    const auto& authored = surfaces[row.index];
    BOOST_CHECK_EQUAL(authored.detectorId, row.detectorId);
    BOOST_CHECK_EQUAL(authored.detectorSurfaceIndex, row.index);
    BOOST_CHECK(authored.kind == row.kind);
    BOOST_CHECK_EQUAL(authored.flags, 0);
  }
}

} // namespace

BOOST_AUTO_TEST_CASE(ITSAuthoredLiteralsMatchProductionRadiiBitExactly)
{
  checkAuthoredLiteralsMatchExpectedTokens(kITSSurfaces, kExpectedITS);
}

BOOST_AUTO_TEST_CASE(MFTAuthoredLiteralsMatchProvenanceTokensBitExactly)
{
  checkAuthoredLiteralsMatchExpectedTokens(kMFTSurfaces, kExpectedMFT);
}

BOOST_AUTO_TEST_CASE(ITSIdentityKindAndIndexingFamily)
{
  checkIdentityAndKind(kITSSurfaces, kExpectedITS);
}

BOOST_AUTO_TEST_CASE(MFTIdentityKindAndIndexingFamily)
{
  checkIdentityAndKind(kMFTSurfaces, kExpectedMFT);
}

BOOST_AUTO_TEST_CASE(ITSMaterialMatchesNominalDefaultsAndRadlRhoFormula)
{
  constexpr std::array<float, ITSNLayers> expectedX0{5.e-3f, 5.e-3f, 5.e-3f, 1.e-2f, 1.e-2f, 1.e-2f, 1.e-2f};
  for (int layer = 0; layer < ITSNLayers; ++layer) {
    const auto& surface = kITSSurfaces[layer];
    BOOST_CHECK_EQUAL(surface.material.xOverX0, expectedX0[layer]);
    BOOST_CHECK_EQUAL(bitsOf(surface.material.arealDensityGPerCm2),
                      bitsOf(expectedX0[layer] * o2::its::constants::Radl * o2::its::constants::Rho));
  }
}

BOOST_AUTO_TEST_CASE(MFTMaterialMatchesNominalDefaultsAndRadlRhoFormula)
{
  for (int layer = 0; layer < MFTNLayers; ++layer) {
    const auto& surface = kMFTSurfaces[layer];
    BOOST_CHECK_EQUAL(surface.material.xOverX0, (0.042f / 5.f));
    BOOST_CHECK_EQUAL(bitsOf(surface.material.arealDensityGPerCm2),
                      bitsOf((0.042f / 5.f) * o2::its::constants::Radl * o2::its::constants::Rho));
  }
}

BOOST_AUTO_TEST_CASE(MFTSurfacesUseTheNominalCAPrescription)
{
  constexpr float expectedSurfaceX0 = 0.0084f;
  float totalX0 = 0.f;
  float totalArealDensity = 0.f;
  for (const auto& surface : kMFTSurfaces) {
    BOOST_CHECK_CLOSE(surface.material.xOverX0, expectedSurfaceX0, 1.e-4f);
    BOOST_CHECK_CLOSE(surface.material.arealDensityGPerCm2,
                      expectedSurfaceX0 * o2::its::constants::Radl * o2::its::constants::Rho, 1.e-4f);
    totalX0 += surface.material.xOverX0;
    totalArealDensity += surface.material.arealDensityGPerCm2;
  }
  BOOST_CHECK_CLOSE(totalX0, 0.084f, 1.e-4f);
  BOOST_CHECK_CLOSE(totalArealDensity,
                    0.084f * o2::its::constants::Radl * o2::its::constants::Rho, 1.e-4f);
}

BOOST_AUTO_TEST_CASE(ChartRangesMatchProductionDefaultsBitExactly)
{
  const o2::its::TrackingParameters productionITS;
  for (int layer = 0; layer < ITSNLayers; ++layer) {
    const auto& range = kITSSurfaces[layer].chartRange;
    BOOST_CHECK_EQUAL(bitsOf(range.min), bitsOf(-productionITS.LayerZ[layer]));
    BOOST_CHECK_EQUAL(bitsOf(range.max), bitsOf(productionITS.LayerZ[layer]));
  }
  for (int layer = 0; layer < MFTNLayers; ++layer) {
    const auto& range = kMFTSurfaces[layer].chartRange;
    BOOST_CHECK_EQUAL(bitsOf(range.min), bitsOf(o2::mft::constants::index_table::RMin[layer]));
    BOOST_CHECK_EQUAL(bitsOf(range.max), bitsOf(o2::mft::constants::index_table::RMax[layer]));
  }
}
