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

#define BOOST_TEST_MODULE Test ITSMFTTracking ITSMFTSurfaceSpecProjection
#include <boost/test/unit_test.hpp>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ios>
#include <string>
#include <vector>

#include "ITSMFTTracking/ITSMFTDetectorDefinitions.h"
#include "ITSMFTTracking/Constants.h"

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
  const char* referenceCoordinateToken; // verbatim from the C1 lossless JSON provenance
  uint8_t detectorId;
  SurfaceKind kind;
};

// Tokens copied verbatim from
// O2-validation-artifacts/itsmft/gate4-b1-slice1-nominal-geometry-validation/
// pp-20ev-run303000-seed20260716-daily20260717/acceptance-cleanup-c1-lossless-json/
// {its,mft}-report.json (geometry SHA-256
// 2a428746b3a0b57179d5ffe631afc9c4afb4ca41cc9baa948ff670099b9204e4; full
// provenance in doc/decisions/0004-its-mft-static-surface-spec-tables.md).
const std::vector<ExpectedSurface> kExpectedITS{
  {0, "2.3259652", 0, SurfaceKind::Cylinder},
  {1, "3.1353536", 0, SurfaceKind::Cylinder},
  {2, "3.9162421", 0, SurfaceKind::Cylinder},
  {3, "19.58824", 0, SurfaceKind::Cylinder},
  {4, "24.527159", 0, SurfaceKind::Cylinder},
  {5, "34.354595", 0, SurfaceKind::Cylinder},
  {6, "39.310642", 0, SurfaceKind::Cylinder},
};

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

template <typename Spec>
void checkAuthoredLiteralsMatchProvenanceTokens(const std::vector<ExpectedSurface>& expected)
{
  BOOST_REQUIRE_EQUAL(Spec::surfaces.size(), expected.size());
  for (const auto& row : expected) {
    const auto& authored = Spec::surfaces[row.index];
    const float fromToken = parseToken(row.referenceCoordinateToken);
    BOOST_CHECK_MESSAGE(bitsOf(authored.nominalReferenceCoordinate) == bitsOf(fromToken),
                        "surface " << row.index << ": authored literal (bits 0x" << std::hex
                                   << bitsOf(authored.nominalReferenceCoordinate) << ") does not bit-match provenance token \""
                                   << row.referenceCoordinateToken << "\" (bits 0x" << bitsOf(fromToken) << ")" << std::dec);
  }
}

template <typename Spec>
void checkIdentityAndKind(const std::vector<ExpectedSurface>& expected)
{
  for (const auto& row : expected) {
    const auto& authored = Spec::surfaces[row.index];
    BOOST_CHECK_EQUAL(authored.identity.detectorId, row.detectorId);
    BOOST_CHECK_EQUAL(authored.identity.detectorSurfaceIndex, row.index);
    BOOST_CHECK(authored.kind == row.kind);
  }
}

template <typename Spec>
void checkProjectionPreservesEveryFieldBitExactly(const std::vector<ExpectedSurface>& expected)
{
  for (const auto& row : expected) {
    const auto& authored = Spec::surfaces[row.index];
    const auto projected = toRuntimeSurfaceDescriptor(authored);
    BOOST_CHECK_EQUAL(projected.detectorSurfaceIndex, authored.identity.detectorSurfaceIndex);
    BOOST_CHECK_EQUAL(projected.detectorId, authored.identity.detectorId);
    BOOST_CHECK(projected.kind == authored.kind);
    BOOST_CHECK_EQUAL(projected.flags, 0);
    BOOST_CHECK_EQUAL(bitsOf(projected.referenceCoordinate), bitsOf(authored.nominalReferenceCoordinate));
    BOOST_CHECK_EQUAL(bitsOf(projected.material.xOverX0), bitsOf(authored.material.xOverX0));
    BOOST_CHECK_EQUAL(bitsOf(projected.material.arealDensityGPerCm2), bitsOf(authored.material.arealDensityGPerCm2));
  }
}
} // namespace

static_assert(SurfaceSpec<ITSSurfaceSpec>);
static_assert(SurfaceSpec<MFTSurfaceSpec>);
static_assert(SurfaceCount<ITSSurfaceSpec> == ITSNLayers);
static_assert(SurfaceCount<MFTSurfaceSpec> == MFTNLayers);
static_assert(SurfaceSpecsCanBeConcatenated<ITSSurfaceSpec, MFTSurfaceSpec>);

BOOST_AUTO_TEST_CASE(ITSAuthoredLiteralsMatchProvenanceTokensBitExactly)
{
  checkAuthoredLiteralsMatchProvenanceTokens<ITSSurfaceSpec>(kExpectedITS);
}

BOOST_AUTO_TEST_CASE(MFTAuthoredLiteralsMatchProvenanceTokensBitExactly)
{
  checkAuthoredLiteralsMatchProvenanceTokens<MFTSurfaceSpec>(kExpectedMFT);
}

BOOST_AUTO_TEST_CASE(ITSIdentityKindAndIndexingFamily)
{
  checkIdentityAndKind<ITSSurfaceSpec>(kExpectedITS);
}

BOOST_AUTO_TEST_CASE(MFTIdentityKindAndIndexingFamily)
{
  checkIdentityAndKind<MFTSurfaceSpec>(kExpectedMFT);
}

BOOST_AUTO_TEST_CASE(ITSMaterialMatchesNominalDefaultsAndRadlRhoFormula)
{
  for (int layer = 0; layer < ITSNLayers; ++layer) {
    const auto& surface = ITSSurfaceSpec::surfaces[layer];
    BOOST_CHECK_EQUAL(surface.material.xOverX0, kNominalITSLayerX0[layer]);
    BOOST_CHECK_CLOSE(surface.material.arealDensityGPerCm2,
                      kNominalITSLayerX0[layer] * o2::its::constants::Radl * o2::its::constants::Rho, 1.e-6f);
  }
}

BOOST_AUTO_TEST_CASE(MFTMaterialMatchesNominalDefaultsAndRadlRhoFormula)
{
  for (int layer = 0; layer < MFTNLayers; ++layer) {
    const auto& surface = MFTSurfaceSpec::surfaces[layer];
    BOOST_CHECK_EQUAL(surface.material.xOverX0, kNominalMFTLayerX0[layer]);
    BOOST_CHECK_CLOSE(surface.material.arealDensityGPerCm2,
                      kNominalMFTLayerX0[layer] * o2::its::constants::Radl * o2::its::constants::Rho, 1.e-6f);
  }
}

BOOST_AUTO_TEST_CASE(MFTSensorPairsShareThePhysicalDiskBudget)
{
  float totalX0 = 0.f;
  float totalArealDensity = 0.f;
  for (int disk = 0; disk < MFTDisks; ++disk) {
    const auto& front = kMFTStaticSurfaceCatalog[2 * disk].material;
    const auto& back = kMFTStaticSurfaceCatalog[2 * disk + 1].material;
    BOOST_CHECK_CLOSE(front.xOverX0 + back.xOverX0, kMFTNominalRadLength / MFTDisks, 1.e-4f);
    totalX0 += front.xOverX0 + back.xOverX0;
    totalArealDensity += front.arealDensityGPerCm2 + back.arealDensityGPerCm2;
  }
  BOOST_CHECK_CLOSE(totalX0, kMFTNominalRadLength, 1.e-4f);
  BOOST_CHECK_CLOSE(totalArealDensity,
                    kMFTNominalRadLength * o2::its::constants::Radl * o2::its::constants::Rho, 1.e-4f);
}

BOOST_AUTO_TEST_CASE(ITSProjectionPreservesEveryFieldBitExactly)
{
  checkProjectionPreservesEveryFieldBitExactly<ITSSurfaceSpec>(kExpectedITS);
}

BOOST_AUTO_TEST_CASE(MFTProjectionPreservesEveryFieldBitExactly)
{
  checkProjectionPreservesEveryFieldBitExactly<MFTSurfaceSpec>(kExpectedMFT);
}
