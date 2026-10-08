// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file Gen3GeometryConfig.h
/// \brief Geometry-specific data for the ALICE 3 Gen3 ACTS blueprint builder
/// \author Paolo Butti
///
/// Ported from actsO2 (ActsAlgorithms/Geometry/include/Gen3GeometryConfig.hpp).
///
/// Runtime geometry-specific parameters for Gen3BlueprintBuilder, loaded from a
/// JSON file instead of being hard-coded. The construction logic stays in the
/// builder; this struct only carries DATA measured from the GDML.
///
/// All lengths in MILLIMETRES - the ACTS base unit, so the builder uses these
/// values directly (1_mm == 1.0). To target a different geometry, supply a
/// different JSON file; see loadGen3GeometryConfig() and the example
/// gen3_geometry_config.json shipped next to each geometry.
///

#ifndef ALICEO2_ALICE3_GEN3GEOMETRYCONFIG_H
#define ALICEO2_ALICE3_GEN3GEOMETRYCONFIG_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace o2::alice3::gen3cfg
{

// Barrel-like passive cylinder, centred on z = zCentre (usually 0).
struct PassiveCylinderCfg {
  std::string name;
  double r = 0.;
  double halfZ = 0.;
  double zCentre = 0.;
};

// Forward service shell: a cylinder present only at large |z|. zMin/zMax are
// |z| ranges; the builder mirrors it to both sides.
struct ForwardCylinderCfg {
  std::string name;
  double r = 0.;
  double zMin = 0.;
  double zMax = 0.;
};

// Passive service disc at a fixed z, spanning [rMin, rMax].
struct PassiveDiscCfg {
  std::string name;
  double z = 0.;
  double rMin = 0.;
  double rMax = 0.;
};

/// Complete geometry description. One instance per geometry, loaded from JSON.
struct Gen3GeometryConfig {
  // Sensor / end-of-stave TGeo volume-name globs.
  std::vector<std::string> sensitiveMatches;
  std::vector<std::string> endOfStaveMatches;
  double endOfStaveRTol = 0.; // mm, radial match window

  // Planar-sensor axis strings (parsed into TGeoAxes at runtime).
  std::string axesThinZ; // e.g. "XYZ" (thin in Z)
  std::string axesThinY; // e.g. "ZXY" (thin in Y)

  // Blueprint region boundaries (mm).
  double rInnerCoreMax = 0.;
  double rMainMax = 0.;
  double zCentralMax = 0.;
  double zMainMax = 0.;

  // Per-subsystem clustering tolerances (mm).
  double tolVertexDetector = 0.;
  double tolTrkBarrel = 0.;
  double tolItof = 0.;
  double tolOtof = 0.;
  double tolFt3Disc = 0.;

  // Forward-disc rMax clip (mm). Informational: the passive-disc rMax values
  // below already carry the clipped number.
  double fwdDiscRMax = 0.;

  // Proto-material (material receiver) binning on the layer faces.
  std::size_t matBinsPhi = 0;
  std::size_t matBinsZ = 0;
  std::size_t matBinsR = 0;

  // Passive material structures (flexible, geometry-dependent sets).
  std::vector<PassiveCylinderCfg> passiveCylinders;
  std::vector<ForwardCylinderCfg> forwardCylinders;
  std::vector<PassiveDiscCfg> passiveDiscs;

  // Pinned volume IDs, one per region.
  std::uint64_t volFwdNeg = 0;
  std::uint64_t volFt3InnerNeg = 0;
  std::uint64_t volInnerBarrel = 0;
  std::uint64_t volOuterTrackerBarrel = 0;
  std::uint64_t volFt3InnerPos = 0;
  std::uint64_t volFwdPos = 0;
  std::uint64_t volOtof = 0;
};

/// Load the Gen3 geometry configuration from a JSON file.
/// Throws std::runtime_error if the file cannot be opened/parsed or if any
/// required field is missing (the JSON is the single source of truth).
Gen3GeometryConfig loadGen3GeometryConfig(const std::string& jsonPath);

} // namespace o2::alice3::gen3cfg

#endif // ALICEO2_ALICE3_GEN3GEOMETRYCONFIG_H
