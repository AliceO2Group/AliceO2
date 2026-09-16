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
/// \file Gen3BlueprintBuilder.h
/// \brief ALICE 3 Gen3 ("Blueprint") ACTS tracking geometry builder
/// \author Paolo Butti
///
/// Ported from actsO2 (ActsAlgorithms/Geometry/src/ALICE3Gen3Geometry.cpp).
/// The construction logic is unchanged, so the volume/layer identifiers it
/// produces stay compatible with the material maps, digitisation and seeding
/// JSON files of the standalone actsO2 chain. The differences are:
///
///   - it reads a TGeoManager handed in by the caller (normally O2's live
///     gGeoManager) instead of importing a ROOT file itself;
///   - the process-lifetime static stores of the standalone version are member
///     state, so several builders can coexist in one process.
///

#ifndef ALICEO2_ALICE3_GEN3BLUEPRINTBUILDER_H
#define ALICEO2_ALICE3_GEN3BLUEPRINTBUILDER_H

#include <string>

#include "Acts/Utilities/Logger.hpp"

#include "ACTSInterface/ITrackingGeometryBuilder.h"

namespace o2::alice3
{

/// Builds the ALICE 3 tracking geometry (TRK vertex detector and barrel, FT3
/// discs, inner and outer TOF) with the ACTS Gen3 Blueprint API.
class Gen3BlueprintBuilder : public o2::acts::ITrackingGeometryBuilder
{
 public:
  struct Config {
    /// Path to gen3_geometry_config.json. All geometry-specific numbers (sensor
    /// name globs, region boundaries, clustering tolerances, passive structures,
    /// pinned volume IDs) come from there; see Gen3GeometryConfig.h.
    std::string geometryConfigFile;

    /// Build the passive layers and attach proto material. Must match what was
    /// used when the material map was produced, otherwise the geometry
    /// identifiers shift and the map no longer applies.
    bool withMaterial = true;

    /// Optional path for a graphviz dump of the blueprint tree.
    std::string graphvizFile;

    Acts::Logging::Level logLevel = Acts::Logging::INFO;
  };

  explicit Gen3BlueprintBuilder(Config config);

  o2::acts::TrackingGeometryOutput build(TGeoManager& tgeo,
                                         const Acts::GeometryContext& gctx) override;

  const char* getName() const override { return "alice3-gen3"; }

  const Config& getConfig() const { return mConfig; }

 private:
  Config mConfig;
};

} // namespace o2::alice3

#endif // ALICEO2_ALICE3_GEN3BLUEPRINTBUILDER_H
