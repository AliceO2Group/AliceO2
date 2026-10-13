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
/// \file TrkTGeoBlueprintBuilder.h
/// \brief Build an ACTS TRK tracking geometry from the TGeo geometry already loaded by O2
///

#ifndef ALICEO2_ALICE3_TRKTGEOBLUEPRINTBUILDER_H
#define ALICEO2_ALICE3_TRKTGEOBLUEPRINTBUILDER_H

#include <string>

#include "Acts/Utilities/Logger.hpp"
#include "ACTSInterface/ITrackingGeometryBuilder.h"

namespace o2::alice3
{

/// First-step automatic ALICE3 builder: TRK sensitive barrel only, no material.
/// The TGeo volume name is a detector-specific selection rule, not a geometry
/// configuration. Positions, shapes, and layer extents come from the live TGeo.
class TrkTGeoBlueprintBuilder : public o2::acts::ITrackingGeometryBuilder
{
 public:
  struct Config {
    std::string graphvizFile;
    Acts::Logging::Level logLevel = Acts::Logging::INFO;
  };

  TrkTGeoBlueprintBuilder();
  explicit TrkTGeoBlueprintBuilder(Config config);
  o2::acts::TrackingGeometryOutput build(TGeoManager& tgeo,
                                         const Acts::GeometryContext& gctx) override;
  const char* getName() const override { return "alice3-trk-tgeo"; }

 private:
  Config mConfig;
};

} // namespace o2::alice3

#endif // ALICEO2_ALICE3_TRKTGEOBLUEPRINTBUILDER_H
