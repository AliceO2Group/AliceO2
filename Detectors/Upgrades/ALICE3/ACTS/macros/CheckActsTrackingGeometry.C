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
/// \file CheckActsTrackingGeometry.C
/// \brief Build the ACTS tracking geometry from a TGeo file and check it against O2
/// \author Paolo Butti
///
/// Standalone check, outside any workflow. It
///   1. loads the TGeo geometry through o2::base::GeometryManager,
///   2. builds the ACTS tracking geometry with the ALICE 3 Gen3 builder,
///   3. prints the volume/layer/surface table -- the authority for the
///      (volume, layer) keys the digitisation and seeding JSON files use,
///   4. locates every O2 TRK chip on an ACTS surface and, for the chips that own
///      their surface, compares the surface centre with the chip origin. That
///      validates the whole TGeo -> ACTS transform chain including the cm -> mm
///      scaling.
///
/// Usage:
/// \code
///   root -l -q 'CheckActsTrackingGeometry.C+("o2sim_geometry-aligned.root", \
///                                            "gen3_geometry_config.json")'
/// \endcode

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <cmath>
#include <cstdio>
#include <string>

#include <TGeoManager.h>

#include "Acts/Definitions/Units.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Surfaces/Surface.hpp"

#include "ACTSInterface/SurfaceIndexMap.h"
#include "ACTSInterface/TrackingGeometryManager.h"
#include "ALICE3ACTS/Gen3BlueprintBuilder.h"
#include "DetectorsBase/GeometryManager.h"
#include "MathUtils/Utils.h"
#include "TRKBase/GeometryTGeo.h"

#endif

int CheckActsTrackingGeometry(const std::string& geomFile = "o2sim_geometry-aligned.root",
                              const std::string& gen3ConfigFile = "gen3_geometry_config.json",
                              const std::string& materialMapFile = "",
                              double toleranceCm = 1e-3)
{
  auto& manager = o2::acts::TrackingGeometryManager::instance();

  o2::alice3::Gen3BlueprintBuilder::Config builderConfig;
  builderConfig.geometryConfigFile = gen3ConfigFile;
  builderConfig.withMaterial = true;
  manager.setBuilder(std::make_unique<o2::alice3::Gen3BlueprintBuilder>(builderConfig));
  if (!materialMapFile.empty()) {
    manager.setMaterialMapFile(materialMapFile);
  }

  manager.buildFromFile(geomFile);
  const auto& geometry = manager.get();

  // The TRK geometry helper needs its local-to-global matrices for the check below.
  auto* trkGeo = o2::trk::GeometryTGeo::Instance();
  trkGeo->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G));

  // Identify the chips by their TGeo node rather than by position: the vertex
  // detector petals are tube segments, so all three layers of a petal share both
  // origin and rotation and a placement-only match cannot separate them.
  const auto& index = manager.getIndex(*trkGeo, toleranceCm, [trkGeo](int chipID) {
    return std::string(trkGeo->getMatrixPath(chipID).Data());
  });

  std::printf("\n--- ACTS <-> O2 sensor check ---\n");
  std::printf("TRK chips              : %d\n", trkGeo->getNumberOfChips());
  std::printf("mapped to ACTS surfaces: %zu (on %zu surfaces)\n", index.size(), index.getNSurfaces());
  if (static_cast<int>(index.size()) != trkGeo->getNumberOfChips()) {
    std::printf("FAILED: not every chip is mapped\n");
    return 1;
  }

  // Compare the ACTS surface centre with the chip origin, for the chips that own
  // their surface. Chips sitting on a whole-layer cylinder or disc (the vertex
  // detector petals) are skipped: that surface's centre is on the beam axis, not
  // at the chip. makeSurfaceIndex() already enforced the tolerance, so this only
  // reports the accuracy actually achieved.
  double maxResidualCm = 0.;
  int worstChip = -1;
  int nPerSensor = 0;
  int nOnLayerSurface = 0;
  for (int chipID = 0; chipID < trkGeo->getNumberOfChips(); ++chipID) {
    const auto* surface = index.getSurface(chipID);
    if (surface == nullptr) {
      std::printf("FAILED: chip %d has no surface\n", chipID);
      return 1;
    }
    if (surface->type() == Acts::Surface::SurfaceType::Cylinder ||
        surface->type() == Acts::Surface::SurfaceType::Disc) {
      ++nOnLayerSurface;
      continue;
    }
    ++nPerSensor;
    double rot[9] = {0.};
    double tra[3] = {0.};
    trkGeo->getMatrixL2G(chipID).GetComponents(rot[0], rot[1], rot[2], tra[0],
                                               rot[3], rot[4], rot[5], tra[1],
                                               rot[6], rot[7], rot[8], tra[2]);
    const auto centre = surface->center(manager.getNominalContext());
    const double residual = std::hypot(centre[0] / Acts::UnitConstants::cm - tra[0],
                                       centre[1] / Acts::UnitConstants::cm - tra[1],
                                       centre[2] / Acts::UnitConstants::cm - tra[2]);
    if (residual > maxResidualCm) {
      maxResidualCm = residual;
      worstChip = chipID;
    }
  }
  std::printf("chips on their own surface : %d\n", nPerSensor);
  std::printf("chips on a layer surface   : %d\n", nOnLayerSurface);
  std::printf("max |surface centre - chip origin| : %.3e cm (chip %d)\n", maxResidualCm, worstChip);

  std::printf("surfaces with material : %zu\n", manager.getNDecoratedSurfaces());
  std::printf("geometry version       : %s\n",
              geometry.geometryVersion() == Acts::TrackingGeometry::GeometryVersion::Gen3 ? "Gen3" : "Gen1");
  std::printf("--- OK ---\n");
  return 0;
}
