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
/// \file SurfaceIndexMap.cxx
/// \author Paolo Butti
///

#include "ACTSInterface/SurfaceIndexMap.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>

#include "Acts/Definitions/Units.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Surfaces/BoundaryTolerance.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Surfaces/SurfacePlacementBase.hpp"
#include "ActsPlugins/Root/TGeoDetectorElement.hpp"

#include <TGeoManager.h>
#include <TGeoMatrix.h>
#include <TGeoNode.h>

#include "DetectorsCommonDataFormats/DetMatrixCache.h"
#include "Framework/Logger.h"

namespace o2::acts
{

namespace
{
const std::vector<int> gNoSensors{};
} // namespace

void SurfaceIndexMap::add(int sensorID, const Acts::Surface& surface)
{
  const auto geoId = surface.geometryId();
  auto [it, inserted] = mBySensor.try_emplace(sensorID, Entry{&surface, geoId});
  if (!inserted) {
    throw std::runtime_error("o2::acts::SurfaceIndexMap: sensor " + std::to_string(sensorID) +
                             " is already mapped");
  }
  mByGeoId[geoId].push_back(sensorID);
}

const Acts::Surface* SurfaceIndexMap::getSurface(int sensorID) const
{
  const auto it = mBySensor.find(sensorID);
  return it == mBySensor.end() ? nullptr : it->second.surface;
}

Acts::GeometryIdentifier SurfaceIndexMap::getGeometryId(int sensorID) const
{
  const auto it = mBySensor.find(sensorID);
  return it == mBySensor.end() ? Acts::GeometryIdentifier{} : it->second.geoId;
}

const std::vector<int>& SurfaceIndexMap::getSensorIDs(Acts::GeometryIdentifier geoId) const
{
  const auto it = mByGeoId.find(geoId);
  return it == mByGeoId.end() ? gNoSensors : it->second;
}

void SurfaceIndexMap::clear()
{
  mBySensor.clear();
  mByGeoId.clear();
}

namespace
{

/// Uniform grid hash over per-sensor surface centres, so matching stays linear
/// in the number of sensors instead of quadratic.
class CentreLookup
{
 public:
  explicit CentreLookup(double cellSize) : mCellSize(cellSize) {}

  void add(const Acts::Surface* surface, const Acts::Vector3& position)
  {
    mEntries.push_back({surface, position});
    mCells[cellOf(position)].push_back(mEntries.size() - 1);
  }

  /// \return the closest surface within \p tolerance, or nullptr
  const Acts::Surface* findNearest(const Acts::Vector3& position, double tolerance) const
  {
    const auto cell = cellOf(position);
    const Acts::Surface* best = nullptr;
    double bestDist2 = tolerance * tolerance;
    for (std::int64_t dx = -1; dx <= 1; ++dx) {
      for (std::int64_t dy = -1; dy <= 1; ++dy) {
        for (std::int64_t dz = -1; dz <= 1; ++dz) {
          const auto it = mCells.find(Cell{cell.x + dx, cell.y + dy, cell.z + dz});
          if (it == mCells.end()) {
            continue;
          }
          for (const auto idx : it->second) {
            const double dist2 = (mEntries[idx].position - position).squaredNorm();
            if (dist2 <= bestDist2) {
              bestDist2 = dist2;
              best = mEntries[idx].surface;
            }
          }
        }
      }
    }
    return best;
  }

  std::size_t size() const { return mEntries.size(); }

 private:
  struct Entry {
    const Acts::Surface* surface = nullptr;
    Acts::Vector3 position;
  };
  struct Cell {
    std::int64_t x = 0, y = 0, z = 0;
    bool operator==(const Cell&) const = default;
  };
  struct CellHash {
    std::size_t operator()(const Cell& c) const noexcept
    {
      std::size_t h = static_cast<std::size_t>(c.x) * 0x9e3779b97f4a7c15ULL;
      h ^= static_cast<std::size_t>(c.y) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
      h ^= static_cast<std::size_t>(c.z) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
      return h;
    }
  };

  Cell cellOf(const Acts::Vector3& p) const
  {
    return Cell{static_cast<std::int64_t>(std::floor(p[0] / mCellSize)),
                static_cast<std::int64_t>(std::floor(p[1] / mCellSize)),
                static_cast<std::int64_t>(std::floor(p[2] / mCellSize))};
  }

  double mCellSize = 1.;
  std::vector<Entry> mEntries;
  std::unordered_map<Cell, std::vector<std::size_t>, CellHash> mCells;
};

} // namespace

namespace
{

/// Collect the sensitive surfaces, split by whether their centre identifies them.
struct SensitiveSurfaces {
  CentreLookup byCentre;
  std::vector<const Acts::Surface*> layerSurfaces;
  std::unordered_map<const TGeoNode*, std::vector<const Acts::Surface*>> byNode;
  std::size_t nWithoutNode = 0;

  explicit SensitiveSurfaces(double cellSize) : byCentre(cellSize) {}
};

SensitiveSurfaces collectSensitive(const Acts::TrackingGeometry& geometry,
                                   const Acts::GeometryContext& gctx, double cellSize)
{
  SensitiveSurfaces out(cellSize);
  geometry.apply([&](const Acts::Surface& surface) {
    const auto* placement = surface.surfacePlacement();
    if (placement == nullptr || !placement->isSensitive()) {
      return;
    }
    const auto* tgeoElement = dynamic_cast<const ActsPlugins::TGeoDetectorElement*>(placement);
    if (tgeoElement != nullptr) {
      out.byNode[&tgeoElement->tgeoNode()].push_back(&surface);
    } else {
      ++out.nWithoutNode;
    }
    const auto type = surface.type();
    if (type == Acts::Surface::SurfaceType::Cylinder || type == Acts::Surface::SurfaceType::Disc) {
      out.layerSurfaces.push_back(&surface);
    } else {
      out.byCentre.add(&surface, surface.center(gctx));
    }
  });
  return out;
}

/// Global translation of an O2 sensor, in ACTS units, plus its rotation.
struct SensorPlacement {
  Acts::Vector3 position;
  Acts::RotationMatrix3 rotation;
};

SensorPlacement sensorPlacementOf(const o2::detectors::DetMatrixCache& cache, int sensorID)
{
  double rot[9] = {0.};
  double tra[3] = {0.};
  cache.getMatrixL2G(sensorID).GetComponents(rot[0], rot[1], rot[2], tra[0],
                                            rot[3], rot[4], rot[5], tra[1],
                                            rot[6], rot[7], rot[8], tra[2]);
  SensorPlacement out;
  out.position = Acts::Vector3{tra[0] * Acts::UnitConstants::cm, tra[1] * Acts::UnitConstants::cm,
                               tra[2] * Acts::UnitConstants::cm};
  out.rotation << rot[0], rot[1], rot[2], rot[3], rot[4], rot[5], rot[6], rot[7], rot[8];
  return out;
}

/// Resolve a sensor through its TGeo node. Exact: the node is an identity, not a
/// measurement. Repeated placements of one volume share a node, so those are
/// separated by the accumulated global translation.
/// \return the surface, or nullptr with \p ambiguous set when several remain
const Acts::Surface* matchByNode(const SensitiveSurfaces& surfaces, const std::string& path,
                                 const Acts::GeometryContext& gctx, const SensorPlacement& placement,
                                 double tolerance, bool& ambiguous)
{
  ambiguous = false;
  if (gGeoManager == nullptr) {
    throw std::runtime_error("o2::acts::makeSurfaceIndex: no TGeo geometry loaded");
  }
  gGeoManager->PushPath();
  const bool ok = gGeoManager->cd(path.c_str());
  const TGeoNode* node = ok ? gGeoManager->GetCurrentNode() : nullptr;
  gGeoManager->PopPath();
  if (node == nullptr) {
    return nullptr;
  }

  const auto it = surfaces.byNode.find(node);
  if (it == surfaces.byNode.end()) {
    return nullptr;
  }
  if (it->second.size() == 1) {
    return it->second.front();
  }
  const Acts::Surface* best = nullptr;
  for (const auto* surface : it->second) {
    if ((surface->localToGlobalTransform(gctx).translation() - placement.position).norm() <= tolerance) {
      if (best != nullptr) {
        ambiguous = true;
        return nullptr;
      }
      best = surface;
    }
  }
  return best;
}

/// Resolve a sensor from its transform alone. Used when no TGeo path is available.
/// \return the surface, or nullptr with \p ambiguous set when several match
const Acts::Surface* matchByPlacement(const SensitiveSurfaces& surfaces,
                                      const Acts::GeometryContext& gctx,
                                      const SensorPlacement& placement, double tolerance,
                                      bool& ambiguous)
{
  ambiguous = false;

  // Rotations are dimensionless; a generous bound on the TGeoHMatrix -> Eigen round trip.
  constexpr double kRotationTolerance = 1e-9;

  if (const auto* surface = surfaces.byCentre.findNearest(placement.position, tolerance);
      surface != nullptr) {
    return surface;
  }

  // A sensor modelled as a whole cylinder or disc: its origin is generally not a
  // useful key (an ALICE 3 vertex-detector petal is a tube segment, so its origin
  // is on the beam axis), so the full transform is compared instead. Every match
  // is collected rather than taking the first: the three layers of one petal share
  // both origin and rotation and differ only in radius, which the matrix cache does
  // not carry, so those are genuinely unresolvable this way and must be reported.
  const Acts::Surface* best = nullptr;
  for (const auto* surface : surfaces.layerSurfaces) {
    const auto& transform = surface->localToGlobalTransform(gctx);
    if ((transform.translation() - placement.position).norm() <= tolerance &&
        (transform.rotation() - placement.rotation).norm() <= kRotationTolerance) {
      if (best != nullptr) {
        ambiguous = true;
        return nullptr;
      }
      best = surface;
    }
  }
  return best;
}

} // namespace

SurfaceIndexMap makeSurfaceIndex(const Acts::TrackingGeometry& geometry,
                                 const Acts::GeometryContext& gctx,
                                 const o2::detectors::DetMatrixCache& cache,
                                 double toleranceCm,
                                 SensorPathProvider pathProvider)
{
  if (!cache.isBuilt()) {
    throw std::runtime_error("o2::acts::makeSurfaceIndex: the detector matrix cache is not built");
  }
  if (cache.getCacheL2G().getSize() == 0) {
    throw std::runtime_error(
      "o2::acts::makeSurfaceIndex: the L2G matrix cache is empty. Call fillMatrixCache() with the "
      "L2G bit set before building the index.");
  }

  // ACTS works in mm, O2 in cm; Acts::UnitConstants::cm is exactly that factor.
  const double tolerance = toleranceCm * Acts::UnitConstants::cm;

  const auto surfaces = collectSensitive(geometry, gctx, std::max(tolerance, 1e-6));
  const int nSensors = cache.getSize();

  LOGP(info,
       "Matching {} {} sensors against {} sensitive ACTS surfaces ({} per-sensor, {} whole-layer) "
       "by {} (tolerance {} cm)",
       nSensors, cache.getDetID().getName(), surfaces.byCentre.size() + surfaces.layerSurfaces.size(),
       surfaces.byCentre.size(), surfaces.layerSurfaces.size(),
       pathProvider ? "TGeo node" : "placement", toleranceCm);
  if (pathProvider && surfaces.nWithoutNode != 0) {
    LOGP(warn, "{} sensitive surfaces have no TGeo detector element and cannot be matched by node",
         surfaces.nWithoutNode);
  }

  SurfaceIndexMap index;
  std::size_t nUnmatched = 0;
  std::size_t nAmbiguous = 0;
  std::unordered_map<Acts::GeometryIdentifier, int> claimedBy;
  std::ostringstream failures;
  int nReported = 0;

  for (int sensorID = 0; sensorID < nSensors; ++sensorID) {
    const auto placement = sensorPlacementOf(cache, sensorID);

    bool ambiguous = false;
    const Acts::Surface* match =
      pathProvider
        ? matchByNode(surfaces, pathProvider(sensorID), gctx, placement, tolerance, ambiguous)
        : matchByPlacement(surfaces, gctx, placement, tolerance, ambiguous);

    if (match == nullptr) {
      ambiguous ? ++nAmbiguous : ++nUnmatched;
      if (nReported < 10) {
        failures << "\n  sensor " << sensorID << " at ("
                 << placement.position[0] / Acts::UnitConstants::cm << ", "
                 << placement.position[1] / Acts::UnitConstants::cm << ", "
                 << placement.position[2] / Acts::UnitConstants::cm << ") cm "
                 << (ambiguous ? "matches several sensitive ACTS surfaces" : "matches no sensitive ACTS surface");
        ++nReported;
      }
      continue;
    }

    // One surface per sensor. Relaxing this hides exactly the failure the
    // ambiguity check above is there to catch, so it is enforced.
    const auto geoId = match->geometryId();
    const auto [it, inserted] = claimedBy.try_emplace(geoId, sensorID);
    if (!inserted) {
      throw std::runtime_error(
        "o2::acts::makeSurfaceIndex: sensors " + std::to_string(it->second) + " and " +
        std::to_string(sensorID) + " both resolve to the ACTS surface with geometry id " +
        std::to_string(geoId.value()) +
        ". They are indistinguishable by the information available, so the index would be wrong.");
    }
    index.add(sensorID, *match);
  }

  if (nUnmatched != 0 || nAmbiguous != 0) {
    std::string hint;
    if (nAmbiguous != 0 && !pathProvider) {
      hint =
        " Pass a SensorPathProvider (e.g. GeometryTGeo::getMatrixPath) so sensors are identified by "
        "their TGeo node instead of their placement.";
    }
    throw std::runtime_error("o2::acts::makeSurfaceIndex: of " + std::to_string(nSensors) +
                             " sensors, " + std::to_string(nUnmatched) + " matched no surface and " +
                             std::to_string(nAmbiguous) + " were ambiguous." + hint + failures.str());
  }

  LOGP(info, "Sensor index built: {} sensors on {} surfaces", index.size(), index.getNSurfaces());

  return index;
}

} // namespace o2::acts
