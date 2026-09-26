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
/// \file SurfaceIndexMap.h
/// \brief Lookup between O2 sensor indices and ACTS surfaces
/// \author Paolo Butti
///

#ifndef ALICEO2_ACTS_SURFACEINDEXMAP_H
#define ALICEO2_ACTS_SURFACEINDEXMAP_H

#include <cstddef>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "Acts/Geometry/GeometryIdentifier.hpp"

namespace Acts
{
class Surface;
class TrackingGeometry;
class GeometryContext;
} // namespace Acts

namespace o2::detectors
{
class DetMatrixCache;
}

namespace o2::acts
{

/// Lookup between an O2 sensor index (chip ID, as used by the detector's
/// DetMatrixCache-derived geometry helper) and the ACTS surface it sits on.
///
/// The relation is many-to-one, not one-to-one. A planar sensor becomes its own
/// ACTS surface, but a sub-detector modelled as a whole cylindrical or disc
/// layer -- as the ALICE 3 vertex detector is -- becomes a single surface that
/// backs every chip of that layer. Hence getSurface() takes a sensor and
/// getSensorIDs() gives back a list.
///
/// ACTS assigns the Acts::GeometryIdentifier inside
/// Acts::Blueprint::construct(), so this map can only be filled once the
/// TrackingGeometry is complete. See makeSurfaceIndex().
class SurfaceIndexMap
{
 public:
  static constexpr int InvalidSensorID = -1;

  /// \throw std::runtime_error if the sensor is already mapped
  void add(int sensorID, const Acts::Surface& surface);

  /// \return nullptr if the sensor is not mapped
  const Acts::Surface* getSurface(int sensorID) const;
  /// \return a default-constructed identifier if the sensor is not mapped
  Acts::GeometryIdentifier getGeometryId(int sensorID) const;

  /// All sensors sitting on the given surface, in increasing sensor order.
  /// \return an empty vector if the surface backs no sensor
  const std::vector<int>& getSensorIDs(Acts::GeometryIdentifier geoId) const;

  bool empty() const { return mBySensor.empty(); }
  /// Number of mapped sensors.
  std::size_t size() const { return mBySensor.size(); }
  /// Number of distinct surfaces the sensors map onto.
  std::size_t getNSurfaces() const { return mByGeoId.size(); }
  void clear();

 private:
  struct Entry {
    const Acts::Surface* surface = nullptr;
    Acts::GeometryIdentifier geoId{};
  };

  std::unordered_map<int, Entry> mBySensor;
  std::unordered_map<Acts::GeometryIdentifier, std::vector<int>> mByGeoId;
};

/// Full TGeo node path of one sensor, e.g. o2::trk::GeometryTGeo::getMatrixPath().
using SensorPathProvider = std::function<std::string(int sensorID)>;

/// Build the sensor <-> surface index.
///
/// Two strategies, and \b which \b one \b is \b used \b matters:
///
///  - **With \p pathProvider (exact, preferred).** Each sensor is resolved to its
///    TGeo node and matched against the node every ACTS surface was built from
///    (ActsPlugins::TGeoDetectorElement::tgeoNode()). This is an identity, not a
///    measurement, so it cannot mis-assign. Repeated placements of one volume
///    share a node and are separated by their accumulated global transform.
///
///  - **Without it (geometric fallback).** Sensors are matched on their
///    local-to-global transform. Exact where a transform identifies a sensor, but
///    it does not always: an ALICE 3 vertex-detector petal is a tube segment, so
///    all three of its layers share one origin *and* one rotation and differ only
///    in radius, which the matrix cache does not carry. Such sensors are reported
///    as ambiguous rather than assigned arbitrarily.
///
/// Both read the same TGeo tree that the surfaces were built from, so a successful
/// match is exact rather than approximate; \p toleranceCm only guards floating-point
/// round-trips.
///
/// \param geometry     the constructed tracking geometry
/// \param gctx         context used to evaluate the surface transforms
/// \param cache        detector geometry helper; requires a filled L2G cache
///                     (call fillMatrixCache(bit2Mask(o2::math_utils::TransformType::L2G)))
/// \param toleranceCm  matching tolerance, in O2 units (cm)
/// \param pathProvider optional TGeo path lookup, as described above
/// \throw std::runtime_error if a sensor cannot be matched, if a match is
///        ambiguous, or if two sensors would claim the same surface
SurfaceIndexMap makeSurfaceIndex(const Acts::TrackingGeometry& geometry,
                                 const Acts::GeometryContext& gctx,
                                 const o2::detectors::DetMatrixCache& cache,
                                 double toleranceCm = 1e-3,
                                 SensorPathProvider pathProvider = {});

} // namespace o2::acts

#endif // ALICEO2_ACTS_SURFACEINDEXMAP_H
