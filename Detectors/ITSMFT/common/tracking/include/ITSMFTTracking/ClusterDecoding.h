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

#ifndef ALICEO2_ITSMFT_TRACKING_CLUSTERDECODING_H_
#define ALICEO2_ITSMFT_TRACKING_CLUSTERDECODING_H_

#include <cstddef>
#include <cstdint>
#include <cmath>

#include <gsl/gsl>

#include "DataFormatsITSMFT/ClusterPattern.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "ITSMFTTracking/GlobalMeasurement.h"
#include "ITSMFTTracking/SurfaceDescriptor.h"
#include "ITSMFTTracking/SurfaceMeasurement.h"

namespace o2::itsmft::tracking
{

struct ClusterShape {
  uint32_t nPixels{0};
  uint16_t rowSpan{0};
  uint16_t columnSpan{0};
};

// Typed failures at the host compact-cluster decoding boundary; the loader
// adds source, ROF, and external-cluster context when it maps them.
enum class ClusterDecodeError : uint8_t {
  None,
  MissingDictionary,
  TruncatedExplicitPattern,
  MalformedExplicitPattern,
  InvalidPatternId,
  InvalidSensor,
  InvalidLayer,
  GeometryUnavailable,
  OtherMalformedInput
};

// Host-only cursor for the source-local explicit-pattern byte stream. It owns
// no storage and checks the complete encoded pattern before using the
// unbounded ClusterPattern iterator.
class BoundedPatternCursor
{
 public:
  explicit BoundedPatternCursor(gsl::span<const unsigned char> bytes) noexcept : mBytes(bytes) {}

  size_t consumed() const noexcept { return mPosition; }
  size_t remaining() const noexcept { return mBytes.size() - mPosition; }
  bool empty() const noexcept { return remaining() == 0; }

  ClusterDecodeError acquirePattern(o2::itsmft::ClusterPattern& pattern) noexcept
  {
    const auto available = remaining();
    if (available < 2) {
      return ClusterDecodeError::TruncatedExplicitPattern;
    }

    const auto rowSpan = mBytes[mPosition];
    const auto columnSpan = mBytes[mPosition + 1];
    if (rowSpan == 0 || columnSpan == 0 ||
        rowSpan > o2::itsmft::ClusterPattern::MaxRowSpan ||
        columnSpan > o2::itsmft::ClusterPattern::MaxColSpan) {
      return ClusterDecodeError::MalformedExplicitPattern;
    }

    const size_t nBits = static_cast<size_t>(rowSpan) * columnSpan;
    const size_t payloadBytes = (nBits + 7) / 8;
    const size_t encodedBytes = 2 + payloadBytes;
    if (available < encodedBytes) {
      return ClusterDecodeError::TruncatedExplicitPattern;
    }

    auto iterator = mBytes.begin() + mPosition;
    o2::itsmft::ClusterPattern decoded{iterator};
    if (decoded.getNPixels() == 0) {
      return ClusterDecodeError::MalformedExplicitPattern;
    }
    pattern = decoded;
    mPosition += encodedBytes;
    return ClusterDecodeError::None;
  }

 private:
  gsl::span<const unsigned char> mBytes{};
  size_t mPosition{0};
};

// Host-side facts produced by compact-cluster and geometry decoding.
struct DecodedCluster {
  GlobalPoint3F global{};
  // ITS geometry supplies its cylindrical tracking frame here. Disk
  // projection uses global coordinates directly.
  SurfaceFramePoint cylinderFrame{};
  // ALPIDE local row/column covariance. The detector projection determines
  // which normalized axes these values describe.
  SurfaceCovariance2F rowColumnCovariance{};
  ClusterShape shape{};
  int layer{-1};
};

// Fallible host-side geometry decode. Source identity, ROF ownership, surface
// mapping, and the tracking/fitting representations are loader concerns.
struct ClusterDecodeResult {
  DecodedCluster decoded{};
  ClusterDecodeError error{ClusterDecodeError::None};

  bool ok() const noexcept { return error == ClusterDecodeError::None; }
};

// Project decoded ITS facts into the accepted cylindrical convention.
inline GlobalMeasurement makeCylinderGlobalMeasurement(const DecodedCluster& decoded, uint32_t clusterId)
{
  const float sine = std::sin(decoded.cylinderFrame.frameAngle);
  const float cosine = std::cos(decoded.cylinderFrame.frameAngle);
  const auto& covariance = decoded.rowColumnCovariance;
  return GlobalMeasurement{
    decoded.global.x,
    decoded.global.y,
    decoded.global.z,
    {sine * sine * covariance.uu,
     -sine * cosine * covariance.uu,
     -sine * covariance.uv,
     cosine * cosine * covariance.uu,
     cosine * covariance.uv,
     covariance.vv},
    std::hypot(decoded.global.x, decoded.global.y),
    std::atan2(decoded.global.y, decoded.global.x),
    clusterId};
}

// Project decoded MFT facts into z-normal, global-x/global-y disk coordinates.
// ALPIDE row is established as global x and column as global y by the MFT
// geometry decoder. No legacy TrackingFrameInfo participates in this mapping.
inline GlobalMeasurement makeDiskGlobalMeasurement(const DecodedCluster& decoded, uint32_t clusterId)
{
  return GlobalMeasurement{
    decoded.global.x,
    decoded.global.y,
    decoded.global.z,
    {decoded.rowColumnCovariance.uu, decoded.rowColumnCovariance.uv, 0.f,
     decoded.rowColumnCovariance.vv, 0.f, 0.f},
    std::hypot(decoded.global.x, decoded.global.y),
    std::atan2(decoded.global.y, decoded.global.x),
    clusterId};
}

inline SurfaceMeasurement makeCylinderSurfaceMeasurement(const DecodedCluster& decoded)
{
  return {decoded.cylinderFrame, decoded.rowColumnCovariance};
}

inline SurfaceMeasurement makeDiskSurfaceMeasurement(const DecodedCluster& decoded)
{
  return {{decoded.global.z, decoded.global.x, decoded.global.y, 0.f},
          decoded.rowColumnCovariance};
}

} // namespace o2::itsmft::tracking

namespace o2::itsmft::ioutils
{
void fillMatrixCache(o2::detectors::DetID::ID detId);

template <o2::detectors::DetID::ID DetId>
o2::itsmft::tracking::ClusterDecodeResult decodeCluster(
  const o2::itsmft::CompClusterExt& c,
  o2::itsmft::tracking::BoundedPatternCursor& patterns,
  const o2::itsmft::TopologyDictionary* dict,
  bool applySysErrors = true);
} // namespace o2::itsmft::ioutils

namespace o2::itsmft::tracking
{

// Host-only loading boundary. Decoder implementations may call detector
// geometry, but this interface and its result never enter device views or CA
// loops.
class ClusterDecoder
{
 public:
  virtual ~ClusterDecoder() = default;

  // Called once per source before its first cluster; no-op by default.
  virtual void prepare() const {}

  virtual ClusterDecodeResult decode(
    const o2::itsmft::CompClusterExt& cluster,
    BoundedPatternCursor& patterns,
    const o2::itsmft::TopologyDictionary* dict,
    uint32_t externalIndex,
    bool applySysErrors) const = 0;
};

// Geometry-backed decoder. It performs the established single-pass geometry,
// pattern, covariance, and systematic-error operations, then maps the decoded
// detector layer to a global LayerId.
template <o2::detectors::DetID::ID DetId>
class GeometryClusterDecoder final : public ClusterDecoder
{
 public:
  void prepare() const override { o2::itsmft::ioutils::fillMatrixCache(DetId); }

  ClusterDecodeResult decode(
    const o2::itsmft::CompClusterExt& cluster,
    BoundedPatternCursor& patterns,
    const o2::itsmft::TopologyDictionary* dict,
    uint32_t,
    bool applySysErrors) const override
  {
    // Check before evaluating GeometryTGeo::Instance(): constructing the
    // geometry singleton without loaded geometry is fatal.
    if (dict == nullptr) {
      ClusterDecodeResult result;
      result.error = ClusterDecodeError::MissingDictionary;
      return result;
    }
    return o2::itsmft::ioutils::decodeCluster<DetId>(cluster, patterns, dict, applySysErrors);
  }
};

using ITSGeometryClusterDecoder = GeometryClusterDecoder<o2::detectors::DetID::ITS>;
using MFTGeometryClusterDecoder = GeometryClusterDecoder<o2::detectors::DetID::MFT>;

} // namespace o2::itsmft::tracking

#endif /* ALICEO2_ITSMFT_TRACKING_CLUSTERDECODING_H_ */
