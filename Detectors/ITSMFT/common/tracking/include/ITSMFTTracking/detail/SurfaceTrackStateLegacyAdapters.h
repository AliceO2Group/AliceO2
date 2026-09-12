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

#ifndef ALICEO2_ITSMFT_TRACKING_SURFACETRACKSTATELEGACYADAPTERS_H_
#define ALICEO2_ITSMFT_TRACKING_SURFACETRACKSTATELEGACYADAPTERS_H_

#include "GPUCommonDef.h"
#include "GPUCommonMath.h"

#if !defined(GPUCA_GPUCODE)

#include <limits>

#include "ITSMFTTracking/SurfaceTrackState.h"
#include "ReconstructionDataFormats/TrackFwd.h"
#include "ReconstructionDataFormats/Track.h"

namespace o2::itsmft::tracking::legacy
{

inline bool canNarrowToFiniteFloat(double value) noexcept
{
  constexpr double maximum = std::numeric_limits<float>::max();
  return value >= -maximum && value <= maximum &&
         o2::gpu::GPUCommonMath::Finite(static_cast<float>(value));
}

// Host-only boundary for retained legacy state conversion. Production state
// operations use SurfaceTrackState directly.
inline bool importBarrelTrackParCov(const o2::track::TrackParCovF& source, SurfaceTrackState& destination) noexcept
{
  SurfaceTrackState scratch{};
  scratch.referenceCoordinate = source.getX();
  scratch.alpha = source.getAlpha();
  for (uint8_t i = 0; i < 5; ++i) {
    scratch.parameters[i] = source.getParam(i);
  }
  for (uint8_t i = 0; i < 15; ++i) {
    scratch.covariance[i] = source.getCov()[i];
  }
  scratch.kind = SurfaceKind::Cylinder;
  scratch.absCharge = static_cast<uint8_t>(source.getAbsCharge());
  scratch.pid = source.getPID();
  destination = scratch;
  return true;
}

inline bool exportBarrelTrackParCov(const SurfaceTrackState& source, o2::track::TrackParCovF& destination) noexcept
{
  if (source.kind != SurfaceKind::Cylinder) {
    return false;
  }
  o2::track::TrackParCovF::params_t parameters{};
  o2::track::TrackParCovF::covMat_t covariance{};
  for (uint8_t i = 0; i < 5; ++i) {
    parameters[i] = source.parameters[i];
  }
  for (uint8_t i = 0; i < 15; ++i) {
    covariance[i] = source.covariance[i];
  }
  const o2::track::TrackParCovF scratch{source.referenceCoordinate, source.alpha, parameters, covariance, source.absCharge, source.pid};
  destination = scratch;
  return true;
}

inline bool importLegacyForwardTrackParCov(const o2::track::TrackParCovFwd& source, SurfaceTrackState& destination) noexcept
{
  SurfaceTrackState scratch{};
  const auto& covariance = source.getCovariances();
  const double parameters[] = {source.getX(), source.getY(), source.getPhi(), source.getTanl(), source.getInvQPt()};
  if (!canNarrowToFiniteFloat(source.getZ())) {
    return false;
  }
  for (uint8_t i = 0; i < 5; ++i) {
    if (!canNarrowToFiniteFloat(parameters[i])) {
      return false;
    }
  }
  for (uint8_t row = 0; row < 5; ++row) {
    for (uint8_t column = 0; column <= row; ++column) {
      if (!canNarrowToFiniteFloat(covariance(row, column))) {
        return false;
      }
    }
  }
  const float referenceCoordinate = static_cast<float>(source.getZ());
  scratch.referenceCoordinate = referenceCoordinate;
  scratch.alpha = 0.f;
  for (uint8_t i = 0; i < 5; ++i) {
    const float value = static_cast<float>(parameters[i]);
    scratch.parameters[i] = value;
  }
  for (uint8_t row = 0; row < 5; ++row) {
    for (uint8_t column = 0; column <= row; ++column) {
      const float value = static_cast<float>(covariance(row, column));
      scratch.covariance[packedCovarianceIndex(row, column)] = value;
    }
  }
  scratch.kind = SurfaceKind::Disk;
  scratch.absCharge = 1;
  scratch.pid = o2::track::PID::Pion;
  destination = scratch;
  return true;
}

// Host-only inverse used by output staging; reconstructs the legacy payload
// from the common float representation.
inline bool exportLegacyForwardTrackParCov(const SurfaceTrackState& source, o2::track::TrackParCovFwd& destination) noexcept
{
  if (source.kind != SurfaceKind::Disk) {
    return false;
  }
  o2::track::SMatrix5 parameters{};
  o2::track::SMatrix55Sym covariance{};
  for (uint8_t i = 0; i < 5; ++i) {
    if (!o2::gpu::GPUCommonMath::Finite(source.parameters[i])) {
      return false;
    }
    parameters[i] = source.parameters[i];
  }
  for (uint8_t row = 0; row < 5; ++row) {
    for (uint8_t column = 0; column <= row; ++column) {
      const auto value = source.covariance[packedCovarianceIndex(row, column)];
      if (!o2::gpu::GPUCommonMath::Finite(value)) {
        return false;
      }
      covariance(row, column) = value;
    }
  }
  if (!o2::gpu::GPUCommonMath::Finite(source.referenceCoordinate)) {
    return false;
  }
  destination = o2::track::TrackParCovFwd{source.referenceCoordinate, parameters, covariance, 0.};
  return true;
}

} // namespace o2::itsmft::tracking::legacy

#endif // !defined(GPUCA_GPUCODE)

#endif // ALICEO2_ITSMFT_TRACKING_SURFACETRACKSTATELEGACYADAPTERS_H_
