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

#ifndef ALICEO2_ITSMFT_TRACKING_SURFACESTATEOPERATIONRESULT_H_
#define ALICEO2_ITSMFT_TRACKING_SURFACESTATEOPERATIONRESULT_H_

#include <cstdint>

namespace o2::itsmft::tracking
{

enum class OperationFailureReason : uint8_t {
  SourceSurfaceKindMismatch = 0,
  NonFiniteInput = 1,
  NonFiniteOutput = 2,
  InvalidCovariance = 3,
  UnreachableTarget = 4,
  PropagationFailure = 5,
  MaterialFailure = 6,
  PredictedChi2Failure = 7,
  UpdateFailure = 8,
  RotationFailure = 9,
  AlphaMismatch = 10,
  ReferenceCoordinateMismatch = 11,
  // Finite-input forward seed with z-ordering or transverse separation at or
  // below the strict 1e-6f geometry boundary; distinct from numeric failures.
  SeedGeometryDegenerate = 12,
  // A present (non-hole) measurement has an invalid LayerId/catalog
  // association; no propagation, material, or chi2 arithmetic ran.
  InvalidSurfaceCatalogAssociation = 14,
  // A completed refit leg failed `|Q2Pt| < maxQoverPt && chi2 <
  // maxChi2NDF*(nCl*2-5)` after per-hit processing.
  LegAcceptanceFailure = 15,
  // The seed-level minimum-pT check failed after the outward leg, keyed by
  // the active traversal count and attached-cluster count.
  MinPtFailure = 16,
  // A nonzero reseedIfShorter is unsupported and is rejected before any leg
  // operation or output mutation.
  ReseedNotSupported = 17,
  // Surface-kind conversion was attempted but failed (invalid target kind
  // or direction boundary), distinct from a
  // source-kind mismatch where conversion was not attempted.
  SurfaceKindConversionFailure = 18
};

static_assert(sizeof(OperationFailureReason) == sizeof(uint8_t));

} // namespace o2::itsmft::tracking

#endif // ALICEO2_ITSMFT_TRACKING_SURFACESTATEOPERATIONRESULT_H_
