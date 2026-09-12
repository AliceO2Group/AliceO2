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

#ifndef ALICEO2_ITSMFT_TRACKING_TRIPLETFITTING_H_
#define ALICEO2_ITSMFT_TRACKING_TRIPLETFITTING_H_

#include <array>
#include <type_traits>

#include "GPUCommonDef.h"
#include "ITSMFTTracking/GlobalMeasurement.h"

namespace o2::itsmft::tracking
{

struct TripletKinkVector {
  float theta{0.f};
  float phi{0.f};
};

// Theta and phi rows of the hit-coordinate Jacobian H.
struct TripletHitJacobian {
  std::array<float, 3> theta{};
  std::array<float, 3> phi{};
};

// Linearized local-triplet factor from Eq. (19) of the General Triplet Track
// Fit. H is evaluated at kappaRef = -Psi_phi / rho_phi and hit slot i maps to
// CellSeed::getClusterReference(i). Measurement and MS covariances are added
// when adjacent triplets are compared.
struct TripletFitFactor {
  TripletKinkVector psi{};
  TripletKinkVector rho{};
  std::array<TripletHitJacobian, 3> h{};

  GPUhdi() bool isValid() const noexcept
  {
    return rho.phi != 0.f;
  }
};

static_assert(std::is_standard_layout_v<TripletFitFactor>);
static_assert(std::is_trivially_copyable_v<TripletFitFactor>);
static_assert(sizeof(TripletFitFactor) == 88);

struct AdjacentTripletFitResult {
  float curvature{0.};
  float curvatureVariance{0.};
  float chi2{0.};
};

bool makeTripletFitFactor(
  const std::array<GlobalMeasurement, 3>& measurements,
  TripletFitFactor& factor) noexcept;

// Minimize Eq. (19) for adjacent triplets sharing one curvature. measurements
// are the four unique ordered hits; angularVariance is the space-angle MS
// variance for each triplet.
bool fitAdjacentTripletFactors(
  const TripletFitFactor& firstFactor,
  const TripletFitFactor& secondFactor,
  const std::array<GlobalMeasurement, 4>& measurements,
  const std::array<float, 2>& angularVariance,
  AdjacentTripletFitResult& result) noexcept;

} // namespace o2::itsmft::tracking

#endif // ALICEO2_ITSMFT_TRACKING_TRIPLETFITTING_H_
