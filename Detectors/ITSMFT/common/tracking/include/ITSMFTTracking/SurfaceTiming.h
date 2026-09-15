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

#ifndef ALICEO2_ITSMFT_TRACKING_SURFACETIMING_H_
#define ALICEO2_ITSMFT_TRACKING_SURFACETIMING_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "GPUCommonDef.h"

#ifndef GPUCA_GPUCODE
#include <gsl/gsl>

#include "CommonDataFormat/InteractionRecord.h"
#include "ITSMFTTracking/ROFLookupTables.h"
#endif

namespace o2::itsmft::tracking
{

// TimeFrame-relative bunch-crossing coordinate. Host timing is signed 64-bit;
// any narrower device representation requires an explicit checked conversion.
using TFBC = int64_t;

// Half-open [begin, end) TF-relative BC interval for one source ROF.
// sourceROF is source-local; cross-source checks use interval intersection.
struct ROFIntervalBC {
  TFBC begin{0};
  TFBC end{0};
  uint32_t sourceROF{std::numeric_limits<uint32_t>::max()};
  uint32_t flags{0};

  // The default interval is invalid. A valid interval has a real source ROF
  // and strictly positive half-open extent.
  GPUhdi() constexpr bool isValid() const noexcept
  {
    return sourceROF != std::numeric_limits<uint32_t>::max() && begin < end;
  }
  // Compute the non-negative width in uint64_t: signed subtraction can
  // overflow for valid intervals spanning more than INT64_MAX BC.
  GPUhdi() constexpr uint64_t length() const noexcept
  {
    return static_cast<uint64_t>(end) - static_cast<uint64_t>(begin);
  }
};

static_assert(std::is_standard_layout_v<ROFIntervalBC>);
static_assert(std::is_trivially_copyable_v<ROFIntervalBC>);

// Detector-neutral half-open time estimate for a track/cell/road. It has no
// source-ROF identity and is standard-layout/trivially copyable for GenericTrack.
struct GenericTrackTimestamp {
  TFBC begin{0};
  TFBC end{0};

  GPUhdi() constexpr bool isValid() const noexcept { return begin < end; }
  // Half-open intersection; adjacent intervals and invalid intervals do not
  // intersect.
  GPUhdi() constexpr bool isCompatible(const GenericTrackTimestamp& other) const noexcept
  {
    return isValid() && other.isValid() && begin < other.end && other.begin < end;
  }
};

static_assert(std::is_standard_layout_v<GenericTrackTimestamp>);
static_assert(std::is_trivially_copyable_v<GenericTrackTimestamp>);
static_assert(sizeof(GenericTrackTimestamp) == 16);
static_assert(alignof(GenericTrackTimestamp) == alignof(TFBC));
static_assert(offsetof(GenericTrackTimestamp, begin) == 0);
static_assert(offsetof(GenericTrackTimestamp, end) == 8);

// Per-source readout timing configuration. ROF start uses the source ROF's
// InteractionRecord plus delay and bias; rofAddTimeErr is applied only by
// widen(), not folded into the stored interval.
struct ROFTimingConfig {
  TFBC rofLength{0};
  TFBC rofDelay{0};
  TFBC rofBias{0};
  TFBC rofAddTimeErr{0};
};

static_assert(std::is_standard_layout_v<ROFTimingConfig>);
static_assert(std::is_trivially_copyable_v<ROFTimingConfig>);

enum class TimingBuildError : uint8_t {
  None,
  InvalidROFLength,
  InvalidSourceROF,
  Overflow
};

struct ROFIntervalBuildResult {
  ROFIntervalBC interval{};
  TimingBuildError error{TimingBuildError::None};

  // Success requires both no error and a valid interval; the default result is
  // therefore not successful even though error defaults to None.
  constexpr bool ok() const noexcept { return error == TimingBuildError::None && interval.isValid(); }
};

enum class WidenError : uint8_t {
  None,
  InvalidInterval,
  InvalidMargin,
  LowerBoundOverflow,
  UpperBoundOverflow
};

struct WidenResult {
  ROFIntervalBC interval{};
  WidenError error{WidenError::None};

  // Match ROFIntervalBuildResult::ok(): a default result is not successful.
  constexpr bool ok() const noexcept { return error == WidenError::None && interval.isValid(); }
};

#ifndef GPUCA_GPUCODE

namespace detail
{
inline bool checkedAddBC(TFBC a, TFBC b, TFBC& out) noexcept
{
  if (b >= 0) {
    if (a > std::numeric_limits<TFBC>::max() - b) {
      return false;
    }
  } else {
    if (a < std::numeric_limits<TFBC>::min() - b) {
      return false;
    }
  }
  out = a + b;
  return true;
}
} // namespace detail

// origin is the explicit InteractionRecord for the loaded frame; rofIR is the
// source ROF's own record, used as-is for continuous and triggered readout.
inline ROFIntervalBuildResult computeROFIntervalBC(const o2::InteractionRecord& rofIR,
                                                   const o2::InteractionRecord& origin,
                                                   const ROFTimingConfig& cfg,
                                                   uint32_t sourceROF) noexcept
{
  if (sourceROF == std::numeric_limits<uint32_t>::max()) {
    return {{}, TimingBuildError::InvalidSourceROF};
  }
  if (cfg.rofLength <= 0) {
    return {{}, TimingBuildError::InvalidROFLength};
  }
  const TFBC anchor = static_cast<TFBC>(rofIR.differenceInBC(origin));
  TFBC begin{0};
  TFBC withDelay{0};
  TFBC end{0};
  if (!detail::checkedAddBC(anchor, cfg.rofDelay, withDelay) ||
      !detail::checkedAddBC(withDelay, cfg.rofBias, begin) ||
      !detail::checkedAddBC(begin, cfg.rofLength, end)) {
    return {{}, TimingBuildError::Overflow};
  }
  return {ROFIntervalBC{begin, end, sourceROF, 0}, TimingBuildError::None};
}

// Widen an interval by an explicit non-negative margin. Invalid input,
// negative margins, and bound overflow are reported rather than wrapped.
inline WidenResult widen(const ROFIntervalBC& interval, TFBC margin) noexcept
{
  if (!interval.isValid()) {
    return {{}, WidenError::InvalidInterval};
  }
  if (margin < 0) {
    return {{}, WidenError::InvalidMargin};
  }
  TFBC newBegin{0};
  if (!detail::checkedAddBC(interval.begin, -margin, newBegin)) {
    return {{}, WidenError::LowerBoundOverflow};
  }
  TFBC newEnd{0};
  if (!detail::checkedAddBC(interval.end, margin, newEnd)) {
    return {{}, WidenError::UpperBoundOverflow};
  }
  return {ROFIntervalBC{newBegin, newEnd, interval.sourceROF, interval.flags}, WidenError::None};
}

struct UniformROFTimingResult {
  ROFTimingConfig config{};
  bool uniform{false};
};

// The returned source-level config is valid only when all layers agree on
// length, delay, bias, and additional timing error; otherwise uniform is false.
inline UniformROFTimingResult deriveUniformROFTimingConfig(gsl::span<const o2::its::LayerTiming> perLayer) noexcept
{
  if (perLayer.empty()) {
    return {};
  }
  const auto& ref = perLayer[0];
  for (const auto& lt : perLayer) {
    if (lt.mROFLength != ref.mROFLength || lt.mROFDelay != ref.mROFDelay ||
        lt.mROFBias != ref.mROFBias || lt.mROFAddTimeErr != ref.mROFAddTimeErr) {
      return {};
    }
  }
  return {ROFTimingConfig{static_cast<TFBC>(ref.mROFLength), static_cast<TFBC>(ref.mROFDelay),
                          static_cast<TFBC>(ref.mROFBias), static_cast<TFBC>(ref.mROFAddTimeErr)},
          true};
}

#endif // GPUCA_GPUCODE

// Cross-source compatibility uses half-open interval intersection, not ROF
// ordinal equality; adjacent or invalid intervals do not intersect.
GPUhdi() constexpr bool intersects(const ROFIntervalBC& a, const ROFIntervalBC& b) noexcept
{
  return a.isValid() && b.isValid() && a.begin < b.end && b.begin < a.end;
}

} // namespace o2::itsmft::tracking

#endif /* ALICEO2_ITSMFT_TRACKING_SURFACETIMING_H_ */
