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

#define BOOST_TEST_MODULE ITSMFT SurfaceTiming
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <array>
#include <limits>

#include "CommonDataFormat/InteractionRecord.h"
#include "ITSMFTTracking/SurfaceTiming.h"

using namespace o2::itsmft::tracking;
using o2::its::LayerTiming;

BOOST_AUTO_TEST_CASE(ROFIntervalBCViewsAreDeviceFriendly)
{
  static_assert(std::is_standard_layout_v<ROFIntervalBC>);
  static_assert(std::is_trivially_copyable_v<ROFIntervalBC>);
  static_assert(std::is_standard_layout_v<ROFTimingConfig>);
  static_assert(std::is_trivially_copyable_v<ROFTimingConfig>);
  BOOST_CHECK(std::is_standard_layout_v<ROFIntervalBC>);
  BOOST_CHECK(std::is_trivially_copyable_v<ROFIntervalBC>);
}

BOOST_AUTO_TEST_CASE(BeginEndFollowLayerTimingSignConvention)
{
  // Mirrors o2::its::LayerTiming::getROFStartInBC/getROFEndInBC: start is the
  // ROF's own BC plus delay plus bias; end is start plus the readout length.
  const o2::InteractionRecord origin{100, 0};
  const o2::InteractionRecord rofIR{150, 0}; // 50 BC after origin
  const ROFTimingConfig cfg{/*rofLength*/ 40, /*rofDelay*/ 5, /*rofBias*/ -2, /*rofAddTimeErr*/ 3};

  const auto built = computeROFIntervalBC(rofIR, origin, cfg, 7);
  BOOST_REQUIRE(built.ok());
  BOOST_CHECK_EQUAL(built.interval.begin, 53); // 50 + 5 - 2
  BOOST_CHECK_EQUAL(built.interval.end, 93);   // begin + rofLength
  BOOST_CHECK_EQUAL(built.interval.sourceROF, 7u);
  BOOST_CHECK(built.interval.isValid());
}

BOOST_AUTO_TEST_CASE(NegativeTFRelativeBCIsLegal)
{
  const o2::InteractionRecord origin{200, 0};
  const o2::InteractionRecord rofIR{50, 0}; // 150 BC before origin
  const ROFTimingConfig cfg{40, 0, 0, 0};

  const auto built = computeROFIntervalBC(rofIR, origin, cfg, 0);
  BOOST_REQUIRE(built.ok());
  BOOST_CHECK_EQUAL(built.interval.begin, -150);
  BOOST_CHECK_EQUAL(built.interval.end, -110);
  BOOST_CHECK(built.interval.isValid());
}

BOOST_AUTO_TEST_CASE(InvalidROFLengthIsRejected)
{
  const o2::InteractionRecord origin{0, 0};
  const ROFTimingConfig cfg{0, 0, 0, 0};
  const auto built = computeROFIntervalBC(origin, origin, cfg, 0);
  BOOST_CHECK(!built.ok());
  BOOST_CHECK(built.error == TimingBuildError::InvalidROFLength);
}

BOOST_AUTO_TEST_CASE(OverflowIsDetectedAndChecked)
{
  const o2::InteractionRecord origin{0, 0};
  const o2::InteractionRecord rofIR{0, 0};
  ROFTimingConfig cfg{1, std::numeric_limits<TFBC>::max(), 1, 0};
  const auto built = computeROFIntervalBC(rofIR, origin, cfg, 0);
  BOOST_CHECK(!built.ok());
  BOOST_CHECK(built.error == TimingBuildError::Overflow);
}

BOOST_AUTO_TEST_CASE(InvalidSourceROFIsRejected)
{
  const o2::InteractionRecord origin{0, 0};
  const ROFTimingConfig cfg{40, 0, 0, 0};
  const auto built = computeROFIntervalBC(origin, origin, cfg, std::numeric_limits<uint32_t>::max());
  BOOST_CHECK(!built.ok());
  BOOST_CHECK(built.error == TimingBuildError::InvalidSourceROF);
}

BOOST_AUTO_TEST_CASE(WidenAppliesSymmetricMarginOnDemandOnly)
{
  const ROFIntervalBC interval{10, 20, 3, 0};
  const auto widened = widen(interval, 5);
  BOOST_REQUIRE(widened.ok());
  BOOST_CHECK_EQUAL(widened.interval.begin, 5);
  BOOST_CHECK_EQUAL(widened.interval.end, 25);
  BOOST_CHECK_EQUAL(widened.interval.sourceROF, 3u);
  // The base interval itself is never mutated by widen().
  BOOST_CHECK_EQUAL(interval.begin, 10);
  BOOST_CHECK_EQUAL(interval.end, 20);
}

BOOST_AUTO_TEST_CASE(WidenRejectsInvalidInterval)
{
  constexpr ROFIntervalBC invalidInterval{};
  const auto widened = widen(invalidInterval, 5);
  BOOST_CHECK(!widened.ok());
  BOOST_CHECK(widened.error == WidenError::InvalidInterval);
}

BOOST_AUTO_TEST_CASE(WidenRejectsNegativeMargin)
{
  const ROFIntervalBC interval{10, 20, 3, 0};
  const auto widened = widen(interval, -1);
  BOOST_CHECK(!widened.ok());
  BOOST_CHECK(widened.error == WidenError::InvalidMargin);
}

BOOST_AUTO_TEST_CASE(WidenDetectsLowerBoundOverflow)
{
  const ROFIntervalBC interval{std::numeric_limits<TFBC>::min() + 5, 20, 3, 0};
  const auto widened = widen(interval, 10);
  BOOST_CHECK(!widened.ok());
  BOOST_CHECK(widened.error == WidenError::LowerBoundOverflow);
}

BOOST_AUTO_TEST_CASE(WidenDetectsUpperBoundOverflow)
{
  const ROFIntervalBC interval{10, std::numeric_limits<TFBC>::max() - 5, 3, 0};
  const auto widened = widen(interval, 10);
  BOOST_CHECK(!widened.ok());
  BOOST_CHECK(widened.error == WidenError::UpperBoundOverflow);
}

BOOST_AUTO_TEST_CASE(EmptyInputIsNotUniform)
{
  const auto result = deriveUniformROFTimingConfig({});
  BOOST_CHECK(!result.uniform);
}

BOOST_AUTO_TEST_CASE(SingleLayerIsTriviallyUniform)
{
  const std::array<LayerTiming, 1> layers{LayerTiming{.mNROFsTF = 10, .mROFLength = 40, .mROFDelay = 5, .mROFBias = 2, .mROFAddTimeErr = 1}};
  const auto result = deriveUniformROFTimingConfig(layers);
  BOOST_REQUIRE(result.uniform);
  BOOST_CHECK_EQUAL(result.config.rofLength, 40);
  BOOST_CHECK_EQUAL(result.config.rofDelay, 5);
  BOOST_CHECK_EQUAL(result.config.rofBias, 2);
  BOOST_CHECK_EQUAL(result.config.rofAddTimeErr, 1);
}

BOOST_AUTO_TEST_CASE(MatchedPerLayerValuesAreUniform)
{
  // Mirrors real ITS/MFT production defaults: every layer resolves to the
  // same shared global value because DPLAlpideParam's per-layer staggering
  // overrides all default to zero.
  std::array<LayerTiming, 7> layers{};
  for (auto& lt : layers) {
    lt = LayerTiming{.mNROFsTF = 100, .mROFLength = 594, .mROFDelay = 0, .mROFBias = 64, .mROFAddTimeErr = 0};
  }
  const auto result = deriveUniformROFTimingConfig(layers);
  BOOST_REQUIRE(result.uniform);
  BOOST_CHECK_EQUAL(result.config.rofLength, 594);
  BOOST_CHECK_EQUAL(result.config.rofBias, 64);
}

BOOST_AUTO_TEST_CASE(DivergentROFLengthIsRejected)
{
  std::array<LayerTiming, 3> layers{
    LayerTiming{.mROFLength = 40, .mROFDelay = 0, .mROFBias = 0, .mROFAddTimeErr = 0},
    LayerTiming{.mROFLength = 40, .mROFDelay = 0, .mROFBias = 0, .mROFAddTimeErr = 0},
    LayerTiming{.mROFLength = 44, .mROFDelay = 0, .mROFBias = 0, .mROFAddTimeErr = 0}}; // staggered length
  BOOST_CHECK(!deriveUniformROFTimingConfig(layers).uniform);
}

BOOST_AUTO_TEST_CASE(DivergentROFDelayIsRejected)
{
  std::array<LayerTiming, 2> layers{
    LayerTiming{.mROFLength = 40, .mROFDelay = 0, .mROFBias = 0, .mROFAddTimeErr = 0},
    LayerTiming{.mROFLength = 40, .mROFDelay = 3, .mROFBias = 0, .mROFAddTimeErr = 0}};
  BOOST_CHECK(!deriveUniformROFTimingConfig(layers).uniform);
}

BOOST_AUTO_TEST_CASE(DivergentROFBiasIsRejected)
{
  std::array<LayerTiming, 2> layers{
    LayerTiming{.mROFLength = 40, .mROFDelay = 0, .mROFBias = 64, .mROFAddTimeErr = 0},
    LayerTiming{.mROFLength = 40, .mROFDelay = 0, .mROFBias = 60, .mROFAddTimeErr = 0}};
  BOOST_CHECK(!deriveUniformROFTimingConfig(layers).uniform);
}

BOOST_AUTO_TEST_CASE(DivergentAddTimeErrIsRejected)
{
  std::array<LayerTiming, 2> layers{
    LayerTiming{.mROFLength = 40, .mROFDelay = 0, .mROFBias = 0, .mROFAddTimeErr = 0},
    LayerTiming{.mROFLength = 40, .mROFDelay = 0, .mROFBias = 0, .mROFAddTimeErr = 5}};
  BOOST_CHECK(!deriveUniformROFTimingConfig(layers).uniform);
}

BOOST_AUTO_TEST_CASE(IntersectionIsHalfOpenAndIgnoresROFOrdinal)
{
  const ROFIntervalBC a{0, 10, 5, 0};
  const ROFIntervalBC touching{10, 20, 5, 0};    // same sourceROF as `a`, adjacent
  const ROFIntervalBC overlapping{9, 15, 99, 0}; // different sourceROF, overlaps
  const ROFIntervalBC disjoint{100, 110, 5, 0};

  BOOST_CHECK(!intersects(a, touching));   // half-open: touching does not intersect
  BOOST_CHECK(intersects(a, overlapping)); // overlap decided by time, not ROF equality
  BOOST_CHECK(!intersects(a, disjoint));
  BOOST_CHECK(intersects(a, a));
}

BOOST_AUTO_TEST_CASE(InvalidIntervalsNeverIntersect)
{
  constexpr ROFIntervalBC invalidInterval{};
  const ROFIntervalBC valid{0, 10, 0, 0};
  const ROFIntervalBC invalidSourceROF{0, 10, std::numeric_limits<uint32_t>::max(), 0};
  const ROFIntervalBC zeroLength{5, 5, 0, 0};
  const ROFIntervalBC reversed{10, 5, 0, 0};

  BOOST_CHECK(!intersects(invalidInterval, valid));
  BOOST_CHECK(!intersects(valid, invalidInterval));
  BOOST_CHECK(!intersects(invalidSourceROF, valid));
  BOOST_CHECK(!intersects(zeroLength, valid));
  BOOST_CHECK(!intersects(reversed, valid));
}

BOOST_AUTO_TEST_CASE(DefaultIntervalIsInvalidSentinel)
{
  constexpr ROFIntervalBC interval{};
  BOOST_CHECK_EQUAL(interval.sourceROF, std::numeric_limits<uint32_t>::max());
  BOOST_CHECK(!interval.isValid()); // sourceROF is the sentinel and begin == end
  BOOST_CHECK_EQUAL(interval.length(), 0);
}

BOOST_AUTO_TEST_CASE(ZeroLengthAndReversedIntervalsAreInvalid)
{
  constexpr ROFIntervalBC zeroLength{5, 5, 0, 0};
  constexpr ROFIntervalBC reversed{10, 5, 0, 0};
  BOOST_CHECK(!zeroLength.isValid());
  BOOST_CHECK(!reversed.isValid());
}

BOOST_AUTO_TEST_CASE(IntervalWithInvalidSourceROFIsInvalidEvenWithPositiveExtent)
{
  constexpr ROFIntervalBC interval{0, 10, std::numeric_limits<uint32_t>::max(), 0};
  BOOST_CHECK(!interval.isValid());
}

BOOST_AUTO_TEST_CASE(LengthIsSafeForExtremeSignedRange)
{
  // Signed `end - begin` overflows TFBC here (INT64_MAX - INT64_MIN does not
  // fit in int64_t); length() must still return the exact, correct distance
  // via unsigned arithmetic instead of invoking signed-overflow UB.
  constexpr ROFIntervalBC interval{std::numeric_limits<TFBC>::min(), std::numeric_limits<TFBC>::max(), 0, 0};
  BOOST_CHECK(interval.isValid());
  BOOST_CHECK_EQUAL(interval.length(), std::numeric_limits<uint64_t>::max());
}
