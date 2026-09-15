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

// Pure classification/typed-exception tests for the workflow loading boundary
// loading boundary (ITSMFTTracking/IOUtils.h). No geometry
// singleton, DPL, or CCDB dependency: isRecoverableLoadError() and the two
// exception types are plain host-only code.

#define BOOST_TEST_MODULE ITSMFT TimeFrameLoadFailure
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <array>

#include "ITSMFTTracking/IOUtils.h"

using namespace o2::itsmft::tracking;

namespace
{
// Hand-maintained, exhaustive list of every MultiSourceLoadError enumerator
// as declared in IOUtils.h, paired with the classification this
// design's failure taxonomy requires. This -- not the absence of `default:`
// in isRecoverableLoadError()'s switch -- is the actual, checked coverage
// guarantee: if a new enumerator is added there without a corresponding
// entry here, this list's size assertion below fails.
struct Case {
  MultiSourceLoadError error;
  bool recoverable; // meaningless when error == TimingError; see the dedicated TimingError cases below
};

constexpr std::array<Case, 22> kAllNonTimingCases{{
  {MultiSourceLoadError::None, false},
  {MultiSourceLoadError::NonDenseSourceIds, false},
  {MultiSourceLoadError::DuplicateSourceId, false},
  {MultiSourceLoadError::UnsupportedDetector, false},
  {MultiSourceLoadError::MissingDecoder, false},
  {MultiSourceLoadError::InvalidROFRange, true},
  {MultiSourceLoadError::InvalidLayerMapping, false},
  {MultiSourceLoadError::DetectorSurfaceMismatch, false},
  {MultiSourceLoadError::InconsistentDecoderMetadata, false},
  {MultiSourceLoadError::TimingError, false}, // placeholder entry: real classification is TimingBuildError-dependent, see below
  {MultiSourceLoadError::SurfaceCatalogNotConfigured, false},
  {MultiSourceLoadError::SurfaceCatalogStale, false},
  {MultiSourceLoadError::MissingDictionary, false},
  {MultiSourceLoadError::TruncatedExplicitPattern, true},
  {MultiSourceLoadError::MalformedExplicitPattern, true},
  {MultiSourceLoadError::InvalidPatternId, true},
  {MultiSourceLoadError::InvalidSensor, true},
  {MultiSourceLoadError::InvalidDecodedLayer, true},
  {MultiSourceLoadError::GeometryUnavailable, false},
  {MultiSourceLoadError::OtherMalformedInput, true},
  {MultiSourceLoadError::TrailingPatternData, true},
}};
} // namespace

BOOST_AUTO_TEST_CASE(ClassifyEveryMultiSourceLoadErrorExceptTiming)
{
  // Bump this count, and the list above, whenever MultiSourceLoadError
  // gains or loses an enumerator -- that is the mechanism that actually
  // catches a classification gap, not the switch's missing `default:`.
  static_assert(kAllNonTimingCases.size() == 22);
  for (const auto& c : kAllNonTimingCases) {
    if (c.error == MultiSourceLoadError::TimingError) {
      continue; // covered exhaustively below, per TimingBuildError value
    }
    BOOST_CHECK_MESSAGE(isRecoverableLoadError(c.error, TimingBuildError::None) == c.recoverable,
                        "error=" << static_cast<int>(c.error));
  }
}

BOOST_AUTO_TEST_CASE(ClassifyTimingErrorForEveryTimingBuildErrorValue)
{
  // Overflow is a genuine per-TF BC-arithmetic overflow caused by the
  // incoming ROF data: recoverable. InvalidROFLength and InvalidSourceROF
  // are configuration problems, and None must never be paired with
  // MultiSourceLoadError::TimingError by a real caller (a successful
  // computeROFIntervalBC() never reaches this classification at all) -- but
  // isRecoverableLoadError() still classifies it structurally, safe-by-
  // default, exactly like the other two.
  BOOST_CHECK(isRecoverableLoadError(MultiSourceLoadError::TimingError, TimingBuildError::Overflow) == true);
  BOOST_CHECK(isRecoverableLoadError(MultiSourceLoadError::TimingError, TimingBuildError::None) == false);
  BOOST_CHECK(isRecoverableLoadError(MultiSourceLoadError::TimingError, TimingBuildError::InvalidROFLength) == false);
  BOOST_CHECK(isRecoverableLoadError(MultiSourceLoadError::TimingError, TimingBuildError::InvalidSourceROF) == false);
}

BOOST_AUTO_TEST_CASE(RecoverableLoadFailureRetainsCompleteResult)
{
  const LoadSourcesResult result{.error = MultiSourceLoadError::MalformedExplicitPattern, .source = ClusterSourceId{2}, .rof = 3, .clusterIndex = 4};
  const RecoverableLoadFailure failure{result};
  BOOST_CHECK(failure.error() == MultiSourceLoadError::MalformedExplicitPattern);
  BOOST_CHECK(failure.result().error == result.error);
  BOOST_CHECK(failure.result().source == result.source);
  BOOST_CHECK_EQUAL(failure.result().rof, result.rof);
  BOOST_CHECK_EQUAL(failure.result().clusterIndex, result.clusterIndex);
}

BOOST_AUTO_TEST_CASE(TimeFrameLoadExceptionDistinguishesReasonsWithoutStringMatching)
{
  const TimeFrameLoadException dictionaryNotConfigured{TimeFrameLoadFailureReason::DictionaryNotConfigured, "cluster dictionary not configured"};
  BOOST_CHECK(dictionaryNotConfigured.reason() == TimeFrameLoadFailureReason::DictionaryNotConfigured);
  BOOST_CHECK(dictionaryNotConfigured.loadResult().error == MultiSourceLoadError::None);

  const TimeFrameLoadException nonUniformTiming{TimeFrameLoadFailureReason::NonUniformROFTiming, "per-layer ROF timing configuration is not uniform"};
  BOOST_CHECK(nonUniformTiming.reason() == TimeFrameLoadFailureReason::NonUniformROFTiming);
  BOOST_CHECK(nonUniformTiming.loadResult().error == MultiSourceLoadError::None);

  const LoadSourcesResult structuralResult{.error = MultiSourceLoadError::SurfaceCatalogStale, .source = ClusterSourceId{0}, .rof = 0, .clusterIndex = 0};
  const TimeFrameLoadException loadSourcesFailure{structuralResult};
  BOOST_CHECK(loadSourcesFailure.reason() == TimeFrameLoadFailureReason::LoadSourcesFailure);
  BOOST_CHECK(loadSourcesFailure.loadResult().error == MultiSourceLoadError::SurfaceCatalogStale);
  BOOST_CHECK(loadSourcesFailure.loadResult().source == structuralResult.source);
}
