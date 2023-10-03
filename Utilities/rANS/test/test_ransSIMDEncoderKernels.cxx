// Copyright 2019-2023 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   test_ransSIMDEncoderKernels.h
/// @author Michael Lettrich
/// @brief  Test rANS SIMD encoder/ decoder kernels

#define BOOST_TEST_MODULE Utility test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>

#include "rANS/internal/common/defines.h"

#ifdef RANS_SIMD

#include <vector>
#include <type_traits>

#include "rANS/internal/encode/simdKernel.h"
#include "rANS/internal/common/typetraits.h"
#include "rANS/internal/common/defaults.h"

using namespace o2::rans::internal::simd;
using namespace o2::rans::internal;
using namespace o2::rans::utils;

// clang-format off
using pd_types = boost::mpl::list<pd_t<SIMDWidth::SSE>
#ifdef RANS_AVX2
                                      , pd_t<SIMDWidth::AVX>
#endif /* RANS_AVX2 */
                                      >;

using epi64_types = boost::mpl::list<epi64_t<SIMDWidth::SSE>
#ifdef RANS_AVX2
                                          , epi64_t<SIMDWidth::AVX>
#endif /* RANS_AVX2 */
                                          >;

using epi32_types = boost::mpl::list<epi32_t<SIMDWidth::SSE>
#ifdef RANS_AVX2
                                          , epi32_t<SIMDWidth::AVX>
#endif /* RANS_AVX2 */
                                          >;
// clang-format on

struct RANSEncodeFixture {

  uint64_t mState{};
  double mNormalization{};
  std::vector<double> mFrequency{};
  std::vector<double> mCumulative{};
  std::vector<uint64_t> mResultState{};

  RANSEncodeFixture()
  {
    const uint64_t state = 1ul << 21;
    const std::vector<uint32_t> frequency{1, 1, 997, 1234};
    const std::vector<uint32_t> cumulative{0, 321, 1, (1u << 16) - 1234};
    const uint64_t normalization = 1ul << 16;

    // copy and convert to double
    mState = static_cast<double>(state);
    mNormalization = static_cast<double>(normalization);
    std::copy(std::begin(frequency), std::end(frequency), std::back_inserter(mFrequency));
    std::copy(std::begin(cumulative), std::end(cumulative), std::back_inserter(mCumulative));

    // calculate result based on RANS formula
    for (size_t i = 0; i < frequency.size(); ++i) {
      uint64_t resultState = normalization * (state / frequency[i]) + (state % frequency[i]) + cumulative[i];
      mResultState.push_back(resultState);
    }
  };
};

BOOST_FIXTURE_TEST_SUITE(testRANSEncode, RANSEncodeFixture)

BOOST_AUTO_TEST_CASE_TEMPLATE(simd_RansEncode, pd_T, pd_types)
{
  using epi64_T = epi64_t<simdWidth_v<pd_T>>;

  const size_t nTests = mFrequency.size();

  for (size_t i = 0; i < nTests; ++i) {
    const epi64_T state{mState};
    const pd_T frequencyPD{mFrequency[i]};
    const pd_T cumulativePD{mCumulative[i]};
    const pd_T normalizationPD{mNormalization};
    epi64_T result{0};

    result = store<uint64_t>(ransEncode(load(state), load(frequencyPD), load(cumulativePD), load(normalizationPD)));

    epi64_T correctStateVector{mResultState[i]};

    BOOST_CHECK_EQUAL_COLLECTIONS(gsl::make_span(correctStateVector).begin(), gsl::make_span(correctStateVector).end(), gsl::make_span(result).begin(), gsl::make_span(result).end());
  }
}
BOOST_AUTO_TEST_SUITE_END()

struct AosToSoaFixture {

  std::vector<Symbol> mSource;
  epi32_t<SIMDWidth::AVX> mFrequencies;
  epi32_t<SIMDWidth::AVX> mCumulative;

  AosToSoaFixture()
  {
    constexpr size_t nElems = getElementCount<uint32_t>(SIMDWidth::AVX);
    uint32_t counter = 0;

    for (size_t i = 0; i < nElems; ++i) {
      const auto freq = counter++;
      const auto cumul = counter++;
      Symbol symbol{freq, cumul, 0};
      mFrequencies(i) = symbol.getFrequency();
      mCumulative(i) = symbol.getCumulative();

      mSource.emplace_back(std::move(symbol));
    }
  };
};
using aosToSoa_T = boost::mpl::list<std::integral_constant<size_t, 2>,
                                    std::integral_constant<size_t, 4>>;

BOOST_FIXTURE_TEST_SUITE(testAostoSoa, AosToSoaFixture)
BOOST_AUTO_TEST_CASE_TEMPLATE(simd_AosToSOA, sizes_T, aosToSoa_T)
{
  constexpr sizes_T nElements;
  std::array<const o2::rans::internal::Symbol*, nElements()> aosPtrs{};
  for (size_t i = 0; i < nElements(); ++i) {
    aosPtrs[i] = &mSource[i];
  }

  UnrolledSymbols u;
  aosToSoa(aosPtrs, &u.frequencies[0], &u.cumulativeFrequencies[0]);

  auto frequencies = store<uint32_t>(u.frequencies[0]);
  auto cumulative = store<uint32_t>(u.cumulativeFrequencies[0]);

  for (size_t i = 0; i < nElements(); ++i) {
    BOOST_CHECK_EQUAL(frequencies(i), mFrequencies(i));
    BOOST_CHECK_EQUAL(cumulative(i), mCumulative(i));
  };
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(testcmpge)

BOOST_AUTO_TEST_CASE_TEMPLATE(simd_cmpgeq_epi64, epi64_T, epi64_types)
{
  epi64_T a{0};
  epi64_T b{1};
  epi64_T res{0x0};
  epi64_T res1 = store<uint64_t>(cmpgeq_epi64(load(a), load(b)));
  BOOST_CHECK_EQUAL_COLLECTIONS(gsl::make_span(res1).begin(), gsl::make_span(res1).end(), gsl::make_span(res).begin(), gsl::make_span(res).end());

  a = epi64_T{1};
  b = epi64_T{1};
  res = epi64_T{0xFFFFFFFFFFFFFFFF};
  res1 = store<uint64_t>(cmpgeq_epi64(load(a), load(b)));
  BOOST_CHECK_EQUAL_COLLECTIONS(gsl::make_span(res1).begin(), gsl::make_span(res1).end(), gsl::make_span(res).begin(), gsl::make_span(res).end());

  a = epi64_T{1};
  b = epi64_T{0};
  res = epi64_T{0xFFFFFFFFFFFFFFFF};
  res1 = store<uint64_t>(cmpgeq_epi64(load(a), load(b)));
  BOOST_CHECK_EQUAL_COLLECTIONS(gsl::make_span(res1).begin(), gsl::make_span(res1).end(), gsl::make_span(res).begin(), gsl::make_span(res).end());
}

BOOST_AUTO_TEST_SUITE_END()

struct SSERenormFixture {
  using count_t = o2::rans::count_t;
  using ransState_t = uint64_t;
  using stream_t = uint32_t;

  SSERenormFixture() = default;

  static constexpr size_t LowerBoundBits = o2::rans::defaults::internal::RenormingLowerBound;
  static constexpr size_t LowerBound = pow2(LowerBoundBits);
  static constexpr size_t SymbolTablePrecisionBits = 16;
  static constexpr size_t StreamBits = o2::rans::utils::toBits<stream_t>();

  uint64_t computeLimitState(count_t frequency)
  {
    return (LowerBound >> SymbolTablePrecisionBits << StreamBits) * static_cast<uint64_t>(frequency);
  };

  template <typename stream_IT>
  inline auto renorm(ransState_t state, stream_IT outputIter, count_t frequency)
  {
    ransState_t maxState = ((LowerBound >> SymbolTablePrecisionBits) << StreamBits) * frequency;
    if (state >= maxState) {
      *outputIter = static_cast<stream_t>(state);
      ++outputIter;
      state >>= StreamBits;
      assert(state < maxState);
    }
    return std::make_tuple(state, outputIter);
  };
  void runRenormingChecksSSE(const epi64_t<SIMDWidth::SSE, 2>& states, const epi32_t<SIMDWidth::SSE>& compactfrequencies)
  {
    const size_t nElems = getElementCount<ransState_t>(SIMDWidth::SSE) * 2;

    std::vector<stream_t> streamOutBuffer = std::vector<stream_t>(nElems, 0);
    std::vector<stream_t> controlBuffer = std::vector<stream_t>(nElems, 0);

    using stream_iterator = decltype(streamOutBuffer.begin());

    epi32_t<SIMDWidth::SSE, 2> frequencies{compactfrequencies(0), compactfrequencies(1), 0x0u, 0x0u, compactfrequencies(2), compactfrequencies(3), 0x0u, 0x0u};

    __m128i frequenciesVec[2];
    __m128i statesVec[2];
    __m128i newStatesVec[2];

    frequenciesVec[0] = load(frequencies[0]);
    frequenciesVec[1] = load(frequencies[1]);

    statesVec[0] = load(states[0]);
    statesVec[1] = load(states[1]);

    [[maybe_unused]] stream_iterator newstreamOutIter = ransRenorm<stream_iterator, LowerBound, StreamBits>(statesVec,
                                                                                                            frequenciesVec,
                                                                                                            SymbolTablePrecisionBits,
                                                                                                            streamOutBuffer.begin(), newStatesVec);

    epi64_t<SIMDWidth::SSE, 2> newStates(0);
    store(newStatesVec[0], newStates[0]);
    store(newStatesVec[1], newStates[1]);

    auto controlIter = controlBuffer.begin();
    epi64_t<SIMDWidth::SSE, 2> controlStates;
    for (size_t i = nElems; i-- > 0;) {
      std::tie(controlStates(i), controlIter) = renorm(states(i), controlIter, compactfrequencies(i));
    }
    for (size_t i = 0; i < nElems; ++i) {
      LOG(trace) << fmt::format("[{}]: {:#0x}; {:#0x}", i, streamOutBuffer[i], controlBuffer[i]);
    }

    BOOST_CHECK_EQUAL_COLLECTIONS(gsl::make_span(newStates).begin(), gsl::make_span(newStates).end(), gsl::make_span(controlStates).begin(), gsl::make_span(controlStates).end());
    BOOST_CHECK_EQUAL_COLLECTIONS(streamOutBuffer.begin(), streamOutBuffer.end(), controlBuffer.begin(), controlBuffer.end());
  }
};

BOOST_FIXTURE_TEST_SUITE(SSErenorm, SSERenormFixture)

BOOST_AUTO_TEST_CASE(renormSSE_0000)
{
  runRenormingChecksSSE({LowerBound, LowerBound, LowerBound, LowerBound}, {0x1u, 0x1u, 0x1u, 0x1u});
}
BOOST_AUTO_TEST_CASE(renormSSE_0001)
{
  epi32_t<SIMDWidth::SSE> frequencies{0x1u, 0x1u, 0x1u, 0x5u};
  runRenormingChecksSSE({LowerBound,
                         LowerBound,
                         LowerBound,
                         computeLimitState(frequencies(3)) + 0xF5},
                        frequencies);
}
BOOST_AUTO_TEST_CASE(renormSSE_0010)
{
  epi32_t<SIMDWidth::SSE> frequencies{0x1u, 0x1u, 0x4u, 0x1u};
  runRenormingChecksSSE({LowerBound,
                         LowerBound,
                         computeLimitState(frequencies(2)) + 0xF4,
                         LowerBound},
                        frequencies);
}
BOOST_AUTO_TEST_CASE(renormSSE_0011)
{
  epi32_t<SIMDWidth::SSE> frequencies{0x1u, 0x1u, 0x4u, 0x5u};
  runRenormingChecksSSE({LowerBound,
                         LowerBound,
                         computeLimitState(frequencies(2)) + 0xF4,
                         computeLimitState(frequencies(3)) + 0xF5},
                        frequencies);
}
BOOST_AUTO_TEST_CASE(renormSSE_0100)
{
  epi32_t<SIMDWidth::SSE> frequencies{0x1u, 0x3u, 0x1u, 0x1u};
  runRenormingChecksSSE({LowerBound,
                         computeLimitState(frequencies(1)) + 0xF3,
                         LowerBound,
                         LowerBound},
                        frequencies);
}
BOOST_AUTO_TEST_CASE(renormSSE_0101)
{
  epi32_t<SIMDWidth::SSE> frequencies{0x1u, 0x3u, 0x1u, 0x5u};
  runRenormingChecksSSE({LowerBound,
                         computeLimitState(frequencies(1)) + 0xF3,
                         LowerBound,
                         computeLimitState(frequencies(3)) + 0xF5},
                        frequencies);
}
BOOST_AUTO_TEST_CASE(renormSSE_0110)
{
  epi32_t<SIMDWidth::SSE> frequencies{0x1u, 0x3u, 0x4u, 0x1u};
  runRenormingChecksSSE({LowerBound,
                         computeLimitState(frequencies(1)) + 0xF3,
                         computeLimitState(frequencies(2)) + 0xF4,
                         LowerBound},
                        frequencies);
}
BOOST_AUTO_TEST_CASE(renormSSE_0111)
{
  epi32_t<SIMDWidth::SSE> frequencies{0x1u, 0x3u, 0x4u, 0x5u};
  runRenormingChecksSSE({LowerBound,
                         computeLimitState(frequencies(1)) + 0xF3,
                         computeLimitState(frequencies(2)) + 0xF4,
                         computeLimitState(frequencies(3)) + 0xF5},
                        frequencies);
}
BOOST_AUTO_TEST_CASE(renormSSE_1000)
{
  epi32_t<SIMDWidth::SSE> frequencies{0x2u, 0x1u, 0x1u, 0x1u};
  runRenormingChecksSSE({computeLimitState(frequencies(0)) + 0xF2,
                         LowerBound,
                         LowerBound,
                         LowerBound},
                        frequencies);
}
BOOST_AUTO_TEST_CASE(renormSSE_1001)
{
  epi32_t<SIMDWidth::SSE> frequencies{0x2u, 0x1u, 0x1u, 0x5u};
  runRenormingChecksSSE({computeLimitState(frequencies(0)) + 0xF2,
                         LowerBound,
                         LowerBound,
                         computeLimitState(frequencies(3)) + 0xF5},
                        frequencies);
}
BOOST_AUTO_TEST_CASE(renormSSE_1010)
{
  epi32_t<SIMDWidth::SSE> frequencies{0x2u, 0x1u, 0x4u, 0x1u};
  runRenormingChecksSSE({computeLimitState(frequencies(0)) + 0xF2,
                         LowerBound,
                         computeLimitState(frequencies(2)) + 0xF4,
                         LowerBound},
                        frequencies);
}
BOOST_AUTO_TEST_CASE(renormSSE_1011)
{
  epi32_t<SIMDWidth::SSE> frequencies{0x2u, 0x1u, 04u, 0x5u};
  runRenormingChecksSSE({computeLimitState(frequencies(0)) + 0xF2,
                         LowerBound,
                         computeLimitState(frequencies(2)) + 0xF4,
                         computeLimitState(frequencies(3)) + 0xF5},
                        frequencies);
}
BOOST_AUTO_TEST_CASE(renormSSE_1100)
{
  epi32_t<SIMDWidth::SSE> frequencies{0x2u, 0x3u, 0x1u, 0x1u};
  runRenormingChecksSSE({computeLimitState(frequencies(0)) + 0xF2,
                         computeLimitState(frequencies(1)) + 0xF3,
                         LowerBound,
                         LowerBound},
                        frequencies);
}
BOOST_AUTO_TEST_CASE(renormSSE_1101)
{
  epi32_t<SIMDWidth::SSE> frequencies{0x2u, 0x3u, 0x1u, 0x5u};
  runRenormingChecksSSE({computeLimitState(frequencies(0)) + 0xF2,
                         computeLimitState(frequencies(1)) + 0xF3,
                         LowerBound,
                         computeLimitState(frequencies(3)) + 0xF5},
                        frequencies);
}
BOOST_AUTO_TEST_CASE(renormSSE_1110)
{
  epi32_t<SIMDWidth::SSE> frequencies{0x2u, 0x3u, 0x4u, 0x1u};
  runRenormingChecksSSE({computeLimitState(frequencies(0)) + 0xF2,
                         computeLimitState(frequencies(1)) + 0xF3,
                         computeLimitState(frequencies(2)) + 0xF4,
                         LowerBound},
                        frequencies);
}
BOOST_AUTO_TEST_CASE(renormSSE_1111)
{
  epi32_t<SIMDWidth::SSE> frequencies{0x2u, 0x3u, 0x4u, 0x5u};
  runRenormingChecksSSE({computeLimitState(frequencies(0)) + 0xF2,
                         computeLimitState(frequencies(1)) + 0xF3,
                         computeLimitState(frequencies(2)) + 0xF4,
                         computeLimitState(frequencies(3)) + 0xF5},
                        frequencies);
}

BOOST_AUTO_TEST_SUITE_END()

#ifndef RANS_AVX2
BOOST_AUTO_TEST_CASE(test_NoAVX2)
{
  BOOST_TEST_WARN("Tests were not Compiled for AVX2, cannot run all tests");
}
#endif

#else /* !defined(RANS_SIMD) */

BOOST_AUTO_TEST_CASE(test_NoSIMD)
{
  BOOST_TEST_WARN("Tests were not Compiled for SIMD, cannot run all tests");
}

#endif /* RANS_SIMD */