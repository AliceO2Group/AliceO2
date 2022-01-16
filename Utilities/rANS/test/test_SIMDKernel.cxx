// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   test_ransSIMDEncoder.h
/// @author Michael Lettrich
/// @since  2020-04-15
/// @brief  Test rANS SIMD encoder/ decoder

#define BOOST_TEST_MODULE Utility test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#ifdef __SSE__

#include <vector>
#include <type_traits>

#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>

#include "rANS/rans.h"
#include "rANS/internal/backend/simd/types.h"
#include "rANS/internal/backend/simd/kernel.h"
#include "rANS/internal/backend/simd/Symbol.h"

using namespace o2::rans::internal::simd;

// clang-format off
using pd_types = boost::mpl::list<pd_t<SIMDWidth::SSE>
#ifdef __AVX2__
                                      , pd_t<SIMDWidth::AVX>
#endif /* __AVX2__ */
                                      >;

using epi64_types = boost::mpl::list<epi64_t<SIMDWidth::SSE>
#ifdef __AVX2__
                                          , epi64_t<SIMDWidth::AVX>
#endif /* __AVX2__ */
                                          >;

using epi32_types = boost::mpl::list<epi32_t<SIMDWidth::SSE>
#ifdef __AVX2__
                                          , epi32_t<SIMDWidth::AVX>
#endif /* __AVX2__ */
                                          >;
// clang-format on

struct ConvertingFixture64 {
  std::vector<uint64_t> uint64Data = {0x0, 0x1, 0xFFFFFFFFFFFFE, 0xFFFFFFFFFFFFF};
  std::vector<double> doubleData;

  ConvertingFixture64()
  {
    for (auto i : uint64Data) {
      doubleData.push_back(static_cast<double>(i));
    }
  };
};

BOOST_FIXTURE_TEST_SUITE(test_SIMDconvert64, ConvertingFixture64)

BOOST_AUTO_TEST_CASE_TEMPLATE(simd_uint64ToDouble, epi64_T, epi64_types)
{
  for (size_t i = 0; i < uint64Data.size(); ++i) {
    const epi64_T src{uint64Data[i]};
    const auto dest = uint64ToDouble(src);

    for (auto elem : dest) {
      BOOST_CHECK_EQUAL(elem, doubleData[i]);
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(simd_doubleToUint64, pd_T, pd_types)
{
  for (size_t i = 0; i < doubleData.size(); ++i) {
    const pd_T src{doubleData[i]};
    const auto dest = doubleToUint64(src);

    for (auto elem : dest) {
      BOOST_CHECK_EQUAL(elem, uint64Data[i]);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()

struct ConvertingFixture32 {
  epi32_t<SIMDWidth::SSE> uint32Data = {0x0, 0x1, 0x7FFFFFFE, 0x7FFFFFFF};
  std::vector<double> doubleData;

  ConvertingFixture32()
  {
    for (auto i : uint32Data) {
      doubleData.push_back(static_cast<double>(i));
    }
  };
};

BOOST_FIXTURE_TEST_SUITE(test_SIMDconvert32, ConvertingFixture32)

BOOST_AUTO_TEST_CASE_TEMPLATE(simd_int32ToDouble, epi32_T, epi32_types)
{
  constexpr SIMDWidth simdWidth_V = simdWidth_v<epi32_T>;

  for (size_t i = 0; i < uint32Data.size(); ++i) {
    const epi32_t<SIMDWidth::SSE> src{uint32Data[i]};
    auto dest = int32ToDouble<simdWidth_V>(src);

    for (auto elem : dest) {
      BOOST_CHECK_EQUAL(elem, doubleData[i]);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()

struct ModDivFixture {
  std::vector<uint32_t> numerator = {1, 6, 17};
  std::vector<uint32_t> denominator = {1, 3, 4};
  // test 1: mod = 0, div correctly rounded
  // test 2: div = 0, mod correclty rounded
  // test 3: mod, div nonzero and correctly rounded
  std::array<uint32_t, 3> mod;
  std::array<uint32_t, 3> div;

  ModDivFixture()
  {
    for (size_t i = 0; i < numerator.size(); ++i) {
      div[i] = numerator[i] / denominator[i];
      mod[i] = numerator[i] % denominator[i];
    }
  };
};

BOOST_FIXTURE_TEST_SUITE(testModDiv, ModDivFixture)

BOOST_AUTO_TEST_CASE_TEMPLATE(modDiv, pd_T, pd_types)
{
  for (size_t i = 0; i < numerator.size(); ++i) {
    const pd_T numeratorPD{static_cast<double>(numerator[i])};
    const pd_T denominatorPD{static_cast<double>(denominator[i])};

    auto [divPD, modPD] = divMod(numeratorPD, denominatorPD);

    pd_T modResult{static_cast<double>(mod[i])};
    pd_T divResult{static_cast<double>(div[i])};

    BOOST_CHECK_EQUAL_COLLECTIONS(divResult.begin(), divResult.end(), divPD.begin(), divPD.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(modResult.begin(), modResult.end(), modPD.begin(), modPD.end());
  }
}

BOOST_AUTO_TEST_SUITE_END()

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

    //copy and convert to double
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

    result = ransEncode(state, frequencyPD, cumulativePD, normalizationPD);

    epi64_T correctStateVector{mResultState[i]};

    BOOST_CHECK_EQUAL_COLLECTIONS(correctStateVector.begin(), correctStateVector.end(), result.begin(), result.end());
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
      Symbol symbol{counter++, counter++};
      mFrequencies[i] = symbol.getFrequency();
      mCumulative[i] = symbol.getCumulative();

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
  std::array<const o2::rans::internal::simd::Symbol*, nElements()> aosPtrs{};
  for (size_t i = 0; i < nElements(); ++i) {
    aosPtrs[i] = &mSource[i];
  }

  auto [frequencies, cumulative] = aosToSoa(aosPtrs);

  for (size_t i = 0; i < nElements(); ++i) {
    BOOST_CHECK_EQUAL(frequencies[i], mFrequencies[i]);
    BOOST_CHECK_EQUAL(cumulative[i], mCumulative[i]);
  };
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(testcmpge)

BOOST_AUTO_TEST_CASE_TEMPLATE(simd_cmpgeq_epi64, epi64_T, epi64_types)
{
  using namespace o2::rans::internal::simd;

  epi64_T a{0};
  epi64_T b{1};
  epi64_T res{0x0};
  epi64_T res1 = cmpgeq_epi64(toConstSIMDView(a), toConstSIMDView(b));
  BOOST_CHECK_EQUAL_COLLECTIONS(res1.begin(), res1.end(), res.begin(), res.end());

  a = epi64_T{1};
  b = epi64_T{1};
  res = epi64_T{0xFFFFFFFFFFFFFFFF};
  res1 = cmpgeq_epi64(toConstSIMDView(a), toConstSIMDView(b));
  BOOST_CHECK_EQUAL_COLLECTIONS(res1.begin(), res1.end(), res.begin(), res.end());

  a = epi64_T{1};
  b = epi64_T{0};
  res = epi64_T{0xFFFFFFFFFFFFFFFF};
  res1 = cmpgeq_epi64(toConstSIMDView(a), toConstSIMDView(b));
  BOOST_CHECK_EQUAL_COLLECTIONS(res1.begin(), res1.end(), res.begin(), res.end());
}

BOOST_AUTO_TEST_SUITE_END()

struct RenormFixture {

  using count_t = typename o2::rans::count_t;
  using ransState_t = uint64_t;
  using stream_t = uint32_t;

  RenormFixture() = default;

  static constexpr size_t LowerBoundBits = 20;
  static constexpr size_t LowerBound = 1ull << LowerBoundBits;
  static constexpr uint8_t SymbolTablePrecisionBits = 16;
  static constexpr size_t StreamBits = o2::rans::internal::toBits(sizeof(stream_t));

  uint64_t computeLimitState(count_t frequency)
  {
    return (LowerBound >> SymbolTablePrecisionBits << StreamBits) * static_cast<uint64_t>(frequency);
  };

  template <typename stream_IT>
  inline auto renorm(ransState_t state, stream_IT outputIter, count_t frequency)
  {
    ransState_t maxState = ((LowerBound >> SymbolTablePrecisionBits) << StreamBits) * frequency;
    if (state >= maxState) {
      *(++outputIter) = static_cast<stream_t>(state);
      state >>= StreamBits;
      assert(state < maxState);
    }
    return std::make_tuple(state, outputIter);
  };

  template <SIMDWidth width_V>
  void runRenormingChecks(const epi64_t<width_V>& states, const epi32_t<SIMDWidth::SSE>& frequencies)
  {
    using namespace o2::rans::internal::simd;

    const size_t nElems = getElementCount<ransState_t>(width_V);

    std::vector<stream_t> streamOutBuffer = std::vector<stream_t>(nElems, 0);
    std::vector<stream_t> controlBuffer = std::vector<stream_t>(nElems, 0);

    auto [newstreamOutIter, newStates] = ransRenorm<decltype(streamOutBuffer.begin()),
                                                    LowerBound,
                                                    StreamBits>(toConstSIMDView(states),
                                                                toConstSIMDView(frequencies),
                                                                SymbolTablePrecisionBits,
                                                                --streamOutBuffer.begin());

    auto controlIter = --controlBuffer.begin();
    epi64_t<width_V> controlStates;
    for (size_t i = nElems; i-- > 0;) {
      std::tie(controlStates[i], controlIter) = renorm(states[i], controlIter, frequencies[i]);
    }
    LOG(trace) << "newStates" << asHex(newStates);
    LOG(trace) << "controlStates" << asHex(controlStates);
    for (size_t i = 0; i < nElems; ++i) {
      LOG(trace) << fmt::format("[{}]: {:#x}; {:#x}", i, streamOutBuffer[i], controlBuffer[i]);
    }

    BOOST_CHECK_EQUAL_COLLECTIONS(newStates.begin(), newStates.end(), controlStates.begin(), controlStates.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(streamOutBuffer.begin(), streamOutBuffer.end(), controlBuffer.begin(), controlBuffer.end());
  }
};

BOOST_FIXTURE_TEST_SUITE(renorm, RenormFixture)

BOOST_AUTO_TEST_CASE(renormSSE_)
{
  using namespace o2::rans::internal::simd;
  runRenormingChecks<SIMDWidth::SSE>({LowerBound}, {0x1u, 0x1u, 0x0u, 0x0u});
}
BOOST_AUTO_TEST_CASE(renormSSE_01)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x1u, 0x3u, 0x0u, 0x0u};
  runRenormingChecks<SIMDWidth::SSE>({LowerBound, computeLimitState(frequencies[1]) + 0xF3}, frequencies);
}
BOOST_AUTO_TEST_CASE(renormSSE_10)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x2u, 0x1u, 0x0u, 0x0u};
  runRenormingChecks<SIMDWidth::SSE>({computeLimitState(frequencies[0]) + 0xF2, LowerBound}, frequencies);
}
BOOST_AUTO_TEST_CASE(renormSSE_11)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x2u, 0x3u, 0x0u, 0x0u};
  runRenormingChecks<SIMDWidth::SSE>({computeLimitState(frequencies[0]) + 0xF2, computeLimitState(frequencies[1]) + 0xF3}, frequencies);
}

BOOST_AUTO_TEST_CASE(renormAVX_0000)
{
  using namespace o2::rans::internal::simd;
  runRenormingChecks<SIMDWidth::AVX>({LowerBound, LowerBound, LowerBound, LowerBound}, {0x1u, 0x1u, 0x1u, 0x1u});
}
BOOST_AUTO_TEST_CASE(renormAVX_0001)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x1u, 0x1u, 0x1u, 0x5u};
  runRenormingChecks<SIMDWidth::AVX>({LowerBound,
                                      LowerBound,
                                      LowerBound,
                                      computeLimitState(frequencies[3]) + 0xF5},
                                     frequencies);
}
BOOST_AUTO_TEST_CASE(renormAVX_0010)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x1u, 0x1u, 0x4u, 0x1u};
  runRenormingChecks<SIMDWidth::AVX>({LowerBound,
                                      LowerBound,
                                      computeLimitState(frequencies[2]) + 0xF4,
                                      LowerBound},
                                     frequencies);
}
BOOST_AUTO_TEST_CASE(renormAVX_0011)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x1u, 0x1u, 0x4u, 0x5u};
  runRenormingChecks<SIMDWidth::AVX>({LowerBound,
                                      LowerBound,
                                      computeLimitState(frequencies[2]) + 0xF4,
                                      computeLimitState(frequencies[3]) + 0xF5},
                                     frequencies);
}
BOOST_AUTO_TEST_CASE(renormAVX_0100)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x1u, 0x3u, 0x1u, 0x1u};
  runRenormingChecks<SIMDWidth::AVX>({LowerBound,
                                      computeLimitState(frequencies[1]) + 0xF3,
                                      LowerBound,
                                      LowerBound},
                                     frequencies);
}
BOOST_AUTO_TEST_CASE(renormAVX_0101)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x1u, 0x3u, 0x1u, 0x5u};
  runRenormingChecks<SIMDWidth::AVX>({LowerBound,
                                      computeLimitState(frequencies[1]) + 0xF3,
                                      LowerBound,
                                      computeLimitState(frequencies[3]) + 0xF5},
                                     frequencies);
}
BOOST_AUTO_TEST_CASE(renormAVX_0110)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x1u, 0x3u, 0x4u, 0x1u};
  runRenormingChecks<SIMDWidth::AVX>({LowerBound,
                                      computeLimitState(frequencies[1]) + 0xF3,
                                      computeLimitState(frequencies[2]) + 0xF4,
                                      LowerBound},
                                     frequencies);
}
BOOST_AUTO_TEST_CASE(renormAVX_0111)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x1u, 0x3u, 0x4u, 0x5u};
  runRenormingChecks<SIMDWidth::AVX>({LowerBound,
                                      computeLimitState(frequencies[1]) + 0xF3,
                                      computeLimitState(frequencies[2]) + 0xF4,
                                      computeLimitState(frequencies[3]) + 0xF5},
                                     frequencies);
}
BOOST_AUTO_TEST_CASE(renormAVX_1000)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x2u, 0x1u, 0x1u, 0x1u};
  runRenormingChecks<SIMDWidth::AVX>({computeLimitState(frequencies[0]) + 0xF2,
                                      LowerBound,
                                      LowerBound,
                                      LowerBound},
                                     frequencies);
}
BOOST_AUTO_TEST_CASE(renormAVX_1001)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x2u, 0x1u, 0x1u, 0x5u};
  runRenormingChecks<SIMDWidth::AVX>({computeLimitState(frequencies[0]) + 0xF2,
                                      LowerBound,
                                      LowerBound,
                                      computeLimitState(frequencies[3]) + 0xF5},
                                     frequencies);
}
BOOST_AUTO_TEST_CASE(renormAVX_1010)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x2u, 0x1u, 0x4u, 0x1u};
  runRenormingChecks<SIMDWidth::AVX>({computeLimitState(frequencies[0]) + 0xF2,
                                      LowerBound,
                                      computeLimitState(frequencies[2]) + 0xF4,
                                      LowerBound},
                                     frequencies);
}
BOOST_AUTO_TEST_CASE(renormAVX_1011)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x2u, 0x1u, 04u, 0x5u};
  runRenormingChecks<SIMDWidth::AVX>({computeLimitState(frequencies[0]) + 0xF2,
                                      LowerBound,
                                      computeLimitState(frequencies[2]) + 0xF4,
                                      computeLimitState(frequencies[3]) + 0xF5},
                                     frequencies);
}
BOOST_AUTO_TEST_CASE(renormAVX_1100)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x2u, 0x3u, 0x1u, 0x1u};
  runRenormingChecks<SIMDWidth::AVX>({computeLimitState(frequencies[0]) + 0xF2,
                                      computeLimitState(frequencies[1]) + 0xF3,
                                      LowerBound,
                                      LowerBound},
                                     frequencies);
}
BOOST_AUTO_TEST_CASE(renormAVX_1101)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x2u, 0x3u, 0x1u, 0x5u};
  runRenormingChecks<SIMDWidth::AVX>({computeLimitState(frequencies[0]) + 0xF2,
                                      computeLimitState(frequencies[1]) + 0xF3,
                                      LowerBound,
                                      computeLimitState(frequencies[3]) + 0xF5},
                                     frequencies);
}
BOOST_AUTO_TEST_CASE(renormAVX_1110)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x2u, 0x3u, 0x4u, 0x1u};
  runRenormingChecks<SIMDWidth::AVX>({computeLimitState(frequencies[0]) + 0xF2,
                                      computeLimitState(frequencies[1]) + 0xF3,
                                      computeLimitState(frequencies[2]) + 0xF4,
                                      LowerBound},
                                     frequencies);
}
BOOST_AUTO_TEST_CASE(renormAVX_1111)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x2u, 0x3u, 0x4u, 0x5u};
  runRenormingChecks<SIMDWidth::AVX>({computeLimitState(frequencies[0]) + 0xF2,
                                      computeLimitState(frequencies[1]) + 0xF3,
                                      computeLimitState(frequencies[2]) + 0xF4,
                                      computeLimitState(frequencies[3]) + 0xF5},
                                     frequencies);
}

BOOST_AUTO_TEST_SUITE_END()

struct SSEDoubleRenormFixture {

  using count_t = typename o2::rans::count_t;
  using ransState_t = uint64_t;
  using stream_t = uint32_t;

  SSEDoubleRenormFixture() = default;

  static constexpr size_t LowerBoundBits = 20;
  static constexpr size_t LowerBound = 1ull << LowerBoundBits;
  static constexpr uint8_t SymbolTablePrecisionBits = 16;
  static constexpr size_t StreamBits = o2::rans::internal::toBits(sizeof(stream_t));

  uint64_t computeLimitState(count_t frequency)
  {
    return (LowerBound >> SymbolTablePrecisionBits << StreamBits) * static_cast<uint64_t>(frequency);
  };

  template <typename stream_IT>
  inline auto renorm(ransState_t state, stream_IT outputIter, count_t frequency)
  {
    ransState_t maxState = ((LowerBound >> SymbolTablePrecisionBits) << StreamBits) * frequency;
    if (state >= maxState) {
      *(++outputIter) = static_cast<stream_t>(state);
      state >>= StreamBits;
      assert(state < maxState);
    }
    return std::make_tuple(state, outputIter);
  };
  void runDoubleRenormingChecksSSE(const epi64_t<SIMDWidth::SSE, 2>& states, const epi32_t<SIMDWidth::SSE>& compactfrequencies)
  {
    using namespace o2::rans::internal::simd;

    const size_t nElems = getElementCount<ransState_t>(SIMDWidth::SSE) * 2;

    std::vector<stream_t> streamOutBuffer = std::vector<stream_t>(nElems, 0);
    std::vector<stream_t> controlBuffer = std::vector<stream_t>(nElems, 0);

    epi32_t<SIMDWidth::SSE, 2> frequencies{compactfrequencies[0], compactfrequencies[1], 0x0u, 0x0u, compactfrequencies[2], compactfrequencies[3], 0x0u, 0x0u};
    auto [newstreamOutIter, newStates] = ransRenorm<decltype(streamOutBuffer.begin()),
                                                    LowerBound,
                                                    StreamBits>(toConstSIMDView(states),
                                                                toConstSIMDView(frequencies),
                                                                SymbolTablePrecisionBits,
                                                                --streamOutBuffer.begin());

    auto controlIter = --controlBuffer.begin();
    epi64_t<SIMDWidth::SSE, 2> controlStates;
    for (size_t i = nElems; i-- > 0;) {
      std::tie(controlStates[i], controlIter) = renorm(states[i], controlIter, compactfrequencies[i]);
    }
    LOG(trace) << "newStates" << asHex(newStates);
    LOG(trace) << "controlStates" << asHex(controlStates);
    for (size_t i = 0; i < nElems; ++i) {
      LOG(trace) << fmt::format("[{}]: {:#x}; {:#x}", i, streamOutBuffer[i], controlBuffer[i]);
    }

    BOOST_CHECK_EQUAL_COLLECTIONS(newStates.begin(), newStates.end(), controlStates.begin(), controlStates.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(streamOutBuffer.begin(), streamOutBuffer.end(), controlBuffer.begin(), controlBuffer.end());
  }
};

BOOST_FIXTURE_TEST_SUITE(SSEdoubleRenorm, SSEDoubleRenormFixture)

BOOST_AUTO_TEST_CASE(doubleRenormSSE_0000)
{
  using namespace o2::rans::internal::simd;
  runDoubleRenormingChecksSSE({LowerBound, LowerBound, LowerBound, LowerBound}, {0x1u, 0x1u, 0x1u, 0x1u});
}
BOOST_AUTO_TEST_CASE(doubleRenormSSE_0001)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x1u, 0x1u, 0x1u, 0x5u};
  runDoubleRenormingChecksSSE({LowerBound,
                               LowerBound,
                               LowerBound,
                               computeLimitState(frequencies[3]) + 0xF5},
                              frequencies);
}
BOOST_AUTO_TEST_CASE(doubleRenormSSE_0010)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x1u, 0x1u, 0x4u, 0x1u};
  runDoubleRenormingChecksSSE({LowerBound,
                               LowerBound,
                               computeLimitState(frequencies[2]) + 0xF4,
                               LowerBound},
                              frequencies);
}
BOOST_AUTO_TEST_CASE(doubleRenormSSE_0011)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x1u, 0x1u, 0x4u, 0x5u};
  runDoubleRenormingChecksSSE({LowerBound,
                               LowerBound,
                               computeLimitState(frequencies[2]) + 0xF4,
                               computeLimitState(frequencies[3]) + 0xF5},
                              frequencies);
}
BOOST_AUTO_TEST_CASE(doubleRenormSSE_0100)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x1u, 0x3u, 0x1u, 0x1u};
  runDoubleRenormingChecksSSE({LowerBound,
                               computeLimitState(frequencies[1]) + 0xF3,
                               LowerBound,
                               LowerBound},
                              frequencies);
}
BOOST_AUTO_TEST_CASE(doubleRenormSSE_0101)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x1u, 0x3u, 0x1u, 0x5u};
  runDoubleRenormingChecksSSE({LowerBound,
                               computeLimitState(frequencies[1]) + 0xF3,
                               LowerBound,
                               computeLimitState(frequencies[3]) + 0xF5},
                              frequencies);
}
BOOST_AUTO_TEST_CASE(doubleRenormSSE_0110)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x1u, 0x3u, 0x4u, 0x1u};
  runDoubleRenormingChecksSSE({LowerBound,
                               computeLimitState(frequencies[1]) + 0xF3,
                               computeLimitState(frequencies[2]) + 0xF4,
                               LowerBound},
                              frequencies);
}
BOOST_AUTO_TEST_CASE(doubleRenormSSE_0111)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x1u, 0x3u, 0x4u, 0x5u};
  runDoubleRenormingChecksSSE({LowerBound,
                               computeLimitState(frequencies[1]) + 0xF3,
                               computeLimitState(frequencies[2]) + 0xF4,
                               computeLimitState(frequencies[3]) + 0xF5},
                              frequencies);
}
BOOST_AUTO_TEST_CASE(doubleRenormSSE_1000)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x2u, 0x1u, 0x1u, 0x1u};
  runDoubleRenormingChecksSSE({computeLimitState(frequencies[0]) + 0xF2,
                               LowerBound,
                               LowerBound,
                               LowerBound},
                              frequencies);
}
BOOST_AUTO_TEST_CASE(doubleRenormSSE_1001)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x2u, 0x1u, 0x1u, 0x5u};
  runDoubleRenormingChecksSSE({computeLimitState(frequencies[0]) + 0xF2,
                               LowerBound,
                               LowerBound,
                               computeLimitState(frequencies[3]) + 0xF5},
                              frequencies);
}
BOOST_AUTO_TEST_CASE(doubleRenormSSE_1010)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x2u, 0x1u, 0x4u, 0x1u};
  runDoubleRenormingChecksSSE({computeLimitState(frequencies[0]) + 0xF2,
                               LowerBound,
                               computeLimitState(frequencies[2]) + 0xF4,
                               LowerBound},
                              frequencies);
}
BOOST_AUTO_TEST_CASE(doubleRenormSSE_1011)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x2u, 0x1u, 04u, 0x5u};
  runDoubleRenormingChecksSSE({computeLimitState(frequencies[0]) + 0xF2,
                               LowerBound,
                               computeLimitState(frequencies[2]) + 0xF4,
                               computeLimitState(frequencies[3]) + 0xF5},
                              frequencies);
}
BOOST_AUTO_TEST_CASE(doubleRenormSSE_1100)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x2u, 0x3u, 0x1u, 0x1u};
  runDoubleRenormingChecksSSE({computeLimitState(frequencies[0]) + 0xF2,
                               computeLimitState(frequencies[1]) + 0xF3,
                               LowerBound,
                               LowerBound},
                              frequencies);
}
BOOST_AUTO_TEST_CASE(doubleRenormSSE_1101)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x2u, 0x3u, 0x1u, 0x5u};
  runDoubleRenormingChecksSSE({computeLimitState(frequencies[0]) + 0xF2,
                               computeLimitState(frequencies[1]) + 0xF3,
                               LowerBound,
                               computeLimitState(frequencies[3]) + 0xF5},
                              frequencies);
}
BOOST_AUTO_TEST_CASE(doubleRenormSSE_1110)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x2u, 0x3u, 0x4u, 0x1u};
  runDoubleRenormingChecksSSE({computeLimitState(frequencies[0]) + 0xF2,
                               computeLimitState(frequencies[1]) + 0xF3,
                               computeLimitState(frequencies[2]) + 0xF4,
                               LowerBound},
                              frequencies);
}
BOOST_AUTO_TEST_CASE(doubleRenormSSE_1111)
{
  using namespace o2::rans::internal::simd;
  epi32_t<SIMDWidth::SSE> frequencies{0x2u, 0x3u, 0x4u, 0x5u};
  runDoubleRenormingChecksSSE({computeLimitState(frequencies[0]) + 0xF2,
                               computeLimitState(frequencies[1]) + 0xF3,
                               computeLimitState(frequencies[2]) + 0xF4,
                               computeLimitState(frequencies[3]) + 0xF5},
                              frequencies);
}

BOOST_AUTO_TEST_SUITE_END()

#endif /* __SSE__ */