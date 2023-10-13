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

/// @file   bench_ransStreaming.cxx
/// @author Michael Lettrich
/// @brief benchmarks streaming out data from rANS state to memory

#include "rANS/internal/common/defines.h"

#include <vector>
#include <cstring>
#include <random>
#include <algorithm>
#ifdef RANS_PARALLEL_STL
#include <execution>
#endif
#include <iterator>

#include <benchmark/benchmark.h>

#include "rANS/factory.h"
#include "rANS/histogram.h"

#include "rANS/internal/common/utils.h"
#include "rANS/internal/common/simdtypes.h"
#include "rANS/internal/common/simdops.h"
#include "rANS/internal/encode/simdKernel.h"

#ifdef ENABLE_VTUNE_PROFILER
#include <ittnotify.h>
#endif

using count_t = uint32_t;
using ransState_t = uint64_t;
using stream_t = uint32_t;

using namespace o2::rans;
using namespace o2::rans::internal;
using namespace o2::rans::utils;

inline constexpr size_t MessageSize = 1ull << 22;
inline constexpr size_t LowerBound = 1ul << 20;
inline constexpr size_t StreamBits = toBits<stream_t>();

template <typename source_T>
class RenormingData
{
 public:
  explicit RenormingData(size_t messageSize)
  {
    std::mt19937 mt(0); // same seed we want always the same distrubution of random numbers;
    const size_t draws = std::min(1ul << 20, static_cast<size_t>(std::numeric_limits<source_T>::max()));
    const double probability = 0.5;
    std::binomial_distribution<source_T> dist(draws, probability);
    const size_t sourceSize = messageSize / sizeof(source_T);
    mSourceMessage.resize(sourceSize);
#ifdef RANS_PARALLEL_STL
    std::generate(std::execution::par_unseq, mSourceMessage.begin(), mSourceMessage.end(), [&dist, &mt]() { return dist(mt); });
#else
    std::generate(mSourceMessage.begin(), mSourceMessage.end(), [&dist, &mt]() { return dist(mt); });
#endif // RANS_PARALLEL_STL

    const auto histogram = makeDenseHistogram::fromSamples(gsl::span<const source_T>(mSourceMessage));
    Metrics<source_T> metrics{histogram};
    mRenormedHistogram = renorm(histogram, metrics);

    double_t expectationValue = std::accumulate(mRenormedHistogram.begin(), mRenormedHistogram.end(), 0.0, [this](const double_t& a, const count_t& b) {
      double_t prb = static_cast<double_t>(b) / static_cast<double_t>(mRenormedHistogram.getNumSamples());
      return a + b * prb;
    });

    mState = ((LowerBound >> mRenormedHistogram.getRenormingBits()) << StreamBits) * expectationValue;
  };

  const auto& getSourceMessage() const { return mSourceMessage; };
  const auto& getRenormedHistogram() const { return mRenormedHistogram; };

  ransState_t getState() const { return mState; };

 private:
  std::vector<source_T> mSourceMessage{};
  RenormedDenseHistogram<source_T> mRenormedHistogram{};
  ransState_t mState{};
};

const RenormingData<uint8_t> Data8(MessageSize);
const RenormingData<uint16_t> Data16(MessageSize);
const RenormingData<uint32_t> Data32(MessageSize);

template <typename T>
const auto& getData()
{
  if constexpr (std::is_same_v<uint8_t, T>) {
    return Data8;
  } else if constexpr (std::is_same_v<uint16_t, T>) {
    return Data16;
  } else {
    return Data32;
  }
};

template <typename source_T>
struct Fixture : public benchmark::Fixture {
  using source_t = source_T;

  void SetUp(const ::benchmark::State& state) final
  {
    const auto& sourceMessage = getData<source_T>().getSourceMessage();
    const auto& renormedHistogram = getData<source_T>().getRenormedHistogram();

    for (auto& symbol : sourceMessage) {
      mFrequencies.push_back(renormedHistogram[symbol]);
    }
  }

  void TearDown(const ::benchmark::State& state) final
  {
    mFrequencies.clear();
  }

  std::vector<count_t> mFrequencies{};
  ransState_t mState = getData<source_T>().getState();
  size_t mRenormingBits = getData<source_T>().getRenormedHistogram().getRenormingBits();
};

#ifdef RANS_SIMD
template <typename source_T, simd::SIMDWidth width_V>
struct SIMDFixture : public benchmark::Fixture {

  using source_t = source_T;

  void
    SetUp(const ::benchmark::State& state) final
  {
    mState[0] = simd::setAll<width_V>(getData<source_T>().getState());
    mState[1] = simd::setAll<width_V>(getData<source_T>().getState());

    const auto& sourceMessage = getData<source_T>().getSourceMessage();
    const auto& renormedHistogram = getData<source_T>().getRenormedHistogram();

    for (size_t i = 0; i < sourceMessage.size(); i += 2 * nElems) {
      if constexpr (width_V == simd::SIMDWidth::SSE) {
        mFrequencies.push_back({{simd::epi32_t<simd::SIMDWidth::SSE>{renormedHistogram[sourceMessage[i + 0]],
                                                                     renormedHistogram[sourceMessage[i + 1]],
                                                                     0x0u,
                                                                     0x0u},
                                 simd::epi32_t<simd::SIMDWidth::SSE>{renormedHistogram[sourceMessage[i + 2]],
                                                                     renormedHistogram[sourceMessage[i + 3]],
                                                                     0x0u,
                                                                     0x0u}}});
      }
      if constexpr (width_V == simd::SIMDWidth::AVX) {
        mFrequencies.push_back({{simd::epi32_t<simd::SIMDWidth::SSE>{renormedHistogram[sourceMessage[i + 0]],
                                                                     renormedHistogram[sourceMessage[i + 1]],
                                                                     renormedHistogram[sourceMessage[i + 2]],
                                                                     renormedHistogram[sourceMessage[i + 3]]},
                                 simd::epi32_t<simd::SIMDWidth::SSE>{renormedHistogram[sourceMessage[i + 4]],
                                                                     renormedHistogram[sourceMessage[i + 5]],
                                                                     renormedHistogram[sourceMessage[i + 6]],
                                                                     renormedHistogram[sourceMessage[i + 7]]}}});
      }
    }
  }

  void TearDown(const ::benchmark::State& state) final
  {
    mFrequencies.clear();
  }

  static constexpr size_t nElems = simd::getElementCount<ransState_t>(width_V);
  std::vector<std::array<simd::epi32_t<simd::SIMDWidth::SSE>, 2>> mFrequencies{};
  simd::simdI_t<width_V> mState[2];
  uint8_t mRenormingBits = getData<source_T>().getRenormedHistogram().getRenormingBits();
};
#endif /* RANS_SIMD */

template <typename stream_IT>
inline std::tuple<ransState_t, stream_IT> renorm(ransState_t state, stream_IT outputIter, count_t frequency, size_t symbolTablePrecision)
{
  ransState_t maxState = ((LowerBound >> symbolTablePrecision) << StreamBits) * frequency; // this turns into a shift.
  if (state >= maxState) {
    *(++outputIter) = static_cast<stream_t>(state);
    state >>= StreamBits;
  }

  return std::make_tuple(state, outputIter);
};

template <class fixture_T>
static void ransRenormingBenchmark(benchmark::State& st, fixture_T& fixture)
{
  std::vector<stream_t> out(fixture.mFrequencies.size() * 4);

  for (auto _ : st) {
    auto outIter = out.data();
    ransState_t newState = fixture.mState;
    for (size_t i = 0; i < fixture.mFrequencies.size(); ++i) {
      std::tie(newState, outIter) = renorm(fixture.mState, outIter, fixture.mFrequencies[i], fixture.mRenormingBits);
    }
    benchmark::ClobberMemory();
  };

  st.SetItemsProcessed(int64_t(st.iterations()) * getData<typename fixture_T::source_t>().getSourceMessage().size());
  st.SetBytesProcessed(int64_t(st.iterations()) * getData<typename fixture_T::source_t>().getSourceMessage().size() * sizeof(typename fixture_T::source_t));
};

#ifdef RANS_SIMD
template <class fixture_T>
static void ransRenormingBenchmarkSIMD(benchmark::State& st, fixture_T& fixture)
{
  std::vector<stream_t> out(fixture.mFrequencies.size() * 4);

#ifdef ENABLE_VTUNE_PROFILER
  __itt_resume();
#endif
  for (auto _ : st) {
    simd::simdIsse_t frequencies[2];
    auto outIter = out.data();
    auto newState = fixture.mState;
    for (size_t i = 0; i < fixture.mFrequencies.size(); ++i) {
      frequencies[0] = load(fixture.mFrequencies[i][0]);
      frequencies[1] = load(fixture.mFrequencies[i][1]);
      outIter = simd::ransRenorm<decltype(outIter),
                                 LowerBound,
                                 StreamBits>(fixture.mState,
                                             frequencies,
                                             fixture.mRenormingBits,
                                             outIter,
                                             newState);
    }
    benchmark::ClobberMemory();
  };
#ifdef ENABLE_VTUNE_PROFILER
  __itt_pause();
#endif

  st.SetItemsProcessed(int64_t(st.iterations()) * getData<typename fixture_T::source_t>().getSourceMessage().size());
  st.SetBytesProcessed(int64_t(st.iterations()) * getData<typename fixture_T::source_t>().getSourceMessage().size() * sizeof(typename fixture_T::source_t));
};
#endif /* RANS_SIMD */

BENCHMARK_TEMPLATE_DEFINE_F(Fixture, renorm_8, uint8_t)
(benchmark::State& st)
{
  ransRenormingBenchmark(st, *this);
};

BENCHMARK_TEMPLATE_DEFINE_F(Fixture, renorm_16, uint16_t)
(benchmark::State& st)
{
  ransRenormingBenchmark(st, *this);
};

BENCHMARK_TEMPLATE_DEFINE_F(Fixture, renorm_32, uint32_t)
(benchmark::State& st)
{
  ransRenormingBenchmark(st, *this);
};

#ifdef RANS_SSE
BENCHMARK_TEMPLATE_DEFINE_F(SIMDFixture, renormSSE_8, uint8_t, simd::SIMDWidth::SSE)
(benchmark::State& st)
{
  ransRenormingBenchmarkSIMD(st, *this);
};

BENCHMARK_TEMPLATE_DEFINE_F(SIMDFixture, renormSSE_16, uint16_t, simd::SIMDWidth::SSE)
(benchmark::State& st)
{
  ransRenormingBenchmarkSIMD(st, *this);
};

BENCHMARK_TEMPLATE_DEFINE_F(SIMDFixture, renormSSE_32, uint32_t, simd::SIMDWidth::SSE)
(benchmark::State& st)
{
  ransRenormingBenchmarkSIMD(st, *this);
};
#endif /* RANS_SSE */

#ifdef RANS_AVX2
BENCHMARK_TEMPLATE_DEFINE_F(SIMDFixture, renormAVX_8, uint8_t, simd::SIMDWidth::AVX)
(benchmark::State& st)
{
  ransRenormingBenchmarkSIMD(st, *this);
};

BENCHMARK_TEMPLATE_DEFINE_F(SIMDFixture, renormAVX_16, uint16_t, simd::SIMDWidth::AVX)
(benchmark::State& st)
{
  ransRenormingBenchmarkSIMD(st, *this);
};

BENCHMARK_TEMPLATE_DEFINE_F(SIMDFixture, renormAVX_32, uint32_t, simd::SIMDWidth::AVX)
(benchmark::State& st)
{
  ransRenormingBenchmarkSIMD(st, *this);
};
#endif /* RANS_AVX2 */

BENCHMARK_REGISTER_F(Fixture, renorm_8);
BENCHMARK_REGISTER_F(Fixture, renorm_16);
BENCHMARK_REGISTER_F(Fixture, renorm_32);

#ifdef RANS_SSE
BENCHMARK_REGISTER_F(SIMDFixture, renormSSE_8);
BENCHMARK_REGISTER_F(SIMDFixture, renormSSE_16);
BENCHMARK_REGISTER_F(SIMDFixture, renormSSE_32);
#endif /* RANS_SSE */

#ifdef RANS_AVX2
BENCHMARK_REGISTER_F(SIMDFixture, renormAVX_8);
BENCHMARK_REGISTER_F(SIMDFixture, renormAVX_16);
BENCHMARK_REGISTER_F(SIMDFixture, renormAVX_32);
#endif /* RANS_AVX2 */

BENCHMARK_MAIN();
