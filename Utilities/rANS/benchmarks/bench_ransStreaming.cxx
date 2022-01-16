// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   bench_ransCombinedIterator.cxx
/// @author Michael Lettrich
/// @since  2021-05-03
/// @brief

#include <vector>
#include <cstring>
#include <random>
#include <algorithm>
#include <execution>
#include <iterator>

#include <benchmark/benchmark.h>

#include "rANS/utils.h"
#include "rANS/rans.h"
#include "rANS/internal/backend/simd/kernel.h"
#include "rANS/internal/backend/simd/types.h"

#ifdef ENABLE_VTUNE_PROFILER
#include <ittnotify.h>
#endif

using count_t = uint32_t;
using ransState_t = uint64_t;
using stream_t = uint32_t;

using namespace o2::rans::internal;

inline constexpr size_t MessageSize = 1ull << 22;
inline constexpr size_t LowerBound = 1ul << 20;
inline constexpr size_t StreamBits = toBits(sizeof(stream_t));

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
    std::generate(std::execution::par_unseq, mSourceMessage.begin(), mSourceMessage.end(), [&dist, &mt]() { return dist(mt); });

    mRenormedFrequencies = o2::rans::renorm(o2::rans::makeFrequencyTableFromSamples(std::begin(mSourceMessage), std::end(mSourceMessage)));

    double_t expectationValue = std::accumulate(mRenormedFrequencies.begin(), mRenormedFrequencies.end(), 0.0, [this](const double_t& a, const count_t& b) {
      double_t prb = static_cast<double_t>(b) / static_cast<double_t>(mRenormedFrequencies.getNumSamples());
      return a + b * prb;
    });

    mState = ((LowerBound >> mRenormedFrequencies.getRenormingBits()) << StreamBits) * expectationValue;
  };

  const auto& getSourceMessage() const { return mSourceMessage; };
  const auto& getRenormedFrequencies() const { return mRenormedFrequencies; };

  ransState_t getState() const { return mState; };

 private:
  std::vector<source_T> mSourceMessage{};
  o2::rans::RenormedFrequencyTable mRenormedFrequencies{};
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
    const auto& frequencyTable = getData<source_T>().getRenormedFrequencies();

    for (auto& symbol : sourceMessage) {
      mFrequencies.push_back(frequencyTable[symbol]);
    }
  }

  void TearDown(const ::benchmark::State& state) final
  {
    mFrequencies.clear();
  }

  std::vector<count_t> mFrequencies{};
  ransState_t mState = getData<source_T>().getState();
  size_t mRenormingBits = getData<source_T>().getRenormedFrequencies().getRenormingBits();
};

template <typename source_T, simd::SIMDWidth width_V>
struct SIMDFixture : public benchmark::Fixture {

  using source_t = source_T;

  void
    SetUp(const ::benchmark::State& state) final
  {
    const auto& sourceMessage = getData<source_T>().getSourceMessage();
    const auto& frequencyTable = getData<source_T>().getRenormedFrequencies();

    for (size_t i = 0; i < sourceMessage.size(); i += nElems) {
      if constexpr (width_V == simd::SIMDWidth::SSE) {
        mFrequencies.emplace_back(frequencyTable[sourceMessage[i]],
                                  frequencyTable[sourceMessage[i + 1]],
                                  0x0u,
                                  0x0u);
      }
      if constexpr (width_V == simd::SIMDWidth::AVX) {
        mFrequencies.emplace_back(frequencyTable[sourceMessage[i]],
                                  frequencyTable[sourceMessage[i + 1]],
                                  frequencyTable[sourceMessage[i + 2]],
                                  frequencyTable[sourceMessage[i + 3]]);
      }
    }
  }

  void TearDown(const ::benchmark::State& state) final
  {
    mFrequencies.clear();
  }

  static constexpr size_t nElems = simd::getElementCount<ransState_t>(width_V);
  std::vector<simd::epi32_t<simd::SIMDWidth::SSE>> mFrequencies{};
  simd::epi64_t<width_V> mState{getData<source_T>().getState()};
  uint8_t mRenormingBits = getData<source_T>().getRenormedFrequencies().getRenormingBits();
};

template <typename source_T, simd::SIMDWidth width_V>
struct InterleavedSIMDFixture : public benchmark::Fixture {

  using source_t = source_T;

  void
    SetUp(const ::benchmark::State& state) final
  {
    const auto& sourceMessage = getData<source_T>().getSourceMessage();
    const auto& frequencyTable = getData<source_T>().getRenormedFrequencies();

    for (size_t i = 0; i < sourceMessage.size(); i += 2 * nElems) {
      if constexpr (width_V == simd::SIMDWidth::SSE) {
        mFrequencies.emplace_back(frequencyTable[sourceMessage[i + 0]],
                                  frequencyTable[sourceMessage[i + 1]],
                                  0x0u,
                                  0x0u,
                                  frequencyTable[sourceMessage[i + 2]],
                                  frequencyTable[sourceMessage[i + 3]],
                                  0x0u,
                                  0x0u);
      }
      if constexpr (width_V == simd::SIMDWidth::AVX) {
        mFrequencies.emplace_back(frequencyTable[sourceMessage[i + 0]],
                                  frequencyTable[sourceMessage[i + 1]],
                                  frequencyTable[sourceMessage[i + 2]],
                                  frequencyTable[sourceMessage[i + 3]],
                                  frequencyTable[sourceMessage[i + 4]],
                                  frequencyTable[sourceMessage[i + 5]],
                                  frequencyTable[sourceMessage[i + 6]],
                                  frequencyTable[sourceMessage[i + 7]]);
      }
    }
  }

  void TearDown(const ::benchmark::State& state) final
  {
    mFrequencies.clear();
  }

  static constexpr size_t nElems = simd::getElementCount<ransState_t>(width_V);
  std::vector<simd::epi32_t<simd::SIMDWidth::SSE, 2>> mFrequencies{};
  simd::epi64_t<width_V, 2> mState{getData<source_T>().getState()};
  uint8_t mRenormingBits = getData<source_T>().getRenormedFrequencies().getRenormingBits();
};

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

template <class fixture_T>
static void ransRenormingBenchmarkSIMD(benchmark::State& st, fixture_T& fixture)
{
  std::vector<stream_t> out(fixture.mFrequencies.size() * 4);

  // #ifdef ENABLE_VTUNE_PROFILER
  //   __itt_resume();
  // #endif
  for (auto _ : st) {
    auto outIter = out.data();
    auto newState = fixture.mState;
    for (size_t i = 0; i < fixture.mFrequencies.size(); ++i) {
      std::tie(outIter, newState) = simd::ransRenorm<decltype(outIter),
                                                     LowerBound,
                                                     StreamBits>(simd::toConstSIMDView(fixture.mState), simd::toConstSIMDView(fixture.mFrequencies[i]), fixture.mRenormingBits, outIter);
    }
    benchmark::ClobberMemory();
  };
  // #ifdef ENABLE_VTUNE_PROFILER
  //   __itt_pause();
  // #endif

  st.SetItemsProcessed(int64_t(st.iterations()) * getData<typename fixture_T::source_t>().getSourceMessage().size());
  st.SetBytesProcessed(int64_t(st.iterations()) * getData<typename fixture_T::source_t>().getSourceMessage().size() * sizeof(typename fixture_T::source_t));
};

template <class fixture_T>
static void ransRenormingBenchmarkInterleavedSIMD(benchmark::State& st, fixture_T& fixture)
{
  std::vector<stream_t> out(fixture.mFrequencies.size() * 4);

#ifdef ENABLE_VTUNE_PROFILER
  __itt_resume();
#endif
  for (auto _ : st) {
    auto outIter = out.data();

    auto newState = fixture.mState;
    for (size_t i = 0; i < fixture.mFrequencies.size(); i += 2) {
      std::tie(outIter, newState) = simd::ransRenorm<decltype(outIter),
                                                     LowerBound,
                                                     StreamBits>(simd::toConstSIMDView(fixture.mState), simd::toConstSIMDView(fixture.mFrequencies[i]), fixture.mRenormingBits, outIter);
    }
    benchmark::ClobberMemory();
  };
#ifdef ENABLE_VTUNE_PROFILER
  __itt_pause();
#endif

  st.SetItemsProcessed(int64_t(st.iterations()) * getData<typename fixture_T::source_t>().getSourceMessage().size());
  st.SetBytesProcessed(int64_t(st.iterations()) * getData<typename fixture_T::source_t>().getSourceMessage().size() * sizeof(typename fixture_T::source_t));
};

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

BENCHMARK_TEMPLATE_DEFINE_F(InterleavedSIMDFixture, renormInterleavedAVX_8, uint8_t, simd::SIMDWidth::AVX)
(benchmark::State& st)
{
  ransRenormingBenchmarkInterleavedSIMD(st, *this);
};

BENCHMARK_TEMPLATE_DEFINE_F(InterleavedSIMDFixture, renormInterleavedAVX_16, uint16_t, simd::SIMDWidth::AVX)
(benchmark::State& st)
{
  ransRenormingBenchmarkInterleavedSIMD(st, *this);
};

BENCHMARK_TEMPLATE_DEFINE_F(InterleavedSIMDFixture, renormInterleavedAVX_32, uint32_t, simd::SIMDWidth::AVX)
(benchmark::State& st)
{
  ransRenormingBenchmarkInterleavedSIMD(st, *this);
};

BENCHMARK_REGISTER_F(Fixture, renorm_8);
BENCHMARK_REGISTER_F(Fixture, renorm_16);
BENCHMARK_REGISTER_F(Fixture, renorm_32);

BENCHMARK_REGISTER_F(SIMDFixture, renormSSE_8);
BENCHMARK_REGISTER_F(SIMDFixture, renormSSE_16);
BENCHMARK_REGISTER_F(SIMDFixture, renormSSE_32);

BENCHMARK_REGISTER_F(SIMDFixture, renormAVX_8);
BENCHMARK_REGISTER_F(SIMDFixture, renormAVX_16);
BENCHMARK_REGISTER_F(SIMDFixture, renormAVX_32);

BENCHMARK_REGISTER_F(InterleavedSIMDFixture, renormInterleavedAVX_8);
BENCHMARK_REGISTER_F(InterleavedSIMDFixture, renormInterleavedAVX_16);
BENCHMARK_REGISTER_F(InterleavedSIMDFixture, renormInterleavedAVX_32);

BENCHMARK_MAIN();
