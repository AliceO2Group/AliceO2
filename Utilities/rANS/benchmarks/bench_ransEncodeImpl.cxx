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

/// @file   bench_ransEncodeImpl.cxx
/// @author Michael Lettrich
/// @brief benchmarks different encoding kernels without the need to renorm and stream data

#include "rANS/internal/common/defines.h"

#include <vector>
#include <cstring>
#include <random>
#include <algorithm>
#ifdef RANS_PARALLEL_STL
#include <execution>
#endif
#include <iterator>

#include <gsl/span>
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

#ifdef RANS_SINGLE_STREAM
__extension__ using uint128_t = unsigned __int128;
#endif /* RANS_SINGLE_STREAM */

using namespace o2::rans;
using namespace o2::rans::internal;
using namespace o2::rans::utils;

inline constexpr size_t MessageSize = 1ull << 22;
inline constexpr size_t LowerBound = 1ul << 20;
inline constexpr size_t StreamBits = toBits<stream_t>();

template <typename source_T>
class SymbolTableData
{
 public:
  explicit SymbolTableData(size_t messageSize)
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
    mRenormedFrequencies = renorm(histogram, metrics);

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
  RenormedDenseHistogram<source_T> mRenormedFrequencies{};
  ransState_t mState{};
};

const SymbolTableData<uint8_t> Data8(MessageSize);
const SymbolTableData<uint16_t> Data16(MessageSize);
const SymbolTableData<uint32_t> Data32(MessageSize);

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
struct SimpleFixture : public benchmark::Fixture {
  using source_t = source_T;
  using symbol_t = Symbol;

  void SetUp(const ::benchmark::State& state) final
  {
    const auto& sourceMessage = getData<source_T>().getSourceMessage();
    DenseSymbolTable<source_t, symbol_t> symbolTable{getData<source_T>().getRenormedFrequencies()};
    for (auto& symbol : sourceMessage) {
      mSymbols.push_back(symbolTable[symbol]);
    }
  }

  void TearDown(const ::benchmark::State& state) final
  {
    mSymbols.clear();
  }
  std::vector<symbol_t> mSymbols{};
  ransState_t mState = getData<source_T>().getState();
  size_t mRenormingBits = getData<source_T>().getRenormedFrequencies().getRenormingBits();
};

template <typename source_T>
struct Fixture : public benchmark::Fixture {
  using source_t = source_T;
  using symbol_t = PrecomputedSymbol;

  void SetUp(const ::benchmark::State& state) final
  {
    const auto& sourceMessage = getData<source_T>().getSourceMessage();
    DenseSymbolTable<source_t, symbol_t> symbolTable{getData<source_T>().getRenormedFrequencies()};
    for (auto& symbol : sourceMessage) {
      mSymbols.push_back(symbolTable[symbol]);
    }
  }

  void TearDown(const ::benchmark::State& state) final
  {
    mSymbols.clear();
  }
  std::vector<symbol_t> mSymbols{};
  ransState_t mState = getData<source_T>().getState();
  size_t mRenormingBits = getData<source_T>().getRenormedFrequencies().getRenormingBits();
};

#ifdef RANS_SIMD

template <typename source_T, simd::SIMDWidth width_V>
struct SIMDFixture : public benchmark::Fixture {
  using source_t = source_T;
  using symbol_t = Symbol;

  void SetUp(const ::benchmark::State& state) final
  {
    mState = simd::setAll<width_V>(getData<source_T>().getState());
    mNSamples = simd::setAll<width_V>(static_cast<double>(pow2(getData<source_T>().getRenormedFrequencies().getRenormingBits())));

    const auto& sourceMessage = getData<source_T>().getSourceMessage();
    DenseSymbolTable<source_t, symbol_t> symbolTable{getData<source_T>().getRenormedFrequencies()};
    for (size_t i = 0; i < sourceMessage.size(); i += nElems) {
      if constexpr (width_V == simd::SIMDWidth::SSE) {
        mSymbols.push_back({
          &symbolTable[sourceMessage[i]],
          &symbolTable[sourceMessage[i + 1]],
        });
      }
      if constexpr (width_V == simd::SIMDWidth::AVX) {
        mSymbols.push_back({
          &symbolTable[sourceMessage[i]],
          &symbolTable[sourceMessage[i + 1]],
          &symbolTable[sourceMessage[i + 2]],
          &symbolTable[sourceMessage[i + 3]],
        });
      }
    }
  }

  void TearDown(const ::benchmark::State& state) final
  {
    mSymbols.clear();
  }

  static constexpr size_t nElems = simd::getElementCount<ransState_t>(width_V);
  std::vector<std::array<const symbol_t*, nElems>> mSymbols{};
  simd::simdI_t<width_V> mState;
  simd::simdD_t<width_V> mNSamples;
};
#endif /* RANS_SIMD */

#ifdef RANS_SINGLE_STREAM
ransState_t encode(ransState_t state, const PrecomputedSymbol& symbol)
{
  // x = C(s,x)
  ransState_t quotient = static_cast<ransState_t>((static_cast<uint128_t>(state) * symbol.getReciprocalFrequency()) >> 64);
  quotient = quotient >> symbol.getReciprocalShift();

  return state + symbol.getCumulative() + quotient * symbol.getFrequencyComplement();
};
#endif /* RANS_SINGLE_STREAM */

ransState_t simpleEncode(ransState_t state, size_t symbolTablePrecision, const Symbol& symbol)
{
  // x = C(s,x)
  return ((state / symbol.getFrequency()) << symbolTablePrecision) + symbol.getCumulative() + (state % symbol.getFrequency());
};

#ifdef RANS_SIMD
template <simd::SIMDWidth width_V>
inline auto SIMDEncode(simd::simdI_t<width_V> states, simd::simdD_t<width_V> nSamples, gsl::span<const Symbol*, simd::getElementCount<ransState_t>(width_V)> symbols)
{
  simd::simdIsse_t frequencies;
  simd::simdIsse_t cumulativeFrequencies;
  simd::aosToSoa(symbols, &frequencies, &cumulativeFrequencies);
  return simd::ransEncode(states, simd::int32ToDouble<width_V>(frequencies), simd::int32ToDouble<width_V>(cumulativeFrequencies), nSamples);
};
#endif /* RANS_SIMD */

template <typename source_T>
static void ransSimpleEncodeBenchmark(benchmark::State& st, SimpleFixture<source_T>& fixture)
{
  for (auto _ : st) {
    for (size_t i = 0; i < fixture.mSymbols.size(); ++i) {
      ransState_t newState = simpleEncode(fixture.mState, fixture.mRenormingBits, fixture.mSymbols[i]);
      benchmark::DoNotOptimize(newState);
    }
  };

  st.SetItemsProcessed(int64_t(st.iterations()) * getData<source_T>().getSourceMessage().size());
  st.SetBytesProcessed(int64_t(st.iterations()) * getData<source_T>().getSourceMessage().size() * sizeof(source_T));
};

#ifdef RANS_SINGLE_STREAM
template <typename source_T>
static void ransEncodeBenchmark(benchmark::State& st, Fixture<source_T>& fixture)
{
  for (auto _ : st) {
    for (size_t i = 0; i < fixture.mSymbols.size(); ++i) {
      ransState_t newState = encode(fixture.mState, fixture.mSymbols[i]);
      benchmark::DoNotOptimize(newState);
    }
  };

  st.SetItemsProcessed(int64_t(st.iterations()) * getData<source_T>().getSourceMessage().size());
  st.SetBytesProcessed(int64_t(st.iterations()) * getData<source_T>().getSourceMessage().size() * sizeof(source_T));
};
#endif /* RANS_SINGLE_STREAM */

#ifdef RANS_SIMD
template <typename source_T, simd::SIMDWidth width_V>
static void ransSIMDEncodeBenchmark(benchmark::State& st, SIMDFixture<source_T, width_V>& fixture)
{
#ifdef ENABLE_VTUNE_PROFILER
  __itt_resume();
#endif
  for (auto _ : st) {
    for (size_t i = 0; i < fixture.mSymbols.size(); ++i) {
      auto newStates = SIMDEncode<width_V>(fixture.mState, fixture.mNSamples, fixture.mSymbols[i]);
      benchmark::DoNotOptimize(newStates);
      benchmark::ClobberMemory();
    }
  }
#ifdef ENABLE_VTUNE_PROFILER
  __itt_pause();
#endif

  st.SetItemsProcessed(int64_t(st.iterations()) * getData<source_T>().getSourceMessage().size());
  st.SetBytesProcessed(int64_t(st.iterations()) * getData<source_T>().getSourceMessage().size() * sizeof(source_T));
};
#endif /* RANS_SIMD */

BENCHMARK_TEMPLATE_DEFINE_F(SimpleFixture, simpleEncode_8, uint8_t)
(benchmark::State& st)
{
  ransSimpleEncodeBenchmark(st, *this);
};
BENCHMARK_TEMPLATE_DEFINE_F(SimpleFixture, simpleEncode_16, uint16_t)
(benchmark::State& st)
{
  ransSimpleEncodeBenchmark(st, *this);
};
BENCHMARK_TEMPLATE_DEFINE_F(SimpleFixture, simpleEncode_32, uint32_t)
(benchmark::State& st)
{
  ransSimpleEncodeBenchmark(st, *this);
};

BENCHMARK_TEMPLATE_DEFINE_F(Fixture, encode_8, uint8_t)
(benchmark::State& st)
{
  ransEncodeBenchmark(st, *this);
};
BENCHMARK_TEMPLATE_DEFINE_F(Fixture, encode_16, uint16_t)
(benchmark::State& st)
{
  ransEncodeBenchmark(st, *this);
};
BENCHMARK_TEMPLATE_DEFINE_F(Fixture, encode_32, uint32_t)
(benchmark::State& st)
{
  ransEncodeBenchmark(st, *this);
};

#ifdef RANS_SSE
BENCHMARK_TEMPLATE_DEFINE_F(SIMDFixture, encodeSSE_8, uint8_t, simd::SIMDWidth::SSE)
(benchmark::State& st)
{
  ransSIMDEncodeBenchmark(st, *this);
};

BENCHMARK_TEMPLATE_DEFINE_F(SIMDFixture, encodeSSE_16, uint16_t, simd::SIMDWidth::SSE)
(benchmark::State& st)
{
  ransSIMDEncodeBenchmark(st, *this);
};

BENCHMARK_TEMPLATE_DEFINE_F(SIMDFixture, encodeSSE_32, uint32_t, simd::SIMDWidth::SSE)
(benchmark::State& st)
{
  ransSIMDEncodeBenchmark(st, *this);
};
#endif /*RANS_SSE*/

#ifdef RANS_AVX2
BENCHMARK_TEMPLATE_DEFINE_F(SIMDFixture, encodeAVX_8, uint8_t, simd::SIMDWidth::AVX)
(benchmark::State& st)
{
  ransSIMDEncodeBenchmark(st, *this);
};

BENCHMARK_TEMPLATE_DEFINE_F(SIMDFixture, encodeAVX_16, uint16_t, simd::SIMDWidth::AVX)
(benchmark::State& st)
{
  ransSIMDEncodeBenchmark(st, *this);
};

BENCHMARK_TEMPLATE_DEFINE_F(SIMDFixture, encodeAVX_32, uint32_t, simd::SIMDWidth::AVX)
(benchmark::State& st)
{
  ransSIMDEncodeBenchmark(st, *this);
};
#endif /* RANS_AVX2 */

BENCHMARK_REGISTER_F(SimpleFixture, simpleEncode_8);
BENCHMARK_REGISTER_F(SimpleFixture, simpleEncode_16);
BENCHMARK_REGISTER_F(SimpleFixture, simpleEncode_32);

#ifdef RANS_SINGLE_STREAM
BENCHMARK_REGISTER_F(Fixture, encode_8);
BENCHMARK_REGISTER_F(Fixture, encode_16);
BENCHMARK_REGISTER_F(Fixture, encode_32);
#endif /* RANS_SINGLE_STREAM */

#ifdef RANS_SSE
BENCHMARK_REGISTER_F(SIMDFixture, encodeSSE_8);
BENCHMARK_REGISTER_F(SIMDFixture, encodeSSE_16);
BENCHMARK_REGISTER_F(SIMDFixture, encodeSSE_32);
#endif /* RANS_SSE */

#ifdef RANS_AVX2
BENCHMARK_REGISTER_F(SIMDFixture, encodeAVX_8);
BENCHMARK_REGISTER_F(SIMDFixture, encodeAVX_16);
BENCHMARK_REGISTER_F(SIMDFixture, encodeAVX_32);
#endif /* RANS_SSE */

BENCHMARK_MAIN();
