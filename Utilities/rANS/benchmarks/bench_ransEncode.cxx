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
#include "rANS/internal/backend/simd/SymbolTable.h"

#ifdef ENABLE_VTUNE_PROFILER
#include <ittnotify.h>
#endif

using count_t = uint32_t;
using ransState_t = uint64_t;
using stream_t = uint32_t;

__extension__ using uint128_t = unsigned __int128;

using namespace o2::rans::internal;

inline constexpr size_t MessageSize = 1ull << 22;
inline constexpr size_t LowerBound = 1ul << 20;
inline constexpr size_t StreamBits = toBits(sizeof(stream_t));

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
  using symbol_t = cpp::DecoderSymbol;

  void SetUp(const ::benchmark::State& state) final
  {
    const auto& sourceMessage = getData<source_T>().getSourceMessage();
    SymbolTable<symbol_t> symbolTable{getData<source_T>().getRenormedFrequencies()};
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
  using symbol_t = cpp::EncoderSymbol<ransState_t>;

  void SetUp(const ::benchmark::State& state) final
  {
    const auto& sourceMessage = getData<source_T>().getSourceMessage();
    SymbolTable<symbol_t> symbolTable{getData<source_T>().getRenormedFrequencies()};
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

template <typename source_T, simd::SIMDWidth width_V>
struct SIMDFixture : public benchmark::Fixture {
  using source_t = source_T;
  using symbol_t = simd::Symbol;

  void SetUp(const ::benchmark::State& state) final
  {
    const auto& sourceMessage = getData<source_T>().getSourceMessage();
    simd::SymbolTable symbolTable{getData<source_T>().getRenormedFrequencies()};
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
  simd::epi64_t<width_V> mState{getData<source_T>().getState()};
  simd::pd_t<width_V> mNSamples{static_cast<double>(pow2(
    getData<source_T>().getRenormedFrequencies().getRenormingBits()))};
};

ransState_t encode(ransState_t state, const cpp::EncoderSymbol<ransState_t>& symbol)
{
  // x = C(s,x)
  ransState_t quotient = static_cast<ransState_t>((static_cast<uint128_t>(state) * symbol.getReciprocalFrequency()) >> 64);
  quotient = quotient >> symbol.getReciprocalShift();

  return state + symbol.getBias() + quotient * symbol.getFrequencyComplement();
};

ransState_t simpleEncode(ransState_t state, size_t symbolTablePrecision, const cpp::DecoderSymbol& symbol)
{
  // x = C(s,x)
  return ((state / symbol.getFrequency()) << symbolTablePrecision) + symbol.getCumulative() + (state % symbol.getFrequency());
};

template <simd::SIMDWidth width_V>
auto SIMDEncode(simd::epi64cV_t<width_V> states,
                simd::pdcV_t<width_V> nSamples,
                simd::ArrayView<const simd::Symbol*, simd::getElementCount<ransState_t>(width_V)> symbols)
{
  auto [frequencies, cumulativeFrequencies] = simd::aosToSoa(symbols);
  return ransEncode(states,
                    simd::int32ToDouble<width_V>(simd::toConstSIMDView(frequencies)),
                    simd::int32ToDouble<width_V>(simd::toConstSIMDView(cumulativeFrequencies)),
                    nSamples);
};

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

template <typename source_T, simd::SIMDWidth width_V>
static void ransSIMDEncodeBenchmark(benchmark::State& st, SIMDFixture<source_T, width_V>& fixture)
{
#ifdef ENABLE_VTUNE_PROFILER
  __itt_resume();
#endif
  for (auto _ : st) {
    for (size_t i = 0; i < fixture.mSymbols.size(); ++i) {
      auto newStates = SIMDEncode(simd::toConstSIMDView(fixture.mState), simd::toConstSIMDView(fixture.mNSamples), fixture.mSymbols[i]);
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

BENCHMARK_REGISTER_F(SimpleFixture, simpleEncode_8);
BENCHMARK_REGISTER_F(SimpleFixture, simpleEncode_16);
BENCHMARK_REGISTER_F(SimpleFixture, simpleEncode_32);

BENCHMARK_REGISTER_F(Fixture, encode_8);
BENCHMARK_REGISTER_F(Fixture, encode_16);
BENCHMARK_REGISTER_F(Fixture, encode_32);

BENCHMARK_REGISTER_F(SIMDFixture, encodeSSE_8);
BENCHMARK_REGISTER_F(SIMDFixture, encodeSSE_16);
BENCHMARK_REGISTER_F(SIMDFixture, encodeSSE_32);

BENCHMARK_REGISTER_F(SIMDFixture, encodeAVX_8);
BENCHMARK_REGISTER_F(SIMDFixture, encodeAVX_16);
BENCHMARK_REGISTER_F(SIMDFixture, encodeAVX_32);

BENCHMARK_MAIN();
