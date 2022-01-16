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
#include "rANS/SIMDEncoder.h"

#ifdef ENABLE_VTUNE_PROFILER
#include <ittnotify.h>
#endif

using count_t = uint32_t;
using ransState_t = uint64_t;
using stream_t = uint32_t;

inline constexpr size_t MessageSize = 1ull << 22;

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
  };

  const auto& getSourceMessage() const { return mSourceMessage; };
  const auto& getRenormedFrequencies() const { return mRenormedFrequencies; };

 private:
  std::vector<source_T> mSourceMessage{};
  o2::rans::RenormedFrequencyTable mRenormedFrequencies{};
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
static void ransSymbolTableAccessLight(benchmark::State& st)
{
  using namespace o2::rans;

  const auto& data = getData<source_T>();

  internal::SymbolTable<internal::cpp::DecoderSymbol> symbolTable(data.getRenormedFrequencies());

  for (auto _ : st) {
    for (size_t i = 0; i < data.getSourceMessage().size(); ++i) {
      benchmark::DoNotOptimize(symbolTable[data.getSourceMessage()[i]]);
      benchmark::ClobberMemory();
    }
  }

  st.SetItemsProcessed(int64_t(st.iterations()) * getData<source_T>().getSourceMessage().size());
  st.SetBytesProcessed(int64_t(st.iterations()) * getData<source_T>().getSourceMessage().size() * sizeof(source_T));
};

template <typename source_T>
static void ransSymbolTableAccessHeavy(benchmark::State& st)
{
  using namespace o2::rans;

  const auto& data = getData<source_T>();

  internal::SymbolTable<internal::cpp::EncoderSymbol<ransState_t>> symbolTable(data.getRenormedFrequencies());

  for (auto _ : st) {
    for (size_t i = 0; i < data.getSourceMessage().size(); ++i) {
      benchmark::DoNotOptimize(symbolTable[data.getSourceMessage()[i]]);
      benchmark::ClobberMemory();
    }
  }

  st.SetItemsProcessed(int64_t(st.iterations()) * getData<source_T>().getSourceMessage().size());
  st.SetBytesProcessed(int64_t(st.iterations()) * getData<source_T>().getSourceMessage().size() * sizeof(source_T));
};

// template <typename source_T, o2::rans::internal::simd::SIMDWidth width_V>
// static void ransSymbolTableAccessSIMD(benchmark::State& st)
// {
//   using namespace o2::rans;
//   static constexpr size_t nElems = o2::rans::internal::simd::getElementCount<ransState_t>(width_V);

//   const auto& data = getData<source_T>();

//   internal::simd::SymbolTable symbolTable(data.getRenormedFrequencies());

//   for (auto _ : st) {
//     for (size_t i = 0; i < data.getSourceMessage().size(); i += nElems) {
//       benchmark::DoNotOptimize(internal::getSymbols<const source_T*, nElems>(&(data.getSourceMessage()[i]), symbolTable));
//       benchmark::ClobberMemory();
//     }
//   }

//   st.SetItemsProcessed(int64_t(st.iterations()) * getData<source_T>().getSourceMessage().size());
//   st.SetBytesProcessed(int64_t(st.iterations()) * getData<source_T>().getSourceMessage().size() * sizeof(source_T));
// };

BENCHMARK_TEMPLATE1(ransSymbolTableAccessLight, uint8_t);
BENCHMARK_TEMPLATE1(ransSymbolTableAccessLight, uint16_t);
BENCHMARK_TEMPLATE1(ransSymbolTableAccessLight, uint32_t);

BENCHMARK_TEMPLATE1(ransSymbolTableAccessHeavy, uint8_t);
BENCHMARK_TEMPLATE1(ransSymbolTableAccessHeavy, uint16_t);
BENCHMARK_TEMPLATE1(ransSymbolTableAccessHeavy, uint32_t);

// BENCHMARK_TEMPLATE2(ransSymbolTableAccessSIMD, uint8_t, o2::rans::internal::simd::SIMDWidth::SSE);
// BENCHMARK_TEMPLATE2(ransSymbolTableAccessSIMD, uint16_t, o2::rans::internal::simd::SIMDWidth::SSE);
// BENCHMARK_TEMPLATE2(ransSymbolTableAccessSIMD, uint32_t, o2::rans::internal::simd::SIMDWidth::SSE);

// BENCHMARK_TEMPLATE2(ransSymbolTableAccessSIMD, uint8_t, o2::rans::internal::simd::SIMDWidth::AVX);
// BENCHMARK_TEMPLATE2(ransSymbolTableAccessSIMD, uint16_t, o2::rans::internal::simd::SIMDWidth::AVX);
// BENCHMARK_TEMPLATE2(ransSymbolTableAccessSIMD, uint32_t, o2::rans::internal::simd::SIMDWidth::AVX);

BENCHMARK_MAIN();
