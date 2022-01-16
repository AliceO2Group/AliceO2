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

// __extension__ using uint128_t = unsigned __int128;

// inline constexpr size_t MessageSize = 1ull << 22;

// using namespace o2::rans;
// using namespace o2::rans::internal;

// template <typename source_T>
// class SymbolTableData
// {
//  public:
//   explicit SymbolTableData(size_t messageSize)
//   {
//     std::mt19937 mt(0); // same seed we want always the same distrubution of random numbers;
//     const size_t draws = std::min(1ul << 20, static_cast<size_t>(std::numeric_limits<source_T>::max()));
//     const double probability = 0.5;
//     std::binomial_distribution<source_T> dist(draws, probability);
//     const size_t sourceSize = messageSize / sizeof(source_T);
//     mSourceMessage.resize(sourceSize);
//     std::generate(std::execution::par_unseq, mSourceMessage.begin(), mSourceMessage.end(), [&dist, &mt]() { return dist(mt); });

//     mRenormedFrequencies = o2::rans::renorm(o2::rans::makeFrequencyTableFromSamples(std::begin(mSourceMessage), std::end(mSourceMessage)));
//   };

//   const auto& getSourceMessage() const { return mSourceMessage; };
//   const auto& getRenormedFrequencies() const { return mRenormedFrequencies; };

//  private:
//   std::vector<source_T> mSourceMessage{};
//   o2::rans::RenormedFrequencyTable mRenormedFrequencies{};
// };

// const SymbolTableData<uint8_t> Data8(MessageSize);
// const SymbolTableData<uint16_t> Data16(MessageSize);
// const SymbolTableData<uint32_t> Data32(MessageSize);

// template <typename T>
// const auto& getData()
// {
//   if constexpr (std::is_same_v<uint8_t, T>) {
//     return Data8;
//   } else if constexpr (std::is_same_v<uint16_t, T>) {
//     return Data16;
//   } else {
//     return Data32;
//   }
// };

// ransState_t encode(ransState_t state, const cpp::EncoderSymbol<ransState_t>& symbol)
// {
//   // x = C(s,x)
//   ransState_t quotient = static_cast<ransState_t>((static_cast<uint128_t>(state) * symbol.getReciprocalFrequency()) >> 64);
//   quotient = quotient >> symbol.getReciprocalShift();

//   return state + symbol.getBias() + quotient * symbol.getFrequencyComplement();
// };

// template <simd::SIMDWidth width_V>
// auto SIMDEncode(simd::epi64cV_t<width_V> states,
//                 simd::pdcV_t<width_V> nSamples,
//                 simd::ArrayView<const simd::Symbol*, simd::getElementCount<ransState_t>(width_V)> symbols)
// {
//   const auto [frequencies, cumulativeFrequencies] = simd::aosToSoa(symbols);
//   return simd::ransEncode(states,
//                           simd::int32ToDouble<width_V>(simd::toConstSIMDView(frequencies)),
//                           simd::int32ToDouble<width_V>(simd::toConstSIMDView(cumulativeFrequencies)),
//                           nSamples);
// };

// template <typename source_T>
// static void rans(benchmark::State& st)
// {
//   using namespace o2::rans;

//   const auto& data = getData<source_T>();
//   internal::SymbolTable<internal::cpp::EncoderSymbol<ransState_t>> symbolTable(data.getRenormedFrequencies());
//   const ransState_t state = 1ull << 31;

//   for (auto _ : st) {
//     for (size_t i = data.getSourceMessage().size(); i-- > 0;) {
//       auto symbol = symbolTable[data.getSourceMessage()[i]];
//       benchmark::DoNotOptimize(encode(state, symbol));
//     }
//   }

//   st.SetItemsProcessed(int64_t(st.iterations()) * getData<source_T>().getSourceMessage().size());
//   st.SetBytesProcessed(int64_t(st.iterations()) * getData<source_T>().getSourceMessage().size() * sizeof(source_T));
// };

// template <typename source_T, simd::SIMDWidth width_V>
// static void ransSIMD(benchmark::State& st)
// {
//   static constexpr size_t nElems = simd::getElementCount<ransState_t>(width_V);
//   const auto& data = getData<source_T>();
//   const auto& sourceMessage = data.getSourceMessage();

//   simd::SymbolTable symbolTable(data.getRenormedFrequencies());
//   const simd::epi64_t<width_V> states{1ull << 20};
//   simd::pd_t<width_V> nSamples{static_cast<double>(pow2(
//     getData<source_T>().getRenormedFrequencies().getRenormingBits()))};

// #ifdef ENABLE_VTUNE_PROFILER
//   __itt_resume();
// #endif
//   for (auto _ : st) {
//     for (auto srcIter = sourceMessage.end(); srcIter != sourceMessage.begin();) {
//       auto [it, symbols] = getSymbols<decltype(srcIter), nElems>(srcIter, symbolTable);
//       benchmark::DoNotOptimize(SIMDEncode(simd::toConstSIMDView(states), simd::toConstSIMDView(nSamples), symbols));
//       srcIter = it;
//     }
//   }
// #ifdef ENABLE_VTUNE_PROFILER
//   __itt_pause();
// #endif

//   st.SetItemsProcessed(int64_t(st.iterations()) * getData<source_T>().getSourceMessage().size());
//   st.SetBytesProcessed(int64_t(st.iterations()) * getData<source_T>().getSourceMessage().size() * sizeof(source_T));
// };

// template <typename source_T>
// static void ransSSE(benchmark::State& st)
// {
//   static constexpr size_t nElems = simd::getElementCount<count_t>(simd::SIMDWidth::SSE);
//   const auto& data = getData<source_T>();
//   const auto& sourceMessage = data.getSourceMessage();

//   simd::SymbolTable symbolTable(data.getRenormedFrequencies());
//   const simd::epi64_t<simd::SIMDWidth::SSE> states{1ull << 20};
//   simd::pd_t<simd::SIMDWidth::SSE> nSamples{static_cast<double_t>(pow2(
//     getData<source_T>().getRenormedFrequencies().getRenormingBits()))};

//   // #ifdef ENABLE_VTUNE_PROFILER
//   //   __itt_resume();
//   // #endif
//   for (auto _ : st) {
//     for (auto srcIter = sourceMessage.end(); srcIter != sourceMessage.begin();) {
//       auto [it, symbols] = getSymbols<decltype(srcIter), nElems>(srcIter, symbolTable);
//       const auto [frequencies, cumulativeFrequencies] = aosToSoa(symbols);
//       const auto frequenciesUpper = getUpper(frequencies);
//       const auto cumulativeUpper = getUpper(cumulativeFrequencies);

//       benchmark::DoNotOptimize(simd::ransEncode(simd::toConstSIMDView(states),
//                                                 simd::int32ToDouble<simd::SIMDWidth::SSE>(simd::toConstSIMDView(frequencies)),
//                                                 simd::int32ToDouble<simd::SIMDWidth::SSE>(simd::toConstSIMDView(cumulativeFrequencies)),
//                                                 simd::toConstSIMDView(nSamples)));
//       benchmark::DoNotOptimize(simd::ransEncode(simd::toConstSIMDView(states),
//                                                 simd::int32ToDouble<simd::SIMDWidth::SSE>(simd::toConstSIMDView(frequenciesUpper)),
//                                                 simd::int32ToDouble<simd::SIMDWidth::SSE>(simd::toConstSIMDView(cumulativeUpper)),
//                                                 simd::toConstSIMDView(nSamples)));
//       srcIter = it;
//     }
//   }
//   // #ifdef ENABLE_VTUNE_PROFILER
//   //   __itt_pause();
//   // #endif

//   st.SetItemsProcessed(int64_t(st.iterations()) * getData<source_T>().getSourceMessage().size());
//   st.SetBytesProcessed(int64_t(st.iterations()) * getData<source_T>().getSourceMessage().size() * sizeof(source_T));
// };

// template <typename source_T>
// static void ransAVX(benchmark::State& st)
// {
//   static constexpr size_t nElems = simd::getElementCount<count_t>(simd::SIMDWidth::AVX);
//   const auto& data = getData<source_T>();
//   const auto& sourceMessage = data.getSourceMessage();
//   const auto renormingBits = getData<source_T>().getRenormedFrequencies().getRenormingBits();

//   simd::SymbolTable symbolTable(data.getRenormedFrequencies());

//   const simd::epi64_t<simd::SIMDWidth::AVX, 2> initialStates = {1ull << 20};
//   simd::pd_t<simd::SIMDWidth::AVX> nSamples{static_cast<double>(pow2(renormingBits))};

//   simd::epi64_t<simd::SIMDWidth::AVX, 2> states = initialStates;
//   simd::epi64_t<simd::SIMDWidth::AVX, 2> newStates = states;

//   std::vector<stream_t> out(data.getSourceMessage().size() * 4);

// #ifdef ENABLE_VTUNE_PROFILER
//   __itt_resume();
// #endif
//   for (auto _ : st) {
//     auto outIter = out.data();
//     states = initialStates;

//     for (auto srcIter = sourceMessage.end(); srcIter != sourceMessage.begin(); srcIter -= 8) {

//       auto [i, symbols] = getSymbols<decltype(srcIter), 8>(srcIter, symbolTable);

//       benchmark::DoNotOptimize(std::tie(outIter, newStates) = simd::ransRenorm<decltype(outIter),
//                                                                                1ull << 20,
//                                                                                32>(simd::toConstSIMDView(states),
//                                                                                    simd::toConstSIMDView(symbols.frequencies),
//                                                                                    renormingBits,
//                                                                                    outIter));
//       simd::ransEncode(simd::toConstSIMDView(newStates).subView<0, 1>(),
//                        simd::int32ToDouble<simd::SIMDWidth::AVX>(simd::toConstSIMDView(frequencies).subView<0, 1>()),
//                        simd::int32ToDouble<simd::SIMDWidth::AVX>(simd::toConstSIMDView(cumulatedFrequencies).subView<0, 1>()),
//                        simd::toConstSIMDView(nSamples),
//                        simd::toSIMDView(states).subView<0, 1>());
//       simd::ransEncode(simd::toConstSIMDView(newStates).subView<1, 1>(),
//                        simd::int32ToDouble<simd::SIMDWidth::AVX>(simd::toConstSIMDView(frequencies).subView<1, 1>()),
//                        simd::int32ToDouble<simd::SIMDWidth::AVX>(simd::toConstSIMDView(cumulatedFrequencies).subView<1, 1>()),
//                        simd::toConstSIMDView(nSamples),
//                        simd::toSIMDView(states).subView<1, 1>());
//     }
//     benchmark::DoNotOptimize(outIter);
//   }
// #ifdef ENABLE_VTUNE_PROFILER
//   __itt_pause();
// #endif

//   st.SetItemsProcessed(int64_t(st.iterations()) * getData<source_T>().getSourceMessage().size());
//   st.SetBytesProcessed(int64_t(st.iterations()) * getData<source_T>().getSourceMessage().size() * sizeof(source_T));
// };

// BENCHMARK_TEMPLATE1(rans, uint8_t);
// BENCHMARK_TEMPLATE1(rans, uint16_t);
// BENCHMARK_TEMPLATE1(rans, uint32_t);

// BENCHMARK_TEMPLATE2(ransSIMD, uint8_t, simd::SIMDWidth::SSE);
// BENCHMARK_TEMPLATE2(ransSIMD, uint16_t, simd::SIMDWidth::SSE);
// BENCHMARK_TEMPLATE2(ransSIMD, uint32_t, simd::SIMDWidth::SSE);

// BENCHMARK_TEMPLATE2(ransSIMD, uint8_t, simd::SIMDWidth::AVX);
// BENCHMARK_TEMPLATE2(ransSIMD, uint16_t, simd::SIMDWidth::AVX);
// BENCHMARK_TEMPLATE2(ransSIMD, uint32_t, simd::SIMDWidth::AVX);

// BENCHMARK_TEMPLATE1(ransSSE, uint8_t);
// BENCHMARK_TEMPLATE1(ransSSE, uint16_t);
// BENCHMARK_TEMPLATE1(ransSSE, uint32_t);

// BENCHMARK_TEMPLATE1(ransAVX, uint8_t);
// BENCHMARK_TEMPLATE1(ransAVX, uint16_t);
// BENCHMARK_TEMPLATE1(ransAVX, uint32_t);

BENCHMARK_MAIN();
