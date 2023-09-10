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

/// @file   bench_ransEncode.cxx
/// @author Michael Lettrich
/// @brief  compares performance of different encoders

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

#ifdef ENABLE_VTUNE_PROFILER
#include <ittnotify.h>
#endif

#include "helpers.h"

using namespace o2::rans;

inline constexpr size_t MessageSize = 1ull << 22;

// template <typename source_T>
// class SourceMessageProxyBinomial
// {
//  public:
//   SourceMessageProxyBinomial(size_t messageSize)
//   {
//     if (mSourceMessage.empty()) {
//       std::mt19937 mt(0); // same seed we want always the same distrubution of random numbers;
//       const size_t draws = std::min(1ul << 27, static_cast<size_t>(std::numeric_limits<source_T>::max()));
//       const double probability = 0.5;
//       std::binomial_distribution<source_T> dist(draws, probability);
//       const size_t sourceSize = messageSize / sizeof(source_T) + 1;
//       mSourceMessage.resize(sourceSize);
// #ifdef RANS_PARALLEL_STL
//       std::generate(std::execution::par_unseq, mSourceMessage.begin(), mSourceMessage.end(), [&dist, &mt]() { return dist(mt); });
// #else
//       std::generate(mSourceMessage.begin(), mSourceMessage.end(), [&dist, &mt]() { return dist(mt); });
// #endif // RANS_PARALLEL_STL
//     }
//   }

//   const auto& get() const { return mSourceMessage; };

//  private:
//   std::vector<source_T> mSourceMessage{};
// };

// inline const SourceMessageProxyBinomial<uint8_t> sourceMessageBinomial8{MessageSize};
// inline const SourceMessageProxyBinomial<uint16_t> sourceMessageBinomial16{MessageSize};
// inline const SourceMessageProxyBinomial<uint32_t> sourceMessageBinomial32{MessageSize};

template <typename source_T>
class SourceMessageProxyUniform
{
 public:
  SourceMessageProxyUniform(size_t messageSize)
  {
    if (mSourceMessage.empty()) {
      std::mt19937 mt(0); // same seed we want always the same distrubution of random numbers;
      const size_t min = 0;
      const double max = std::min(1ul << 27, static_cast<size_t>(std::numeric_limits<source_T>::max()));
      std::uniform_int_distribution<source_T> dist(min, max);
      const size_t sourceSize = messageSize / sizeof(source_T) + 1;
      mSourceMessage.resize(sourceSize);
#ifdef RANS_PARALLEL_STL
      std::generate(std::execution::par_unseq, mSourceMessage.begin(), mSourceMessage.end(), [&dist, &mt]() { return dist(mt); });
#else
      std::generate(mSourceMessage.begin(), mSourceMessage.end(), [&dist, &mt]() { return dist(mt); });
#endif // RANS_PARALLEL_STL
    }
  }

  const auto& get() const { return mSourceMessage; };

 private:
  std::vector<source_T> mSourceMessage{};
};

inline const SourceMessageProxyUniform<uint8_t> sourceMessageUniform8{MessageSize};
inline const SourceMessageProxyUniform<uint16_t> sourceMessageUniform16{MessageSize};
inline const SourceMessageProxyUniform<uint32_t> sourceMessageUniform32{MessageSize};

template <class... Args>
void ransDecodeBenchmark(benchmark::State& st, Args&&... args)
{

  auto args_tuple = std::make_tuple(std::move(args)...);

  const auto& inputData = std::get<0>(args_tuple).get();

  using input_data_type = std::remove_cv_t<std::remove_reference_t<decltype(inputData)>>;
  using source_type = typename input_data_type::value_type;

  EncodeBuffer<source_type> encodeBuffer{inputData.size()};
  DecodeBuffer<source_type> decodeBuffer{inputData.size()};

  const auto histogram = makeDenseHistogram::fromSamples(gsl::span<const source_type>(inputData));
  Metrics<source_type> metrics{histogram};
  const auto renormedHistogram = renorm(histogram, metrics, RenormingPolicy::Auto, 10);

  auto encoder = makeDenseEncoder<>::fromRenormed(renormedHistogram);
  encodeBuffer.encodeBufferEnd = encoder.process(inputData.data(), inputData.data() + inputData.size(), encodeBuffer.buffer.data());

  auto decoder = makeDecoder<>::fromRenormed(renormedHistogram);
#ifdef ENABLE_VTUNE_PROFILER
  __itt_resume();
#endif
  for (auto _ : st) {
    decoder.process(encodeBuffer.encodeBufferEnd, decodeBuffer.buffer.data(), inputData.size(), encoder.getNStreams());
  }
#ifdef ENABLE_VTUNE_PROFILER
  __itt_pause();
#endif

  if (!(decodeBuffer == inputData)) {
    st.SkipWithError("Missmatch between encoded and decoded Message");
  }

  const auto& datasetProperties = metrics.getDatasetProperties();
  st.SetItemsProcessed(static_cast<int64_t>(inputData.size()) * static_cast<int64_t>(st.iterations()));
  st.SetBytesProcessed(static_cast<int64_t>(inputData.size()) * sizeof(source_type) * static_cast<int64_t>(st.iterations()));
  st.counters["AlphabetRangeBits"] = datasetProperties.alphabetRangeBits;
  st.counters["nUsedAlphabetSymbols"] = datasetProperties.nUsedAlphabetSymbols;
  st.counters["SymbolTablePrecision"] = renormedHistogram.getRenormingBits();
  st.counters["Entropy"] = datasetProperties.entropy;
  st.counters["ExpectedCodewordLength"] = computeExpectedCodewordLength(histogram, renormedHistogram);
  st.counters["SourceSize"] = inputData.size() * sizeof(source_type);
  st.counters["CompressedSize"] = std::distance(encodeBuffer.buffer.data(), encodeBuffer.encodeBufferEnd) * sizeof(typename decltype(encoder)::stream_type);
  st.counters["Compression"] = st.counters["SourceSize"] / static_cast<double>(st.counters["CompressedSize"]);
  st.counters["LowerBound"] = inputData.size() * (static_cast<double>(st.counters["Entropy"]) / 8);
  st.counters["CompressionWRTEntropy"] = st.counters["CompressedSize"] / st.counters["LowerBound"];
};

// BENCHMARK_CAPTURE(ransDecodeBenchmark, decode_binomial_8, sourceMessageBinomial8);
// BENCHMARK_CAPTURE(ransDecodeBenchmark, decode_binomial_16, sourceMessageBinomial16);
// BENCHMARK_CAPTURE(ransDecodeBenchmark, decode_binomial_32, sourceMessageBinomial32);

BENCHMARK_CAPTURE(ransDecodeBenchmark, decode_uniform_8, sourceMessageUniform8);
BENCHMARK_CAPTURE(ransDecodeBenchmark, decode_uniform_16, sourceMessageUniform16);
BENCHMARK_CAPTURE(ransDecodeBenchmark, decode_uniform_32, sourceMessageUniform32);

BENCHMARK_MAIN();