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

template <typename source_T>
class SourceMessageProxy
{
 public:
  SourceMessageProxy(size_t messageSize)
  {
    if (mSourceMessage.empty()) {
      std::mt19937 mt(0); // same seed we want always the same distrubution of random numbers;
      const size_t draws = std::min(utils::pow2(20), static_cast<size_t>(std::numeric_limits<source_T>::max()));
      const double probability = 0.5;
      std::binomial_distribution<source_T> dist(draws, probability);
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

inline const SourceMessageProxy<uint8_t> sourceMessage8{MessageSize};
inline const SourceMessageProxy<uint16_t> sourceMessage16{MessageSize};
inline const SourceMessageProxy<uint32_t> sourceMessage32{MessageSize};

template <typename T>
const auto& getMessage()
{
  if constexpr (std::is_same_v<uint8_t, T>) {
    return sourceMessage8.get();
  } else if constexpr (std::is_same_v<uint16_t, T>) {
    return sourceMessage16.get();
  } else {
    return sourceMessage32.get();
  }
};

template <typename source_T, CoderTag coderTag_V>
void ransCompressionBenchmark(benchmark::State& st)
{
  using source_type = source_T;

  const auto& inputData = getMessage<source_type>();
  EncodeBuffer<source_type> encodeBuffer{inputData.size()};
  DecodeBuffer<source_type> decodeBuffer{inputData.size()};

  const auto histogram = makeDenseHistogram::fromSamples(gsl::span<const source_type>(inputData));
  Metrics<source_type> metrics{histogram};
  const auto renormedHistogram = renorm(histogram, metrics, RenormingPolicy::Auto, 10);
  auto encoder = makeDenseEncoder<coderTag_V>::fromRenormed(renormedHistogram);

#ifdef ENABLE_VTUNE_PROFILER
  __itt_resume();
#endif
  for (auto _ : st) {
    benchmark::DoNotOptimize(encodeBuffer.encodeBufferEnd = encoder.process(inputData.data(), inputData.data() + inputData.size(), encodeBuffer.buffer.data()));
  }
#ifdef ENABLE_VTUNE_PROFILER
  __itt_pause();
#endif

  auto decoder = makeDecoder<>::fromRenormed(renormedHistogram);
  decoder.process(encodeBuffer.encodeBufferEnd, decodeBuffer.buffer.data(), inputData.size(), encoder.getNStreams());
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

template <typename source_T, CoderTag coderTag_V>
void ransLiteralCompressionBenchmark(benchmark::State& st)
{
  using source_type = source_T;

  const auto& inputData = getMessage<source_type>();
  EncodeBuffer<source_type> encodeBuffer{inputData.size()};
  encodeBuffer.literals.resize(inputData.size(), 0);
  encodeBuffer.literalsEnd = encodeBuffer.literals.data();
  DecodeBuffer<source_type> decodeBuffer{inputData.size()};

  const auto histogram = makeDenseHistogram::fromSamples(gsl::span<const source_type>(inputData));
  Metrics<source_type> metrics{histogram};
  const auto renormedHistogram = renorm(histogram, metrics);
  auto encoder = makeDenseEncoder<coderTag_V>::fromRenormed(renormedHistogram);

#ifdef ENABLE_VTUNE_PROFILER
  __itt_resume();
#endif
  for (auto _ : st) {
    encodeBuffer.literalsEnd = encodeBuffer.literals.data();
    benchmark::DoNotOptimize(std::tie(encodeBuffer.encodeBufferEnd, encodeBuffer.literalsEnd) = encoder.process(inputData.data(), inputData.data() + inputData.size(), encodeBuffer.buffer.data(), encodeBuffer.literalsEnd));
  }
#ifdef ENABLE_VTUNE_PROFILER
  __itt_pause();
#endif

  auto decoder = makeDecoder<>::fromRenormed(renormedHistogram);
  decoder.process(encodeBuffer.encodeBufferEnd, decodeBuffer.buffer.data(), inputData.size(), encoder.getNStreams(), encodeBuffer.literalsEnd);
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

template <typename source_T, CoderTag coderTag_V>
void ransAdaptiveCompressionBenchmark(benchmark::State& st)
{
  using source_type = source_T;

  const auto& inputData = getMessage<source_type>();
  EncodeBuffer<source_type> encodeBuffer{inputData.size()};
  DecodeBuffer<source_type> decodeBuffer{inputData.size()};

  const auto histogram = makeDenseHistogram::fromSamples(gsl::span<const source_type>(inputData));
  auto adaptiveHistogram = makeAdaptiveHistogram::fromSamples(gsl::span<const source_type>(inputData));
  Metrics<source_type> metrics{histogram};
  Metrics<source_type> adaptiveMetrics{adaptiveHistogram};
  const auto renormedHistogram = renorm(histogram, metrics, RenormingPolicy::Auto, 10);
  const auto renormedAdaptiveHistogram = renorm(std::move(adaptiveHistogram), adaptiveMetrics, RenormingPolicy::Auto, 10);

  auto encoder = makeAdaptiveEncoder<coderTag_V>::fromRenormed(renormedAdaptiveHistogram);

#ifdef ENABLE_VTUNE_PROFILER
  __itt_resume();
#endif
  for (auto _ : st) {
    benchmark::DoNotOptimize(encodeBuffer.encodeBufferEnd = encoder.process(inputData.data(), inputData.data() + inputData.size(), encodeBuffer.buffer.data()));
  }
#ifdef ENABLE_VTUNE_PROFILER
  __itt_pause();
#endif

  auto decoder = makeDecoder<>::fromRenormed(renormedHistogram);
  decoder.process(encodeBuffer.encodeBufferEnd, decodeBuffer.buffer.data(), inputData.size(), encoder.getNStreams());
  if (!(decodeBuffer == inputData)) {
    st.SkipWithError("Missmatch between encoded and decoded Message");
  }

  const auto& datasetProperties = adaptiveMetrics.getDatasetProperties();
  st.SetItemsProcessed(static_cast<int64_t>(inputData.size()) * static_cast<int64_t>(st.iterations()));
  st.SetBytesProcessed(static_cast<int64_t>(inputData.size()) * sizeof(source_type) * static_cast<int64_t>(st.iterations()));
  st.counters["AlphabetRangeBits"] = datasetProperties.alphabetRangeBits;
  st.counters["nUsedAlphabetSymbols"] = datasetProperties.nUsedAlphabetSymbols;
  st.counters["SymbolTablePrecision"] = renormedAdaptiveHistogram.getRenormingBits();
  st.counters["Entropy"] = datasetProperties.entropy;
  st.counters["ExpectedCodewordLength"] = computeExpectedCodewordLength(histogram, renormedHistogram);
  st.counters["SourceSize"] = inputData.size() * sizeof(source_type);
  st.counters["CompressedSize"] = std::distance(encodeBuffer.buffer.data(), encodeBuffer.encodeBufferEnd) * sizeof(typename decltype(encoder)::stream_type);
  st.counters["Compression"] = st.counters["SourceSize"] / static_cast<double>(st.counters["CompressedSize"]);
  st.counters["LowerBound"] = inputData.size() * (static_cast<double>(st.counters["Entropy"]) / 8);
  st.counters["CompressionWRTEntropy"] = st.counters["CompressedSize"] / st.counters["LowerBound"];
};

template <typename source_T, CoderTag coderTag_V>
void ransAdaptiveLiteralCompressionBenchmark(benchmark::State& st)
{
  using source_type = source_T;

  const auto& inputData = getMessage<source_type>();
  EncodeBuffer<source_type> encodeBuffer{inputData.size()};
  encodeBuffer.literals.resize(inputData.size(), 0);
  encodeBuffer.literalsEnd = encodeBuffer.literals.data();
  DecodeBuffer<source_type> decodeBuffer{inputData.size()};

  const auto histogram = makeDenseHistogram::fromSamples(gsl::span<const source_type>(inputData));
  auto adaptiveHistogram = makeAdaptiveHistogram::fromSamples(gsl::span<const source_type>(inputData));
  Metrics<source_type> metrics{histogram};
  Metrics<source_type> adaptiveMetrics{adaptiveHistogram};
  const auto renormedHistogram = renorm(histogram, metrics);
  const auto renormedAdaptiveHistogram = renorm(std::move(adaptiveHistogram), adaptiveMetrics);

  auto encoder = makeAdaptiveEncoder<coderTag_V>::fromRenormed(renormedAdaptiveHistogram);

#ifdef ENABLE_VTUNE_PROFILER
  __itt_resume();
#endif
  for (auto _ : st) {
    encodeBuffer.literalsEnd = encodeBuffer.literals.data();
    benchmark::DoNotOptimize(std::tie(encodeBuffer.encodeBufferEnd, encodeBuffer.literalsEnd) = encoder.process(inputData.data(), inputData.data() + inputData.size(), encodeBuffer.buffer.data(), encodeBuffer.literalsEnd));
  }
#ifdef ENABLE_VTUNE_PROFILER
  __itt_pause();
#endif

  auto decoder = makeDecoder<>::fromRenormed(renormedHistogram);
  decoder.process(encodeBuffer.encodeBufferEnd, decodeBuffer.buffer.data(), inputData.size(), encoder.getNStreams(), encodeBuffer.literalsEnd);
  if (!(decodeBuffer == inputData)) {
    st.SkipWithError("Missmatch between encoded and decoded Message");
  }

  const auto& datasetProperties = adaptiveMetrics.getDatasetProperties();
  st.SetItemsProcessed(static_cast<int64_t>(inputData.size()) * static_cast<int64_t>(st.iterations()));
  st.SetBytesProcessed(static_cast<int64_t>(inputData.size()) * sizeof(source_type) * static_cast<int64_t>(st.iterations()));
  st.counters["AlphabetRangeBits"] = datasetProperties.alphabetRangeBits;
  st.counters["nUsedAlphabetSymbols"] = datasetProperties.nUsedAlphabetSymbols;
  st.counters["SymbolTablePrecision"] = renormedAdaptiveHistogram.getRenormingBits();
  st.counters["Entropy"] = datasetProperties.entropy;
  st.counters["ExpectedCodewordLength"] = computeExpectedCodewordLength(histogram, renormedHistogram);
  st.counters["SourceSize"] = inputData.size() * sizeof(source_type);
  st.counters["CompressedSize"] = std::distance(encodeBuffer.buffer.data(), encodeBuffer.encodeBufferEnd) * sizeof(typename decltype(encoder)::stream_type);
  st.counters["Compression"] = st.counters["SourceSize"] / static_cast<double>(st.counters["CompressedSize"]);
  st.counters["LowerBound"] = inputData.size() * (static_cast<double>(st.counters["Entropy"]) / 8);
  st.counters["CompressionWRTEntropy"] = st.counters["CompressedSize"] / st.counters["LowerBound"];
};

BENCHMARK(ransCompressionBenchmark<uint8_t, CoderTag::Compat>);
BENCHMARK(ransCompressionBenchmark<uint16_t, CoderTag::Compat>);
BENCHMARK(ransCompressionBenchmark<uint32_t, CoderTag::Compat>);

BENCHMARK(ransAdaptiveCompressionBenchmark<uint32_t, CoderTag::Compat>);
//########################################################################################

#ifdef RANS_SINGLE_STREAM
BENCHMARK(ransCompressionBenchmark<uint8_t, CoderTag::SingleStream>);
BENCHMARK(ransCompressionBenchmark<uint16_t, CoderTag::SingleStream>);
BENCHMARK(ransCompressionBenchmark<uint32_t, CoderTag::SingleStream>);

BENCHMARK(ransAdaptiveCompressionBenchmark<uint32_t, CoderTag::SingleStream>);
#endif /* RANS_SINGLE_STREAM */

//########################################################################################

#ifdef RANS_SSE
BENCHMARK(ransCompressionBenchmark<uint8_t, CoderTag::SSE>);
BENCHMARK(ransCompressionBenchmark<uint16_t, CoderTag::SSE>);
BENCHMARK(ransCompressionBenchmark<uint32_t, CoderTag::SSE>);

BENCHMARK(ransAdaptiveCompressionBenchmark<uint32_t, CoderTag::SSE>);
#endif /* RANS SSE */

// //########################################################################################

#ifdef RANS_AVX2
BENCHMARK(ransCompressionBenchmark<uint8_t, CoderTag::AVX2>);
BENCHMARK(ransCompressionBenchmark<uint16_t, CoderTag::AVX2>);
BENCHMARK(ransCompressionBenchmark<uint32_t, CoderTag::AVX2>);

BENCHMARK(ransAdaptiveCompressionBenchmark<uint32_t, CoderTag::AVX2>);
#endif /* RANS_AVX2 */

//########################################################################################

BENCHMARK(ransLiteralCompressionBenchmark<uint8_t, CoderTag::Compat>);
BENCHMARK(ransLiteralCompressionBenchmark<uint16_t, CoderTag::Compat>);
BENCHMARK(ransLiteralCompressionBenchmark<uint32_t, CoderTag::Compat>);

BENCHMARK(ransAdaptiveLiteralCompressionBenchmark<uint32_t, CoderTag::Compat>);

//########################################################################################

#ifdef RANS_SINGLE_STREAM
BENCHMARK(ransLiteralCompressionBenchmark<uint8_t, CoderTag::SingleStream>);
BENCHMARK(ransLiteralCompressionBenchmark<uint16_t, CoderTag::SingleStream>);
BENCHMARK(ransLiteralCompressionBenchmark<uint32_t, CoderTag::SingleStream>);

BENCHMARK(ransAdaptiveLiteralCompressionBenchmark<uint32_t, CoderTag::SingleStream>);
#endif /* RANS_SINGLE_STREAM */

//########################################################################################

#ifdef RANS_SSE
BENCHMARK(ransLiteralCompressionBenchmark<uint8_t, CoderTag::SSE>);
BENCHMARK(ransLiteralCompressionBenchmark<uint16_t, CoderTag::SSE>);
BENCHMARK(ransLiteralCompressionBenchmark<uint32_t, CoderTag::SSE>);

BENCHMARK(ransAdaptiveLiteralCompressionBenchmark<uint32_t, CoderTag::SSE>);
#endif /* RANS_SSE */

//########################################################################################

#ifdef RANS_AVX2
BENCHMARK(ransLiteralCompressionBenchmark<uint8_t, CoderTag::AVX2>);
BENCHMARK(ransLiteralCompressionBenchmark<uint16_t, CoderTag::AVX2>);
BENCHMARK(ransLiteralCompressionBenchmark<uint32_t, CoderTag::AVX2>);

BENCHMARK(ransAdaptiveLiteralCompressionBenchmark<uint32_t, CoderTag::AVX2>);
#endif /* RANS_AVX2 */

BENCHMARK_MAIN();