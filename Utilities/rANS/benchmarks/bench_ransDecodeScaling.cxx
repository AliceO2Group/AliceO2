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
class SourceMessageUniform
{
 public:
  SourceMessageUniform(size_t messageSize, size_t max) : mMax{max}
  {
    std::mt19937 mt(0); // same seed we want always the same distrubution of random numbers;
    std::uniform_int_distribution<source_T> dist(0, max);
    const size_t sourceSize = messageSize / sizeof(source_T) + 1;
    mSourceMessage.resize(sourceSize);
#ifdef RANS_PARALLEL_STL
    std::generate(std::execution::par_unseq, mSourceMessage.begin(), mSourceMessage.end(), [&dist, &mt]() { return dist(mt); });
#else
    std::generate(mSourceMessage.begin(), mSourceMessage.end(), [&dist, &mt]() { return dist(mt); });
#endif // RANS_PARALLEL_STL
  }

  const auto& get() const { return mSourceMessage; };
  size_t getMax() const { return mMax; };

 private:
  size_t mMax{};
  std::vector<source_T> mSourceMessage{};
};

SourceMessageUniform<uint32_t> sourceMessage{0, 0};

void ransDecodeBenchmark(benchmark::State& st)
{

  using source_type = uint32_t;
  size_t max = utils::pow2(st.range(0));

  if (max != sourceMessage.getMax()) {
    sourceMessage = SourceMessageUniform<uint32_t>{MessageSize, max};
  }
  const auto& inputData = sourceMessage.get();
  EncodeBuffer<source_type>
    encodeBuffer{inputData.size()};
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

BENCHMARK(ransDecodeBenchmark)->DenseRange(8, 27, 1);

BENCHMARK_MAIN();