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
#include "rANS/internal/containers/DenseHistogram.h"
#include "rANS/internal/containers/AdaptiveHistogram.h"

#ifdef ENABLE_VTUNE_PROFILER
#include <ittnotify.h>
#endif

#include "helpers.h"

using namespace o2::rans;

inline constexpr size_t MessageSize = 1ull << 22;

template <typename source_T>
class SourceMessageProxyBinomial
{
 public:
  SourceMessageProxyBinomial() = default;

  const auto& get(size_t messageSize)
  {
    if (mSourceMessage.empty()) {
      LOG(info) << "generating Binomial distribution";

      std::mt19937 mt(0); // same seed we want always the same distrubution of random numbers;
      const size_t draws = std::min(1ul << 27, static_cast<size_t>(std::numeric_limits<source_T>::max()));
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

    return mSourceMessage;
  };

 private:
  std::vector<source_T> mSourceMessage{};
};

inline SourceMessageProxyBinomial<uint32_t> sourceMessageBinomial32{};

template <typename source_T>
class SourceMessageProxyUniform
{
 public:
  SourceMessageProxyUniform() = default;

  const auto& get(size_t messageSize)
  {
    if (mSourceMessage.empty()) {
      LOG(info) << "generating Uniform distribution";

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
#endif
    }
    return mSourceMessage;
  };

 private:
  std::vector<source_T> mSourceMessage{};
};

inline SourceMessageProxyUniform<uint32_t> sourceMessageUniform32{};

template <class... Args>
void ransMakeHistogramBenchmark(benchmark::State& st, Args&&... args)
{

  auto args_tuple = std::make_tuple(std::move(args)...);

  const auto& inputData = std::get<0>(args_tuple).get(MessageSize);

  using histogram_type = std::remove_cv_t<std::remove_reference_t<decltype(std::get<1>(args_tuple))>>;
  using input_data_type = std::remove_cv_t<std::remove_reference_t<decltype(inputData)>>;
  using source_type = typename input_data_type::value_type;

  const auto histogram = makeDenseHistogram::fromSamples(gsl::span<const source_type>(inputData));
  Metrics<source_type> metrics{histogram};

#ifdef ENABLE_VTUNE_PROFILER
  __itt_resume();
#endif
  for (auto _ : st) {
    histogram_type hist{};
    // hist.addSamples(gsl::span<const source_type>(inputData));
    hist.addSamples(inputData.begin(), inputData.end());
    benchmark::DoNotOptimize(hist);
  }
#ifdef ENABLE_VTUNE_PROFILER
  __itt_pause();
#endif

  bool isSame = true;
  histogram_type hist{};
  hist.addSamples(gsl::span<const source_type>(inputData));

  for (std::ptrdiff_t symbol = histogram.getOffset(); symbol != histogram.getOffset() + static_cast<std::ptrdiff_t>(histogram.size()); ++symbol) {
    if (histogram[symbol] > 0) {
      LOG_IF(info, histogram[symbol] != hist[symbol]) << fmt::format("[{}]: {} != {}", symbol, hist[symbol], histogram[symbol]);
      isSame = isSame && (histogram[symbol] == hist[symbol]);
    }
  }

  if (!(isSame)) {
    st.SkipWithError("Missmatch between encoded and decoded Message");
  }

  const auto& datasetProperties = metrics.getDatasetProperties();
  st.SetItemsProcessed(static_cast<int64_t>(inputData.size()) * static_cast<int64_t>(st.iterations()));
  st.SetBytesProcessed(static_cast<int64_t>(inputData.size()) * sizeof(source_type) * static_cast<int64_t>(st.iterations()));
  st.counters["AlphabetRangeBits"] = datasetProperties.alphabetRangeBits;
  st.counters["nUsedAlphabetSymbols"] = datasetProperties.nUsedAlphabetSymbols;
  st.counters["HistogramSize"] = hist.size() * sizeof(source_type);
  st.counters["Entropy"] = datasetProperties.entropy;
  st.counters["SourceSize"] = inputData.size() * sizeof(source_type);
  st.counters["LowerBound"] = inputData.size() * (static_cast<double>(st.counters["Entropy"]) / 8);
};

template <class... Args>
void ransAccessHistogramBenchmark(benchmark::State& st, Args&&... args)
{

  auto args_tuple = std::make_tuple(std::move(args)...);
  const auto& inputData = std::get<0>(args_tuple).get(MessageSize);

  using histogram_type = std::remove_cv_t<std::remove_reference_t<decltype(std::get<1>(args_tuple))>>;
  using input_data_type = std::remove_cv_t<std::remove_reference_t<decltype(inputData)>>;
  using source_type = typename input_data_type::value_type;

  const auto histogram = makeDenseHistogram::fromSamples(gsl::span<const source_type>(inputData));
  Metrics<source_type> metrics{histogram};

  histogram_type hist{};
  hist.addSamples(gsl::span<const source_type>(inputData));

#ifdef ENABLE_VTUNE_PROFILER
  __itt_resume();
#endif
  for (auto _ : st) {
    for (auto& symbol : inputData) {
      const uint32_t t = hist[symbol];
      benchmark::DoNotOptimize(t);
    }
  }
#ifdef ENABLE_VTUNE_PROFILER
  __itt_pause();
#endif

  bool isSame = true;
  for (std::ptrdiff_t symbol = histogram.getOffset(); symbol != histogram.getOffset() + static_cast<std::ptrdiff_t>(histogram.size()); ++symbol) {
    if (histogram[symbol] > 0) {
      LOG_IF(info, histogram[symbol] != hist[symbol]) << fmt::format("[{}]: {} != {}", symbol, hist[symbol], histogram[symbol]);
      isSame = isSame && (histogram[symbol] == hist[symbol]);
    }
  }

  if (!(isSame)) {
    st.SkipWithError("Missmatch between encoded and decoded Message");
  }

  const auto& datasetProperties = metrics.getDatasetProperties();
  st.SetItemsProcessed(static_cast<int64_t>(inputData.size()) * static_cast<int64_t>(st.iterations()));
  st.SetBytesProcessed(static_cast<int64_t>(inputData.size()) * sizeof(source_type) * static_cast<int64_t>(st.iterations()));
  st.counters["AlphabetRangeBits"] = datasetProperties.alphabetRangeBits;
  st.counters["nUsedAlphabetSymbols"] = datasetProperties.nUsedAlphabetSymbols;
  st.counters["HistogramSize"] = hist.size() * sizeof(source_type);
  st.counters["Entropy"] = datasetProperties.entropy;
  st.counters["SourceSize"] = inputData.size() * sizeof(source_type);
  st.counters["LowerBound"] = inputData.size() * (static_cast<double>(st.counters["Entropy"]) / 8);
};

BENCHMARK_CAPTURE(ransMakeHistogramBenchmark, makeHistogram_Vector_binomial_32, std::reference_wrapper(sourceMessageBinomial32), DenseHistogram<uint32_t>{});
BENCHMARK_CAPTURE(ransMakeHistogramBenchmark, makeHistogram_Vector_uniform_32, std::reference_wrapper(sourceMessageUniform32), DenseHistogram<uint32_t>{});

BENCHMARK_CAPTURE(ransMakeHistogramBenchmark, makeHistogram_SparseVector_binomial_32, std::reference_wrapper(sourceMessageBinomial32), AdaptiveHistogram<uint32_t>{});
BENCHMARK_CAPTURE(ransMakeHistogramBenchmark, makeHistogram_SparseVector_uniform_32, std::reference_wrapper(sourceMessageUniform32), AdaptiveHistogram<uint32_t>{});

BENCHMARK_CAPTURE(ransAccessHistogramBenchmark, accessHistogram_Vector_binomial_32, std::reference_wrapper(sourceMessageBinomial32), DenseHistogram<uint32_t>{});
BENCHMARK_CAPTURE(ransAccessHistogramBenchmark, accessHistogram_Vector_uniform_32, std::reference_wrapper(sourceMessageUniform32), DenseHistogram<uint32_t>{});

BENCHMARK_CAPTURE(ransAccessHistogramBenchmark, accessHistogram_SparseVector_binomial_32, std::reference_wrapper(sourceMessageBinomial32), AdaptiveHistogram<uint32_t>{});
BENCHMARK_CAPTURE(ransAccessHistogramBenchmark, accessHistogram_SparseVector_uniform_32, std::reference_wrapper(sourceMessageUniform32), AdaptiveHistogram<uint32_t>{});

BENCHMARK_MAIN();