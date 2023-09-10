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

/// @file   bench_ransUnPack.h
/// @author Michael Lettrich
/// @brief  benchmark unpacking of data compared to memcpy

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

#include "rANS/internal/pack/pack.h"

#ifdef ENABLE_VTUNE_PROFILER
#include <ittnotify.h>
#endif

#include "helpers.h"

using namespace o2::rans;

inline constexpr size_t MessageSize = 1ull << 22;

using source_type = uint32_t;

template <typename source_T>
std::vector<source_T> makeRandomUniformVector(size_t nelems, source_T min = std::numeric_limits<source_T>::max(), source_T max = std::numeric_limits<source_T>::max())
{
  std::vector<source_T> result(nelems, 0);
  std::mt19937 mt(0); // same seed we want always the same distrubution of random numbers;
  std::uniform_int_distribution<source_T> dist(min, max);

#ifdef RANS_PARALLEL_STL
  std::generate(std::execution::par_unseq, result.begin(), result.end(), [&dist, &mt]() { return dist(mt); });
#else
  std::generate(result.begin(), result.end(), [&dist, &mt]() { return dist(mt); });
#endif // RANS_PARALLEL_STL
  return result;
};

static void copyBenchmark(benchmark::State& state)
{
  std::vector<source_type> src = makeRandomUniformVector<source_type>(MessageSize);
  std::vector<source_type> dst(MessageSize, 0);
  for (auto _ : state) {
    std::copy(src.begin(), src.end(), dst.begin());
  };
  state.SetItemsProcessed(src.size() * state.iterations());
  state.SetBytesProcessed(src.size() * sizeof(source_type) * state.iterations());
};

static void unpackingBenchmark(benchmark::State& state)
{
  size_t packingBits = state.range(0);

  std::vector<source_type> src = makeRandomUniformVector<source_type>(MessageSize, 0, utils::pow2(packingBits) - 1);
  std::vector<uint32_t> dst(MessageSize, 0);

  BitPtr iter{dst.data()};
  for (auto i : src) {
    iter = internal::pack(iter, i, packingBits);
  }

  std::vector<uint32_t> unpacked(MessageSize, 0);

  for (auto _ : state) {
    BitPtr iter{dst.data()};
    for (size_t i = 0; i < src.size(); ++i) {
      unpacked[i] = internal::unpack<uint32_t>(iter, packingBits);
      iter += packingBits;
    }
  }

  state.SetItemsProcessed(src.size() * state.iterations());
  state.SetBytesProcessed(src.size() * sizeof(uint32_t) * state.iterations());

  if (!std::equal(unpacked.begin(), unpacked.end(), src.begin())) {
    state.SkipWithError("error in packing");
  }
};

BENCHMARK(copyBenchmark);
BENCHMARK(unpackingBenchmark)->DenseRange(1, 32, 1);
BENCHMARK_MAIN();
