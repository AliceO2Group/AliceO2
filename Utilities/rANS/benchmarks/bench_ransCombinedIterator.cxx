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
/// @since  Nov 18, 2020
/// @brief

#include <vector>

#include <benchmark/benchmark.h>

#include "rANS/utils.h"

static void BM_Array_Read_Copy(benchmark::State& state)
{
  std::vector<uint32_t> a(state.range(0), 0x0);
  std::vector<uint32_t> b(state.range(0), 0x1);
  std::vector<uint32_t> c(state.range(0), 0x0);
  for (auto _ : state) {
    std::vector<uint32_t> tmp(state.range(0));

    for (size_t i = 0; i < a.size(); ++i) {
      tmp[i] = b[i] + (a[i] << 16);
    }

    for (size_t i = 0; i < tmp.size(); ++i) {
      c[i] = tmp[i] + 1;
    }
  }
}

BENCHMARK(BM_Array_Read_Copy)->RangeMultiplier(2)->Range(10e4, 2 * 10e6);

static void BM_Array_Read_Iterator(benchmark::State& state)
{
  std::vector<uint32_t> a(state.range(0), 0x0);
  std::vector<uint32_t> b(state.range(0), 0x1);
  std::vector<uint32_t> c(state.range(0), 0x0);

  for (auto _ : state) {
    auto readOP = [](auto iterA, auto iterB) -> uint32_t {
      return *iterB + (*iterA << 16);
    };

    const o2::rans::utils::CombinedInputIterator begin(a.begin(), b.begin(), readOP);
    const o2::rans::utils::CombinedInputIterator end(a.end(), b.end(), readOP);

    auto cIter = c.begin();

    for (auto iter = begin; iter != end; ++iter) {
      *cIter = *iter + 1;
      ++cIter;
    }
  }
}

BENCHMARK(BM_Array_Read_Iterator)->RangeMultiplier(2)->Range(10e4, 2 * 10e6);

static void BM_Array_Write_Copy(benchmark::State& state)
{
  std::vector<uint32_t> a(state.range(0), 0x0);
  std::vector<uint32_t> b(state.range(0), 0x0);
  std::vector<uint32_t> c(state.range(0), 0x0001000f);
  for (auto _ : state) {
    std::vector<uint32_t> tmp(state.range(0));

    for (size_t i = 0; i < c.size(); ++i) {
      tmp[i] = c[i] + 1;
    }

    for (size_t i = 0; i < a.size(); ++i) {
      const uint32_t shift = 16;
      a[i] = tmp[i] >> shift;
      b[i] = tmp[i] & ((1 << shift) - 1);
    }
  }
}

BENCHMARK(BM_Array_Write_Copy)->RangeMultiplier(2)->Range(10e4, 2 * 10e6);

static void BM_Array_Write_Iterator(benchmark::State& state)
{
  std::vector<uint32_t> a(state.range(0), 0x0);
  std::vector<uint32_t> b(state.range(0), 0x0);
  std::vector<uint32_t> c(state.range(0), 0x0001000f);

  for (auto _ : state) {
    auto writeOP = [](auto iterA, auto iterB, uint32_t value) -> void {
      const uint32_t shift = 16;
      *iterA = value >> shift;
      *iterB = value & ((1 << shift) - 1);
    };

    auto out = o2::rans::utils::CombinedOutputIteratorFactory<uint32_t>::makeIter(a.begin(), b.begin(), writeOP);

    for (auto iter = c.begin(); iter != c.end(); ++iter) {
      *out = *iter + 1;
      ++out;
    }
  }
}

BENCHMARK(BM_Array_Write_Iterator)->RangeMultiplier(2)->Range(10e4, 2 * 10e6);

BENCHMARK_MAIN();
