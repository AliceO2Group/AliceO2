// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDBase/PersistentVector.h
/// \brief  Benches the performance of the persistent vector for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   14 June 2019

#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include <iostream>
#include <algorithm>
#include "MIDBase/PersistentVector.h"

namespace o2
{
namespace mid
{
struct MySubStruct {
  std::array<float, 3> array;
  float number = 0;
};

struct MyTestData {
  std::vector<MySubStruct> subStruct; ///< Position
  std::array<float, 3> array = {};    ///< Direction
  int number = 0;
};

} // namespace mid
} // namespace o2

template <typename T>
o2::mid::MyTestData getElement(T mt)
{
  std::uniform_int_distribution<int> nSubStructs(1, 20);
  std::uniform_real_distribution<float> floatGen(-200., 200.);
  std::uniform_int_distribution<int> intGen(0, 1000);
  o2::mid::MyTestData element;
  int nSubs = nSubStructs(mt);
  for (int ival = 0; ival < nSubs; ++ival) {
    o2::mid::MySubStruct subStruct;
    subStruct.array = { floatGen(mt), floatGen(mt), floatGen(mt) };
    subStruct.number = intGen(mt);
    element.subStruct.emplace_back(subStruct);
  }
  element.array = { floatGen(mt), floatGen(mt), floatGen(mt) };
  element.number = intGen(mt);
  return std::move(element);
}

std::vector<std::vector<o2::mid::MyTestData>> getData(int nevents = 1000)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<int> nValues(1, 30);

  std::vector<std::vector<o2::mid::MyTestData>> fullData;
  std::vector<o2::mid::MyTestData> vec;
  for (int ievent = 0; ievent < nevents; ++ievent) {
    int nval = nValues(mt);
    for (int ival = 0; ival < nval; ++ival) {
      vec.emplace_back(getElement(mt));
    }
    fullData.emplace_back(vec);
  }
  return fullData;
}

static void CustomArguments(benchmark::internal::Benchmark* bench)
{
  for (int i1 = 1; i1 <= 1000; i1 *= 10) {
    for (int i2 = 10; i2 <= 30; i2 += 10) {
      bench->Args({ i1, i2 });
    }
  }
}

static void BM_STLVEC(benchmark::State& state)
{

  int nevents = state.range(0);

  auto data = getData(nevents);

  std::vector<o2::mid::MyTestData> vec;
  vec.reserve(state.range(1));

  for (auto _ : state) {
    for (auto& event : data) {
      for (auto& val : event) {
        vec.emplace_back(val);
      }
      vec.clear();
    }
  }
}

static void BM_MYVEC(benchmark::State& state)
{

  int nevents = state.range(0);

  auto data = getData(nevents);

  o2::mid::PersistentVector<o2::mid::MyTestData> vec;
  vec.reserve(state.range(1));

  for (auto _ : state) {
    for (auto& event : data) {
      for (auto& val : event) {
        auto& curr = vec.next();
        curr = val;
      }
      vec.rewind();
    }
  }
}

BENCHMARK(BM_STLVEC)->Apply(CustomArguments);
BENCHMARK(BM_MYVEC)->Apply(CustomArguments);

BENCHMARK_MAIN();
