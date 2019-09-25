// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "benchmark/benchmark.h"
#include "MCHSimulation/Digit.h"
#include "DigitMerging.h"
#include <ctime>
#include <cstdlib>

using o2::mch::Digit;

// createDigits generates N digits with random id
// (hence some might get duplicated).
std::vector<Digit> createDigits(int N)
{
  std::vector<Digit> digits;
  float dummyadc{42.0};
  std::srand(std::time(nullptr)); // use current time as seed for random generator
  float dummytime{0.0};
  int dummydetID = 100; //to be improved, timing depending on that

  for (auto i = 0; i < N; i++) {
    int randomPadID = std::rand() * N;
    digits.emplace_back(dummytime, dummydetID, randomPadID, dummyadc);
  }

  return digits;
}

std::vector<o2::MCCompLabel> createLabels(int N)
{
  std::vector<o2::MCCompLabel> labels;
  int dummyEventID{1000};
  std::srand(std::time(nullptr)); // use current time as seed for random generator
  float dummysrcID{10};

  for (auto i = 0; i < N; i++) {
    int randomTrackID = std::rand() * N;
    labels.emplace_back(randomTrackID, dummyEventID, dummysrcID, false);
  }

  return labels;
}

// benchDigitMerging create fake digits and merges them
// using one of the merging functions.
static void benchDigitMerging(benchmark::State& state)
{
  auto digits = createDigits(100);
  auto labels = createLabels(100);

  auto mergingFunction = mergingFunctions()[state.range(0)];

  for (auto _ : state) {
    mergingFunction(digits, labels);
  }
}

// mergingFunctionIndices generate arguments from 0 to # merging functions - 1
// to be used in the BENCHMARK macro below.
static void mergingFunctionIndices(benchmark::internal::Benchmark* b)
{
  for (auto i = 0; i < mergingFunctions().size(); i++) {
    b->Args({i});
  }
}

// This effectively register as many benchmarks as there are functions
// in the mergingFunctions vector.
BENCHMARK(benchDigitMerging)->Apply(mergingFunctionIndices);

BENCHMARK_MAIN();
