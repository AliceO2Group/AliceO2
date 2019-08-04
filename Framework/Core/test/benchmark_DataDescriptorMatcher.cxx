// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <benchmark/benchmark.h>
#include "Headers/DataHeader.h"
#include "Framework/DataDescriptorMatcher.h"

using namespace o2::header;
using namespace o2::framework::data_matcher;

static void BM_MatchedSingleQuery(benchmark::State& state)
{
  DataHeader header;
  header.dataOrigin = "TRD";
  header.dataDescription = "TRACKLET";
  header.subSpecification = 0;

  DataDescriptorMatcher matcher{
    DataDescriptorMatcher::Op::Just,
    OriginValueMatcher{"TRD"}};

  VariableContext context;

  for (auto _ : state) {
    matcher.match(header, context);
  }
}

BENCHMARK(BM_MatchedSingleQuery);

static void BM_MatchedFullQuery(benchmark::State& state)
{
  DataHeader header;
  header.dataOrigin = "TRD";
  header.dataDescription = "TRACKLET";
  header.subSpecification = 0;

  DataDescriptorMatcher matcher{
    DataDescriptorMatcher::Op::And,
    OriginValueMatcher{"TRD"},
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      DescriptionValueMatcher{"TRACKLET"},
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        SubSpecificationTypeValueMatcher{0},
        ConstantValueMatcher{true}))};

  VariableContext context;

  for (auto _ : state) {
    matcher.match(header, context);
  }
}
// Register the function as a benchmark
BENCHMARK(BM_MatchedFullQuery);

static void BM_UnmatchedSingleQuery(benchmark::State& state)
{
  DataHeader header;
  header.dataOrigin = "TRD";
  header.dataDescription = "TRACKLET";
  header.subSpecification = 0;

  DataDescriptorMatcher matcher{
    DataDescriptorMatcher::Op::And,
    OriginValueMatcher{"TDR"},
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      DescriptionValueMatcher{"TRACKLET"},
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        SubSpecificationTypeValueMatcher{1},
        ConstantValueMatcher{true}))};

  VariableContext context;

  for (auto _ : state) {
    matcher.match(header, context);
  }
}

// Register the function as a benchmark
BENCHMARK(BM_UnmatchedSingleQuery);

static void BM_UnmatchedFullQuery(benchmark::State& state)
{
  DataHeader header;
  header.dataOrigin = "TRD";
  header.dataDescription = "TRACKLET";
  header.subSpecification = 0;

  DataDescriptorMatcher matcher{
    DataDescriptorMatcher::Op::And,
    OriginValueMatcher{"TRD"},
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      DescriptionValueMatcher{"TRACKLET"},
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        SubSpecificationTypeValueMatcher{1},
        ConstantValueMatcher{true}))};

  VariableContext context;

  for (auto _ : state) {
    matcher.match(header, context);
  }
}
// Register the function as a benchmark
BENCHMARK(BM_UnmatchedFullQuery);

static void BM_OneVariableFullMatch(benchmark::State& state)
{
  DataHeader header;
  header.dataOrigin = "TRD";
  header.dataDescription = "TRACKLET";
  header.subSpecification = 0;

  DataDescriptorMatcher matcher{
    DataDescriptorMatcher::Op::And,
    OriginValueMatcher{ContextRef{0}},
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      DescriptionValueMatcher{"TRACKLET"},
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        SubSpecificationTypeValueMatcher{1},
        ConstantValueMatcher{true}))};

  VariableContext context;

  for (auto _ : state) {
    matcher.match(header, context);
    context.discard();
  }
}
// Register the function as a benchmark
BENCHMARK(BM_OneVariableFullMatch);

static void BM_OneVariableMatchUnmatch(benchmark::State& state)
{
  DataHeader header0;
  header0.dataOrigin = "TRD";
  header0.dataDescription = "TRACKLET";
  header0.subSpecification = 0;

  DataHeader header1;
  header1.dataOrigin = "TPC";
  header1.dataDescription = "CLUSTERS";
  header1.subSpecification = 0;

  DataDescriptorMatcher matcher{
    DataDescriptorMatcher::Op::And,
    OriginValueMatcher{ContextRef{0}},
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      DescriptionValueMatcher{"TRACKLET"},
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        SubSpecificationTypeValueMatcher{1},
        ConstantValueMatcher{true}))};

  VariableContext context;

  for (auto _ : state) {
    matcher.match(header0, context);
    matcher.match(header1, context);
    context.discard();
  }
}
// Register the function as a benchmark
BENCHMARK(BM_OneVariableMatchUnmatch);

BENCHMARK_MAIN();
