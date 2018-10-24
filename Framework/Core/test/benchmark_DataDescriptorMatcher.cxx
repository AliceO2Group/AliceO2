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
    OriginValueMatcher{ "TRD" }
  };

  std::vector<ContextElement> context;

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
    OriginValueMatcher{ "TRD" },
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      DescriptionValueMatcher{ "TRACKLET" },
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        SubSpecificationTypeValueMatcher{ 0 },
        ConstantValueMatcher{ true }))
  };

  std::vector<ContextElement> context;

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
    OriginValueMatcher{ "TDR" },
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      DescriptionValueMatcher{ "TRACKLET" },
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        SubSpecificationTypeValueMatcher{ 1 },
        ConstantValueMatcher{ true }))
  };

  std::vector<ContextElement> context;

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
    OriginValueMatcher{ "TRD" },
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      DescriptionValueMatcher{ "TRACKLET" },
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        SubSpecificationTypeValueMatcher{ 1 },
        ConstantValueMatcher{ true }))
  };

  std::vector<ContextElement> context;

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
    OriginValueMatcher{ ContextRef{ 0 } },
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      DescriptionValueMatcher{ "TRACKLET" },
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        SubSpecificationTypeValueMatcher{ 1 },
        ConstantValueMatcher{ true }))
  };

  std::vector<ContextElement> context(1);

  for (auto _ : state) {
    context[0].value = None{};
    matcher.match(header, context);
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
    OriginValueMatcher{ ContextRef{ 0 } },
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      DescriptionValueMatcher{ "TRACKLET" },
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        SubSpecificationTypeValueMatcher{ 1 },
        ConstantValueMatcher{ true }))
  };

  std::vector<ContextElement> context(1);

  for (auto _ : state) {
    context[0].value = None{};
    matcher.match(header0, context);
    matcher.match(header1, context);
  }
}
// Register the function as a benchmark
BENCHMARK(BM_OneVariableMatchUnmatch);

BENCHMARK_MAIN();
