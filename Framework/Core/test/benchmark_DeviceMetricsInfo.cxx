// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/DeviceMetricsInfo.h"

#include <benchmark/benchmark.h>
#include <regex>

// This is the fastest we could ever get.
static void BM_MemcmpBaseline(benchmark::State& state)
{
  using namespace o2::framework;
  std::string metric;
  ParsedMetricMatch match;
  DeviceMetricsInfo info;

  metric = "[METRIC] bkey,0 12 1789372894 hostname=test.cern.ch";
  for (auto _ : state) {
    // Parse a simple metric
    benchmark::DoNotOptimize(metric == "[METRIC] bkey,0 12 1789372894 hostname=test.cern.ch");
  }
  state.SetBytesProcessed(state.iterations() * metric.size());
}

BENCHMARK(BM_MemcmpBaseline);

static void BM_RegexBaseline(benchmark::State& state)
{
  using namespace o2::framework;
  std::string metric;
  ParsedMetricMatch match;
  DeviceMetricsInfo info;
  std::regex metricsRE(R"regex(\[METRIC\] ([a-zA-Z0-9/_-]+),(0|1|2|4) ([0-9.a-zA-Z_/" <>()<$:-]+) ([0-9]+))regex", std::regex::optimize);
  metric = "[METRIC] bkey,0 12 1789372894 hostname=test.cern.ch";
  char const* key;
  char const* type;
  char const* value;
  char const* timestamp;
  for (auto _ : state) {
    std::cregex_token_iterator it(metric.data(), metric.data() + metric.length(), metricsRE, {1, 2, 3, 4});
    key = it->first;
    ++it;
    type = it->first;
    ++it;
    value = it->first;
    ++it;
    timestamp = it->first;
  }
  state.SetBytesProcessed(state.iterations() * metric.size());
}
BENCHMARK(BM_RegexBaseline);

static void BM_ParseIntMetric(benchmark::State& state)
{
  using namespace o2::framework;
  std::string metric;
  ParsedMetricMatch match;
  DeviceMetricsInfo info;

  metric = "[METRIC] bkey,0 12 1789372894 hostname=test.cern.ch";
  for (auto _ : state) {
    // Parse a simple metric
    DeviceMetricsHelper::parseMetric(metric, match);
  }
  state.SetBytesProcessed(state.iterations() * metric.size());
}

BENCHMARK(BM_ParseIntMetric);

static void BM_ProcessIntMetric(benchmark::State& state)
{
  using namespace o2::framework;
  std::string metric;
  ParsedMetricMatch match;
  DeviceMetricsInfo info;

  metric = "[METRIC] bkey,0 12 1789372894 hostname=test.cern.ch";
  std::vector<std::string> metrics{1000, metric};
  // Add the first metric to the store
  for (auto _ : state) {
    for (auto& s : metrics) {
      DeviceMetricsHelper::parseMetric(s, match);
      DeviceMetricsHelper::processMetric(match, info);
    }
  }
  state.SetBytesProcessed(state.iterations() * metrics.size() * metric.size());
}

BENCHMARK(BM_ProcessIntMetric);

static void BM_ParseFloatMetric(benchmark::State& state)
{
  using namespace o2::framework;
  std::string metric;
  ParsedMetricMatch match;
  DeviceMetricsInfo info;

  // Parse a fourth metric, now a float one
  metric = "[METRIC] key3,2 16.0 1789372894 hostname=test.cern.ch";
  for (auto _ : state) {
    DeviceMetricsHelper::parseMetric(metric, match);
  }
  state.SetBytesProcessed(state.iterations() * metric.size());
}

BENCHMARK(BM_ParseFloatMetric);

static void BM_ProcessFloatMetric(benchmark::State& state)
{
  using namespace o2::framework;
  std::string metric;
  ParsedMetricMatch match;
  DeviceMetricsInfo info;

  metric = "[METRIC] key3,2 16.0 1789372894 hostname=test.cern.ch";
  for (auto _ : state) {
    DeviceMetricsHelper::parseMetric(metric, match);
    DeviceMetricsHelper::processMetric(match, info);
  }
  state.SetBytesProcessed(state.iterations() * metric.size());
}

BENCHMARK(BM_ProcessFloatMetric);

static void BM_ProcessStringMetric(benchmark::State& state)
{
  using namespace o2::framework;
  std::string metric;
  ParsedMetricMatch match;
  DeviceMetricsInfo info;

  metric = "[METRIC] key3,1 some_string 1789372895 hostname=test.cern.ch";
  for (auto _ : state) {
    // Parse a string metric
    DeviceMetricsHelper::parseMetric(metric, match);
    DeviceMetricsHelper::processMetric(match, info);
  }
  state.SetBytesProcessed(state.iterations() * metric.size());
}

BENCHMARK(BM_ProcessStringMetric);

static void BM_ProcessMismatchedMetric(benchmark::State& state)
{
  using namespace o2::framework;
  std::string metric;
  ParsedMetricMatch match;
  DeviceMetricsInfo info;

  metric = "[METRICA] key3,1 some_string 1789372895 hostname=test.cern.ch";
  for (auto _ : state) {
    DeviceMetricsHelper::parseMetric(metric, match);
  }
  state.SetBytesProcessed(state.iterations() * metric.size());
}

BENCHMARK(BM_ProcessMismatchedMetric);

BENCHMARK_MAIN();
