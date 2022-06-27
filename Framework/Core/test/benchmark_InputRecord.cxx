// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <benchmark/benchmark.h>

#include "Headers/DataHeader.h"
#include "Headers/Stack.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/DataRelayer.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/InputRecord.h"
#include "Framework/InputSpan.h"
#include <Monitoring/Monitoring.h>
#include <fairmq/TransportFactory.h>
#include <cstring>

using Monitoring = o2::monitoring::Monitoring;
using namespace o2::framework;
using DataHeader = o2::header::DataHeader;
using Stack = o2::header::Stack;

static void BM_InputRecordGenericGetters(benchmark::State& state)
{
  // Create the routes we want for the InputRecord
  InputSpec spec1{"x", "TPC", "CLUSTERS", 0, Lifetime::Timeframe};
  InputSpec spec2{"y", "ITS", "CLUSTERS", 0, Lifetime::Timeframe};
  InputSpec spec3{"z", "TST", "EMPTY", 0, Lifetime::Timeframe};

  size_t i = 0;
  auto createRoute = [&i](const char* source, InputSpec& spec) {
    return InputRoute{
      spec,
      i++,
      source};
  };

  std::vector<InputRoute> schema = {
    createRoute("x_source", spec1),
    createRoute("y_source", spec2),
    createRoute("z_source", spec3)};
  // First of all we test if an empty registry behaves as expected, raising a
  // bunch of exceptions.
  InputSpan span{[](size_t) { return DataRef{nullptr, nullptr, nullptr}; }, 0};
  ServiceRegistry registry;
  InputRecord emptyRecord(schema, span, registry);

  std::vector<void*> inputs;

  auto createMessage = [&inputs](DataHeader& dh, int value) {
    DataProcessingHeader dph{0, 1};
    Stack stack{dh, dph};
    void* header = malloc(stack.size());
    void* payload = malloc(sizeof(int));
    memcpy(header, stack.data(), stack.size());
    memcpy(payload, &value, sizeof(int));
    inputs.emplace_back(header);
    inputs.emplace_back(payload);
  };

  auto createEmpty = [&inputs]() {
    inputs.emplace_back(nullptr);
    inputs.emplace_back(nullptr);
  };

  DataHeader dh1;
  dh1.dataDescription = "CLUSTERS";
  dh1.dataOrigin = "TPC";
  dh1.subSpecification = 0;
  dh1.payloadSerializationMethod = o2::header::gSerializationMethodNone;
  DataHeader dh2;
  dh2.dataDescription = "CLUSTERS";
  dh2.dataOrigin = "ITS";
  dh2.subSpecification = 0;
  dh2.payloadSerializationMethod = o2::header::gSerializationMethodNone;
  createMessage(dh1, 1);
  createMessage(dh2, 2);
  createEmpty();
  InputSpan span2{[&inputs](size_t i) { return DataRef{nullptr, static_cast<char const*>(inputs[2 * i]), static_cast<char const*>(inputs[2 * i + 1])}; }, inputs.size() / 2};
  InputRecord record{schema, span2, registry};

  for (auto _ : state) {
    // Checking we can get the whole ref by name
    [[maybe_unused]] auto ref00 = record.get("x");
    [[maybe_unused]] auto ref10 = record.get("y");
    [[maybe_unused]] auto ref20 = record.get("z");

    // Or we can get it positionally
    [[maybe_unused]] auto ref01 = record.getByPos(0);
    [[maybe_unused]] auto ref11 = record.getByPos(1);

    record.isValid("x");
    record.isValid("y");
    record.isValid("z");
  }
}

BENCHMARK(BM_InputRecordGenericGetters);

BENCHMARK_MAIN();
