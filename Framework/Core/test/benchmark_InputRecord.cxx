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
#include "Headers/Stack.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/DataRelayer.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/InputRecord.h"
#include <Monitoring/Monitoring.h>
#include <fairmq/FairMQTransportFactory.h>
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

  auto createRoute = [](const char* source, InputSpec& spec) {
    return InputRoute{
      spec,
      source};
  };

  std::vector<InputRoute> schema = {
    createRoute("x_source", spec1),
    createRoute("y_source", spec2),
    createRoute("z_source", spec3)};
  // First of all we test if an empty registry behaves as expected, raising a
  // bunch of exceptions.
  InputRecord emptyRecord(schema, {[](size_t) { return nullptr; }, 0});

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
  InputSpan span{[&inputs](size_t i) { return static_cast<char const*>(inputs[i]); }, inputs.size()};
  InputRecord record{schema, std::move(span)};

  for (auto _ : state) {
    // Checking we can get the whole ref by name
    auto ref00 = record.get("x");
    auto ref10 = record.get("y");
    auto ref20 = record.get("z");

    // Or we can get it positionally
    auto ref01 = record.getByPos(0);
    auto ref11 = record.getByPos(1);

    record.isValid("x");
    record.isValid("y");
    record.isValid("z");
  }
}

BENCHMARK(BM_InputRecordGenericGetters);

BENCHMARK_MAIN();
