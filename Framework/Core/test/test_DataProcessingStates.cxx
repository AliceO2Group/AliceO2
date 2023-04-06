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

#include "Framework/DataProcessingStates.h"
#include "Framework/TimingHelpers.h"
#include "Framework/DeviceState.h"
#include <catch_amalgamated.hpp>
#include <uv.h>

using namespace o2::framework;

enum TestMetricsId {
  DummyMetric = 0,
  DummyMetric2 = 1,
  Missing = 2
};

using namespace o2::framework;

TEST_CASE("DataProcessingStates")
{
  DataProcessingStates states(TimingHelpers::defaultRealtimeBaseConfigurator(0, uv_default_loop()),
                              TimingHelpers::defaultCPUTimeConfigurator(uv_default_loop()));

  states.registerState({"dummy_metric", DummyMetric});
  /// Registering twice should throw.
  REQUIRE_THROWS(states.registerState({"dummy_metric", DummyMetric2}));
  /// Registering with a different name should throw.
  REQUIRE_THROWS(states.registerState({"dummy_metric2", DummyMetric}));

  o2::framework::DataProcessingStates::CommandHeader header;

  states.registerState({"dummy_metric2", DummyMetric2});
  INFO("Next state is " << states.nextState.load());
  states.updateState({DummyMetric, (int)strlen("foo"), "foo"});
  memcpy(&header, states.store.data() + states.nextState.load(), sizeof(header));
  REQUIRE(header.size == 3);
  REQUIRE(header.id == DummyMetric);
  INFO("Timestamp is " << header.timestamp);
  REQUIRE_THROWS(states.updateState({Missing, int(strlen("foo")), "foo"}));
  INFO("Next state is " << states.nextState.load());
  REQUIRE(states.nextState.load() == (1 << 16) - sizeof(DataProcessingStates::CommandHeader) - 3);
  REQUIRE(states.updatedMetricsLapse.load() == 1);
  REQUIRE(states.pushedMetricsLapse == 0);
  REQUIRE(states.publishedMetricsLapse == 0);

  INFO("Next state is " << states.nextState.load());
  states.updateState({DummyMetric, (int)strlen("barbar"), "barbar"});
  INFO("Next state is " << states.nextState.load());
  REQUIRE(states.nextState.load() == (1 << 16) - 2 * sizeof(DataProcessingStates::CommandHeader) - 3 - 6);
  memcpy(&header, states.store.data() + states.nextState.load(), sizeof(header));
  REQUIRE(header.size == 6);
  REQUIRE(header.id == DummyMetric);
  INFO("Timestamp is " << header.timestamp);
  REQUIRE(std::string_view(states.store.data() + states.nextState.load() + sizeof(header), header.size) == "barbar");
  REQUIRE(states.updatedMetricsLapse.load() == 2);
  REQUIRE(states.pushedMetricsLapse == 0);
  REQUIRE(states.publishedMetricsLapse == 0);
  /// Nothing has been published yet.
  REQUIRE(states.statesViews[0].first == 0);
  REQUIRE(states.statesViews[0].capacity == 0);
  REQUIRE(states.statesViews[0].size == 0);

  states.processCommandQueue();
  REQUIRE(states.nextState.load() == (1 << 16));

  REQUIRE(states.statesViews[0].first == 0);
  REQUIRE(states.statesViews[0].size == 6);
  REQUIRE(states.statesViews[0].capacity == 64);

  std::vector<std::string> updated;
  std::vector<std::string> values;
  auto simpleFlush = [&updated, &values](std::string const& name, int64_t timestamp, std::string_view value) {
    updated.emplace_back(name);
    values.emplace_back(value);
  };

  states.flushChangedStates(simpleFlush);
  REQUIRE(states.updatedMetricsLapse.load() == 2);
  CHECK(states.pushedMetricsLapse == 1);
  CHECK(states.publishedMetricsLapse == 1);
  REQUIRE(updated.size() == 1);
  REQUIRE(updated[0] == "dummy_metric");
  REQUIRE(values.size() == 1);
  REQUIRE(values[0] == "barbar");

  states.updateState({DummyMetric, (int)strlen("foofo"), "foofo"});
  REQUIRE(states.nextState.load() == DataProcessingStates::STATES_BUFFER_SIZE - sizeof(DataProcessingStates::CommandHeader) - 5);
  memcpy(&header, states.store.data() + states.nextState.load(), sizeof(header));
  REQUIRE(header.size == 5);
  REQUIRE(header.id == DummyMetric);
  INFO("Timestamp is " << header.timestamp);
  states.processCommandQueue();

  REQUIRE(states.nextState.load() == (1 << 16));

  REQUIRE(states.statesViews[0].first == 0);
  REQUIRE(states.statesViews[0].size == 5);
  REQUIRE(states.statesViews[0].capacity == 64);
  // Test the insertion of a differet state
  states.updateState({DummyMetric2, (int)strlen("foofo"), "foofo"});
  REQUIRE(states.nextState.load() == DataProcessingStates::STATES_BUFFER_SIZE - sizeof(DataProcessingStates::CommandHeader) - 5);
  memcpy(&header, states.store.data() + states.nextState.load(), sizeof(header));
  REQUIRE(header.size == 5);
  REQUIRE(header.id == DummyMetric2);
  INFO("Timestamp is " << header.timestamp);
  states.processCommandQueue();

  REQUIRE(states.nextState.load() == (1 << 16));
  REQUIRE(states.statesViews[0].first == 0);
  REQUIRE(states.statesViews[0].size == 5);
  REQUIRE(states.statesViews[0].capacity == 64);
  REQUIRE(states.statesViews[1].first == 64);
  REQUIRE(states.statesViews[1].size == 5);
  REQUIRE(states.statesViews[1].capacity == 64);

  // Test going above capacity
  SECTION("Test capacity handling")
  {
    states.updateState({DummyMetric2, (int)strlen("foofofoofo"), "foofofoofo"});
    states.updateState({DummyMetric, 70, "01234567890123456789012345678901234567890123456789012345678901234567890123456789"});
    states.processCommandQueue();
    REQUIRE(states.nextState.load() == (1 << 16));
    CHECK(states.statesViews[0].first == 128);
    CHECK(states.statesViews[0].size == 70);
    CHECK(states.statesViews[0].capacity == 70);
    CHECK(states.statesViews[1].first == 64);
    CHECK(states.statesViews[1].size == 10);
    CHECK(states.statesViews[1].capacity == 64);
    states.updateState({DummyMetric2, 70, "01234567890123456789012345678901234567890123456789012345678901234567890123456789"});
    states.processCommandQueue();
    CHECK(states.statesViews[0].first == 128);
    CHECK(states.statesViews[0].size == 70);
    CHECK(states.statesViews[0].capacity == 70);
    CHECK(states.statesViews[1].first == 128 + 70);
    CHECK(states.statesViews[1].size == 70);
    CHECK(states.statesViews[1].capacity == 70);
    states.updateState({DummyMetric2, 70, "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"});
    states.processCommandQueue();
    CHECK(states.statesViews[1].first == 128 + 70);
    CHECK(states.statesViews[1].size == 70);
    CHECK(states.statesViews[1].capacity == 70);
    states.updateState({DummyMetric2, 68, "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"});
    states.processCommandQueue();
    CHECK(states.statesViews[1].first == 128 + 70);
    CHECK(states.statesViews[1].size == 68);
    CHECK(states.statesViews[1].capacity == 70);
    states.repack();
    CHECK(states.statesViews[0].first == 0);
    CHECK(states.statesViews[0].size == 70);
    CHECK(states.statesViews[0].capacity == 70);
    CHECK(states.statesViews[1].first == 70);
    CHECK(states.statesViews[1].size == 68);
    CHECK(states.statesViews[1].capacity == 68);
  }
}
