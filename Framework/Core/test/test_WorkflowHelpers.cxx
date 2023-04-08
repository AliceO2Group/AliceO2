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
#include "Mocking.h"
#include "test_HelperMacros.h"
#include "Framework/ConfigContext.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/SimpleOptionsRetriever.h"
#include "Framework/LifetimeHelpers.h"
#include "../src/WorkflowHelpers.h"
#include <catch_amalgamated.hpp>
#include <algorithm>
#include <memory>
#include <list>

using namespace o2::framework;

TEST_CASE("TestVerifyWorkflow")
{
  using namespace o2::framework;
  auto checkIncompleteInput = [](WorkflowSpec const& workflow) {
    // Empty workflows should be invalid.
    REQUIRE_THROWS_AS((void)WorkflowHelpers::verifyWorkflow(workflow), std::runtime_error);
  };

  auto checkSpecialChars = [](WorkflowSpec const& workflow) {
    // Empty workflows should be invalid.
    REQUIRE_THROWS_AS((void)WorkflowHelpers::verifyWorkflow(workflow), std::runtime_error);
  };

  auto checkOk = [](WorkflowSpec const& workflow) {
    // Empty workflows should be invalid.
    REQUIRE_NOTHROW((void)WorkflowHelpers::verifyWorkflow(workflow));
  };

  auto checkNotOk = [](WorkflowSpec const& workflow) {
    // Empty workflows should be invalid.
    REQUIRE_THROWS_AS((void)WorkflowHelpers::verifyWorkflow(workflow), std::runtime_error);
  };

  // A non fully specified input is an error, given the result is ambiguous.
  // Completely ambiguous.
  checkIncompleteInput(WorkflowSpec{{"A", {InputSpec{"", "", ""}}}});
  // missing origin and description
  checkIncompleteInput(WorkflowSpec{{"A", {InputSpec{"x", "", ""}}}});
  // missing description
  checkIncompleteInput(WorkflowSpec{{"A", {InputSpec{"x", "TST", ""}}}});
  // comma is not allowed
  checkSpecialChars(WorkflowSpec{{"A,B", {}}});
  // This is fine, since by default both subSpec == 0 and
  // Timeframe are assumed.
  checkOk(WorkflowSpec{{"A", {InputSpec{"x", "TST", "A"}}}});
  // Check for duplicate DataProcessorSpecs names
  checkNotOk(WorkflowSpec{{"A"}, {"A"}});
}

TEST_CASE("TestWorkflowHelpers")
{
  using namespace o2::framework;
  using Edges = std::vector<std::pair<int, int>>;
  // No edges

  Edges edges0 = {};
  auto result0 = WorkflowHelpers::topologicalSort(1,
                                                  &(edges0[0].first),
                                                  &(edges0[0].second),
                                                  sizeof(edges0[0]),
                                                  0);
  std::vector<TopoIndexInfo> expected0 = {{0, 0}};
  REQUIRE(result0 == expected0);

  // Already sorted
  Edges edges1 = {
    {0, 1}, // 1 depends on 0
    {1, 2},
    {2, 3}};
  auto result1 = WorkflowHelpers::topologicalSort(4,
                                                  &(edges1[0].first),
                                                  &(edges1[0].second),
                                                  sizeof(edges1[0]),
                                                  3);
  std::vector<TopoIndexInfo> expected1 = {{0, 0}, {1, 1}, {2, 2}, {3, 3}};
  REQUIRE(result1 == expected1);
  // Inverse sort
  Edges edges2 = {
    {3, 2},
    {2, 1},
    {1, 0}};
  auto result2 = WorkflowHelpers::topologicalSort(4,
                                                  &edges2[0].first,
                                                  &edges2[0].second,
                                                  sizeof(edges2[0]),
                                                  3);
  std::vector<TopoIndexInfo> expected2 = {{3, 0}, {2, 1}, {1, 2}, {0, 1}};
  REQUIRE(result2 == expected2);
  //     2
  //    / \
  // 4-3   0-5
  //    \ /
  //     1
  Edges edges3 = {
    {0, 5},
    {4, 3},
    {3, 2},
    {2, 0},
    {1, 0},
    {3, 1},
  };
  auto result3 = WorkflowHelpers::topologicalSort(6,
                                                  &(edges3[0].first),
                                                  &(edges3[0].second),
                                                  sizeof(edges3[0]),
                                                  6);
  std::vector<TopoIndexInfo> expected3 = {{4, 0}, {3, 1}, {1, 2}, {2, 2}, {0, 3}, {5, 4}};
  REQUIRE(result3 == expected3);

  // 0 -> 1 -----\
  //              \
  //               5
  //              /
  // 2 -> 3 -> 4-/
  Edges edges4 = {
    {0, 1},
    {2, 3},
    {3, 4},
    {4, 5},
    {1, 5}};
  auto result4 = WorkflowHelpers::topologicalSort(6,
                                                  &(edges4[0].first),
                                                  &(edges4[0].second),
                                                  sizeof(edges4[0]),
                                                  5);
  std::vector<TopoIndexInfo> expected4 = {{0, 0}, {2, 0}, {1, 1}, {3, 1}, {4, 2}, {5, 3}};
  REQUIRE(result4 == expected4);

  // 0 -> 1
  // 2 -> 3 -> 4
  Edges edges5 = {
    {0, 1},
    {2, 3},
    {3, 4},
  };
  auto result5 = WorkflowHelpers::topologicalSort(5,
                                                  &(edges5[0].first),
                                                  &(edges5[0].second),
                                                  sizeof(edges5[0]),
                                                  3);
  std::vector<TopoIndexInfo> expected5 = {{0, 0}, {2, 0}, {1, 1}, {3, 1}, {4, 2}};
  REQUIRE(result5 == expected5);

  // 0 <-> 1
  Edges edges6 = {
    {0, 1},
    {1, 0}};
  auto result6 = WorkflowHelpers::topologicalSort(2,
                                                  &(edges6[0].first),
                                                  &(edges6[0].second),
                                                  sizeof(edges6[0]),
                                                  2);
  /// FIXME: Circular dependencies not possible for now. Should they actually
  ///        be supported?
  std::vector<TopoIndexInfo> expected6 = {};
  REQUIRE(result6 == expected6);

  /// We actually support using node indexes which are not
  /// std::pair<size_t, size_t> as long as they occupy 64 bit
  struct SlotEdge {
    int nodeIn;
    int slotIn;
    int nodeOut;
    int slotOut;
  };

  // (0,0) -> (1,0) or 0 -> 1
  // (0,1) -> (2,0) or 0 -> 2
  // (0,2) -> (2,1) or 0 -> 2
  std::vector<SlotEdge> edges7 = {
    {0, 0, 1, 0},
    {0, 1, 2, 0},
    {0, 2, 2, 1},
  };
  auto result7 = WorkflowHelpers::topologicalSort(3,
                                                  &(edges7[0].nodeIn),
                                                  &(edges7[0].nodeOut),
                                                  sizeof(edges7[0]),
                                                  3);
  std::vector<TopoIndexInfo> expected7 = {{0, 0}, {1, 1}, {2, 1}};
  REQUIRE(result7 == expected7);
}

// Test a single connection
//
// A->B becomes Enum -> A -> B
TEST_CASE("TestSimpleConnection")
{
  std::vector<InputSpec> expectedInputs = {InputSpec{"y", "TST", "A"}};
  std::vector<OutputSpec> expectedOutputs = {
    OutputSpec{"TST", "A"},
    OutputSpec{"DPL", "SUMMARY", compile_time_hash("A"), Lifetime::Timeframe},
    OutputSpec{"DPL", "ENUM", 0, Lifetime::Enumeration}};
  WorkflowSpec workflow{
    {"A",
     {},
     Outputs{expectedOutputs[0]}},
    {"B", {expectedInputs[0]}}};
  std::vector<DeviceConnectionEdge> logicalEdges;
  std::vector<OutputSpec> outputs;
  std::vector<LogicalForwardInfo> availableForwardsInfo;

  auto result = WorkflowHelpers::verifyWorkflow(workflow);
  REQUIRE(result == WorkflowParsingState::Valid);
  auto context = makeEmptyConfigContext();
  WorkflowHelpers::injectServiceDevices(workflow, *context);
  // The fourth one is the dummy sink for the
  // timeframe reporting messages
  std::vector<std::string> expectedNames = {"A", "B", "internal-dpl-clock", "internal-dpl-injected-dummy-sink"};
  REQUIRE(workflow.size() == expectedNames.size());
  for (size_t wi = 0, we = workflow.size(); wi != we; ++wi) {
    SECTION("With parameter wi = " + std::to_string(wi))
    {
      REQUIRE(workflow[wi].name == expectedNames[wi]);
    }
  }
  WorkflowHelpers::constructGraph(workflow, logicalEdges,
                                  outputs,
                                  availableForwardsInfo);
  std::vector<DeviceConnectionEdge> expectedEdges{
    {2, 0, 0, 0, 2, 0, false, ConnectionKind::Out},
    {0, 1, 0, 0, 0, 0, false, ConnectionKind::Out},
    {1, 3, 0, 1, 1, 0, false, ConnectionKind::Out},
  };
  REQUIRE(expectedOutputs.size() == outputs.size());
  for (size_t oi = 0, oe = expectedOutputs.size(); oi != oe; ++oi) {
    INFO("With parameter oi = " << oi);
    REQUIRE(expectedOutputs[oi].lifetime == outputs[oi].lifetime);
  }
  REQUIRE(expectedEdges.size() == logicalEdges.size());
  for (size_t ei = 0, ee = expectedEdges.size(); ei != ee; ++ei) {
    SECTION("With parameter ei = " + std::to_string(ei))
    {
      REQUIRE(expectedEdges[ei].consumer == logicalEdges[ei].consumer);
      REQUIRE(expectedEdges[ei].producer == logicalEdges[ei].producer);
      REQUIRE(expectedEdges[ei].outputGlobalIndex == logicalEdges[ei].outputGlobalIndex);
    }
  }
}

// Test a simple forward in case of two parallel consumers
//   B
//  /
// A      becomes A -> B -> C
//  \
//   C
TEST_CASE("TestSimpleForward")
{
  std::vector<InputSpec> expectedInputs = {InputSpec{"y", "TST", "A"}};
  std::vector<OutputSpec> expectedOutputs = {
    OutputSpec{"TST", "A"},
    OutputSpec{"DPL", "SUMMARY", compile_time_hash("B"), Lifetime::Timeframe},
    OutputSpec{"DPL", "SUMMARY", compile_time_hash("C"), Lifetime::Timeframe},
    OutputSpec{"DPL", "SUMMARY", compile_time_hash("D"), Lifetime::Timeframe},
    OutputSpec{"DPL", "TIMER", 0, Lifetime::Timer}};
  WorkflowSpec workflow{
    {"A", {}, Outputs{expectedOutputs[0]}},
    {"B", {expectedInputs[0]}},
    {"C", {expectedInputs[0]}},
    {"D", {expectedInputs[0]}}};
  std::vector<DeviceConnectionEdge> logicalEdges;
  std::vector<OutputSpec> outputs;
  std::vector<LogicalForwardInfo> availableForwardsInfo;
  REQUIRE(WorkflowHelpers::verifyWorkflow(workflow) == WorkflowParsingState::Valid);
  auto context = makeEmptyConfigContext();
  WorkflowHelpers::injectServiceDevices(workflow, *context);
  WorkflowHelpers::constructGraph(workflow, logicalEdges,
                                  outputs,
                                  availableForwardsInfo);

  std::vector<DeviceConnectionEdge> expectedEdges{
    {4, 0, 0, 0, 4, 0, false, ConnectionKind::Out},
    {0, 1, 0, 0, 0, 0, false, ConnectionKind::Out},
    {1, 2, 0, 0, 0, 0, true, ConnectionKind::Out},
    {2, 3, 0, 0, 0, 0, true, ConnectionKind::Out},

    {1, 5, 0, 0, 1, 0, true, ConnectionKind::Out},
    {2, 5, 0, 0, 2, 1, true, ConnectionKind::Out},
    {3, 5, 0, 0, 3, 2, true, ConnectionKind::Out},
  };
  REQUIRE(expectedOutputs.size() == outputs.size());
  REQUIRE(expectedEdges.size() == logicalEdges.size());
  for (size_t ei = 0, ee = expectedEdges.size(); ei != ee; ++ei) {
    SECTION("with ei: " + std::to_string(ei))
    {
      REQUIRE(expectedEdges[ei].consumer == logicalEdges[ei].consumer);
      REQUIRE(expectedEdges[ei].producer == logicalEdges[ei].producer);
      REQUIRE(expectedEdges[ei].outputGlobalIndex == logicalEdges[ei].outputGlobalIndex);
      REQUIRE(expectedEdges[ei].consumerInputIndex == logicalEdges[ei].consumerInputIndex);
    }
  }
}

TEST_CASE("TestGraphConstruction")
{
  WorkflowSpec workflow{
    {"A",
     Inputs{},
     Outputs{
       OutputSpec{"TST", "A"}}},
    timePipeline({
                   "B",
                   Inputs{InputSpec{"b", "TST", "A"}},
                   Outputs{OutputSpec{"TST", "B"}},
                 },
                 3),
    timePipeline({"C", Inputs{InputSpec{"c", "TST", "B"}}}, 2)};

  std::vector<DeviceConnectionEdge> expected{
    {3, 0, 0, 0, 3, 0, false, ConnectionKind::Out},
    {0, 1, 0, 0, 0, 0, false, ConnectionKind::Out},
    {0, 1, 1, 0, 0, 0, false, ConnectionKind::Out},
    {0, 1, 2, 0, 0, 0, false, ConnectionKind::Out},
    {1, 2, 0, 0, 1, 0, false, ConnectionKind::Out},
    {1, 2, 0, 1, 1, 0, false, ConnectionKind::Out},
    {1, 2, 0, 2, 1, 0, false, ConnectionKind::Out},
    {1, 2, 1, 0, 1, 0, false, ConnectionKind::Out},
    {1, 2, 1, 1, 1, 0, false, ConnectionKind::Out},
    {1, 2, 1, 2, 1, 0, false, ConnectionKind::Out},

    {2, 4, 0, 0, 2, 0, false, ConnectionKind::Out}, // DPL/SUMMARY routes
    {2, 4, 0, 1, 2, 0, false, ConnectionKind::Out},
  };
  std::list<LogicalOutputInfo> availableOutputsInfo;
  std::vector<DeviceConnectionEdge> logicalEdges;
  std::vector<LogicalForwardInfo> availableForwardsInfo;

  // This is a temporary store for inputs and outputs, including forwarded
  // channels, so that we can construct them before assigning to a device.
  std::vector<OutputSpec> outputs;

  REQUIRE(WorkflowHelpers::verifyWorkflow(workflow) == WorkflowParsingState::Valid);
  auto context = makeEmptyConfigContext();
  WorkflowHelpers::injectServiceDevices(workflow, *context);
  WorkflowHelpers::constructGraph(workflow, logicalEdges,
                                  outputs,
                                  availableForwardsInfo);
  std::vector<ConcreteDataMatcher> expectedMatchers = {
    ConcreteDataMatcher{"TST", "A", 0},
    ConcreteDataMatcher{"TST", "B", 0},
    ConcreteDataMatcher{"DPL", "SUMMARY", compile_time_hash("C")}, // Summary value
    ConcreteDataMatcher{"DPL", "ENUM", compile_time_hash("A")},    // Enums value
  };

  std::vector<Lifetime> expectedLifetimes = {
    Lifetime::Timeframe,
    Lifetime::Timeframe,
    Lifetime::Timeframe,
    Lifetime::Enumeration,
  };

  REQUIRE(expectedMatchers.size() == expectedLifetimes.size());
  REQUIRE(outputs.size() == expectedMatchers.size());
  ; // FIXME: Is this what we actually want? We need
    // different matchers depending on the different timeframe ID.

  for (size_t i = 0; i < outputs.size(); ++i) {
    SECTION("with i: " + std::to_string(i))
    {
      auto concrete = DataSpecUtils::asConcreteDataMatcher(outputs[i]);
      REQUIRE(concrete.origin.as<std::string>() == expectedMatchers[i].origin.as<std::string>());
      REQUIRE(concrete.description.as<std::string>() == expectedMatchers[i].description.as<std::string>());
      REQUIRE(concrete.subSpec == expectedMatchers[i].subSpec);
      REQUIRE(outputs[i].lifetime == expectedLifetimes[i]);
    }
  }

  REQUIRE(expected.size() == logicalEdges.size());
  for (size_t i = 0; i < logicalEdges.size(); ++i) {
    SECTION("with i: " + std::to_string(i))
    {
      REQUIRE(logicalEdges[i].producer == expected[i].producer);
      REQUIRE(logicalEdges[i].consumer == expected[i].consumer);
      REQUIRE(logicalEdges[i].timeIndex == expected[i].timeIndex);
      REQUIRE(logicalEdges[i].producerTimeIndex == expected[i].producerTimeIndex);
      REQUIRE(logicalEdges[i].outputGlobalIndex == expected[i].outputGlobalIndex);
    }
  }

  std::vector<size_t> inIndex;
  std::vector<size_t> outIndex;
  WorkflowHelpers::sortEdges(inIndex, outIndex, logicalEdges);
  // Notice that zero is at the end because the first edge in the topological
  // sort is the timer and that gets added last.
  std::vector<size_t> expectedOutIndex{
    1, 2, 3, 4, 7, 5, 8, 6, 9, 10, 11, 0};

  std::vector<size_t> expectedInIndex{
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  REQUIRE_THAT(expectedOutIndex, Catch::Matchers::RangeEquals(outIndex));
  REQUIRE_THAT(expectedInIndex, Catch::Matchers::RangeEquals(inIndex));

  auto actions = WorkflowHelpers::computeOutEdgeActions(logicalEdges,
                                                        outIndex);

  std::vector<EdgeAction> expectedActionsOut{
    EdgeAction{true, true}, // timer device with first timer channel
    EdgeAction{true, true}, // actual first edge
    EdgeAction{false, true},
    EdgeAction{false, true},
    EdgeAction{true, true},
    EdgeAction{true, true},
    EdgeAction{true, true},
    EdgeAction{false, true},
    EdgeAction{false, true},
    EdgeAction{false, true},
    EdgeAction{true, true},
    EdgeAction{true, true},
  };

  REQUIRE(expectedActionsOut.size() == actions.size());
  for (size_t i = 0; i < outIndex.size(); i++) {
    size_t j = outIndex[i];
    SECTION(std::to_string(i) + " " + std::to_string(j))
    {
      REQUIRE(expectedActionsOut[j].requiresNewDevice == actions[j].requiresNewDevice);
    }
  }

  std::vector<EdgeAction> expectedActionsIn{
    EdgeAction{true, true}, // timer device with first timer channel
    EdgeAction{true, true}, // actual first edge
    EdgeAction{true, true},
    EdgeAction{true, true},
    EdgeAction{true, true},
    EdgeAction{false, true},
    EdgeAction{false, true},
    EdgeAction{true, true},
    EdgeAction{false, true},
    EdgeAction{false, true},
    EdgeAction{true, true},
    EdgeAction{false, true},
  };
  auto inActions = WorkflowHelpers::computeInEdgeActions(logicalEdges,
                                                         inIndex);

  REQUIRE(expectedActionsIn.size() == inActions.size());
  for (size_t i = 0; i < inIndex.size(); i++) {
    size_t j = inIndex[i];
    auto expectedValue = expectedActionsIn[j].requiresNewDevice;
    auto actualValue = inActions[j].requiresNewDevice;

    SECTION(std::to_string(i) + " " + std::to_string(j))
    {
      REQUIRE(expectedValue == actualValue);
    }
  }
}

// This is to test a workflow where the input is not of type Timeframe and
// therefore requires a dangling channel.
// The topology is
//
// TST/A     TST/B
// ----> (A) ---->
//
TEST_CASE("TestExternalInput")
{
  WorkflowSpec workflow{
    {.name = "A",
     .inputs = {
       InputSpec{"external", "TST", "A", 0, Lifetime::Timer}},
     .outputs = {OutputSpec{"TST", "B"}}}};
  REQUIRE(WorkflowHelpers::verifyWorkflow(workflow) == WorkflowParsingState::Valid);
  std::vector<DeviceConnectionEdge> logicalEdges;
  std::vector<OutputSpec> outputs;
  std::vector<LogicalForwardInfo> availableForwardsInfo;

  REQUIRE(workflow.size() == 1);

  auto context = makeEmptyConfigContext();
  WorkflowHelpers::injectServiceDevices(workflow, *context);
  // The added devices are the one which should connect to
  // the condition DB and the sink for the dangling outputs.
  REQUIRE(workflow.size() == 3);
  WorkflowHelpers::constructGraph(workflow, logicalEdges,
                                  outputs,
                                  availableForwardsInfo);
}

TEST_CASE("DetermineDanglingOutputs")
{
  WorkflowSpec workflow0{
    {.name = "A", .outputs = {OutputSpec{"TST", "A"}}},
    {.name = "B", .inputs = {InputSpec{"a", "TST", "A"}}}};

  WorkflowSpec workflow1{
    {.name = "A",
     .outputs = {OutputSpec{"TST", "A"}}}};

  WorkflowSpec workflow2{
    {.name = "A", .outputs = {OutputSpec{"TST", "A"}}},
    {.name = "B", .inputs = {InputSpec{"a", "TST", "B"}}}};

  WorkflowSpec workflow3{
    {.name = "A", .outputs = {OutputSpec{"TST", "A"}, OutputSpec{"TST", "B"}}},
    {.name = "B", .inputs = {InputSpec{"a", "TST", "A"}}}};

  WorkflowSpec workflow4{
    {.name = "A", .outputs = {OutputSpec{"TST", "A"}, OutputSpec{"TST", "B"}, OutputSpec{"TST", "C"}}},
    {.name = "B", .inputs = {InputSpec{"a", "TST", "A"}}}};

  auto dangling0 = WorkflowHelpers::computeDanglingOutputs(workflow0);
  std::vector<InputSpec> expected0{};
  REQUIRE_THAT(dangling0, Catch::Matchers::RangeEquals(expected0));

  auto dangling1 = WorkflowHelpers::computeDanglingOutputs(workflow1);
  std::vector<InputSpec> expected1{InputSpec{"dangling0", "TST", "A"}};
  REQUIRE_THAT(dangling1, Catch::Matchers::RangeEquals(expected1));

  auto dangling2 = WorkflowHelpers::computeDanglingOutputs(workflow2);
  std::vector<InputSpec> expected2{InputSpec{"dangling0", "TST", "A"}};
  REQUIRE_THAT(dangling2, Catch::Matchers::RangeEquals(expected2));

  auto dangling3 = WorkflowHelpers::computeDanglingOutputs(workflow3);
  std::vector<InputSpec> expected3{InputSpec{"dangling0", "TST", "B"}};
  REQUIRE_THAT(dangling3, Catch::Matchers::RangeEquals(expected3));

  auto dangling4 = WorkflowHelpers::computeDanglingOutputs(workflow4);
  std::vector<InputSpec> expected4{InputSpec{"dangling0", "TST", "B"}, InputSpec{"dangling1", "TST", "C"}};
  REQUIRE_THAT(dangling4, Catch::Matchers::RangeEquals(expected4));
}

TEST_CASE("TEST_SELECT")
{
  auto res = o2::framework::select();
  REQUIRE(res.empty());
  auto res1 = o2::framework::select("x:TST/C1/0");
  REQUIRE(res1.size() == 1);
}

// Test the case in which two outputs are matched by the same generic input on B
// A/1
//     \
//      B becomes Timer -> A -> B
//     /
// A/2
TEST_CASE("TestOriginWildcard")
{
  std::vector<InputSpec> expectedInputs = {InputSpec{"x", DataSpecUtils::dataDescriptorMatcherFrom(o2::header::DataOrigin{"A"})}};
  std::vector<OutputSpec> expectedOutputs = {
    OutputSpec{"A", "1"},
    OutputSpec{"A", "2"},
    OutputSpec{"DPL", "TIMER", 0, Lifetime::Timer},
    OutputSpec{"DPL", "SUMMARY", compile_time_hash("B"), Lifetime::Timeframe}};
  WorkflowSpec workflow{
    {"A", {}, {expectedOutputs[0], expectedOutputs[1]}},
    {"B", expectedInputs, {}},
  };
  std::vector<DeviceConnectionEdge> logicalEdges;
  std::vector<OutputSpec> outputs;
  std::vector<LogicalForwardInfo> availableForwardsInfo;

  REQUIRE(WorkflowHelpers::verifyWorkflow(workflow) == WorkflowParsingState::Valid);
  auto context = makeEmptyConfigContext();
  WorkflowHelpers::injectServiceDevices(workflow, *context);
  REQUIRE(workflow.size() == 4);
  REQUIRE(workflow.size() >= 4);
  REQUIRE(workflow[0].name == "A");
  REQUIRE(workflow[1].name == "B");
  REQUIRE(workflow[2].name == "internal-dpl-clock");
  REQUIRE(workflow[3].name == "internal-dpl-injected-dummy-sink");
  for (size_t wi = 4; wi < workflow.size(); ++wi) {
    REQUIRE(workflow[wi].name == "");
  }
  WorkflowHelpers::constructGraph(workflow, logicalEdges,
                                  outputs,
                                  availableForwardsInfo);

  std::vector<DeviceConnectionEdge> expectedEdges{
    {2, 0, 0, 0, 3, 0, false, ConnectionKind::Out},
    {0, 1, 0, 0, 0, 0, false, ConnectionKind::Out},
    {0, 1, 0, 0, 1, 0, false, ConnectionKind::Out},
    {1, 3, 0, 0, 2, 0, false, ConnectionKind::Out},
  };

  std::vector<size_t> expectedOutEdgeIndex = {1, 2, 3, 0};
  std::vector<size_t> expectedInEdgeIndex = {0, 1, 2, 3};
  std::vector<EdgeAction> expectedActions = {
    {true, true},   // to go from timer to A (new channel and new device)
    {true, true},   // to go from A/1 to B (new channel and new device)
    {false, false}, // to go from A/2 to B (device is the same as A/1, device is the same as B?)
    {true, true}    // to go from B to sink
  };

  // Not sure I understand...
  std::vector<EdgeAction> expectedInActions = {
    {true, true},
    {true, true},
    {false, false},
    {true, true} // to go from B to sink
  };

  REQUIRE(expectedOutputs.size() == outputs.size());
  REQUIRE(expectedEdges.size() == logicalEdges.size());
  for (size_t ei = 0, ee = expectedEdges.size(); ei != ee; ++ei) {
    SECTION("ei : " + std::to_string(ei))
    {
      REQUIRE(expectedEdges[ei].consumer == logicalEdges[ei].consumer);
      REQUIRE(expectedEdges[ei].producer == logicalEdges[ei].producer);
      REQUIRE(expectedEdges[ei].outputGlobalIndex == logicalEdges[ei].outputGlobalIndex);
      REQUIRE(expectedEdges[ei].consumerInputIndex == logicalEdges[ei].consumerInputIndex);
    }
  }

  std::vector<size_t> inEdgeIndex;
  std::vector<size_t> outEdgeIndex;
  WorkflowHelpers::sortEdges(inEdgeIndex, outEdgeIndex, logicalEdges);
  REQUIRE_THAT(outEdgeIndex, Catch::Matchers::RangeEquals(expectedOutEdgeIndex));
  REQUIRE_THAT(inEdgeIndex, Catch::Matchers::RangeEquals(expectedInEdgeIndex));
  REQUIRE(inEdgeIndex.size() == 4);

  std::vector<EdgeAction> outActions = WorkflowHelpers::computeOutEdgeActions(logicalEdges, outEdgeIndex);
  REQUIRE(outActions.size() == expectedActions.size());
  for (size_t ai = 0; ai < outActions.size(); ++ai) {
    SECTION("ai : " + std::to_string(ai))
    {
      REQUIRE(outActions[ai].requiresNewDevice == expectedActions[ai].requiresNewDevice);
      REQUIRE(outActions[ai].requiresNewChannel == expectedActions[ai].requiresNewChannel);
    }
  }

  // Crete the connections on the inverse map for all of them
  // lookup for port and add as input of the current device.
  std::vector<EdgeAction> inActions = WorkflowHelpers::computeInEdgeActions(logicalEdges, inEdgeIndex);
  REQUIRE(inActions.size() == expectedInActions.size());
  for (size_t ai = 0; ai < inActions.size(); ++ai) {
    SECTION("ai : " + std::to_string(ai))
    {
      REQUIRE(inActions[ai].requiresNewDevice == expectedInActions[ai].requiresNewDevice);
      REQUIRE(inActions[ai].requiresNewChannel == expectedInActions[ai].requiresNewChannel);
    }
  }
}
