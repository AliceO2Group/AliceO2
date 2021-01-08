// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test Framework WorkflowHelpers
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Mocking.h"
#include "test_HelperMacros.h"
#include "Framework/ConfigContext.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/SimpleOptionsRetriever.h"
#include "../src/WorkflowHelpers.h"
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/detail/per_element_manip.hpp>
#include <algorithm>
#include <memory>

using namespace o2::framework;

BOOST_AUTO_TEST_CASE(TestVerifyWorkflow)
{
  using namespace o2::framework;
  auto checkIncompleteInput = [](WorkflowSpec const& workflow) {
    // Empty workflows should be invalid.
    BOOST_CHECK_THROW((void)WorkflowHelpers::verifyWorkflow(workflow), std::runtime_error);
  };

  auto checkOk = [](WorkflowSpec const& workflow) {
    // Empty workflows should be invalid.
    BOOST_CHECK_NO_THROW((void)WorkflowHelpers::verifyWorkflow(workflow));
  };

  auto checkNotOk = [](WorkflowSpec const& workflow) {
    // Empty workflows should be invalid.
    BOOST_CHECK_THROW((void)WorkflowHelpers::verifyWorkflow(workflow), std::runtime_error);
  };

  // A non fully specified input is an error, given the result is ambiguous.
  // Completely ambiguous.
  checkIncompleteInput(WorkflowSpec{{"A", {InputSpec{"", "", ""}}}});
  // missing origin and description
  checkIncompleteInput(WorkflowSpec{{"A", {InputSpec{"x", "", ""}}}});
  // missing description
  checkIncompleteInput(WorkflowSpec{{"A", {InputSpec{"x", "TST", ""}}}});
  // This is fine, since by default both subSpec == 0 and
  // Timeframe are assumed.
  checkOk(WorkflowSpec{{"A", {InputSpec{"x", "TST", "A"}}}});
  // Check for duplicate DataProcessorSpecs names
  checkNotOk(WorkflowSpec{{"A"}, {"A"}});
}

BOOST_AUTO_TEST_CASE(TestWorkflowHelpers)
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
  BOOST_TEST(result0 == expected0, boost::test_tools::per_element());

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
  BOOST_TEST(result1 == expected1, boost::test_tools::per_element());
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
  BOOST_TEST(result2 == expected2, boost::test_tools::per_element());
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
  BOOST_TEST(result3 == expected3, boost::test_tools::per_element());

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
  BOOST_TEST(result4 == expected4, boost::test_tools::per_element());

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
  BOOST_TEST(result5 == expected5, boost::test_tools::per_element());

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
  BOOST_TEST(result6 == expected6, boost::test_tools::per_element());

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
  BOOST_TEST(result7 == expected7, boost::test_tools::per_element());
}

// Test a single connection
//
// A->B becomes Enum -> A -> B
BOOST_AUTO_TEST_CASE(TestSimpleConnection)
{
  std::vector<InputSpec> expectedInputs = {InputSpec{"y", "TST", "A"}};
  std::vector<OutputSpec> expectedOutputs = {
    OutputSpec{"TST", "A"},
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
  BOOST_REQUIRE(result == WorkflowParsingState::Valid);
  auto context = makeEmptyConfigContext();
  WorkflowHelpers::injectServiceDevices(workflow, *context);
  BOOST_CHECK_EQUAL(workflow.size(), 3);
  WorkflowHelpers::constructGraph(workflow, logicalEdges,
                                  outputs,
                                  availableForwardsInfo);
  std::vector<DeviceConnectionEdge> expectedEdges{
    {2, 0, 0, 0, 1, 0, false, ConnectionKind::Out},
    {0, 1, 0, 0, 0, 0, false, ConnectionKind::Out},
  };
  BOOST_REQUIRE_EQUAL(expectedOutputs.size(), outputs.size());
  for (size_t oi = 0, oe = expectedOutputs.size(); oi != oe; ++oi) {
    BOOST_CHECK(expectedOutputs[oi].lifetime == outputs[oi].lifetime);
  }
  BOOST_REQUIRE_EQUAL(expectedEdges.size(), logicalEdges.size());
  for (size_t ei = 0, ee = expectedEdges.size(); ei != ee; ++ei) {
    BOOST_CHECK_EQUAL(expectedEdges[ei].consumer, logicalEdges[ei].consumer);
    BOOST_CHECK_EQUAL(expectedEdges[ei].producer, logicalEdges[ei].producer);
    BOOST_CHECK_EQUAL(expectedEdges[ei].outputGlobalIndex, logicalEdges[ei].outputGlobalIndex);
  }
}

// Test a simple forward in case of two parallel consumers
//   B
//  /
// A      becomes A -> B -> C
//  \
//   C
BOOST_AUTO_TEST_CASE(TestSimpleForward)
{
  std::vector<InputSpec> expectedInputs = {InputSpec{"y", "TST", "A"}};
  std::vector<OutputSpec> expectedOutputs = {OutputSpec{"TST", "A"}, OutputSpec{"DPL", "TIMER", 0, Lifetime::Timer}};
  WorkflowSpec workflow{
    {"A", {}, Outputs{expectedOutputs[0]}},
    {"B", {expectedInputs[0]}},
    {"C", {expectedInputs[0]}},
    {"D", {expectedInputs[0]}}};
  std::vector<DeviceConnectionEdge> logicalEdges;
  std::vector<OutputSpec> outputs;
  std::vector<LogicalForwardInfo> availableForwardsInfo;
  BOOST_REQUIRE(WorkflowHelpers::verifyWorkflow(workflow) == WorkflowParsingState::Valid);
  auto context = makeEmptyConfigContext();
  WorkflowHelpers::injectServiceDevices(workflow, *context);
  WorkflowHelpers::constructGraph(workflow, logicalEdges,
                                  outputs,
                                  availableForwardsInfo);

  std::vector<DeviceConnectionEdge> expectedEdges{
    {4, 0, 0, 0, 1, 0, false, ConnectionKind::Out},
    {0, 1, 0, 0, 0, 0, false, ConnectionKind::Out},
    {1, 2, 0, 0, 0, 0, true, ConnectionKind::Out},
    {2, 3, 0, 0, 0, 0, true, ConnectionKind::Out},
  };
  BOOST_REQUIRE_EQUAL(expectedOutputs.size(), outputs.size());
  BOOST_REQUIRE_EQUAL(expectedEdges.size(), logicalEdges.size());
  for (size_t ei = 0, ee = expectedEdges.size(); ei != ee; ++ei) {
    BOOST_CHECK_EQUAL(expectedEdges[ei].consumer, logicalEdges[ei].consumer);
    BOOST_CHECK_EQUAL(expectedEdges[ei].producer, logicalEdges[ei].producer);
    BOOST_CHECK_EQUAL(expectedEdges[ei].outputGlobalIndex, logicalEdges[ei].outputGlobalIndex);
    BOOST_CHECK_EQUAL(expectedEdges[ei].consumerInputIndex, logicalEdges[ei].consumerInputIndex);
  }
}

BOOST_AUTO_TEST_CASE(TestGraphConstruction)
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
    {3, 0, 0, 0, 2, 0, false, ConnectionKind::Out},
    {0, 1, 0, 0, 0, 0, false, ConnectionKind::Out},
    {0, 1, 1, 0, 0, 0, false, ConnectionKind::Out},
    {0, 1, 2, 0, 0, 0, false, ConnectionKind::Out},
    {1, 2, 0, 0, 1, 0, false, ConnectionKind::Out},
    {1, 2, 0, 1, 1, 0, false, ConnectionKind::Out},
    {1, 2, 0, 2, 1, 0, false, ConnectionKind::Out},
    {1, 2, 1, 0, 1, 0, false, ConnectionKind::Out},
    {1, 2, 1, 1, 1, 0, false, ConnectionKind::Out},
    {1, 2, 1, 2, 1, 0, false, ConnectionKind::Out}};
  std::list<LogicalOutputInfo> availableOutputsInfo;
  std::vector<DeviceConnectionEdge> logicalEdges;
  std::vector<LogicalForwardInfo> availableForwardsInfo;

  // This is a temporary store for inputs and outputs, including forwarded
  // channels, so that we can construct them before assigning to a device.
  std::vector<OutputSpec> outputs;

  BOOST_REQUIRE(WorkflowHelpers::verifyWorkflow(workflow) == WorkflowParsingState::Valid);
  auto context = makeEmptyConfigContext();
  WorkflowHelpers::injectServiceDevices(workflow, *context);
  WorkflowHelpers::constructGraph(workflow, logicalEdges,
                                  outputs,
                                  availableForwardsInfo);
  std::vector<ConcreteDataMatcher> expectedMatchers = {
    ConcreteDataMatcher{"TST", "A", 0},
    ConcreteDataMatcher{"TST", "B", 0},
    ConcreteDataMatcher{"DPL", "ENUM", compile_time_hash("A")}, // Enums value
  };

  std::vector<Lifetime> expectedLifetimes = {
    Lifetime::Timeframe,
    Lifetime::Timeframe,
    Lifetime::Enumeration};

  BOOST_REQUIRE_EQUAL(expectedMatchers.size(), expectedLifetimes.size());
  BOOST_CHECK_EQUAL(outputs.size(), expectedMatchers.size()); // FIXME: Is this what we actually want? We need
                                                              // different matchers depending on the different timeframe ID.

  for (size_t i = 0; i < outputs.size(); ++i) {
    auto concrete = DataSpecUtils::asConcreteDataMatcher(outputs[i]);
    BOOST_CHECK_EQUAL(concrete.origin.as<std::string>(), expectedMatchers[i].origin.as<std::string>());
    BOOST_CHECK_EQUAL(concrete.description.as<std::string>(), expectedMatchers[i].description.as<std::string>());
    BOOST_CHECK_EQUAL(concrete.subSpec, expectedMatchers[i].subSpec);
    BOOST_CHECK_EQUAL(static_cast<int>(outputs[i].lifetime), static_cast<int>(expectedLifetimes[i]));
  }

  for (size_t i = 0; i < logicalEdges.size(); ++i) {
    BOOST_CHECK_EQUAL(logicalEdges[i].producer, expected[i].producer);
    BOOST_CHECK_EQUAL(logicalEdges[i].consumer, expected[i].consumer);
    BOOST_CHECK_EQUAL(logicalEdges[i].timeIndex, expected[i].timeIndex);
    BOOST_CHECK_EQUAL(logicalEdges[i].producerTimeIndex, expected[i].producerTimeIndex);
    BOOST_CHECK_EQUAL(logicalEdges[i].outputGlobalIndex, expected[i].outputGlobalIndex);
  }

  std::vector<size_t> inIndex;
  std::vector<size_t> outIndex;
  WorkflowHelpers::sortEdges(inIndex, outIndex, logicalEdges);
  // Notice that zero is at the end because the first edge in the topological
  // sort is the timer and that gets added last.
  std::vector<size_t> expectedOutIndex{
    1, 2, 3, 4, 7, 5, 8, 6, 9, 0};

  std::vector<size_t> expectedInIndex{
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  BOOST_CHECK_EQUAL(expectedOutIndex.size(), outIndex.size());
  BOOST_CHECK_EQUAL(expectedInIndex.size(), inIndex.size());

  for (size_t i = 0; i < outIndex.size(); ++i) {
    BOOST_CHECK_EQUAL(expectedOutIndex[i], outIndex[i]);
  }
  for (size_t i = 0; i < inIndex.size(); ++i) {
    BOOST_CHECK_EQUAL(expectedInIndex[i], inIndex[i]);
  }
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
  };

  BOOST_REQUIRE_EQUAL(expectedActionsOut.size(), actions.size());
  for (size_t i = 0; i < outIndex.size(); i++) {
    size_t j = outIndex[i];
    BOOST_CHECK_EQUAL_MESSAGE(expectedActionsOut[j].requiresNewDevice, actions[j].requiresNewDevice, i << " " << j);
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
  };
  auto inActions = WorkflowHelpers::computeInEdgeActions(logicalEdges,
                                                         inIndex);

  BOOST_REQUIRE_EQUAL(expectedActionsIn.size(), inActions.size());
  for (size_t i = 0; i < inIndex.size(); i++) {
    size_t j = inIndex[i];
    auto expectedValue = expectedActionsIn[j].requiresNewDevice;
    auto actualValue = inActions[j].requiresNewDevice;

    BOOST_CHECK_EQUAL_MESSAGE(expectedValue, actualValue, i << " " << j);
  }
}

// This is to test a workflow where the input is not of type Timeframe and
// therefore requires a dangling channel.
// The topology is
//
// TST/A     TST/B
// ----> (A) ---->
//
BOOST_AUTO_TEST_CASE(TestExternalInput)
{
  WorkflowSpec workflow{
    {"A",
     Inputs{
       InputSpec{"external", "TST", "A", 0, Lifetime::Timer}},
     Outputs{
       OutputSpec{"TST", "B"}}}};
  BOOST_REQUIRE(WorkflowHelpers::verifyWorkflow(workflow) == WorkflowParsingState::Valid);
  std::vector<DeviceConnectionEdge> logicalEdges;
  std::vector<OutputSpec> outputs;
  std::vector<LogicalForwardInfo> availableForwardsInfo;

  BOOST_CHECK_EQUAL(workflow.size(), 1);

  auto context = makeEmptyConfigContext();
  WorkflowHelpers::injectServiceDevices(workflow, *context);
  // The added devices are the one which should connect to
  // the condition DB and the sink for the dangling outputs.
  BOOST_CHECK_EQUAL(workflow.size(), 3);
  WorkflowHelpers::constructGraph(workflow, logicalEdges,
                                  outputs,
                                  availableForwardsInfo);
}

BOOST_AUTO_TEST_CASE(DetermineDanglingOutputs)
{
  WorkflowSpec workflow0{
    {"A", Inputs{}, {OutputSpec{"TST", "A"}}},
    {"B", {InputSpec{"a", "TST", "A"}}, Outputs{}}};

  WorkflowSpec workflow1{
    {"A",
     Inputs{},
     Outputs{OutputSpec{"TST", "A"}}}};

  WorkflowSpec workflow2{
    {"A", Inputs{}, {OutputSpec{"TST", "A"}}},
    {"B", {InputSpec{"a", "TST", "B"}}, Outputs{}}};

  WorkflowSpec workflow3{
    {"A", Inputs{}, {OutputSpec{"TST", "A"}, OutputSpec{"TST", "B"}}},
    {"B", {InputSpec{"a", "TST", "A"}}, Outputs{}}};

  WorkflowSpec workflow4{
    {"A", Inputs{}, {OutputSpec{"TST", "A"}, OutputSpec{"TST", "B"}, OutputSpec{"TST", "C"}}},
    {"B", {InputSpec{"a", "TST", "A"}}, Outputs{}}};

  auto dangling0 = WorkflowHelpers::computeDanglingOutputs(workflow0);
  std::vector<InputSpec> expected0{};
  BOOST_TEST(dangling0 == expected0, boost::test_tools::per_element());

  auto dangling1 = WorkflowHelpers::computeDanglingOutputs(workflow1);
  std::vector<InputSpec> expected1{InputSpec{"dangling0", "TST", "A"}};
  BOOST_TEST(dangling1 == expected1, boost::test_tools::per_element());

  auto dangling2 = WorkflowHelpers::computeDanglingOutputs(workflow2);
  std::vector<InputSpec> expected2{InputSpec{"dangling0", "TST", "A"}};
  BOOST_TEST(dangling2 == expected2, boost::test_tools::per_element());

  auto dangling3 = WorkflowHelpers::computeDanglingOutputs(workflow3);
  std::vector<InputSpec> expected3{InputSpec{"dangling0", "TST", "B"}};
  BOOST_TEST(dangling3 == expected3, boost::test_tools::per_element());

  auto dangling4 = WorkflowHelpers::computeDanglingOutputs(workflow4);
  std::vector<InputSpec> expected4{InputSpec{"dangling0", "TST", "B"}, InputSpec{"dangling1", "TST", "C"}};
  BOOST_TEST(dangling4 == expected4, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(TEST_SELECT)
{
  auto res = o2::framework::select();
  BOOST_CHECK(res.empty());
  auto res1 = o2::framework::select("x:TST/C1/0");
  BOOST_CHECK(res1.size() == 1);
}

// Test the case in which two outputs are matched by the same generic input on B
// A/1
//     \
//      B becomes Timer -> A -> B
//     /
// A/2
BOOST_AUTO_TEST_CASE(TestOriginWildcard)
{
  std::vector<InputSpec> expectedInputs = {InputSpec{"x", DataSpecUtils::dataDescriptorMatcherFrom(o2::header::DataOrigin{"A"})}};
  std::vector<OutputSpec> expectedOutputs = {OutputSpec{"A", "1"}, OutputSpec{"A", "2"}, OutputSpec{"DPL", "TIMER", 0, Lifetime::Timer}};
  WorkflowSpec workflow{
    {"A", {}, {expectedOutputs[0], expectedOutputs[1]}},
    {"B", expectedInputs, {}},
  };
  std::vector<DeviceConnectionEdge> logicalEdges;
  std::vector<OutputSpec> outputs;
  std::vector<LogicalForwardInfo> availableForwardsInfo;

  BOOST_REQUIRE(WorkflowHelpers::verifyWorkflow(workflow) == WorkflowParsingState::Valid);
  auto context = makeEmptyConfigContext();
  WorkflowHelpers::injectServiceDevices(workflow, *context);
  BOOST_CHECK_EQUAL(workflow.size(), 3);
  BOOST_REQUIRE(workflow.size() >= 3);
  BOOST_CHECK_EQUAL(workflow[0].name, "A");
  BOOST_CHECK_EQUAL(workflow[1].name, "B");
  BOOST_CHECK_EQUAL(workflow[2].name, "internal-dpl-clock");
  for (size_t wi = 3; wi < workflow.size(); ++wi) {
    BOOST_CHECK_EQUAL(workflow[wi].name, "");
  }
  WorkflowHelpers::constructGraph(workflow, logicalEdges,
                                  outputs,
                                  availableForwardsInfo);

  std::vector<DeviceConnectionEdge> expectedEdges{
    {2, 0, 0, 0, 2, 0, false, ConnectionKind::Out},
    {0, 1, 0, 0, 0, 0, false, ConnectionKind::Out},
    {0, 1, 0, 0, 1, 0, false, ConnectionKind::Out},
  };

  std::vector<size_t> expectedOutEdgeIndex = {1, 2, 0};
  std::vector<EdgeAction> expectedActions = {
    {true, true},  // to go from timer to A (new channel and new device)
    {true, true},  // to go from A/1 to B (new channel and new device)
    {false, false} // to go from A/2 to B (device is the same as A/1, device is the same as B?)
  };

  // Not sure I understand...
  std::vector<EdgeAction> expectedInActions = {
    {true, true},
    {true, true},
    {false, false},
  };

  BOOST_REQUIRE_EQUAL(expectedOutputs.size(), outputs.size());
  BOOST_REQUIRE_EQUAL(expectedEdges.size(), logicalEdges.size());
  for (size_t ei = 0, ee = expectedEdges.size(); ei != ee; ++ei) {
    BOOST_CHECK_EQUAL(expectedEdges[ei].consumer, logicalEdges[ei].consumer);
    BOOST_CHECK_EQUAL(expectedEdges[ei].producer, logicalEdges[ei].producer);
    BOOST_CHECK_EQUAL(expectedEdges[ei].outputGlobalIndex, logicalEdges[ei].outputGlobalIndex);
    BOOST_CHECK_EQUAL(expectedEdges[ei].consumerInputIndex, logicalEdges[ei].consumerInputIndex);
  }

  std::vector<size_t> inEdgeIndex;
  std::vector<size_t> outEdgeIndex;
  WorkflowHelpers::sortEdges(inEdgeIndex, outEdgeIndex, logicalEdges);
  BOOST_REQUIRE_EQUAL(inEdgeIndex.size(), 3);
  BOOST_REQUIRE_EQUAL(outEdgeIndex.size(), 3);
  for (size_t ei = 0; ei < outEdgeIndex.size(); ++ei) {
    BOOST_CHECK_EQUAL(outEdgeIndex[ei], expectedOutEdgeIndex[ei]);
  }

  std::vector<EdgeAction> outActions = WorkflowHelpers::computeOutEdgeActions(logicalEdges, outEdgeIndex);
  BOOST_REQUIRE_EQUAL(outActions.size(), expectedActions.size());
  for (size_t ai = 0; ai < outActions.size(); ++ai) {
    BOOST_CHECK_EQUAL(outActions[ai].requiresNewDevice, expectedActions[ai].requiresNewDevice);
    BOOST_CHECK_EQUAL(outActions[ai].requiresNewChannel, expectedActions[ai].requiresNewChannel);
  }

  // Crete the connections on the inverse map for all of them
  // lookup for port and add as input of the current device.
  std::vector<EdgeAction> inActions = WorkflowHelpers::computeInEdgeActions(logicalEdges, inEdgeIndex);
  BOOST_REQUIRE_EQUAL(inActions.size(), expectedInActions.size());
  for (size_t ai = 0; ai < inActions.size(); ++ai) {
    BOOST_CHECK_EQUAL(inActions[ai].requiresNewDevice, expectedInActions[ai].requiresNewDevice);
    BOOST_CHECK_EQUAL(inActions[ai].requiresNewChannel, expectedInActions[ai].requiresNewChannel);
  }
}
