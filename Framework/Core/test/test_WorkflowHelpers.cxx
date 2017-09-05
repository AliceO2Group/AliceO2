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

#include "test_HelperMacros.h"
#include "Framework/WorkflowSpec.h"
#include "../src/WorkflowHelpers.h"
#include <boost/test/unit_test.hpp>

using namespace o2::framework;


BOOST_AUTO_TEST_CASE(TestVerifyWorkflow) {
  using namespace o2::framework;
  auto checkIncompleteInput = [](WorkflowSpec const &workflow) {
    // Empty workflows should be invalid.
    BOOST_CHECK_THROW(WorkflowHelpers::verifyWorkflow(workflow), std::runtime_error);
  };

  auto checkOk = [](WorkflowSpec const &workflow) {
    // Empty workflows should be invalid.
    BOOST_CHECK_NO_THROW(WorkflowHelpers::verifyWorkflow(workflow));
  };

  // A non fully specified input is an error, given the result is ambiguous.
  // Completely ambiguous.
  checkIncompleteInput(WorkflowSpec{{"A", {InputSpec{}}}});
  // missing origin and description
  checkIncompleteInput(WorkflowSpec{{"A", {InputSpec{"x"}}}});
  // missing description
  checkIncompleteInput(WorkflowSpec{{"A", {InputSpec{"x", "TST"}}}});
  // This is fine, since by default both subSpec == 0 and 
  // Timeframe are assumed.
  checkOk(WorkflowSpec{{"A", {InputSpec{"x", "TST", "A"}}}});
}

BOOST_AUTO_TEST_CASE(TestWorkflowHelpers) {
  using namespace o2::framework;
  using Edges = std::vector<std::pair<size_t, size_t>>;
  // Already sorted
  Edges edges1 = {
    {0,1}, // 1 depends on 0
    {1,2},
    {2,3}
  };
  auto result1 = WorkflowHelpers::topologicalSort(4,
                                                  &(edges1[0].first),
                                                  &(edges1[0].second),
                                                  sizeof(edges1[0])/sizeof(size_t),
                                                  3);
  auto expected1 = {0, 1, 2, 3};
  BOOST_CHECK(std::equal(result1.begin(), result1.end(), expected1.begin()));
  // Inverse sort
  Edges edges2 = {
    {3,2},
    {2,1},
    {1,0}
  };
  auto result2 = WorkflowHelpers::topologicalSort(4,
                                                  &edges2[0].first,
                                                  &edges2[0].second,
                                                  sizeof(edges2[0])/sizeof(size_t),
                                                  3);
  auto expected2 = {3, 2, 1, 0};
  BOOST_CHECK(std::equal(result2.begin(), result2.end(), expected2.begin()));
  //     2
  //    / \
  // 4-3   0-5
  //    \ /
  //     1
  Edges edges3 = {
    {0,5},
    {4,3},
    {3,2},
    {2,0},
    {1,0},
    {3,1},
  };
  auto result3 = WorkflowHelpers::topologicalSort(6,
                                                  &(edges3[0].first),
                                                  &(edges3[0].second),
                                                  sizeof(edges3[0])/sizeof(size_t),
                                                  6);
  auto expected3 = {4,3,1,2,0,5};
  BOOST_CHECK(std::equal(result3.begin(), result3.end(), expected3.begin()));
}

// Test a single connection
//
// A->B
BOOST_AUTO_TEST_CASE(TestSimpleConnection) {
  std::vector<InputSpec> expectedInputs = {InputSpec{"y", "TST", "A", InputSpec::Timeframe}};
  std::vector<OutputSpec> expectedOutputs = {OutputSpec{"TST", "A", OutputSpec::Timeframe}};
  WorkflowSpec workflow{
    {
      "A",
      {},
      Outputs{expectedOutputs[0]}
    },
    {
      "B", {expectedInputs[0]}
    }
  };
  std::vector<DeviceConnectionEdge> logicalEdges;
  std::vector<OutputSpec> outputs;
  std::vector<LogicalForwardInfo> availableForwardsInfo;

  WorkflowHelpers::verifyWorkflow(workflow);
  WorkflowHelpers::constructGraph(workflow, logicalEdges,
                                  outputs,
                                  availableForwardsInfo);
  std::vector<DeviceConnectionEdge> expectedEdges {
    {0, 1, 0, 0, 0, 0, false, ConnectionKind::Out},
  };
  BOOST_REQUIRE_EQUAL(expectedOutputs.size(), outputs.size());
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
BOOST_AUTO_TEST_CASE(TestSimpleForward) {
  std::vector<InputSpec> expectedInputs = {InputSpec{"y", "TST", "A", InputSpec::Timeframe}};
  std::vector<OutputSpec> expectedOutputs = {OutputSpec{"TST", "A", OutputSpec::Timeframe}};
  WorkflowSpec workflow{
    {
      "A",
      {},
      Outputs{expectedOutputs[0]}
    },
    {
      "B", {expectedInputs[0]}
    },
    {
      "C",
      {expectedInputs[0]}
    },
    {
      "D",
      {expectedInputs[0]}
    }
  };
  std::vector<DeviceConnectionEdge> logicalEdges;
  std::vector<OutputSpec> outputs;
  std::vector<LogicalForwardInfo> availableForwardsInfo;

  WorkflowHelpers::verifyWorkflow(workflow);
  WorkflowHelpers::constructGraph(workflow, logicalEdges,
                                  outputs,
                                  availableForwardsInfo);

  std::vector<DeviceConnectionEdge> expectedEdges {
    {0, 1, 0, 0, 0, 0, false, ConnectionKind::Out},
    {1, 2, 0, 0, 0, 0, true,  ConnectionKind::Out},
    {2, 3, 0, 0, 0, 0, true,  ConnectionKind::Out},
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

BOOST_AUTO_TEST_CASE(TestGraphConstruction) {
  WorkflowSpec workflow{
    {
      "A",
      Inputs{},
      Outputs{
        OutputSpec{"TST", "A", OutputSpec::Timeframe}
      }
    },
    timePipeline({
      "B",
      Inputs{InputSpec{"b", "TST", "A", InputSpec::Timeframe}},
      Outputs{OutputSpec{"TST", "B", OutputSpec::Timeframe}},
    }, 3),
    timePipeline({
      "C",
      Inputs{InputSpec{"c", "TST", "B", InputSpec::Timeframe}}
    }, 2)
  };

  std::vector<DeviceConnectionEdge> expected {
    {0, 1, 0, 0, 0, 0, false, ConnectionKind::Out},
    {0, 1, 1, 0, 0, 0, false, ConnectionKind::Out},
    {0, 1, 2, 0, 0, 0, false, ConnectionKind::Out},
    {1, 2, 0, 0, 1, 0, false, ConnectionKind::Out},
    {1, 2, 0, 1, 1, 0, false, ConnectionKind::Out},
    {1, 2, 0, 2, 1, 0, false, ConnectionKind::Out},
    {1, 2, 1, 0, 1, 0, false, ConnectionKind::Out},
    {1, 2, 1, 1, 1, 0, false, ConnectionKind::Out},
    {1, 2, 1, 2, 1, 0, false, ConnectionKind::Out}
  };
  std::list<LogicalOutputInfo> availableOutputsInfo;
  std::vector<DeviceConnectionEdge> logicalEdges;
  std::vector<LogicalForwardInfo> availableForwardsInfo;

  // This is a temporary store for inputs and outputs, including forwarded
  // channels, so that we can construct them before assigning to a device.
  std::vector<OutputSpec> outputs;

  WorkflowHelpers::verifyWorkflow(workflow);
  WorkflowHelpers::constructGraph(workflow, logicalEdges,
                                  outputs,
                                  availableForwardsInfo);

  BOOST_CHECK_EQUAL(outputs.size(), 2); // FIXME: Is this what we actually want? We need
                                        // different matchers depending on the different timeframe ID.
  Outputs expectedOutputs = {
    OutputSpec{"TST", "A", OutputSpec::Timeframe},
    OutputSpec{"TST", "B", OutputSpec::Timeframe},
  };

  for (size_t i = 0; i < outputs.size(); ++i) {
    BOOST_CHECK(outputs[i].description == expectedOutputs[i].description);
    BOOST_CHECK(outputs[i].origin == expectedOutputs[i].origin);
    BOOST_CHECK(outputs[i].lifetime == expectedOutputs[i].lifetime);
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
  std::vector<size_t> expectedOutIndex {
    0,1,2,3,6,4,7,5,8
  };

  std::vector<size_t> expectedInIndex {
    0,1,2,3,4,5,6,7,8
  };

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

  std::vector<EdgeAction> expectedActionsOut {
    EdgeAction{true, true},
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

  std::vector<EdgeAction> expectedActionsIn {
    EdgeAction{true, true},
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
