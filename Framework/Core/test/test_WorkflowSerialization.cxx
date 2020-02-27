// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test Framework WorkflowSerializationHelpers
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/WorkflowSpec.h"
#include "../src/WorkflowSerializationHelpers.h"
#include <boost/test/unit_test.hpp>

using namespace o2::framework;

BOOST_AUTO_TEST_CASE(TestVerifyWorkflow)
{
  using namespace o2::framework;
  WorkflowSpec w0{
    DataProcessorSpec{"A",
                      {InputSpec{"foo", "A", "COLLISIONCONTEXT", 1, Lifetime::Condition}},
                      {OutputSpec{{"bar"}, "C", "D", 2, Lifetime::Timeframe}},
                      AlgorithmSpec{[](ProcessingContext& ctx) {}},
                      {ConfigParamSpec{"aInt", VariantType::Int, 0, {"An Int"}},
                       ConfigParamSpec{"aFloat", VariantType::Float, 1.3, {"A Float"}},
                       ConfigParamSpec{"aBool", VariantType::Bool, true, {"A Bool"}},
                       ConfigParamSpec{"aString", VariantType::String, "some string", {"A String"}}}},
    DataProcessorSpec{"B",
                      {InputSpec{"foo", "C", "D"}},
                      {OutputSpec{{"bar1"}, "E", "F", 0},
                       OutputSpec{{"bar2"}, "E", "F", 1}},
                      AlgorithmSpec{[](ProcessingContext& ctx) {}},
                      {}},
    DataProcessorSpec{"C",
                      {},
                      {OutputSpec{{"bar"}, "G", "H"}},
                      AlgorithmSpec{[](ProcessingContext& ctx) {}},
                      {}},
    DataProcessorSpec{"D",
                      {InputSpec{"foo", {"C", "D"}}},
                      {OutputSpec{{"bar"}, {"I", "L"}}},
                      AlgorithmSpec{[](ProcessingContext& ctx) {}},
                      {}}};

  std::vector<DataProcessorInfo> metadataOut{
    {"A", "test_Framework_test_SerializationWorkflow", {"foo"}, {ConfigParamSpec{"aBool", VariantType::Bool, true, {"A Bool"}}}},
    {"B", "test_Framework_test_SerializationWorkflow", {"b-bar", "bfoof", "fbdbfaso"}},
    {"C", "test_Framework_test_SerializationWorkflow", {}},
    {"D", "test_Framework_test_SerializationWorkflow", {}},
  };

  std::vector<DataProcessorInfo> metadataIn{};

  std::ostringstream firstDump;
  WorkflowSerializationHelpers::dump(firstDump, w0, metadataOut);
  std::istringstream is;
  is.str(firstDump.str());
  WorkflowSpec w1;
  WorkflowSerializationHelpers::import(is, w1, metadataIn);

  std::ostringstream secondDump;
  WorkflowSerializationHelpers::dump(secondDump, w1, metadataIn);

  BOOST_REQUIRE_EQUAL(w0.size(), 4);
  BOOST_REQUIRE_EQUAL(w0.size(), w1.size());
  BOOST_CHECK_EQUAL(firstDump.str(), secondDump.str());
}
