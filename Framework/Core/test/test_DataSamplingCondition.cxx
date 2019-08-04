// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test Framework DataSamplingCondition
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <vector>
#include <boost/test/unit_test.hpp>

#include "Framework/DataSamplingConditionFactory.h"
#include "Framework/DataRef.h"
#include "Framework/DataProcessingHeader.h"
#include "Headers/DataHeader.h"
#include "Headers/Stack.h"

using namespace o2::framework;
using namespace o2::header;

BOOST_AUTO_TEST_CASE(DataSamplingConditionRandom)
{
  auto conditionRandom = DataSamplingConditionFactory::create("random");
  BOOST_REQUIRE(conditionRandom);

  boost::property_tree::ptree config;
  config.put("fraction", "0.5");
  config.put("seed", "943753948");
  conditionRandom->configure(config);

  // PRNG should behave the same every time and on every machine.
  // Of course, the test does not cover full range of timesliceIDs, but at least gives an idea about its determinism.
  {
    std::vector<bool> correctDecision{
      true, false, true, false, true, false, false, true, false, false, true, true, false, false, true, false, false,
      true, false, false, true, true, true, false, false, false, true, false, true, true, true, false, false, true,
      false, false, false, false, false, false, true, false, false, true, false, false, true, false, false};
    for (DataProcessingHeader::StartTime id = 1; id < 50; id++) {
      DataProcessingHeader dph{id, 0};
      o2::header::Stack headerStack{dph};
      DataRef dr{nullptr, reinterpret_cast<const char*>(headerStack.data()), nullptr};
      BOOST_CHECK_EQUAL(correctDecision[id - 1], conditionRandom->decide(dr));
    }
  }

  // random access check
  {
    std::vector<std::pair<DataProcessingHeader::StartTime, bool>> correctDecision{
      {222, true},
      {222, true},
      {222, true},
      {230, false},
      {210, true},
      {230, false},
      {250, false},
      {251, false},
      {222, true},
      {230, false}};
    for (const auto& check : correctDecision) {
      DataProcessingHeader dph{check.first, 0};
      o2::header::Stack headerStack{dph};
      DataRef dr{nullptr, reinterpret_cast<const char*>(headerStack.data()), nullptr};
      BOOST_CHECK_EQUAL(check.second, conditionRandom->decide(dr));
    }
  }
}

BOOST_AUTO_TEST_CASE(DataSamplingConditionPayloadSize)
{
  auto conditionPayloadSize = DataSamplingConditionFactory::create("payloadSize");
  BOOST_REQUIRE(conditionPayloadSize);

  boost::property_tree::ptree config;
  config.put("upperLimit", 500);
  config.put("lowerLimit", 30);
  conditionPayloadSize->configure(config);

  std::vector<std::pair<size_t, bool>> testCases{
    {0, false},
    {29, false},
    {30, true},
    {200, true},
    {500, true},
    {501, false}};

  for (const auto& t : testCases) {
    DataHeader dh;
    dh.payloadSize = t.first;
    o2::header::Stack headerStack{dh};
    DataRef dr{nullptr, reinterpret_cast<const char*>(headerStack.data()), nullptr};
    BOOST_CHECK_EQUAL(conditionPayloadSize->decide(dr), t.second);
  }
}
BOOST_AUTO_TEST_CASE(DataSamplingConditionNConsecutive)
{
  auto conditionNConsecutive = DataSamplingConditionFactory::create("nConsecutive");
  BOOST_REQUIRE(conditionNConsecutive);

  boost::property_tree::ptree config;
  config.put("samplesNumber", 3);
  config.put("cycleSize", 10);
  conditionNConsecutive->configure(config);

  std::vector<std::pair<size_t, bool>> testCases{
    {0, true},
    {1, true},
    {2, true},
    {3, false},
    {8, false},
    {9, false},
    {9999999999999, false},
    {10000000000000, true},
    {10000000000001, true},
    {10000000000002, true},
    {10000000000003, false}};

  for (const auto& t : testCases) {
    DataProcessingHeader dph{t.first, 0};
    o2::header::Stack headerStack{dph};
    DataRef dr{nullptr, reinterpret_cast<const char*>(headerStack.data()), nullptr};
    BOOST_CHECK_EQUAL(conditionNConsecutive->decide(dr), t.second);
  }
}
