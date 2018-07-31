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

using namespace o2::framework;
using namespace o2::header;

BOOST_AUTO_TEST_CASE(DataSamplingConditionRandom)
{
  auto conditionRandom = DataSamplingConditionFactory::create("random");
  BOOST_REQUIRE(conditionRandom);

  // PRNG should behave the same every time and on every machine.
  // Of course, the test does not cover full range of timesliceIDs.
  std::vector<bool> correctDecision{
    false, true, false, true, true, false, true, false, false, true, false, true, false, false, false, false, false,
    true, false, false, true, true, false, false, true, true, false, false, false, false, true, true, false, false,
    true, true, false, false, false, false, false, true, false, false, false, false, false, true, false
  };
  boost::property_tree::ptree config;
  config.put("fraction", 0.5);
  config.put("seed", 943753948);
  conditionRandom->configure(config);

  for (DataProcessingHeader::StartTime id = 1; id < 50; id++) {
    DataProcessingHeader dph{ id, 0 };
    o2::header::Stack headerStack{ dph };
    DataRef dr{ nullptr, reinterpret_cast<const char*>(headerStack.data()), nullptr };
    BOOST_CHECK_EQUAL(correctDecision[id - 1], conditionRandom->decide(dr));
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
    { 0, false },
    { 29, false },
    { 30, true },
    { 200, true },
    { 500, true },
    { 501, false }
  };

  for (const auto& t : testCases) {
    DataHeader dh;
    dh.payloadSize = t.first;
    o2::header::Stack headerStack{ dh };
    DataRef dr{ nullptr, reinterpret_cast<const char*>(headerStack.data()), nullptr };
    BOOST_CHECK_EQUAL(conditionPayloadSize->decide(dr), t.second);
  }
}
