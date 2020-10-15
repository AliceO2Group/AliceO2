// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test Framework DataSamplingPolicy
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/property_tree/ptree.hpp>

#include "DataSampling/DataSamplingConditionFactory.h"
#include "DataSampling/DataSamplingPolicy.h"
#include "Framework/DataRef.h"
#include "Framework/DataProcessingHeader.h"

using namespace o2::framework;
using namespace o2::utilities;
using namespace o2::header;

// an example of DataSamplingPolicy JSON object
// {
//   "id" : "policy_example1",
//   "active" : "false",
//   "machines" : [
//     "aidlalala1",
//     "aidlalala2"
//   ]
//   "query" : "c:TST/CHLEB/33;m:TST/MLEKO/33",
//   "samplingConditions" : [
//     {
//       "condition" : "random",
//       "fraction" : "0.1",
//       "seed" : "2137"
//     }
//   ],
//   "blocking" : "false"
// }

BOOST_AUTO_TEST_CASE(DataSamplingPolicyFromConfiguration)
{
  using boost::property_tree::ptree;

  ptree config;
  config.put("id", "my_policy");
  config.put("active", "true");
  config.put("query", "c:TST/CHLEB/33;m:TST/MLEKO/33");
  ptree samplingConditions;
  ptree conditionRandom;
  conditionRandom.put("condition", "random");
  conditionRandom.put("fraction", "0.1");
  conditionRandom.put("seed", "2137");
  samplingConditions.push_back(std::make_pair("", conditionRandom));
  config.add_child("samplingConditions", samplingConditions);
  config.put("blocking", "false");

  {
    auto policy = std::move(DataSamplingPolicy::fromConfiguration(config));

    BOOST_CHECK_EQUAL(policy.getName(), "my_policy");
    BOOST_CHECK((policy.prepareOutput(ConcreteDataMatcher{"TST", "CHLEB", 33})) == (Output{"DS", "my_policy0", 33}));
    BOOST_CHECK((policy.prepareOutput(ConcreteDataMatcher{"TST", "MLEKO", 33})) == (Output{"DS", "my_policy1", 33}));
    const auto& map = policy.getPathMap();
    BOOST_CHECK((*map.find(ConcreteDataMatcher{"TST", "CHLEB", 33})).second == (OutputSpec{"DS", "my_policy0", 33}));
    BOOST_CHECK((*map.find(ConcreteDataMatcher{"TST", "MLEKO", 33})).second == (OutputSpec{"DS", "my_policy1", 33}));
    BOOST_CHECK_EQUAL(map.size(), 2);

    BOOST_CHECK(policy.match(ConcreteDataMatcher{"TST", "CHLEB", 33}));
    BOOST_CHECK(!policy.match(ConcreteDataMatcher{"TST", "SZYNKA", 33}));

    DataProcessingHeader dph{555, 0};
    o2::header::Stack headerStack{dph};
    DataRef dr{nullptr, reinterpret_cast<const char*>(headerStack.data()), nullptr};
    policy.decide(dr); // just make sure it does not crash
  }

  config.put("id", "too-long-policy-name");

  {
    auto policy = std::move(DataSamplingPolicy::fromConfiguration(config));
    BOOST_CHECK_EQUAL(policy.getName(), "too-long-policy-name");
    BOOST_CHECK((policy.prepareOutput(ConcreteDataMatcher{"TST", "CHLEB", 33})) == (Output{"DS", "too-long-polic0", 33}));
    BOOST_CHECK((policy.prepareOutput(ConcreteDataMatcher{"TST", "MLEKO", 33})) == (Output{"DS", "too-long-polic1", 33}));
  }
}

BOOST_AUTO_TEST_CASE(DataSamplingPolicyFromMethods)
{
  DataSamplingPolicy policy("my_policy");
  auto conditionNConsecutive = DataSamplingConditionFactory::create("nConsecutive");
  BOOST_REQUIRE(conditionNConsecutive);

  boost::property_tree::ptree config;
  config.put("samplesNumber", 3);
  config.put("cycleSize", 10);
  conditionNConsecutive->configure(config);

  policy.registerCondition(std::move(conditionNConsecutive));

  policy.registerPath({"tststs", {"TST", "CHLEB"}}, {{"asdf"}, "AA", "BBBB"});
  BOOST_CHECK((policy.prepareOutput(ConcreteDataMatcher{"TST", "CHLEB", 33})) == (Output{"AA", "BBBB", 33}));
}

BOOST_AUTO_TEST_CASE(DataSamplingPolicyStaticMethods)
{
  BOOST_CHECK(DataSamplingPolicy::createPolicyDataOrigin() == DataOrigin("DS"));
  BOOST_CHECK(DataSamplingPolicy::createPolicyDataDescription("asdf", 0) == DataDescription("asdf0"));
  BOOST_CHECK(DataSamplingPolicy::createPolicyDataDescription("asdfasdfasdfasdf", 0) == DataDescription("asdfasdfasdfas0"));
  BOOST_CHECK(DataSamplingPolicy::createPolicyDataDescription("asdfasdfasdfasdf", 10) == DataDescription("asdfasdfasdfas10"));
}