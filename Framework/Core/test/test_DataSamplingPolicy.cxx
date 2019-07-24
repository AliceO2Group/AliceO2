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

#include "Framework/DataSamplingPolicy.h"
#include "Framework/DataRef.h"
#include "Framework/DataProcessingHeader.h"

using namespace o2::framework;

// an example of DataSamplingPolicy JSON object
// {
//   "id" : "policy_example1",
//   "active" : "false",
//   "machines" : [
//     "aidlalala1",
//     "aidlalala2"
//   ]
//   "dataHeaders" : [
//     {
//       "binding" : "clusters",
//       "dataOrigin" : "TPC",
//       "dataDescription" : "CLUSTERS"
//     },
//     {
//       "binding" : "tracks",
//       "dataOrigin" : "TPC",
//       "dataDescription" : "TRACKS"
//     }
//   ],
//   "subSpec" : "*",
//   "samplingConditions" : [
//     {
//       "condition" : "random",
//       "fraction" : "0.1",
//       "seed" : "2137"
//     }
//   ],
//   "blocking" : "false"
// }

BOOST_AUTO_TEST_CASE(DataSamplingPolicyConfiguration)
{
  using boost::property_tree::ptree;
  DataSamplingPolicy policy;

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

  policy.configure(config);

  BOOST_CHECK_EQUAL(policy.getName(), "my_policy");
  BOOST_CHECK((policy.prepareOutput(ConcreteDataMatcher{ "TST", "CHLEB", 33 })) == (Output{ "DS", "my_policy-0", 33 }));
  BOOST_CHECK((policy.prepareOutput(ConcreteDataMatcher{ "TST", "MLEKO", 33 })) == (Output{ "DS", "my_policy-1", 33 }));
  const auto& map = policy.getPathMap();
  BOOST_CHECK((*map.find(ConcreteDataMatcher{ "TST", "CHLEB", 33 })).second == (OutputSpec{ "DS", "my_policy-0", 33 }));
  BOOST_CHECK((*map.find(ConcreteDataMatcher{ "TST", "MLEKO", 33 })).second == (OutputSpec{ "DS", "my_policy-1", 33 }));
  BOOST_CHECK_EQUAL(map.size(), 2);

  BOOST_CHECK(policy.match(ConcreteDataMatcher{ "TST", "CHLEB", 33 }));
  BOOST_CHECK(!policy.match(ConcreteDataMatcher{ "TST", "SZYNKA", 33 }));

  DataProcessingHeader dph{ 555, 0 };
  o2::header::Stack headerStack{ dph };
  DataRef dr{ nullptr, reinterpret_cast<const char*>(headerStack.data()), nullptr };
  policy.decide(dr); // just make sure it does not crash

  config.put("id", "too-long-policy-name");
  policy.configure(config);
  BOOST_CHECK_EQUAL(policy.getName(), "too-long-polic");
  BOOST_CHECK((policy.prepareOutput(ConcreteDataMatcher{ "TST", "CHLEB", 33 })) == (Output{ "DS", "too-long-polic-0", 33 }));
  BOOST_CHECK((policy.prepareOutput(ConcreteDataMatcher{ "TST", "MLEKO", 33 })) == (Output{ "DS", "too-long-polic-1", 33 }));
  BOOST_CHECK_EQUAL(policy.getPathMap().size(), 2); // previous paths should be cleared
}
