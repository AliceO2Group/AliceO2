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

#include <boost/test/unit_test.hpp>

#include "Framework/DataSamplingCondition.h"
#include "Framework/DataRef.h"
#include "Framework/DataProcessingHeader.h"

using namespace o2::framework;

BOOST_AUTO_TEST_CASE(DataSamplingConditionRandom)
{
  auto conditionRandom = DataSamplingCondition::getDataSamplingConditionRandom();

  // PRNG should behave the same every time and on every machine.
  // Of course, the test does not cover full range of timesliceIDs.
  std::vector<bool> correctDecision{
    false, true, false, true, true, false, true, false, false, true, false, true, false, false, false, false, false,
    true, false, false, true, true, false, false, true, true, false, false, false, false, true, true, false, false,
    true, true, false, false, false, false, false, true, false, false, false, false, false, true, false
  };
  conditionRandom->configure({ { "fraction", 0.5 }, { "seed", 943753948 } });

  for (DataProcessingHeader::StartTime id = 1; id < 50; id++) {
    DataProcessingHeader dph{ id, 0 };
    o2::header::Stack headerStack{ dph };
    DataRef dr{ nullptr, reinterpret_cast<const char*>(headerStack.data()), nullptr };
    BOOST_CHECK_EQUAL(correctDecision[id - 1], conditionRandom->decide(dr));
  }
}