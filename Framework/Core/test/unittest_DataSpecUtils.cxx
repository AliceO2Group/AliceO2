// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework DataSpecUtils
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/DataSpecUtils.h"
#include <boost/test/unit_test.hpp>

#include <string>

using namespace o2::framework;

BOOST_AUTO_TEST_CASE(DataSpecUtilsConcreteTest)
{
  OutputSpec spec{
    "TEST",
    "FOOO",
    1,
    Lifetime::Timeframe
  };
  ConcreteDataMatcher concrete = DataSpecUtils::asConcreteDataMatcher(spec);
  BOOST_CHECK_EQUAL(std::string(concrete.origin.as<std::string>()), "TEST");
  BOOST_CHECK_EQUAL(std::string(concrete.description.as<std::string>()), "FOOO");
  BOOST_CHECK_EQUAL(concrete.subSpec, 1);

  ConcreteDataTypeMatcher dataType = DataSpecUtils::asConcreteDataTypeMatcher(spec);
  BOOST_CHECK_EQUAL(std::string(dataType.origin.as<std::string>()), "TEST");
  BOOST_CHECK_EQUAL(std::string(dataType.description.as<std::string>()), "FOOO");
}
