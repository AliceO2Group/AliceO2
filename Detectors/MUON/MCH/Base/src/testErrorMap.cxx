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

#define BOOST_TEST_MODULE errormap test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "MCHBase/ErrorMap.h"
#include <fmt/core.h>

using o2::mch::ErrorMap;

BOOST_AUTO_TEST_CASE(DefaultErrorMapShouldBeEmpty)
{
  ErrorMap em;
  BOOST_CHECK_EQUAL(o2::mch::numberOfErrorTypes(em), 0);
  BOOST_CHECK_EQUAL(o2::mch::totalNumberOfErrors(em), 0);
}

BOOST_AUTO_TEST_CASE(AddingErrorType)
{
  ErrorMap em;
  em.add(0, 0, 0);
  em.add(1, 0, 0);
  em.add(2, 0, 0);
  BOOST_CHECK_EQUAL(o2::mch::numberOfErrorTypes(em), 3);
}

BOOST_AUTO_TEST_CASE(AddingError)
{
  ErrorMap em;
  em.add(0, 0, 0);
  em.add(0, 0, 0);
  em.add(0, 0, 0);
  BOOST_CHECK_EQUAL(o2::mch::numberOfErrorTypes(em), 1);
  BOOST_CHECK_EQUAL(o2::mch::totalNumberOfErrors(em), 3);
}

BOOST_AUTO_TEST_CASE(ErrorFunction)
{
  ErrorMap em;
  em.add(0, 0, 0);
  em.add(0, 0, 0);
  em.add(0, 0, 0);
  em.add(0, 1, 2);

  std::vector<std::string> lines;
  auto f = [&lines](uint32_t errorType, uint32_t id0, uint32_t id1,
                    uint64_t count) {
    lines.emplace_back(fmt::format("ET {} ID [{},{}] seen {} time(s)", errorType, id0, id1, count));
  };
  em.forEach(f);
  for (auto s : lines) {
    std::cout << s << "\n";
  }
  BOOST_REQUIRE_EQUAL(lines.size(), 2);
  BOOST_CHECK_EQUAL(lines[0], "ET 0 ID [0,0] seen 3 time(s)");
  BOOST_CHECK_EQUAL(lines[1], "ET 0 ID [1,2] seen 1 time(s)");
}
