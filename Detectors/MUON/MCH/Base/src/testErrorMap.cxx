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

using o2::mch::Error;
using o2::mch::ErrorGroup;
using o2::mch::ErrorMap;
using o2::mch::ErrorType;

BOOST_AUTO_TEST_CASE(DefaultErrorMapShouldBeEmpty)
{
  ErrorMap em;
  BOOST_CHECK_EQUAL(em.getNumberOfErrorTypes(), 0);
  BOOST_CHECK_EQUAL(em.getNumberOfErrors(), 0);
}

BOOST_AUTO_TEST_CASE(AddingErrorType)
{
  ErrorMap em;
  em.add(ErrorType{0}, 0, 0);
  em.add(ErrorType{1}, 0, 0);
  em.add(ErrorType{2}, 0, 0);
  BOOST_CHECK_EQUAL(em.getNumberOfErrorTypes(), 3);
}

BOOST_AUTO_TEST_CASE(AddingError)
{
  ErrorMap em;
  em.add(ErrorType{0}, 0, 0);
  em.add(ErrorType{1}, 0, 0, 2);
  em.add(Error{ErrorType{0}, 1, 2, 3});
  BOOST_CHECK_EQUAL(em.getNumberOfErrorTypes(), 2);
  BOOST_CHECK_EQUAL(em.getNumberOfErrors(), 6);
  BOOST_CHECK_EQUAL(em.getNumberOfErrors(ErrorType{0}), 4);
  BOOST_CHECK_EQUAL(em.getNumberOfErrors(ErrorGroup{0}), 6);
}

BOOST_AUTO_TEST_CASE(MergingError)
{
  ErrorMap em1;
  em1.add(ErrorType{0}, 0, 0);
  em1.add(ErrorType{1}, 0, 0);
  ErrorMap em2;
  em2.add(ErrorType{0}, 1, 2);
  em2.add(ErrorType{0}, 0, 0);
  em2.add(em1);
  BOOST_CHECK_EQUAL(em2.getNumberOfErrorTypes(), 2);
  BOOST_CHECK_EQUAL(em2.getNumberOfErrors(), 4);
}

BOOST_AUTO_TEST_CASE(ErrorFunction)
{
  ErrorMap em;
  em.add(ErrorType{0}, 0, 0);
  em.add(ErrorType{0}, 0, 0);
  em.add(ErrorType{1}, 0, 0);
  em.add(ErrorType{0}, 1, 2);

  std::vector<std::string> lines;
  auto f = [&lines](Error error) {
    lines.emplace_back(fmt::format("ET {} ID [{},{}] seen {} time(s)",
                                   static_cast<uint32_t>(error.type), error.id0, error.id1, error.count));
  };
  em.forEach(f);
  BOOST_REQUIRE_EQUAL(lines.size(), 3);
  BOOST_CHECK_EQUAL(lines[0], "ET 0 ID [0,0] seen 2 time(s)");
  BOOST_CHECK_EQUAL(lines[1], "ET 0 ID [1,2] seen 1 time(s)");
  BOOST_CHECK_EQUAL(lines[2], "ET 1 ID [0,0] seen 1 time(s)");

  uint64_t n(0);
  em.forEach(ErrorType{0}, [&n](Error error) {
    if (error.id0 == 0) {
      n += error.count;
    }
  });
  BOOST_CHECK_EQUAL(n, 2);
}
