// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Ex4
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include "Ex2/A.h"
#include "Ex3/A.h"
#include "Ex4/A.h"

#include <iostream>
#include <sstream>

BOOST_AUTO_TEST_CASE(AValueShouldBeTheAnswer)
{
  ex4::A a;
  BOOST_CHECK_EQUAL(a.value(), 42);
}

BOOST_AUTO_TEST_CASE(ACtorShouldSayHello)
{
  std::stringstream buffer;
  std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());

  ex4::A a;

  std::string text = buffer.str();

  std::cout.rdbuf(old);

  BOOST_CHECK_EQUAL(text, "Hello from ex4::A ctor\n");
}

BOOST_AUTO_TEST_CASE(AAACtorShouldSayHello)
{
  std::stringstream buffer;
  std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());

  ex2::A a;
  ex3::A b;
  ex4::A c;

  std::string text = buffer.str();

  std::cout.rdbuf(old);

  BOOST_CHECK_EQUAL(text,
                    "Hello from ex2::A ctor\n"
                    "Hello from ex3::A ctor\n"
                    "Hello from ex4::A ctor\n");
}
