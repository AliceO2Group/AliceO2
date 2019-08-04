// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test Framework VariantTest
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "Framework/Variant.h"
#include <stdexcept>
#include <sstream>
#include <cstring>

using namespace o2::framework;

bool unknown_type(std::runtime_error const& ex)
{
  return strcmp(ex.what(), "Mismatch between types") == 0;
}

BOOST_AUTO_TEST_CASE(VariantTest)
{
  std::ostringstream ss{};
  Variant a(10);
  BOOST_CHECK(a.get<int>() == 10);
  ss << a;
  Variant b(10.1f);
  BOOST_CHECK(b.get<float>() == 10.1f);
  ss << b;
  Variant c(10.2);
  BOOST_CHECK(c.get<double>() == 10.2);
  ss << c;
  BOOST_CHECK_EXCEPTION(a.get<char*>(), std::runtime_error, unknown_type);
  Variant d("foo");
  ss << d;
  BOOST_CHECK(std::string(d.get<const char*>()) == "foo");

  Variant e(true);
  BOOST_CHECK_EQUAL(e.get<bool>(), true);

  Variant f(false);
  BOOST_CHECK_EQUAL(f.get<bool>(), false);

  BOOST_CHECK(ss.str() == "1010.110.2foo");
  // Spotted valgrind error while deleting a vector of variants.
  std::vector<Variant> vector{1, 1.2, 1.1f, "foo"};
  Variant sa("foo");
  Variant sb(sa);            // Copy constructor
  Variant sc(std::move(sa)); // Move constructor
  Variant sd = sc;           // Copy operator

  BOOST_CHECK(std::string(sb.get<const char*>()) == "foo");
  BOOST_CHECK(std::string(sc.get<const char*>()) == "foo");
  BOOST_CHECK(std::string(sd.get<const char*>()) == "foo");
  BOOST_CHECK(sa.get<const char*>() == nullptr);
}
