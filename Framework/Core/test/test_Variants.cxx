// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
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

using namespace o2::framework;

bool unknown_type( std::runtime_error const &ex ) { return strcmp(ex.what(), "Unknown type") == 0; }


BOOST_AUTO_TEST_CASE(VariantTest) {
  Variant a(10);
  BOOST_CHECK(a.get<int>() == 10);
  Variant b(10.f);
  BOOST_CHECK(b.get<float>() == 10.f);
  Variant c(10.);
  BOOST_CHECK(c.get<double>() == 10.);
  BOOST_REQUIRE_EXCEPTION(a.get<char*>(), std::runtime_error, unknown_type);
}

