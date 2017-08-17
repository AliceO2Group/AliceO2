// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework Collection
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/Collection.h"
#include <boost/test/unit_test.hpp>
#include <iostream>


BOOST_AUTO_TEST_CASE(TestCollection) {
  using namespace o2::framework;
  struct Foo {
    int a;
    int b;
    int c;
  };
  Foo a[100];
  {
    Collection<Foo> ca(a, 100);
    int i = 0;
    for (auto &ci : ca) {
      assert(i < 200);
      assert(i < 100);
      ci.a = i;
      ci.b = i;
      ci.c = i;
      i++;
    }
    BOOST_CHECK(ca.size() == 100);
  }
  int i = 0;
  for (size_t ci = 0; i < 100; ++ci) {
    BOOST_CHECK(a[ci].a == i);
    BOOST_CHECK(a[ci].b == i);
    BOOST_CHECK(a[ci].c == i);
    i++;
  }

}
