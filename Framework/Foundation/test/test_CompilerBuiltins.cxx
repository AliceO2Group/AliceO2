// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework Traits
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "Framework/CompilerBuiltins.h"

struct Foo {
  int foo;
  int O2_VISIBILITY_HIDDEN someHiddenMethod() { return 0; };
};

struct O2_VISIBILITY_HIDDEN Bar {
  int someMethod() { return 0; };
};

BOOST_AUTO_TEST_CASE(TestPrefetch)
{
  int a[10];
  int b[10];

  Foo foo;
  Bar bar;

  for (int i = 0; i < 10; i++) {
    a[i] = a[i] + b[i];
    O2_BUILTIN_PREFETCH(&a[i + 1], 1, 1);
    O2_BUILTIN_PREFETCH(&b[i + 1], 0, 1);
    /* … */
  }
}
