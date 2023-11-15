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

#include <catch_amalgamated.hpp>
#include "Framework/CompilerBuiltins.h"

struct Foo {
  int foo;
  int O2_VISIBILITY_HIDDEN someHiddenMethod() { return 0; };
};

struct O2_VISIBILITY_HIDDEN Bar {
  int someMethod() { return 0; };
};

TEST_CASE("TestPrefetch")
{
  int a[10];
  int b[10];

  [[maybe_unused]] Foo foo;
  [[maybe_unused]] Bar bar;

  for (int i = 0; i < 10; i++) {
    a[i] = a[i] + b[i];
    O2_BUILTIN_PREFETCH(&a[i + 1], 1, 1);
    O2_BUILTIN_PREFETCH(&b[i + 1], 0, 1);
    /* … */
  }
}
