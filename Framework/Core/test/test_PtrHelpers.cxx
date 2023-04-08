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
#include "Framework/TypeTraits.h"
#include <memory>

/// exclude from doxygen, TODO: we might want to do this on a higher level
/// because of doxygen's autolinking of references, all 'A' are displayed as
/// reference to this struct.
/// @cond
struct Base {
  int v;
};
struct A : public Base {
  A(int v_, int va_) : Base{v_}, va{va_} {}
  int va;
};
struct B : public Base {
  B(int v_, int vb_) : Base{v_}, vb{vb_} {}
  int vb;
};

TEST_CASE("MatchingPtrMaker")
{

  std::shared_ptr<Base> s = std::move(o2::framework::make_matching<decltype(s), A>(1, 3));
  std::unique_ptr<Base> u = std::move(o2::framework::make_matching<decltype(u), B>(2, 4));

  REQUIRE(s != nullptr);
  REQUIRE(u != nullptr);
  REQUIRE(s->v == 1);
  REQUIRE(u->v == 2);
  REQUIRE(static_cast<A*>(s.get())->va == 3);
  REQUIRE(static_cast<B*>(u.get())->vb == 4);
}
// @endcond
