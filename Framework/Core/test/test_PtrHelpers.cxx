// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework PtrHelpers
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
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

BOOST_AUTO_TEST_CASE(MatchingPtrMaker)
{

  std::shared_ptr<Base> s = std::move(o2::framework::make_matching<decltype(s), A>(1, 3));
  std::unique_ptr<Base> u = std::move(o2::framework::make_matching<decltype(u), B>(2, 4));

  BOOST_REQUIRE(s != nullptr);
  BOOST_REQUIRE(u != nullptr);
  BOOST_CHECK_EQUAL(s->v, 1);
  BOOST_CHECK_EQUAL(u->v, 2);
  BOOST_CHECK_EQUAL(static_cast<A*>(s.get())->va, 3);
  BOOST_CHECK_EQUAL(static_cast<B*>(u.get())->vb, 4);
}
// @endcond
