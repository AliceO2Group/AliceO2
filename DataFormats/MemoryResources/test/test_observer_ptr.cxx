// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test observer_ptr
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <functional>
#include "MemoryResources/observer_ptr.h"

/// exclude from doxygen, TODO: we might want to do this on a higher level
/// because of doxygen's autolinking of references, all 'A' are displayed as
/// reference to this struct.
/// @cond
struct A {
  int i{2};
  int get() const { return i; }
};

BOOST_AUTO_TEST_CASE(observer_ptr_A)
{
  using namespace o2;
  A t;
  int i{1};

  {
    observer_ptr<int> p{nullptr};
    BOOST_CHECK(p == nullptr);
  }

  {
    observer_ptr<int> p;
    BOOST_CHECK(p == nullptr);
  }

  {
    observer_ptr<int> b{&i};
    observer_ptr<void> a(b);
    BOOST_CHECK(a == b);
  }

  {
    observer_ptr<int> b{&i};
    observer_ptr<int> a(b);
    BOOST_CHECK(a == b);
  }

  {
    const observer_ptr<int> b{&i};
    observer_ptr<int> a(b);
    BOOST_CHECK(a == b);
  }

  {
    observer_ptr<A> pt{&t};
    A* ptt = pt.release();
    BOOST_CHECK(ptt == &t);
    BOOST_CHECK(pt == nullptr);
  }

  {
    observer_ptr<A> p{&t};
    p.reset();
    BOOST_CHECK(p == nullptr);
    p.reset(&t);
    BOOST_CHECK(p.get() == &t);
  }

  {
    A tt{4};
    observer_ptr<A> pp{&tt};
    observer_ptr<A> p{&t};
    p.swap(pp);
    BOOST_CHECK(p->get() == 4 && pp->get() == 2);
    p.swap(nullptr);
    BOOST_CHECK(p == nullptr);
    std::swap(p, pp);
    BOOST_CHECK(pp == nullptr);
    BOOST_CHECK(p->get() == 2);
  }

  {
    observer_ptr<A> p{&t};
    BOOST_CHECK(p.get() == &t);
  }

  {
    observer_ptr<A> p{&t};
    BOOST_CHECK((*p).i == 2);
  }

  {
    observer_ptr<A> p;
    p = &t;
    BOOST_CHECK(p.get() == &t);
    observer_ptr<A> pp;
    pp = p;
    BOOST_CHECK(pp.get() == &t);
  }

  //comparisons
  {
    A tt{4};
    observer_ptr<A> pp{&tt};
    observer_ptr<A> p{&t};
    observer_ptr<A> p_{&t};
    observer_ptr<A> n{nullptr};
    BOOST_CHECK(p != pp);
    BOOST_CHECK(p == p_);
    BOOST_CHECK(p != nullptr);
    BOOST_CHECK(n == nullptr);
    BOOST_CHECK((&tt > &t) ? pp > p : p > pp);
    BOOST_CHECK((&tt < &t) ? pp < p : p < pp);
    BOOST_CHECK((&tt >= &t) ? pp >= p : p >= pp);
    BOOST_CHECK((&tt <= &t) ? pp <= p : p <= pp);
    BOOST_CHECK((&t >= &t) ? p >= p : p >= p);
    BOOST_CHECK((&t <= &t) ? p <= p : p <= p);
  }

  //hash
  {
    observer_ptr<int> p{&i};
    BOOST_CHECK(std::hash<o2::observer_ptr<int>>()(p) == std::hash<int*>()(p.get()));
  }
}
// @endcond
