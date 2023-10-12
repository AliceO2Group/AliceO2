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

/// @file   test_ransHistogramView.cxx
/// @author Michael Lettrich
/// @brief

#define BOOST_TEST_MODULE Utility test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <vector>
#include <algorithm>

#include <boost/test/unit_test.hpp>
#include <boost/mpl/vector.hpp>

#include "rANS/utils/HistogramView.h"

struct ReferenceState {
  ReferenceState(std::vector<int32_t> v) : mV{std::move(v)} {};

  int32_t getMin() { return *std::min_element(mV.begin(), mV.end()); };
  int32_t getMax() { return *std::max_element(mV.begin(), mV.end()); };
  size_t size() { return mV.size(); };
  int32_t getOffset() { return this->getMin(); };
  auto begin() { return mV.begin(); };
  auto end() { return mV.end(); };

  std::vector<int32_t> mV{};
};

BOOST_AUTO_TEST_CASE(test_emptyHistogramView)
{
  std::vector<int32_t> a{};
  o2::rans::utils::HistogramView v{a.begin(), a.end()};

  BOOST_CHECK_EQUAL(v.size(), 0);
  BOOST_CHECK_EQUAL(v.getOffset(), 0);
  BOOST_CHECK_EQUAL(v.getMin(), 0);
  BOOST_CHECK_EQUAL(v.getMax(), 0);
  BOOST_CHECK((v.begin() == a.begin()));
  BOOST_CHECK((v.end() == a.end()));
  BOOST_CHECK((v.rbegin().base() == a.end()));
  BOOST_CHECK((v.rend().base() == a.begin()));
};

struct HistogramViewFixture_one {
  ReferenceState expected{{-2}};
  o2::rans::utils::HistogramView<decltype(expected.begin())> view{expected.begin(), expected.end(), -2};
};

struct HistogramViewFixture_plus {
  ReferenceState expected{{2, 3, 4, 5}};
  o2::rans::utils::HistogramView<decltype(expected.begin())> view{expected.begin(), expected.end(), 2};
};

struct HistogramViewFixture_minus {
  ReferenceState expected{{-3, -2}};
  o2::rans::utils::HistogramView<decltype(expected.begin())> view{expected.begin(), expected.end(), -3};
};

struct HistogramViewFixture_plusminus {
  ReferenceState expected{{-3, -2, -1, 0, 1, 2, 3, 4, 5}};
  o2::rans::utils::HistogramView<decltype(expected.begin())> view{expected.begin(), expected.end(), -3};
};

using histogramViewFixtures_t = boost::mpl::vector<HistogramViewFixture_one,
                                                   HistogramViewFixture_plus,
                                                   HistogramViewFixture_minus,
                                                   HistogramViewFixture_plusminus>;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_histogramView, fixture, histogramViewFixtures_t)
{
  fixture f;

  ReferenceState& expected = f.expected;
  auto& view = f.view;

  BOOST_CHECK_EQUAL(view.size(), expected.size());
  BOOST_CHECK_EQUAL(view.getOffset(), expected.getOffset());
  BOOST_CHECK_EQUAL(view.getMin(), expected.getMin());
  BOOST_CHECK_EQUAL(view.getMax(), expected.getMax());
  BOOST_CHECK((view.begin() == expected.begin()));
  BOOST_CHECK((view.end() == expected.end()));
  BOOST_CHECK((view.rbegin().base() == expected.end()));
  BOOST_CHECK((view.rend().base() == expected.begin()));
};

BOOST_AUTO_TEST_CASE(test_trimEmpty)
{
  std::vector<int32_t> a{};
  o2::rans::utils::HistogramView v{a.begin(), a.end()};
  v = o2::rans::utils::trim(v);

  BOOST_CHECK_EQUAL(v.size(), 0);
  BOOST_CHECK_EQUAL(v.getOffset(), 0);
  BOOST_CHECK_EQUAL(v.getMin(), 0);
  BOOST_CHECK_EQUAL(v.getMax(), 0);
  BOOST_CHECK((v.begin() == a.begin()));
  BOOST_CHECK((v.end() == a.end()));
  BOOST_CHECK((v.rbegin().base() == a.end()));
  BOOST_CHECK((v.rend().base() == a.begin()));
};

BOOST_AUTO_TEST_CASE(test_trimFull)
{
  std::vector<int32_t> a{0, 0, 0, 0, 0};
  o2::rans::utils::HistogramView v{a.begin(), a.end()};
  v = o2::rans::utils::trim(v);

  BOOST_CHECK_EQUAL(v.size(), 0);
  BOOST_CHECK_EQUAL(v.getOffset(), 0);
  BOOST_CHECK_EQUAL(v.getMin(), 0);
  BOOST_CHECK_EQUAL(v.getMax(), 0);
  BOOST_CHECK((v.begin() == a.end()));
  BOOST_CHECK((v.end() == a.end()));
  BOOST_CHECK((v.rbegin().base() == a.end()));
  BOOST_CHECK((v.rend().base() == a.end()));
};

struct trimFixture_left {
  ReferenceState initial{{0, -3, -2, -1, 0, 1, 2, 3, 4, 5}};
  ReferenceState expected{{-3, -2, -1, 0, 1, 2, 3, 4, 5}};
  decltype(initial.begin()) begin = ++initial.begin();
  decltype(initial.end()) end = initial.end();
  o2::rans::utils::HistogramView<decltype(initial.begin())> view{initial.begin(), initial.end(), -4};
};

struct trimFixture_right {
  ReferenceState initial{{-3, -2, -1, 0, 1, 2, 3, 4, 5, 0}};
  ReferenceState expected{{-3, -2, -1, 0, 1, 2, 3, 4, 5}};
  decltype(initial.begin()) begin = initial.begin();
  decltype(initial.end()) end = --initial.end();
  o2::rans::utils::HistogramView<decltype(initial.begin())> view{initial.begin(), initial.end(), -3};
};

struct trimFixture_both {
  ReferenceState initial{{0, -3, -2, -1, 0, 1, 2, 3, 4, 5, 0}};
  ReferenceState expected{{-3, -2, -1, 0, 1, 2, 3, 4, 5}};
  decltype(initial.begin()) begin = ++initial.begin();
  decltype(initial.end()) end = --initial.end();
  o2::rans::utils::HistogramView<decltype(initial.begin())> view{initial.begin(), initial.end(), -4};
};

struct trimFixture_none {
  ReferenceState expected{{-3, -2, -1, 0, 1, 2, 3, 4, 5}};
  decltype(expected.begin()) begin = expected.begin();
  decltype(expected.end()) end = expected.end();
  o2::rans::utils::HistogramView<decltype(expected.begin())> view{expected.begin(), expected.end(), -3};
};

using trimFixture_t = boost::mpl::vector<trimFixture_left,
                                         trimFixture_right,
                                         trimFixture_both,
                                         trimFixture_none>;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_trim, fixture, trimFixture_t)
{
  fixture f;

  ReferenceState& expected = f.expected;
  f.view = o2::rans::utils::trim(f.view);
  auto& view = f.view;

  BOOST_CHECK_EQUAL(view.size(), expected.size());
  BOOST_CHECK_EQUAL(view.getOffset(), expected.getOffset());
  BOOST_CHECK_EQUAL(view.getMin(), expected.getMin());
  BOOST_CHECK_EQUAL(view.getMax(), expected.getMax());
  BOOST_CHECK((view.begin() == f.begin));
  BOOST_CHECK((view.end() == f.end));
  BOOST_CHECK((view.rbegin().base() == f.end));
  BOOST_CHECK((view.rend().base() == f.begin));
};

BOOST_AUTO_TEST_CASE(intersection_disjointLeft)
{
  std::vector<int32_t> a{-3, -2, -1, 0, 1, 2, 3, 4, 5};
  std::vector<int32_t> b{-10, -9};
  o2::rans::utils::HistogramView av{a.begin(), a.end(), -3};
  o2::rans::utils::HistogramView bv{b.begin(), b.end(), -10};

  av = o2::rans::utils::intersection(av, bv);

  BOOST_CHECK_EQUAL(av.size(), 0);
  BOOST_CHECK_EQUAL(av.getOffset(), 0);
  BOOST_CHECK_EQUAL(av.getMin(), 0);
  BOOST_CHECK_EQUAL(av.getMax(), 0);
  BOOST_CHECK((av.begin() == a.end()));
  BOOST_CHECK((av.end() == a.end()));
  BOOST_CHECK((av.rbegin().base() == a.end()));
  BOOST_CHECK((av.rend().base() == a.end()));
}

BOOST_AUTO_TEST_CASE(intersection_disjointRight)
{
  std::vector<int32_t> a{-3, -2, -1, 0, 1, 2, 3, 4, 5};
  std::vector<int32_t> b{9, 10};
  o2::rans::utils::HistogramView av{a.begin(), a.end(), -3};
  o2::rans::utils::HistogramView bv{b.begin(), b.end(), 9};

  av = o2::rans::utils::intersection(av, bv);

  BOOST_CHECK_EQUAL(av.size(), 0);
  BOOST_CHECK_EQUAL(av.getOffset(), 0);
  BOOST_CHECK_EQUAL(av.getMin(), 0);
  BOOST_CHECK_EQUAL(av.getMax(), 0);
  BOOST_CHECK((av.begin() == a.end()));
  BOOST_CHECK((av.end() == a.end()));
  BOOST_CHECK((av.rbegin().base() == a.end()));
  BOOST_CHECK((av.rend().base() == a.end()));
}

BOOST_AUTO_TEST_CASE(intersection_leftOverlap)
{
  std::vector<int32_t> a{-3, -2, -1, 0, 1, 2, 3, 4, 5};
  std::vector<int32_t> b{-5, -4, -3, -2, -1};
  o2::rans::utils::HistogramView av{a.begin(), a.end(), -3};
  o2::rans::utils::HistogramView bv{b.begin(), b.end(), -5};

  av = o2::rans::utils::intersection(av, bv);

  BOOST_CHECK_EQUAL(av.size(), 3);
  BOOST_CHECK_EQUAL(av.getOffset(), -3);
  BOOST_CHECK_EQUAL(av.getMin(), -3);
  BOOST_CHECK_EQUAL(av.getMax(), -1);
  BOOST_CHECK((av.begin() == a.begin()));
  BOOST_CHECK((av.end() == a.begin() + 3));
  BOOST_CHECK((av.rbegin().base() == a.begin() + 3));
  BOOST_CHECK((av.rend() == a.rend()));
}

BOOST_AUTO_TEST_CASE(intersection_rightOverlap)
{
  std::vector<int32_t> a{-3, -2, -1, 0, 1, 2, 3, 4, 5};
  std::vector<int32_t> b{4, 5, 6, 7, 8};
  o2::rans::utils::HistogramView av{a.begin(), a.end(), -3};
  o2::rans::utils::HistogramView bv{b.begin(), b.end(), 4};

  av = o2::rans::utils::intersection(av, bv);

  BOOST_CHECK_EQUAL(av.size(), 2);
  BOOST_CHECK_EQUAL(av.getOffset(), 4);
  BOOST_CHECK_EQUAL(av.getMin(), 4);
  BOOST_CHECK_EQUAL(av.getMax(), 5);
  BOOST_CHECK((av.begin() == a.end()) - 2);
  BOOST_CHECK((av.end() == a.end()));
  BOOST_CHECK((av.rbegin().base() == a.end()));
  BOOST_CHECK((av.rend().base() == a.end() - 2));
}

BOOST_AUTO_TEST_CASE(intersection_fullOverlap)
{
  std::vector<int32_t> a{-3, -2, -1, 0, 1, 2, 3, 4, 5};
  std::vector<int32_t> b{-1, 0, 1, 2};
  o2::rans::utils::HistogramView av{a.begin(), a.end(), -3};
  o2::rans::utils::HistogramView bv{b.begin(), b.end(), -1};

  av = o2::rans::utils::intersection(av, bv);

  BOOST_CHECK_EQUAL(av.size(), 4);
  BOOST_CHECK_EQUAL(av.getOffset(), -1);
  BOOST_CHECK_EQUAL(av.getMin(), -1);
  BOOST_CHECK_EQUAL(av.getMax(), 2);
  BOOST_CHECK((av.begin() == a.begin() + 2));
  BOOST_CHECK((av.end() == a.end() - 3));
  BOOST_CHECK((av.rbegin().base() == a.end() - 3));
  BOOST_CHECK((av.rend().base() == a.begin() + 2));
}

BOOST_AUTO_TEST_CASE(intersection_emptyB)
{
  std::vector<int32_t> a{-3, -2, -1, 0, 1, 2, 3, 4, 5};
  std::vector<int32_t> b{};
  o2::rans::utils::HistogramView av{a.begin(), a.end(), -3};
  o2::rans::utils::HistogramView bv{b.begin(), b.end(), 0};

  av = o2::rans::utils::intersection(av, bv);

  BOOST_CHECK_EQUAL(av.size(), 0);
  BOOST_CHECK_EQUAL(av.getOffset(), 0);
  BOOST_CHECK_EQUAL(av.getMin(), 0);
  BOOST_CHECK_EQUAL(av.getMax(), 0);
  BOOST_CHECK((av.begin() == a.end()));
  BOOST_CHECK((av.end() == a.end()));
  BOOST_CHECK((av.rbegin().base() == a.end()));
  BOOST_CHECK((av.rend().base() == a.end()));
}

BOOST_AUTO_TEST_CASE(intersection_emptyA)
{
  std::vector<int32_t> a{};
  std::vector<int32_t> b{-10, -9};
  o2::rans::utils::HistogramView av{a.begin(), a.end(), 0};
  o2::rans::utils::HistogramView bv{b.begin(), b.end(), -10};

  av = o2::rans::utils::intersection(av, bv);

  BOOST_CHECK_EQUAL(av.size(), 0);
  BOOST_CHECK_EQUAL(av.getOffset(), 0);
  BOOST_CHECK_EQUAL(av.getMin(), 0);
  BOOST_CHECK_EQUAL(av.getMax(), 0);
  BOOST_CHECK((av.begin() == a.end()));
  BOOST_CHECK((av.end() == a.end()));
  BOOST_CHECK((av.rbegin().base() == a.end()));
  BOOST_CHECK((av.rend().base() == a.end()));
}

BOOST_AUTO_TEST_CASE(intersection_empty)
{
  std::vector<int32_t> a{};
  std::vector<int32_t> b{};
  o2::rans::utils::HistogramView av{a.begin(), a.end(), 0};
  o2::rans::utils::HistogramView bv{b.begin(), b.end(), 0};

  av = o2::rans::utils::intersection(av, bv);

  BOOST_CHECK_EQUAL(av.size(), 0);
  BOOST_CHECK_EQUAL(av.getOffset(), 0);
  BOOST_CHECK_EQUAL(av.getMin(), 0);
  BOOST_CHECK_EQUAL(av.getMax(), 0);
  BOOST_CHECK((av.begin() == a.end()));
  BOOST_CHECK((av.end() == a.end()));
  BOOST_CHECK((av.rbegin().base() == a.end()));
  BOOST_CHECK((av.rend().base() == a.end()));
}

BOOST_AUTO_TEST_CASE(tails_empty)
{
  std::vector<int32_t> a{};
  std::vector<int32_t> b{};

  o2::rans::utils::HistogramView av{a.begin(), a.end()};
  o2::rans::utils::HistogramView bv{b.begin(), b.end()};

  auto v = o2::rans::utils::leftTail(av, bv);
  BOOST_CHECK_EQUAL(v.size(), 0);
  BOOST_CHECK_EQUAL(v.getOffset(), 0);
  BOOST_CHECK_EQUAL(v.getMin(), 0);
  BOOST_CHECK_EQUAL(v.getMax(), 0);
  BOOST_CHECK((v.begin() == a.end()));
  BOOST_CHECK((v.end() == a.end()));
  BOOST_CHECK((v.rbegin().base() == a.end()));
  BOOST_CHECK((v.rend().base() == a.end()));

  v = o2::rans::utils::rightTail(av, bv);
  BOOST_CHECK_EQUAL(v.size(), 0);
  BOOST_CHECK_EQUAL(v.getOffset(), 0);
  BOOST_CHECK_EQUAL(v.getMin(), 0);
  BOOST_CHECK_EQUAL(v.getMax(), 0);
  BOOST_CHECK((v.begin() == a.end()));
  BOOST_CHECK((v.end() == a.end()));
  BOOST_CHECK((v.rbegin().base() == a.end()));
  BOOST_CHECK((v.rend().base() == a.end()));
};

BOOST_AUTO_TEST_CASE(tails_emptyA)
{
  std::vector<int32_t> a{};
  std::vector<int32_t> b{-10, -9};

  o2::rans::utils::HistogramView av{a.begin(), a.end()};
  o2::rans::utils::HistogramView bv{b.begin(), b.end(), -10};

  auto v = o2::rans::utils::leftTail(av, bv);
  BOOST_CHECK_EQUAL(v.size(), av.size());
  BOOST_CHECK_EQUAL(v.getOffset(), av.getOffset());
  BOOST_CHECK_EQUAL(v.getMin(), av.getMin());
  BOOST_CHECK_EQUAL(v.getMax(), av.getMax());
  BOOST_CHECK((v.begin() == av.begin()));
  BOOST_CHECK((v.end() == av.end()));
  BOOST_CHECK((v.rbegin().base() == av.rend().base()));
  BOOST_CHECK((v.rend().base() == av.rend().base()));

  v = o2::rans::utils::rightTail(av, bv);
  BOOST_CHECK_EQUAL(v.size(), av.size());
  BOOST_CHECK_EQUAL(v.getOffset(), av.getOffset());
  BOOST_CHECK_EQUAL(v.getMin(), av.getMin());
  BOOST_CHECK_EQUAL(v.getMax(), av.getMax());
  BOOST_CHECK((v.begin() == av.begin()));
  BOOST_CHECK((v.end() == av.end()));
  BOOST_CHECK((v.rbegin().base() == av.rbegin().base()));
  BOOST_CHECK((v.rend().base() == av.rend().base()));
};

BOOST_AUTO_TEST_CASE(tails_emptyB)
{
  std::vector<int32_t> a{-3, -2, -1, 0, 1, 2, 3, 4, 5};
  std::vector<int32_t> b{};

  o2::rans::utils::HistogramView av{a.begin(), a.end(), -3};
  o2::rans::utils::HistogramView bv{b.begin(), b.end()};

  auto v = o2::rans::utils::leftTail(av, bv);
  BOOST_CHECK_EQUAL(v.size(), av.size());
  BOOST_CHECK_EQUAL(v.getOffset(), av.getOffset());
  BOOST_CHECK_EQUAL(v.getMin(), av.getMin());
  BOOST_CHECK_EQUAL(v.getMax(), av.getMax());
  BOOST_CHECK((v.begin() == av.begin()));
  BOOST_CHECK((v.end() == av.end()));
  BOOST_CHECK((v.rbegin().base() == av.rbegin().base()));
  BOOST_CHECK((v.rend().base() == av.rend().base()));

  v = o2::rans::utils::rightTail(av, bv);
  BOOST_CHECK_EQUAL(v.size(), av.size());
  BOOST_CHECK_EQUAL(v.getOffset(), av.getOffset());
  BOOST_CHECK_EQUAL(v.getMin(), av.getMin());
  BOOST_CHECK_EQUAL(v.getMax(), av.getMax());
  BOOST_CHECK((v.begin() == av.begin()));
  BOOST_CHECK((v.end() == av.end()));
  BOOST_CHECK((v.rbegin().base() == av.rbegin().base()));
  BOOST_CHECK((v.rend().base() == av.rend().base()));
};

BOOST_AUTO_TEST_CASE(tails_leftTail)
{
  std::vector<int32_t> a{-3, -2, -1, 0, 1, 2, 3, 4, 5};
  std::vector<int32_t> b{4, 5, 6, 7, 8};
  o2::rans::utils::HistogramView av{a.begin(), a.end(), -3};
  o2::rans::utils::HistogramView bv{b.begin(), b.end(), 4};

  auto v = o2::rans::utils::leftTail(av, bv);
  BOOST_CHECK_EQUAL(v.size(), 7);
  BOOST_CHECK_EQUAL(v.getOffset(), -3);
  BOOST_CHECK_EQUAL(v.getMin(), -3);
  BOOST_CHECK_EQUAL(v.getMax(), 3);
  BOOST_CHECK((v.begin() == a.begin()));
  BOOST_CHECK((v.end() == a.begin() + 7));
  BOOST_CHECK((v.rbegin().base() == a.begin() + 7));
  BOOST_CHECK((v.rend().base() == a.begin()));

  v = o2::rans::utils::rightTail(av, bv);
  BOOST_CHECK_EQUAL(v.size(), 0);
  BOOST_CHECK_EQUAL(v.getOffset(), 0);
  BOOST_CHECK_EQUAL(v.getMin(), 0);
  BOOST_CHECK_EQUAL(v.getMax(), 0);
  BOOST_CHECK((v.begin() == a.end()));
  BOOST_CHECK((v.end() == a.end()));
  BOOST_CHECK((v.rbegin().base() == a.end()));
  BOOST_CHECK((v.rend().base() == a.end()));
};

BOOST_AUTO_TEST_CASE(tails_rightTail)
{
  std::vector<int32_t> a{-3, -2, -1, 0, 1, 2, 3, 4, 5};
  std::vector<int32_t> b{-4, -3, -2, -1, 0};
  o2::rans::utils::HistogramView av{a.begin(), a.end(), -3};
  o2::rans::utils::HistogramView bv{b.begin(), b.end(), -4};

  auto v = o2::rans::utils::leftTail(av, bv);
  BOOST_CHECK_EQUAL(v.size(), 0);
  BOOST_CHECK_EQUAL(v.getOffset(), 0);
  BOOST_CHECK_EQUAL(v.getMin(), 0);
  BOOST_CHECK_EQUAL(v.getMax(), 0);
  BOOST_CHECK((v.begin() == a.end()));
  BOOST_CHECK((v.end() == a.end()));
  BOOST_CHECK((v.rbegin().base() == a.end()));
  BOOST_CHECK((v.rend().base() == a.end()));

  v = o2::rans::utils::rightTail(av, bv);
  BOOST_CHECK_EQUAL(v.size(), 5);
  BOOST_CHECK_EQUAL(v.getOffset(), 1);
  BOOST_CHECK_EQUAL(v.getMin(), 1);
  BOOST_CHECK_EQUAL(v.getMax(), 5);
  BOOST_CHECK((v.begin() == a.end() - 5));
  BOOST_CHECK((v.end() == a.end()));
  BOOST_CHECK((v.rbegin().base() == a.end()));
  BOOST_CHECK((v.rend().base() == a.end() - 5));
};

BOOST_AUTO_TEST_CASE(tails_bothTail)
{
  std::vector<int32_t> a{-3, -2, -1, 0, 1, 2, 3, 4, 5};
  std::vector<int32_t> b{-1, 0, 1, 2};
  o2::rans::utils::HistogramView av{a.begin(), a.end(), -3};
  o2::rans::utils::HistogramView bv{b.begin(), b.end(), -1};

  auto v = o2::rans::utils::leftTail(av, bv);
  BOOST_CHECK_EQUAL(v.size(), 2);
  BOOST_CHECK_EQUAL(v.getOffset(), -3);
  BOOST_CHECK_EQUAL(v.getMin(), -3);
  BOOST_CHECK_EQUAL(v.getMax(), -2);
  BOOST_CHECK((v.begin() == a.begin()));
  BOOST_CHECK((v.end() == a.begin() + 2));
  BOOST_CHECK((v.rbegin().base() == a.begin() + 2));
  BOOST_CHECK((v.rend().base() == a.begin()));

  v = o2::rans::utils::rightTail(av, bv);
  BOOST_CHECK_EQUAL(v.size(), 3);
  BOOST_CHECK_EQUAL(v.getOffset(), 3);
  BOOST_CHECK_EQUAL(v.getMin(), 3);
  BOOST_CHECK_EQUAL(v.getMax(), 5);
  BOOST_CHECK((v.begin() == a.end() - 3));
  BOOST_CHECK((v.end() == a.end()));
  BOOST_CHECK((v.rbegin().base() == a.end()));
  BOOST_CHECK((v.rend().base() == a.end() - 3));
};