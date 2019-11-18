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
#include "Framework/FunctionalHelpers.h"

using namespace o2::framework;

template <typename T>
using is_int_t = std::is_same<typename std::decay_t<T>, int>;

template <int A, int B>
struct TestStruct {
};
BOOST_AUTO_TEST_CASE(TestOverride)
{
  static_assert(pack_size(pack<int, float>{}) == 2, "Bad size for pack");
  static_assert(has_type_v<int, pack<int, float>> == true, "int should be in pack");
  static_assert(has_type_v<double, pack<int, float>> == false, "int should be in pack");

  static_assert(std::is_same_v<selected_pack<is_int_t, int, float, char>, pack<int>>, "selector should select int");
  static_assert(std::is_same_v<filtered_pack<is_int_t, int, float, char>, pack<float, char>>, "filter should remove int");
  static_assert(std::is_same_v<intersected_pack_t<pack<int, float, char>, pack<float, double>>, pack<float>>, "filter intersect two packs");
  static_assert(has_type_v<pack_element_t<0, pack<int>>, pack<int>> == true, "foo");
  print_pack<intersected_pack_t<pack<int>, pack<int>>>();
  print_pack<intersected_pack_t<pack<TestStruct<0, -1>, int>, pack<TestStruct<0, -1>, float>>>();
  static_assert(std::is_same_v<intersected_pack_t<pack<TestStruct<0, -1>, int>, pack<TestStruct<0, -1>, float>>, pack<TestStruct<0, -1>>>, "filter intersect two packs");
  static_assert(std::is_same_v<concatenated_pack_t<pack<int, float, char>, pack<float, double>>, pack<int, float, char, float, double>>, "pack should be concatenated");
}
