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

#define BOOST_TEST_MODULE Test Framework Traits
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "Framework/FunctionalHelpers.h"
#include "Framework/Pack.h"
#include "Framework/CheckTypes.h"

using namespace o2::framework;

template <typename T>
using is_int_t = std::is_same<typename std::decay_t<T>, int>;

template <typename T, typename T2>
using is_same_as_second_t = std::is_same<typename std::decay_t<T>, T2>;

template <int A, int B>
struct TestStruct {
};
BOOST_AUTO_TEST_CASE(TestOverride)
{
  static_assert(pack_size(pack<int, float>{}) == 2, "Bad size for the pack");
  static_assert(has_type_v<int, pack<int, float>> == true, "int should be in the pack");
  static_assert(has_type_v<double, pack<int, float>> == false, "double should not be in the pack");
  static_assert(has_type_conditional_v<std::is_same, int, pack<int, float>> == true, "int should be in the pack");
  static_assert(has_type_conditional_v<std::is_same, double, pack<int, float>> == false, "double should not be in the pack");

  pack<float, char, int, bool> pck;
  static_assert(has_type_at<int>(pck) == 2, "int should be at 2");
  static_assert(has_type_at<double>(pck) == pack_size(pck) + 1, "double is not in the pack so the function returns size + 1");
  static_assert(has_type_at_conditional<std::is_same, bool>(pack<int, float, bool>()) == 2, "bool should be at 2");
  static_assert(has_type_at_conditional<std::is_same, bool>(pack<int, float, double>()) == 3 + 1, "bool is not in the pack so the function returns size + 1");

  static_assert(std::is_same_v<selected_pack<is_int_t, int, float, char>, pack<int>>, "selector should select int");
  static_assert(std::is_same_v<selected_pack_multicondition<is_same_as_second_t, pack<int>, pack<int, float, char>>, pack<int>>, "multiselector should select int");
  static_assert(std::is_same_v<filtered_pack<is_int_t, int, float, char>, pack<float, char>>, "filter should remove int");
  static_assert(std::is_same_v<intersected_pack_t<pack<int, float, char>, pack<float, double>>, pack<float>>, "filter intersect two packs");
  static_assert(has_type_v<pack_element_t<0, pack<int>>, pack<int>> == true, "foo");
  print_pack<intersected_pack_t<pack<int>, pack<int>>>();
  print_pack<intersected_pack_t<pack<TestStruct<0, -1>, int>, pack<TestStruct<0, -1>, float>>>();
  static_assert(std::is_same_v<intersected_pack_t<pack<TestStruct<0, -1>, int>, pack<TestStruct<0, -1>, float>>, pack<TestStruct<0, -1>>>, "filter intersect two packs");
  static_assert(std::is_same_v<concatenated_pack_t<pack<int, float, char>, pack<float, double>>, pack<int, float, char, float, double>>, "pack should be concatenated");
  static_assert(std::is_same_v<concatenated_pack_t<pack<int, float, char>, pack<float, double>, pack<char, short>>, pack<int, float, char, float, double, char, short>>, "pack should be concatenated");

  using p1 = pack<int, float, bool>;
  using p2 = pack<int, double, char>;
  using p3 = concatenated_pack_unique_t<p1, p2>;
  print_pack<p3>();
  static_assert(std::is_same_v<p3, pack<float, bool, int, double, char>>, "pack should not have duplicated types");

  static_assert(std::is_same_v<unique_pack_t<pack<int, float, int, float, char, char>>, pack<char, float, int>>, "pack should not have duplicated types");
  static_assert(std::is_same_v<interleaved_pack_t<pack<int, float, int>, pack<char, bool, char>>, pack<int, char, float, bool, int, char>>, "interleaved packs of the same size");
  static_assert(std::is_same_v<pack_to_tuple_t<pack<int, float, char>>, std::tuple<int, float, char>>, "pack should become a tuple");
  static_assert(std::is_same_v<repeated_type_pack_t<float, 5>, pack<float, float, float, float, float>>, "pack should have float repeated 5 times");

  struct ForwardDeclared;
  static_assert(is_type_complete_v<ForwardDeclared> == false, "This should not be complete because the struct is simply forward declared.");

  struct Declared {
  };
  static_assert(is_type_complete_v<Declared> == true, "This should be complete because the struct is fully declared.");

  // Notice this will not work because the static assert above and the one below
  // conflict and the first one wins. We can use is_type_complete_v only once per
  // compilation unit.
  //
  // struct ForwardDeclared { int a;};
  // static_assert(is_type_complete_v<ForwardDeclared> == true, "This should be complete because the struct is now fully declared.");

  bool flag = false;
  call_if_defined<struct Undeclared>([&flag](auto*) { flag = true; });
  BOOST_REQUIRE_EQUAL(flag, false);

  flag = false;
  call_if_defined<struct Declared>([&flag](auto*) { flag = true; });
  BOOST_REQUIRE_EQUAL(flag, true);
}
