// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_STRUCTTOTUPLE_H_
#define O2_FRAMEWORK_STRUCTTOTUPLE_H_

#include <Framework/Traits.h>
#include <vector>

namespace o2::framework
{
struct any_type {
  template <class T>
  constexpr operator T();  // non explicit
};

template <class T, typename... Args>
decltype(void(T{std::declval<Args>()...}), std::true_type())
  test(int);

template <class T, typename... Args>
std::false_type
  test(...);

template <class T, typename... Args>
struct is_braces_constructible : decltype(test<T, Args...>(0)) {
};

#if __cplusplus >= 202002L
struct UniversalType {
  template <typename T>
  operator T()
  {
  }
};

template <typename T>
consteval auto brace_constructible_size(auto... Members)
{
  if constexpr (requires { T{Members...}; } == false)
    return sizeof...(Members) - 1;
  else
    return brace_constructible_size<T>(Members..., UniversalType{});
}
#else
template <typename T>
constexpr long brace_constructible_size()
{
  using A = any_type;
  using type = std::decay_t<T>;

  if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A>{}) {
    return 39;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A>{}) {
    return 38;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A>{}) {
    return 37;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A>{}) {
    return 36;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A>{}) {
    return 35;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A>{}) {
    return 34;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A>{}) {
    return 33;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A>{}) {
    return 32;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A>{}) {
    return 31;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A>{}) {
    return 30;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A>{}) {
    return 29;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A>{}) {
    return 28;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A>{}) {
    return 27;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A>{}) {
    return 26;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A>{}) {
    return 25;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A>{}) {
    return 24;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A>{}) {
    return 23;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A>{}) {
    return 22;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A>{}) {
    return 21;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A>{}) {
    return 20;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A>{}) {
    return 19;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A>{}) {
    return 18;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A>{}) {
    return 17;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A>{}) {
    return 16;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A>{}) {
    return 15;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A, A, A, A, A, A, A, A>{}) {
    return 14;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A, A, A, A, A, A, A>{}) {
    return 13;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A, A, A, A, A, A>{}) {
    return 12;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A, A, A, A, A>{}) {
    return 11;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A, A, A, A>{}) {
    return 10;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A, A, A>{}) {
    return 9;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A, A>{}) {
    return 8;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A, A>{}) {
    return 7;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A, A>{}) {
    return 6;
  } else if constexpr (is_braces_constructible<type, A, A, A, A, A>{}) {
    return 5;
  } else if constexpr (is_braces_constructible<type, A, A, A, A>{}) {
    return 4;
  } else if constexpr (is_braces_constructible<type, A, A, A>{}) {
    return 3;
  } else if constexpr (is_braces_constructible<type, A, A>{}) {
    return 2;
  } else if constexpr (is_braces_constructible<type, A>{}) {
    return 1;
  } else {
    return 0;
  }
}
#endif

template <typename L, class T>
auto constexpr homogeneous_apply_refs(L l, T&& object) noexcept
{
  using type = std::decay_t<T>;
  constexpr unsigned long numElements = brace_constructible_size<T>();
  if constexpr (numElements == 40) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30, p31, p32, p33, p34, p35, p36, p37, p38, p39] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7), l(p8), l(p9), l(p10), l(p11), l(p12), l(p13), l(p14), l(p15), l(p16), l(p17), l(p18), l(p19), l(p20), l(p21), l(p22), l(p23), l(p24), l(p25), l(p26), l(p27), l(p28), l(p29), l(p30), l(p31), l(p32), l(p33), l(p34), l(p35), l(p36), l(p37), l(p38), l(p39)};
  } else if constexpr (numElements == 39) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30, p31, p32, p33, p34, p35, p36, p37, p38] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7), l(p8), l(p9), l(p10), l(p11), l(p12), l(p13), l(p14), l(p15), l(p16), l(p17), l(p18), l(p19), l(p20), l(p21), l(p22), l(p23), l(p24), l(p25), l(p26), l(p27), l(p28), l(p29), l(p30), l(p31), l(p32), l(p33), l(p34), l(p35), l(p36), l(p37), l(p38)};
  } else if constexpr (numElements == 38) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30, p31, p32, p33, p34, p35, p36, p37] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7), l(p8), l(p9), l(p10), l(p11), l(p12), l(p13), l(p14), l(p15), l(p16), l(p17), l(p18), l(p19), l(p20), l(p21), l(p22), l(p23), l(p24), l(p25), l(p26), l(p27), l(p28), l(p29), l(p30), l(p31), l(p32), l(p33), l(p34), l(p35), l(p36), l(p37)};
  } else if constexpr (numElements == 37) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30, p31, p32, p33, p34, p35, p36] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7), l(p8), l(p9), l(p10), l(p11), l(p12), l(p13), l(p14), l(p15), l(p16), l(p17), l(p18), l(p19), l(p20), l(p21), l(p22), l(p23), l(p24), l(p25), l(p26), l(p27), l(p28), l(p29), l(p30), l(p31), l(p32), l(p33), l(p34), l(p35), l(p36)};
  } else if constexpr (numElements == 36) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30, p31, p32, p33, p34, p35] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7), l(p8), l(p9), l(p10), l(p11), l(p12), l(p13), l(p14), l(p15), l(p16), l(p17), l(p18), l(p19), l(p20), l(p21), l(p22), l(p23), l(p24), l(p25), l(p26), l(p27), l(p28), l(p29), l(p30), l(p31), l(p32), l(p33), l(p34), l(p35)};
  } else if constexpr (numElements == 35) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30, p31, p32, p33, p34] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7), l(p8), l(p9), l(p10), l(p11), l(p12), l(p13), l(p14), l(p15), l(p16), l(p17), l(p18), l(p19), l(p20), l(p21), l(p22), l(p23), l(p24), l(p25), l(p26), l(p27), l(p28), l(p29), l(p30), l(p31), l(p32), l(p33), l(p34)};
  } else if constexpr (numElements == 34) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30, p31, p32, p33] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7), l(p8), l(p9), l(p10), l(p11), l(p12), l(p13), l(p14), l(p15), l(p16), l(p17), l(p18), l(p19), l(p20), l(p21), l(p22), l(p23), l(p24), l(p25), l(p26), l(p27), l(p28), l(p29), l(p30), l(p31), l(p32), l(p33)};
  } else if constexpr (numElements == 33) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30, p31, p32] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7), l(p8), l(p9), l(p10), l(p11), l(p12), l(p13), l(p14), l(p15), l(p16), l(p17), l(p18), l(p19), l(p20), l(p21), l(p22), l(p23), l(p24), l(p25), l(p26), l(p27), l(p28), l(p29), l(p30), l(p31), l(p32)};
  } else if constexpr (numElements == 32) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30, p31] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7), l(p8), l(p9), l(p10), l(p11), l(p12), l(p13), l(p14), l(p15), l(p16), l(p17), l(p18), l(p19), l(p20), l(p21), l(p22), l(p23), l(p24), l(p25), l(p26), l(p27), l(p28), l(p29), l(p30), l(p31)};
  } else if constexpr (numElements == 31) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7), l(p8), l(p9), l(p10), l(p11), l(p12), l(p13), l(p14), l(p15), l(p16), l(p17), l(p18), l(p19), l(p20), l(p21), l(p22), l(p23), l(p24), l(p25), l(p26), l(p27), l(p28), l(p29), l(p30)};
  } else if constexpr (numElements == 30) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7), l(p8), l(p9), l(p10), l(p11), l(p12), l(p13), l(p14), l(p15), l(p16), l(p17), l(p18), l(p19), l(p20), l(p21), l(p22), l(p23), l(p24), l(p25), l(p26), l(p27), l(p28), l(p29)};
  } else if constexpr (numElements == 29) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7), l(p8), l(p9), l(p10), l(p11), l(p12), l(p13), l(p14), l(p15), l(p16), l(p17), l(p18), l(p19), l(p20), l(p21), l(p22), l(p23), l(p24), l(p25), l(p26), l(p27), l(p28)};
  } else if constexpr (numElements == 28) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7), l(p8), l(p9), l(p10), l(p11), l(p12), l(p13), l(p14), l(p15), l(p16), l(p17), l(p18), l(p19), l(p20), l(p21), l(p22), l(p23), l(p24), l(p25), l(p26), l(p27)};
  } else if constexpr (numElements == 27) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7), l(p8), l(p9), l(p10), l(p11), l(p12), l(p13), l(p14), l(p15), l(p16), l(p17), l(p18), l(p19), l(p20), l(p21), l(p22), l(p23), l(p24), l(p25), l(p26)};
  } else if constexpr (numElements == 26) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7), l(p8), l(p9), l(p10), l(p11), l(p12), l(p13), l(p14), l(p15), l(p16), l(p17), l(p18), l(p19), l(p20), l(p21), l(p22), l(p23), l(p24), l(p25)};
  } else if constexpr (numElements == 25) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7), l(p8), l(p9), l(p10), l(p11), l(p12), l(p13), l(p14), l(p15), l(p16), l(p17), l(p18), l(p19), l(p20), l(p21), l(p22), l(p23), l(p24)};
  } else if constexpr (numElements == 24) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7), l(p8), l(p9), l(p10), l(p11), l(p12), l(p13), l(p14), l(p15), l(p16), l(p17), l(p18), l(p19), l(p20), l(p21), l(p22), l(p23)};
  } else if constexpr (numElements == 23) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7), l(p8), l(p9), l(p10), l(p11), l(p12), l(p13), l(p14), l(p15), l(p16), l(p17), l(p18), l(p19), l(p20), l(p21), l(p22)};
  } else if constexpr (numElements == 22) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7), l(p8), l(p9), l(p10), l(p11), l(p12), l(p13), l(p14), l(p15), l(p16), l(p17), l(p18), l(p19), l(p20), l(p21)};
  } else if constexpr (numElements == 21) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7), l(p8), l(p9), l(p10), l(p11), l(p12), l(p13), l(p14), l(p15), l(p16), l(p17), l(p18), l(p19), l(p20)};
  } else if constexpr (numElements == 20) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7), l(p8), l(p9), l(p10), l(p11), l(p12), l(p13), l(p14), l(p15), l(p16), l(p17), l(p18), l(p19)};
  } else if constexpr (numElements == 19) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7), l(p8), l(p9), l(p10), l(p11), l(p12), l(p13), l(p14), l(p15), l(p16), l(p17), l(p18)};
  } else if constexpr (numElements == 18) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7), l(p8), l(p9), l(p10), l(p11), l(p12), l(p13), l(p14), l(p15), l(p16), l(p17)};
  } else if constexpr (numElements == 17) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7), l(p8), l(p9), l(p10), l(p11), l(p12), l(p13), l(p14), l(p15), l(p16)};
  } else if constexpr (numElements == 16) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7), l(p8), l(p9), l(p10), l(p11), l(p12), l(p13), l(p14), l(p15)};
  } else if constexpr (numElements == 15) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7), l(p8), l(p9), l(p10), l(p11), l(p12), l(p13), l(p14)};
  } else if constexpr (numElements == 14) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7), l(p8), l(p9), l(p10), l(p11), l(p12), l(p13)};
  } else if constexpr (numElements == 13) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7), l(p8), l(p9), l(p10), l(p11), l(p12)};
  } else if constexpr (numElements == 12) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7), l(p8), l(p9), l(p10), l(p11)};
  } else if constexpr (numElements == 11) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7), l(p8), l(p9), l(p10)};
  } else if constexpr (numElements == 10) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7), l(p8), l(p9)};
  } else if constexpr (numElements == 9) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7, p8] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7), l(p8)};
  } else if constexpr (numElements == 8) {
    auto&& [p0, p1, p2, p3, p4, p5, p6, p7] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6), l(p7)};
  } else if constexpr (numElements == 7) {
    auto&& [p0, p1, p2, p3, p4, p5, p6] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5), l(p6)};
  } else if constexpr (numElements == 6) {
    auto&& [p0, p1, p2, p3, p4, p5] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4), l(p5)};
  } else if constexpr (numElements == 5) {
    auto&& [p0, p1, p2, p3, p4] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3), l(p4)};
  } else if constexpr (numElements == 4) {
    auto&& [p0, p1, p2, p3] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2), l(p3)};
  } else if constexpr (numElements == 3) {
    auto&& [p0, p1, p2] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1), l(p2)};
  } else if constexpr (numElements == 2) {
    auto&& [p0, p1] = object;
    return std::vector<decltype(l(p0))>{l(p0), l(p1)};
  } else if constexpr (numElements == 1) {
    auto&& [p0] = object;
    return std::vector<decltype(l(p0))>{l(p0)};
  } else {
    return false;
  }
}

}  // namespace o2::framework

#endif  // O2_FRAMEWORK_STRUCTTOTUPLE_H_
