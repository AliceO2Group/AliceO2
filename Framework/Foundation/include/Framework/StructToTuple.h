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
#ifndef O2_FRAMEWORK_STRUCTTOTUPLE_H_
#define O2_FRAMEWORK_STRUCTTOTUPLE_H_

#include <Framework/Traits.h>
#include <array>

// Structured binding packs (P1061) are C++26, but clang implements them as an
// extension in every language mode and advertises them via the feature test
// macro, so we can use them while still compiling as C++20.
#if defined(__cpp_structured_bindings) && __cpp_structured_bindings >= 202411L
#define DPL_STRUCTURED_BINDING_PACKS 1
#endif

#ifndef DPL_STRUCTURED_BINDING_PACKS
namespace
{
template <class T, typename... Args>
decltype(void(T{std::declval<Args>()...}), std::true_type())
  brace_test(int);

template <class T, typename... Args>
std::false_type
  brace_test(...);
} // namespace
#endif

namespace o2::framework
{
#ifndef DPL_STRUCTURED_BINDING_PACKS
struct any_type {
  template <class T>
  constexpr operator T(); // non explicit
};

template <class T, typename... Args>
struct is_braces_constructible : decltype(brace_test<T, Args...>(0)) {
};
#endif

struct UniversalType {
  template <typename T>
  operator T()
  {
  }
};

template <typename T>
consteval auto brace_constructible_size(auto... Members)
{
  if constexpr (requires { T{Members...}; } == false) {
    static_assert(sizeof...(Members) != 0, "You need to make sure that you have implicit constructors or that you call the explicit constructor correctly.");
    return sizeof...(Members) - 1;
  } else {
    return brace_constructible_size<T>(Members..., UniversalType{});
  }
}

template <bool B, typename T>
consteval int nested_brace_constructible_size()
{
  using type = std::decay_t<T>;
  constexpr int nesting = B ? 1 : 0;
  return brace_constructible_size<type>() - nesting;
}

/// The size to be passed to homogeneous_apply_refs_sized for T. Structured
/// binding packs do not need to know how many members T has, so we can skip
/// counting them altogether.
template <bool B, typename T>
consteval int homogeneous_apply_refs_size()
{
#ifdef DPL_STRUCTURED_BINDING_PACKS
  return 0;
#else
  return nested_brace_constructible_size<B, T>() / 10;
#endif
}

#ifdef DPL_STRUCTURED_BINDING_PACKS
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++26-extensions"
#endif
template <bool B = false, class T, int D = 0, typename L>
constexpr auto homogeneous_apply_refs(L l, T&& object)
{
  auto&& [... members] = object;
  if constexpr (sizeof...(members) == 0) {
    return std::array<bool, 0>();
  } else {
    return std::array{l(members)...};
  }
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif

#else // DPL_STRUCTURED_BINDING_PACKS

#define DPL_ENUM_0(pre, post)
#define DPL_ENUM_1(pre, post) pre##0##post
#define DPL_ENUM_2(pre, post) pre##0##post, pre##1##post
#define DPL_ENUM_3(pre, post) pre##0##post, pre##1##post, pre##2##post
#define DPL_ENUM_4(pre, post) pre##0##post, pre##1##post, pre##2##post, pre##3##post
#define DPL_ENUM_5(pre, post) pre##0##post, pre##1##post, pre##2##post, pre##3##post, pre##4##post
#define DPL_ENUM_6(pre, post) pre##0##post, pre##1##post, pre##2##post, pre##3##post, pre##4##post, pre##5##post
#define DPL_ENUM_7(pre, post) pre##0##post, pre##1##post, pre##2##post, pre##3##post, pre##4##post, pre##5##post, pre##6##post
#define DPL_ENUM_8(pre, post) pre##0##post, pre##1##post, pre##2##post, pre##3##post, pre##4##post, pre##5##post, pre##6##post, pre##7##post
#define DPL_ENUM_9(pre, post) pre##0##post, pre##1##post, pre##2##post, pre##3##post, pre##4##post, pre##5##post, pre##6##post, pre##7##post, pre##8##post
#define DPL_ENUM_10(pre, post) pre##0##post, pre##1##post, pre##2##post, pre##3##post, pre##4##post, pre##5##post, pre##6##post, pre##7##post, pre##8##post, pre##9##post

#define DPL_ENUM_20(pre, post) DPL_ENUM_10(pre, post), DPL_ENUM_10(pre##1, post)
#define DPL_ENUM_30(pre, post) DPL_ENUM_20(pre, post), DPL_ENUM_10(pre##2, post)
#define DPL_ENUM_40(pre, post) DPL_ENUM_30(pre, post), DPL_ENUM_10(pre##3, post)
#define DPL_ENUM_50(pre, post) DPL_ENUM_40(pre, post), DPL_ENUM_10(pre##4, post)
#define DPL_ENUM_60(pre, post) DPL_ENUM_50(pre, post), DPL_ENUM_10(pre##5, post)
#define DPL_ENUM_70(pre, post) DPL_ENUM_60(pre, post), DPL_ENUM_10(pre##6, post)
#define DPL_ENUM_80(pre, post) DPL_ENUM_70(pre, post), DPL_ENUM_10(pre##7, post)
#define DPL_ENUM_90(pre, post) DPL_ENUM_80(pre, post), DPL_ENUM_10(pre##8, post)
#define DPL_ENUM_100(pre, post) DPL_ENUM_90(pre, post), DPL_ENUM_10(pre##9, post)

#define DPL_ENUM(pre, post, d, u) DPL_ENUM_##d##0(pre, post), DPL_ENUM_##u(pre##d, post)

#define DPL_FENUM_0(f, pre, post)
#define DPL_FENUM_1(f, pre, post) f(pre##0##post)
#define DPL_FENUM_2(f, pre, post) f(pre##0##post), f(pre##1##post)
#define DPL_FENUM_3(f, pre, post) f(pre##0##post), f(pre##1##post), f(pre##2##post)
#define DPL_FENUM_4(f, pre, post) f(pre##0##post), f(pre##1##post), f(pre##2##post), f(pre##3##post)
#define DPL_FENUM_5(f, pre, post) f(pre##0##post), f(pre##1##post), f(pre##2##post), f(pre##3##post), f(pre##4##post)
#define DPL_FENUM_6(f, pre, post) f(pre##0##post), f(pre##1##post), f(pre##2##post), f(pre##3##post), f(pre##4##post), f(pre##5##post)
#define DPL_FENUM_7(f, pre, post) f(pre##0##post), f(pre##1##post), f(pre##2##post), f(pre##3##post), f(pre##4##post), f(pre##5##post), f(pre##6##post)
#define DPL_FENUM_8(f, pre, post) f(pre##0##post), f(pre##1##post), f(pre##2##post), f(pre##3##post), f(pre##4##post), f(pre##5##post), f(pre##6##post), f(pre##7##post)
#define DPL_FENUM_9(f, pre, post) f(pre##0##post), f(pre##1##post), f(pre##2##post), f(pre##3##post), f(pre##4##post), f(pre##5##post), f(pre##6##post), f(pre##7##post), f(pre##8##post)
#define DPL_FENUM_10(f, pre, post) f(pre##0##post), f(pre##1##post), f(pre##2##post), f(pre##3##post), f(pre##4##post), f(pre##5##post), f(pre##6##post), f(pre##7##post), f(pre##8##post), f(pre##9##post)

#define DPL_FENUM_20(f, pre, post) DPL_FENUM_10(f, pre, post), DPL_FENUM_10(f, pre##1, post)
#define DPL_FENUM_30(f, pre, post) DPL_FENUM_20(f, pre, post), DPL_FENUM_10(f, pre##2, post)
#define DPL_FENUM_40(f, pre, post) DPL_FENUM_30(f, pre, post), DPL_FENUM_10(f, pre##3, post)
#define DPL_FENUM_50(f, pre, post) DPL_FENUM_40(f, pre, post), DPL_FENUM_10(f, pre##4, post)
#define DPL_FENUM_60(f, pre, post) DPL_FENUM_50(f, pre, post), DPL_FENUM_10(f, pre##5, post)
#define DPL_FENUM_70(f, pre, post) DPL_FENUM_60(f, pre, post), DPL_FENUM_10(f, pre##6, post)
#define DPL_FENUM_80(f, pre, post) DPL_FENUM_70(f, pre, post), DPL_FENUM_10(f, pre##7, post)
#define DPL_FENUM_90(f, pre, post) DPL_FENUM_80(f, pre, post), DPL_FENUM_10(f, pre##8, post)
#define DPL_FENUM_100(f, pre, post) DPL_FENUM_90(f, pre, post), DPL_FENUM_10(f, pre##9, post)

#define DPL_FENUM(f, pre, post, d, u) DPL_FENUM_##d##0(f, pre, post), DPL_FENUM_##u(f, pre##d, post)

#define DPL_HOMOGENEOUS_APPLY_ENTRY_LOW(u)                        \
  constexpr(numElements == u)                                     \
  {                                                               \
    auto&& [DPL_ENUM_##u(p, )] = object;                          \
    return std::array<decltype(l(p0)), u>{DPL_FENUM_##u(l, p, )}; \
  }

#define DPL_HOMOGENEOUS_APPLY_ENTRY(d, u)                              \
  constexpr(numElements == d##u)                                       \
  {                                                                    \
    auto&& [DPL_ENUM(p, , d, u)] = object;                             \
    return std::array<decltype(l(p0)), d##u>{DPL_FENUM(l, p, , d, u)}; \
  }

#define DPL_HOMOGENEOUS_APPLY_ENTRY_TENS(d)                             \
  constexpr(numElements == d##0)                                        \
  {                                                                     \
    auto&& [DPL_ENUM_##d##0(p, )] = object;                             \
    return std::array<decltype(l(p0)), d##0>{DPL_FENUM_##d##0(l, p, )}; \
  }

template <bool B = false, class T, int D = nested_brace_constructible_size<B, T>() / 10, typename L>
  requires(D == 9)
constexpr auto homogeneous_apply_refs(L l, T&& object)
{
  constexpr int numElements = nested_brace_constructible_size<B, T>();
  // clang-format off
  if DPL_HOMOGENEOUS_APPLY_ENTRY (9, 9)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (9, 8)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (9, 7)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (9, 6)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (9, 5)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (9, 4)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (9, 3)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (9, 2)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (9, 1)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY_TENS (9)
  else { return std::array<bool,0>(); }
  // clang-format on
}

template <bool B = false, class T, int D = nested_brace_constructible_size<B, T>() / 10, typename L>
  requires(D == 8)
constexpr auto homogeneous_apply_refs(L l, T&& object)
{
  constexpr int numElements = nested_brace_constructible_size<B, T>();
  // clang-format off
  if DPL_HOMOGENEOUS_APPLY_ENTRY (8, 9)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (8, 8)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (8, 7)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (8, 6)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (8, 5)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (8, 4)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (8, 3)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (8, 2)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (8, 1)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY_TENS (8)
  else { return std::array<bool,0>(); }
  // clang-format on
}

template <bool B = false, class T, int D = nested_brace_constructible_size<B, T>() / 10, typename L>
  requires(D == 7)
constexpr auto homogeneous_apply_refs(L l, T&& object)
{
  constexpr int numElements = nested_brace_constructible_size<B, T>();
  // clang-format off
  if DPL_HOMOGENEOUS_APPLY_ENTRY (7, 9)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (7, 8)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (7, 7)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (7, 6)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (7, 5)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (7, 4)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (7, 3)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (7, 2)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (7, 1)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY_TENS (7)
  else { return std::array<bool,0>(); }
  // clang-format on
}

template <bool B = false, class T, int D = nested_brace_constructible_size<B, T>() / 10, typename L>
  requires(D == 6)
constexpr auto homogeneous_apply_refs(L l, T&& object)
{
  constexpr int numElements = nested_brace_constructible_size<B, T>();
  // clang-format off
  if DPL_HOMOGENEOUS_APPLY_ENTRY (6, 9)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (6, 8)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (6, 7)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (6, 6)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (6, 5)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (6, 4)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (6, 3)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (6, 2)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (6, 1)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY_TENS (6)
  else { return std::array<bool,0>(); }
  // clang-format on
}

template <bool B = false, class T, int D = nested_brace_constructible_size<B, T>() / 10, typename L>
  requires(D == 5)
constexpr auto homogeneous_apply_refs(L l, T&& object)
{
  constexpr int numElements = nested_brace_constructible_size<B, T>();
  // clang-format off
  if DPL_HOMOGENEOUS_APPLY_ENTRY (5, 9)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (5, 8)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (5, 7)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (5, 6)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (5, 5)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (5, 4)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (5, 3)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (5, 2)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (5, 1)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY_TENS (5)
  else { return std::array<bool,0>(); }
  // clang-format on
}

template <bool B = false, class T, int D = nested_brace_constructible_size<B, T>() / 10, typename L>
  requires(D == 4)
constexpr auto homogeneous_apply_refs(L l, T&& object)
{
  constexpr int numElements = nested_brace_constructible_size<B, T>();
  // clang-format off
  if DPL_HOMOGENEOUS_APPLY_ENTRY (4, 9)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (4, 8)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (4, 7)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (4, 6)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (4, 5)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (4, 4)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (4, 3)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (4, 2)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (4, 1)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY_TENS (4)
  else { return std::array<bool,0>(); }
  // clang-format on
}

template <bool B = false, class T, int D = nested_brace_constructible_size<B, T>() / 10, typename L>
  requires(D == 3)
constexpr auto homogeneous_apply_refs(L l, T&& object)
{
  constexpr int numElements = nested_brace_constructible_size<B, T>();
  // clang-format off
  if DPL_HOMOGENEOUS_APPLY_ENTRY (3, 9)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (3, 8)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (3, 7)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (3, 6)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (3, 5)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (3, 4)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (3, 3)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (3, 2)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (3, 1)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY_TENS (3)
  else { return std::array<bool,0>(); }
  // clang-format on
}

template <bool B = false, class T, int D = nested_brace_constructible_size<B, T>() / 10, typename L>
  requires(D == 2)
constexpr auto homogeneous_apply_refs(L l, T&& object)
{
  constexpr int numElements = nested_brace_constructible_size<B, T>();
  // clang-format off
  if DPL_HOMOGENEOUS_APPLY_ENTRY (2, 9)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (2, 8)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (2, 7)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (2, 6)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (2, 5)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (2, 4)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (2, 3)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (2, 2)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (2, 1)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY_TENS (2)
  else { return std::array<bool,0>(); }
  // clang-format on
}

template <bool B = false, class T, int D = nested_brace_constructible_size<B, T>() / 10, typename L>
  requires(D == 1)
constexpr auto homogeneous_apply_refs(L l, T&& object)
{
  constexpr int numElements = nested_brace_constructible_size<B, T>();
  // clang-format off
  if DPL_HOMOGENEOUS_APPLY_ENTRY (1, 9)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (1, 8)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (1, 7)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (1, 6)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (1, 5)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (1, 4)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (1, 3)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (1, 2)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY (1, 1)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY_TENS (1)
  else { return std::array<bool,0>(); }
  // clang-format on
}

template <bool B = false, class T, int D = nested_brace_constructible_size<B, T>() / 10, typename L>
  requires(D == 0)
constexpr auto homogeneous_apply_refs(L l, T&& object)
{
  constexpr int numElements = nested_brace_constructible_size<B, T>();
  // clang-format off
  if DPL_HOMOGENEOUS_APPLY_ENTRY_LOW (9)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY_LOW (8)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY_LOW (7)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY_LOW (6)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY_LOW (5)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY_LOW (4)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY_LOW (3)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY_LOW (2)
  else if DPL_HOMOGENEOUS_APPLY_ENTRY_LOW (1)
  else { return std::array<bool,0>(); }
  // clang-format on
}

#endif // DPL_STRUCTURED_BINDING_PACKS

template <int D, typename T, typename L>
constexpr auto homogeneous_apply_refs_sized(L l, T&& object)
{
  return homogeneous_apply_refs<false, T, D, L>(l, object);
}

} // namespace o2::framework

#endif // O2_FRAMEWORK_STRUCTTOTUPLE_H_
