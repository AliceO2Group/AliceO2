// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <Framework/Traits.h>
#include <tuple>
#include <type_traits>

namespace o2::framework
{
struct any_type {
  template <class T>
  constexpr operator T();  // non explicit
};

template <class T, typename... Args>
decltype(void(T{ std::declval<Args>()... }), std::true_type())
  test(int);

template <class T, typename... Args>
std::false_type
  test(...);

template <class T, typename... Args>
struct is_braces_constructible : decltype(test<T, Args...>(0)) {
};

/// Helper function to convert a brace-initialisable struct to
/// a tuple.
template <class T>
auto constexpr to_tuple(T&& object) noexcept
{
  using type = std::decay_t<T>;
  if constexpr (is_braces_constructible<type, any_type, any_type, any_type, any_type>{}) {
    auto&& [p0, p1, p2, p3] = object;
    return std::make_tuple(p0, p1, p2, p3);
  } else if constexpr (is_braces_constructible<type, any_type, any_type, any_type>{}) {
    auto&& [p0, p1, p2] = object;
    return std::make_tuple(p0, p1, p2);
  } else if constexpr (is_braces_constructible<type, any_type, any_type>{}) {
    auto&& [p0, p1] = object;
    return std::make_tuple(p0, p1);
  } else if constexpr (is_braces_constructible<type, any_type>{}) {
    auto&& [p0] = object;
    return std::make_tuple(p0);
  } else {
    static_assert(always_static_assert<type>(), "Empty struct");
  }
}

/// Helper function to convert a brace-initialisable struct to
/// a tuple.
template <class T>
auto constexpr to_tuple_refs(T&& object) noexcept
{
  using type = std::decay_t<T>;
  if constexpr (is_braces_constructible<type, any_type, any_type, any_type, any_type>{}) {
    auto&& [p0, p1, p2, p3] = object;
    return std::tie(p0, p1, p2, p3);
  } else if constexpr (is_braces_constructible<type, any_type, any_type, any_type>{}) {
    auto&& [p0, p1, p2] = object;
    return std::tie(p0, p1, p2);
  } else if constexpr (is_braces_constructible<type, any_type, any_type>{}) {
    auto&& [p0, p1] = object;
    return std::tie(p0, p1);
  } else if constexpr (is_braces_constructible<type, any_type>{}) {
    auto&& [p0] = object;
    return std::make_tuple(p0);
  } else {
    static_assert(always_static_assert<type>(), "Empty struct");
  }
}

}  // namespace o2::framework
