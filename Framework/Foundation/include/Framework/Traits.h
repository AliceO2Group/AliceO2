// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FRAMEWORK_TRAITS_H_
#define O2_FRAMEWORK_TRAITS_H_

#include <type_traits>

namespace o2
{
namespace framework
{

template <typename A, typename B>
struct is_overriding : public std::bool_constant<std::is_same_v<A, B> == false && std::is_member_function_pointer_v<A> && std::is_member_function_pointer_v<B>> {
};

template <typename... T>
struct always_static_assert : std::false_type {
};

template <typename... T>
inline constexpr bool always_static_assert_v = always_static_assert<T...>::value;

} // namespace framework
} // namespace o2

#endif // O2_FRAMEWORK_TRAITS_H_
