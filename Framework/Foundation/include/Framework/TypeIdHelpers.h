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

#ifndef O2_FRAMEWORK_TYPEIDHELPERS_H_
#define O2_FRAMEWORK_TYPEIDHELPERS_H_

#include <string_view>
#include <sstream>
#if __cplusplus >= 202002L
#include <source_location>
#endif
#include "Framework/StringHelpers.h"

namespace o2::framework
{
#if defined(__CLING__) || __cplusplus < 202002L
template <typename T>
struct unique_type_id {
  static constexpr auto get() noexcept
  {
    constexpr std::string_view full_name{__PRETTY_FUNCTION__};
    return full_name;
  }

  static constexpr std::string_view value{get()};
};

template <typename T>
inline constexpr auto unique_type_id_v = unique_type_id<T>::value;
#endif

#if !defined(__CLING__) && __cplusplus >= 202002L
/// Workaround GCC optimizing out unused template parameter
template <typename T>
consteval static std::string_view type_name_impl(T*)
{
  return std::source_location::current().function_name();
}
#endif

/// Return pure type name with no namespaces etc.
/// Works with GCC and CLANG
template <typename T>
constexpr static std::string_view type_name()
{
#if defined(__CLING__) || __cplusplus < 202002L
  constexpr std::string_view wrapped_name{unique_type_id_v<T>};
#else
  constexpr std::string_view wrapped_name = type_name_impl<T>(nullptr);
#endif
  const std::string_view left_marker{"T = "};
#if !defined(__clang__) && __cplusplus >= 202002L
  const std::string_view right_marker{";"};
#else
  const std::string_view right_marker{"]"};
#endif

  const auto left_marker_index = wrapped_name.find(left_marker);
  const auto start_index = left_marker_index + left_marker.size();
  const auto end_index = wrapped_name.find(right_marker, left_marker_index);
  const auto length = end_index - start_index;
  return wrapped_name.substr(start_index, length);
}

struct TypeIdHelpers {
  template <typename T>
  constexpr static uint32_t uniqueId()
  {
#ifdef __CLING__
    constexpr uint32_t r = crc32(unique_type_id_v<T>.data(), unique_type_id_v<T>.size());
    return r;
#else
    return compile_time_hash(type_name<T>().data());
#endif
  }
};

/// Convert a CamelCase task struct name to snake-case task name
inline static std::string type_to_task_name(std::string_view& camelCase)
{
  std::ostringstream str;
  str << static_cast<char>(std::tolower(camelCase[0]));

  for (auto it = camelCase.begin() + 1; it != camelCase.end(); ++it) {
    if (std::isupper(*it) && *(it - 1) != '-') {
      str << "-";
    }
    str << static_cast<char>(std::tolower(*it));
  }

  return str.str();
}

} // namespace o2::framework

#endif // O2_FRAMEWORK_TYPEIDHELPERS_H_
