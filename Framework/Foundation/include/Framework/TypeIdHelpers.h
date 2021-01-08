// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FRAMEWORK_TYPEIDHELPERS_H_
#define O2_FRAMEWORK_TYPEIDHELPERS_H_

#include <string_view>
#include "Framework/StringHelpers.h"

#if defined(__GNUC__) && (__GNUC__ < 8)
#define PRETTY_FUNCTION_CONSTEXPR const
#else
#define PRETTY_FUNCTION_CONSTEXPR constexpr
#endif

namespace o2::framework
{

template <typename T>
struct unique_type_id {
  static constexpr auto get() noexcept
  {
    PRETTY_FUNCTION_CONSTEXPR std::string_view full_name{__PRETTY_FUNCTION__};
    return full_name;
  }

  static constexpr std::string_view value{get()};
};

template <typename T>
inline constexpr auto unique_type_id_v = unique_type_id<T>::value;

struct TypeIdHelpers {
  /// Return a unique id for a given type
  /// This works just fine with GCC and CLANG,
  /// C++20 will allow us to use:
  ///    std::source_location::current().function_name();
  template <typename T>
  constexpr static uint32_t uniqueId()
  {
    PRETTY_FUNCTION_CONSTEXPR uint32_t r = crc32(unique_type_id_v<T>.data(), unique_type_id_v<T>.size());
    return r;
  }
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_TYPEIDHELPERS_H_
