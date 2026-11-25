// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file qconfig_check.inc
/// \author David Rohr

// Provides static inline qConfigCheck(objs...) in the namespace it is included in: evaluates the qcheck(...) settings of all options of the passed config objects, returns the number of failed checks.
// Requires <algorithm>, <cstdio>, <iterator>, <string>, <tuple>, <type_traits>, "qconfig_helpers.h", "qconfig_internal_helpers.h" to be included before (outside of the namespace), and the config structs to be visible unqualified.
// Options with the default value are not checked. Each type may be passed only once, sub configs are not included automatically and must be passed explicitly. Objects of other types can be passed to be used in the checks via get<T>().

#ifndef QCONFIG_CHECK_INC
#define QCONFIG_CHECK_INC

template <typename T>
static inline bool qConfigCheckIsDefault(const T& val, const T& def)
{
  if constexpr (std::is_array_v<T>) {
    return std::equal(std::begin(val), std::end(val), std::begin(def));
  } else {
    return val == def;
  }
}

template <typename T, typename... Args>
static inline int32_t qConfigCheckOption([[maybe_unused]] const char* configStruct, [[maybe_unused]] const char* name, [[maybe_unused]] const T& val, [[maybe_unused]] const T& def, const char* /*help*/, const Args&... args) // Evaluate the qcheck_result settings of one option, if its value differs from the default
{
  int32_t retVal = 0;
  if constexpr (sizeof...(Args) > 0) {
    if (qConfigCheckIsDefault(val, def)) {
      return 0;
    }
  }
  (
    [&](const auto& arg) {
      if constexpr (std::is_same_v<std::decay_t<decltype(arg)>, ::qConfig::qcheck_result>) {
        if (!arg.ok) {
          std::string value = "";
          if constexpr (std::is_arithmetic_v<T> || std::is_enum_v<T> || std::is_same_v<T, std::string>) {
            value = " = " + ::qConfig::print_type(val);
          }
          printf("Invalid setting %s.%s%s, check failed: %s\n", configStruct, name, value.c_str(), arg.expression);
          retVal++;
        }
      }
    }(args),
    ...);
  return retVal;
}

template <typename... Args>
class qConfigChecker
{
 public:
  qConfigChecker(const Args&... objs) : mObjs{objs...} {}
  int32_t check() const
  {
    using namespace ::qConfig; // For the settings helpers (min, max, ...) used in the settings lists
    int32_t retVal = 0;
#define QCONFIG_CHECK
#include "qconfig_internal.h"
#undef QCONFIG_CHECK
    return retVal;
  }

 private:
  template <typename T>
  const T& get() const
  {
    static_assert((std::is_same_v<T, Args> || ...), "qcheck uses get<T>() with a type not passed to qConfigCheck");
    return std::get<const T&>(mObjs);
  }
  std::tuple<const Args&...> mObjs;
};

template <typename T, typename... Args>
constexpr int32_t qConfigCheckTypeCount = (std::is_same_v<T, Args> + ... + 0);

template <typename... Args>
static inline int32_t qConfigCheck(const Args&... objs)
{
  static_assert(((qConfigCheckTypeCount<Args, Args...> == 1) && ...), "qConfigCheck: each type may be passed only once");
  return qConfigChecker<Args...>(objs...).check();
}

#endif // QCONFIG_CHECK_INC
