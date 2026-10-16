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

/// \file qconfig_internal_helpers.h
/// \author David Rohr

#ifndef QCONFIG_INTERNAL_HELPERS_H
#define QCONFIG_INTERNAL_HELPERS_H

#include "qconfig_helpers.h"
#include <tuple>
#include <type_traits>

namespace qConfig
{
#define QCONFIG_SETTING(name, type)                     \
  struct qon_mxcat3(q, name, _t) {                      \
    type v;                                             \
    constexpr qon_mxcat3(q, name, _t)(type s) : v(s) {} \
  };                                                    \
  constexpr qon_mxcat3(q, name, _t) name(type v) { return (qon_mxcat3(q, name, _t)(v)); }

#define QCONFIG_SETTING_TEMPLATE(name)                      \
  template <typename T>                                     \
  struct qon_mxcat3(q, name, _t) {                          \
    T v;                                                    \
    constexpr qon_mxcat3(q, name, _t)(const T& s) : v(s) {} \
  };                                                        \
  template <typename T>                                     \
  constexpr qon_mxcat3(q, name, _t)<T> name(T v)            \
  {                                                         \
    return (qon_mxcat3(q, name, _t) < T > (v));             \
  }

QCONFIG_SETTING(message, const char*)
QCONFIG_SETTING_TEMPLATE(min)
QCONFIG_SETTING_TEMPLATE(max)
QCONFIG_SETTING_TEMPLATE(set)
QCONFIG_SETTING_TEMPLATE(def)

template <typename T>
struct qConfigSettings {
  qConfigSettings() : checkMin(false), checkMax(false), doSet(false), doDefault(false), min(), max(), set(), message(nullptr), allowEmpty(false) {};
  template <typename S>
  qConfigSettings(const qConfigSettings<S> v) : checkMin(false), checkMax(false), doSet(false), doDefault(false), min(), max(), set(), message(v.message), allowEmpty(v.allowEmpty) {};
  bool checkMin, checkMax;
  bool doSet, doDefault;
  T min, max;
  T set;
  const char* message;
  bool allowEmpty;
};

template <typename T>
struct qSettingsType {
  typedef T settingsType;
};
template <typename... X>
struct qSettingsType<std::tuple<X...>> {
  typedef int32_t settingsType;
};

template <typename T>
struct qConfigSettingsParser { // Parse the additional settings (min, max, set, def, message) of an option
  // Recursive handling of additional settings
  static inline void qProcessSetting(qConfigSettings<T>& settings, qmin_t<T> minval)
  {
    static_assert(!std::is_same_v<T, bool>, "min option not supported for boolean settings");
    settings.checkMin = true;
    settings.min = minval.v;
  }
  static inline void qProcessSetting(qConfigSettings<T>& settings, qmax_t<T> maxval)
  {
    static_assert(!std::is_same_v<T, bool>, "max option not supported for boolean settings");
    settings.checkMax = true;
    settings.max = maxval.v;
  }
  static inline void qProcessSetting(qConfigSettings<T>& settings, qmessage_t msg) { settings.message = msg.v; }
  template <typename S>
  static inline void qProcessSetting(qConfigSettings<T>& settings, qset_t<S> set)
  {
    settings.doSet = true;
    settings.set = set.v;
  }
  template <typename S>
  static inline void qProcessSetting(qConfigSettings<T>& settings, qdef_t<S> set)
  {
    settings.doDefault = true;
    settings.set = set.v;
  }

  static inline void qAddOptionSettings(qConfigSettings<T>& /*settings*/) {}
  template <typename Arg1, typename... Args>
  static inline void qAddOptionSettings(qConfigSettings<T>& settings, Arg1&& arg1, Args&&... args)
  {
    qProcessSetting(settings, arg1);
    qAddOptionSettings(settings, args...);
  }

  template <typename... Args>
  static inline auto qConfigGetSettings(Args&... args)
  {
    qConfigSettings<typename qSettingsType<T>::settingsType> settings;
    qConfigSettingsParser<typename qSettingsType<T>::settingsType>::qAddOptionSettings(settings, args...);
    return (settings);
  }
};
} // namespace qConfig

#endif
