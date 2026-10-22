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
} // namespace qConfig

#endif
