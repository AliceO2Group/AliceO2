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

/// \file qconfig.h
/// \author David Rohr

#include "qconfig_general.h"

#if !defined(QCONFIG_HEADER_GUARD_NO_INCLUDE) || defined(QCONFIG_GENRTC)

#if defined(QCONFIG_HEADER_GUARD) && !defined(QCONFIG_GENRTC)
#define QCONFIG_HEADER_GUARD_NO_INCLUDE
#else
#define QCONFIG_HEADER_GUARD

#ifndef BeginNamespace
#define BeginNamespace(name) \
  namespace name             \
  {
#define EndNamespace() }
#endif
#if !defined(QCONFIG_GPU)
#define AddOption(name, type, default, optname, optnameshort, help, ...) type name = default;
#define AddVariable(name, type, default) type name = default;
#define _AddOptionArray_INTERNAL_EXPAND(...) __VA_ARGS__
#define AddOptionArray(name, type, count, default, optname, optnameshort, help, ...) type name[count] = {_AddOptionArray_INTERNAL_EXPAND default};
#define AddOptionVec(name, type, optname, optnameshort, help, ...) std::vector<type> name;
#else
#define AddOption(name, type, default, optname, optnameshort, help, ...) type name;
#define AddVariable(name, type, default) type name;
#define AddOptionArray(name, type, count, default, optname, optnameshort, help, ...) type name[count];
#define AddOptionVec(name, type, optname, optnameshort, help, ...) void* name[sizeof(std::vector<type>) / sizeof(void*)];
#endif
#ifdef QCONFIG_GENRTC
#define AddVariableRTC(name, type, default) \
  static constexpr type name = default;     \
  type _dummy_##name = default;
#define AddOptionRTC(name, type, default, optname, optnameshort, help, ...) AddVariableRTC(name, type, default)
#define _AddOptionArray_INTERNAL_EXPAND(...) __VA_ARGS__
#define AddOptionArrayRTC(name, type, count, default, optname, optnameshort, help, ...) \
  static constexpr type name[count] = {_AddOptionArray_INTERNAL_EXPAND default};        \
  type _dummy_##name[count] = {_AddOptionArray_INTERNAL_EXPAND default};
#else
#define AddCustomCPP(...) __VA_ARGS__
#endif
#define AddOptionSet(name, type, value, optname, optnameshort, help, ...)
#define AddSubConfig(name, instance) name instance;
#if !defined(QCONFIG_GENRTC) && !defined(GPUCA_GPUCODE_DEVICE)
#define BeginConfig(name, instance) \
  struct name {                     \
    bool operator==(const name&) const = default;
#else
#define BeginConfig(name, instance) struct name {
#endif
#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr, ...) BeginConfig(name, instance)
// clang-format off
#define EndConfig() \
  };
// clang-format on

#endif

#include "qconfig_wrapinclude.h"

#endif // QCONFIG_HEADER_GUARD_NO_INCLUDE
