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

/// \file GPUGetConstexpr.h
/// \author David Rohr

#ifndef GPUGETCONSTEXPR_H
#define GPUGETCONSTEXPR_H

#include "GPUCommonDef.h"
#ifndef GPUCA_GPUCODE_DEVICE
#include <type_traits>
#endif

// This is a temporary workaround required for clang (with c++20), until we can go to C++23 with P2280R4, which allows getting constexpr static values from references

#if (defined(__clang__) && __cplusplus >= 202002L && __cplusplus < 202302L) || defined(__CPUDEVICE__) // BUG: The workaround is needed for GCC if used inside a templated function.

namespace o2::gpu::internal
{

#define GPUCA_GET_CONSTEXPR(obj, val) ( \
  std::is_member_pointer_v<decltype(&std::remove_reference_t<decltype(obj)>::val)> ? o2::gpu::internal::getConstexpr(&std::remove_reference_t<decltype(obj)>::val, o2::gpu::internal::getConstexprHelper<decltype(&std::remove_reference_t<decltype(obj)>::val), decltype(&obj)>(&obj).value) : o2::gpu::internal::getConstexpr(&std::remove_reference_t<decltype(obj)>::val, o2::gpu::internal::getConstexprHelper<decltype(&std::remove_reference_t<decltype(obj)>::val), decltype(&obj)>().value))

template <class T, class S>
struct getConstexprHelper;

template <class T, class S>
  requires(!std::is_member_pointer_v<T>)
struct getConstexprHelper<T, S> {
  GPUdi() constexpr getConstexprHelper(const void* = nullptr) {}
  static constexpr const void* value = nullptr;
};

template <class T, class S>
  requires(std::is_member_pointer_v<T>)
struct getConstexprHelper<T, S> {
  GPUdi() constexpr getConstexprHelper(const S& v) : value(v) {}
  GPUdDefault() constexpr getConstexprHelper() = default;
  const S value = nullptr;
};

GPUdi() constexpr auto getConstexpr(const auto* v, const void* = nullptr)
{
  return *v;
}

GPUdi() constexpr auto getConstexpr(const auto v, const auto w)
{
  return w->*v;
}

} // namespace o2::gpu::internal

#else // __clang__

#define GPUCA_GET_CONSTEXPR(obj, val) (obj).val

#endif

#endif // GPUGETCONSTEXPR_H
