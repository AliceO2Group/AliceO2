// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUCommonTypeTraits.h
/// \author David Rohr

#ifndef GPUCOMMONTYPETRAITS_H
#define GPUCOMMONTYPETRAITS_H

#if !defined(GPUCA_GPUCODE_DEVICE) || defined(__CUDACC__) || defined(__HIPCC__)
#include <type_traits>
#elif !defined(__OPENCL__) || defined(__OPENCLCPP__)
// We just reimplement some type traits in std for the GPU
namespace std
{
template <bool B, class T, class F>
struct conditional {
  typedef T type;
};
template <class T, class F>
struct conditional<false, T, F> {
  typedef F type;
};
template <bool B, class T, class F>
using contitional_t = typename conditional<B, T, F>::type;
template <class T, class U>
struct is_same {
  static constexpr bool value = false;
};
template <class T>
struct is_same<T, T> {
  static constexpr bool value = true;
};
template <class T, class U>
static constexpr bool is_same_v = is_same<T, U>::value;
} // namespace std
#endif

#endif
