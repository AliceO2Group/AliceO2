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

/// \file GPUCommonArray.h
/// \author David Rohr

#ifndef GPUCOMMONARRAY_H
#define GPUCOMMONARRAY_H

#if !defined(GPUCA_GPUCODE_DEVICE) || defined(__CUDACC__) || defined(__HIPCC__) // TODO: Get rid of GPUCommonArray once OpenCL supports <array>
#ifndef GPUCA_GPUCODE_COMPILEKERNELS
#include <array>
#endif
#else

#include "GPUCommonDef.h"
namespace std
{
#ifdef __METAL__
using ::array;
#elif defined(GPUCA_GPUCODE_DEVICE)
template <typename T, size_t N>
struct array {
  GPUd() T& operator[](size_t i) { return m_internal_V__[i]; };
  GPUd() const T& operator[](size_t i) const { return m_internal_V__[i]; };
  GPUd() T* data() { return m_internal_V__; };
  GPUd() const T* data() const { return m_internal_V__; };
  GPUd() void fill(const T& t)
  {
    for (size_t i{0}; i < N; ++i) {
      m_internal_V__[i] = t;
    }
  }
  T m_internal_V__[N];
};
template <class T, class... E>
GPUd() array(T, E...)->array<T, 1 + sizeof...(E)>;
#else
template <typename T, size_t N>
using array = std::array<T, N>;
#endif
} // namespace std
#endif

#endif // GPUCOMMONARRAY_H
