// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUCommonArray.h
/// \author David Rohr

#ifndef GPUCOMMONARRAY_H
#define GPUCOMMONARRAY_H

#ifndef GPUCA_GPUCODE_DEVICE
#include <array>
#endif

#include "GPUCommonDef.h"
namespace o2::gpu::gpustd
{
#ifdef GPUCA_GPUCODE_DEVICE
template <typename T, size_t N>
struct array {
  GPUd() T& operator[](size_t i) { return m_internal_V__[i]; };
  GPUd() const T& operator[](size_t i) const { return m_internal_V__[i]; };
  GPUd() T* data() { return m_internal_V__; };
  GPUd() const T* data() const { return m_internal_V__; };
  T m_internal_V__[N];
};
template <class T, class... E>
GPUd() array(T, E...)->array<T, 1 + sizeof...(E)>;
#else
template <typename T, size_t N>
using array = std::array<T, N>;
#endif
} // namespace o2::gpu::gpustd

#endif
