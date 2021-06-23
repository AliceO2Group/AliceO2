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
///
/// \file ArrayHIP.h
/// \brief
///

#ifndef O2_ITS_TRACKING_INCLUDE_ARRAY_HIP_H_
#define O2_ITS_TRACKING_INCLUDE_ARRAY_HIP_H_

#include <hip/hip_runtime.h>

#include "GPUCommonDef.h"

namespace o2
{
namespace its
{
namespace gpu
{

namespace
{
template <typename T, size_t Size>
struct ArrayTraitsHIP final {
  typedef T InternalArray[Size];

  GPUhd() static constexpr T& getReference(const InternalArray& internalArray, size_t index) noexcept
  {
    return const_cast<T&>(internalArray[index]);
  }

  GPUhd() static constexpr T* getPointer(const InternalArray& internalArray) noexcept
  {
    return const_cast<T*>(internalArray);
  }
};
} // namespace

template <typename T, size_t Size>
struct ArrayHIP final {

  void copy(const ArrayHIP<T, Size>& t)
  {
#ifdef __OPENCL__
    for (size_t i{0}; i < Size; ++i) {
      InternalArray[i] = t[i];
    }
#else
    memcpy(InternalArray, t.data(), Size * sizeof(T));
#endif
  }

  GPUhd() T* data() noexcept { return const_cast<T*>(InternalArray); }
  GPUhd() const T* data() const noexcept { return const_cast<T*>(InternalArray); }
  GPUhd() T& operator[](const int index) noexcept { return const_cast<T&>(InternalArray[index]); }
  GPUhd() constexpr T& operator[](const int index) const noexcept { return const_cast<T&>(InternalArray[index]); }
  GPUhd() size_t size() const noexcept { return Size; }

  T InternalArray[Size];
};
} // namespace gpu
} // namespace its
} // namespace o2

#endif /* O2_ITS_TRACKING_INCLUDE_ARRAY_HIP_H_ */
