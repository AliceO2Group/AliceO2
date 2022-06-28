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
/// \file Array.h
/// \brief
///

#ifndef ITSTRACKINGGPU_ARRAY_H_
#define ITSTRACKINGGPU_ARRAY_H_

#include "GPUCommonDef.h"
#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#endif

namespace o2
{
namespace its
{
namespace gpu
{

namespace
{
template <typename T, size_t Size>
struct ArrayTraits final {
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
struct Array final {

  void copy(const Array<T, Size>& t)
  {
    memcpy(InternalArray, t.data(), Size * sizeof(T));
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

#endif
