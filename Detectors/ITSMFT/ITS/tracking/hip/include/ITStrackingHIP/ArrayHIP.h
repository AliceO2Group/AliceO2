// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
namespace GPU
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
} // namespace GPU
} // namespace its
} // namespace o2

#endif /* O2_ITS_TRACKING_INCLUDE_ARRAY_HIP_H_ */
