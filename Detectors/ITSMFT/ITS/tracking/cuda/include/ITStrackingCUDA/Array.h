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
/// \file Array.h
/// \brief
///

#ifndef TRAKINGITSU_INCLUDE_GPU_ARRAY_H_
#define TRAKINGITSU_INCLUDE_GPU_ARRAY_H_

#include "ITStracking/Definitions.h"

namespace o2
{
namespace ITS
{
namespace GPU
{

namespace
{
template <typename T, std::size_t Size>
struct ArrayTraits final {
  typedef T InternalArray[Size];

  GPU_HOST_DEVICE static constexpr T& getReference(const InternalArray& internalArray, std::size_t index) noexcept
  {
    return const_cast<T&>(internalArray[index]);
  }

  GPU_HOST_DEVICE static constexpr T* getPointer(const InternalArray& internalArray) noexcept
  {
    return const_cast<T*>(internalArray);
  }
};
}

template <typename T, std::size_t Size>
struct Array final {

  GPU_HOST_DEVICE T* data() noexcept { return const_cast<T*>(InternalArray); }
  GPU_HOST_DEVICE const T* data() const noexcept { return const_cast<T*>(InternalArray); }
  GPU_HOST_DEVICE T& operator[](const int index) noexcept { return const_cast<T&>(InternalArray[index]); }
  GPU_HOST_DEVICE constexpr T& operator[](const int index) const noexcept { return const_cast<T&>(InternalArray[index]); }
  GPU_HOST_DEVICE std::size_t size() const noexcept { return Size; }

  T InternalArray[Size];

};

}
}
}

#endif /* TRAKINGITSU_INCLUDE_GPU_CAGPUVECTOR_H_ */
