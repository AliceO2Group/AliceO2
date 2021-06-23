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
/// \file UniquePointer.h
/// \brief
///

#ifndef TRAKINGITSU_INCLUDE_GPU_CAGPUUNIQUE_POINTER_H_
#define TRAKINGITSU_INCLUDE_GPU_CAGPUUNIQUE_POINTER_H_

#include "ITStrackingCUDA/Utils.h"

namespace o2
{
namespace its
{
namespace gpu
{

namespace
{
template <typename T>
struct UniquePointerTraits final {
  typedef T* InternalPointer;

  GPU_HOST_DEVICE static constexpr T& getReference(const InternalPointer& internalPointer) noexcept
  {
    return const_cast<T&>(*internalPointer);
  }

  GPU_HOST_DEVICE static constexpr T* getPointer(const InternalPointer& internalPointer) noexcept
  {
    return const_cast<T*>(internalPointer);
  }
};
} // namespace

template <typename T>
class UniquePointer final
{
  typedef UniquePointerTraits<T> PointerTraits;

 public:
  UniquePointer();
  explicit UniquePointer(const T&);
  ~UniquePointer();

  UniquePointer(const UniquePointer&) = delete;
  UniquePointer& operator=(const UniquePointer&) = delete;

  UniquePointer(UniquePointer&&);
  UniquePointer& operator=(UniquePointer&&);

  GPU_HOST_DEVICE T* get() noexcept;
  GPU_HOST_DEVICE const T* get() const noexcept;
  GPU_HOST_DEVICE T& operator*() noexcept;
  GPU_HOST_DEVICE const T& operator*() const noexcept;

 protected:
  void destroy();

 private:
  typename PointerTraits::InternalPointer mDevicePointer;
};

template <typename T>
UniquePointer<T>::UniquePointer() : mDevicePointer{nullptr}
{
  // Nothing to do
}

template <typename T>
UniquePointer<T>::UniquePointer(const T& ref)
{
  try {

    utils::host::gpuMalloc(reinterpret_cast<void**>(&mDevicePointer), sizeof(T));
    utils::host::gpuMemcpyHostToDevice(mDevicePointer, &ref, sizeof(T));

  } catch (...) {

    destroy();

    throw;
  }
}

template <typename T>
UniquePointer<T>::~UniquePointer()
{
  destroy();
}

template <typename T>
UniquePointer<T>::UniquePointer(UniquePointer<T>&& other) : mDevicePointer{other.mDevicePointer}
{
  // Nothing to do
}

template <typename T>
UniquePointer<T>& UniquePointer<T>::operator=(UniquePointer<T>&& other)
{
  mDevicePointer = other.mDevicePointer;
  other.mDevicePointer = nullptr;

  return *this;
}

template <typename T>
void UniquePointer<T>::destroy()
{
  if (mDevicePointer != nullptr) {

    utils::host::gpuFree(mDevicePointer);
  }
}

template <typename T>
GPU_HOST_DEVICE T* UniquePointer<T>::get() noexcept
{
  return PointerTraits::getPointer(mDevicePointer);
}

template <typename T>
GPU_HOST_DEVICE const T* UniquePointer<T>::get() const noexcept
{
  return PointerTraits::getPointer(mDevicePointer);
}

template <typename T>
GPU_HOST_DEVICE T& UniquePointer<T>::operator*() noexcept
{
  return PointerTraits::getReference(mDevicePointer);
}

template <typename T>
GPU_HOST_DEVICE const T& UniquePointer<T>::operator*() const noexcept
{
  return PointerTraits::getReference(mDevicePointer);
}
} // namespace gpu
} // namespace its
} // namespace o2

#endif /* TRAKINGITSU_INCLUDE_GPU_CAGPUUNIQUE_POINTER_H_ */
