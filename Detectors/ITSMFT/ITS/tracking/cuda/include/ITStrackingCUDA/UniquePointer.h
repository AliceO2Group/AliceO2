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
namespace GPU
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

    Utils::Host::gpuMalloc(reinterpret_cast<void**>(&mDevicePointer), sizeof(T));
    Utils::Host::gpuMemcpyHostToDevice(mDevicePointer, &ref, sizeof(T));

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

    Utils::Host::gpuFree(mDevicePointer);
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
} // namespace GPU
} // namespace its
} // namespace o2

#endif /* TRAKINGITSU_INCLUDE_GPU_CAGPUUNIQUE_POINTER_H_ */
