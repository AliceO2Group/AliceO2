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
/// \file VectorHIP.h
/// \brief
///

#ifndef O2_ITS_TRACKING_INCLUDE_VECTOR_HIP_H_
#define O2_ITS_TRACKING_INCLUDE_VECTOR_HIP_H_

#include <cassert>
#include <new>
#include <type_traits>
#include <vector>

#include "ITStrackingHIP/StreamHIP.h"
#include "ITStrackingHIP/UtilsHIP.h"
#include "GPUCommonDef.h"

namespace o2
{
namespace its
{
namespace gpu
{

template <typename T>
class VectorHIP final
{
  static_assert(std::is_trivially_destructible<T>::value, "VectorHIP only supports trivially destructible objects.");

 public:
  VectorHIP();
  explicit VectorHIP(const int, const int = 0);
  VectorHIP(const T* const, const int, const int = 0);
  GPUhd() ~VectorHIP();

  VectorHIP(const VectorHIP&) = delete;
  VectorHIP& operator=(const VectorHIP&) = delete;

  GPUhd() VectorHIP(VectorHIP&&);
  VectorHIP& operator=(VectorHIP&&);

  int getSizeFromDevice() const;

  T getElementFromDevice(const int) const;

  void resize(const int);
  void reset(const int, const int = 0);
  void reset(const T* const, const int, const int = 0);
  void copyIntoVector(std::vector<T>&, const int);
  void copyIntoSizedVector(std::vector<T>&);

  GPUhd() T* get() const;
  GPUhd() int capacity() const;
  GPUhd() VectorHIP<T> getWeakCopy() const;
  GPUhd() T& operator[](const int) const;

  GPUhd() int size() const;
  GPUd() int extend(const int) const;
  GPUhd() void dump();

  template <typename... Args>
  GPUd() void emplace(const int, Args&&...);

 protected:
  void destroy();

 private:
  GPUhd() VectorHIP(const VectorHIP&, const bool);

  T* mArrayPointer = nullptr;
  int* mDeviceSize = nullptr;
  int mCapacity;
  bool mIsWeak;
};

template <typename T>
VectorHIP<T>::VectorHIP() : VectorHIP{nullptr, 0}
{
  // Nothing to do
}

template <typename T>
VectorHIP<T>::VectorHIP(const int capacity, const int initialSize) : VectorHIP{nullptr, capacity, initialSize}
{
  // Nothing to do
}

template <typename T>
VectorHIP<T>::VectorHIP(const T* const source, const int size, const int initialSize) : mCapacity{size}, mIsWeak{false}
{
  if (size > 0) {
    try {

      utils::host_hip::gpuMalloc(reinterpret_cast<void**>(&mArrayPointer), size * sizeof(T));
      utils::host_hip::gpuMalloc(reinterpret_cast<void**>(&mDeviceSize), sizeof(int));

      if (source != nullptr) {

        utils::host_hip::gpuMemcpyHostToDevice(mArrayPointer, source, size * sizeof(T));
        utils::host_hip::gpuMemcpyHostToDevice(mDeviceSize, &size, sizeof(int));

      } else {

        utils::host_hip::gpuMemcpyHostToDevice(mDeviceSize, &initialSize, sizeof(int));
      }

    } catch (...) {

      destroy();

      throw;
    }
  }
}

template <typename T>
GPUhdi() VectorHIP<T>::VectorHIP(const VectorHIP& other, const bool isWeak)
  : mArrayPointer{other.mArrayPointer},
    mDeviceSize{other.mDeviceSize},
    mCapacity{other.mCapacity},
    mIsWeak{isWeak}
{
  // Nothing to do
}

template <typename T>
GPUhd() VectorHIP<T>::~VectorHIP()
{
  if (mIsWeak) {

    return;

  } else {
#if defined(TRACKINGITSU_GPU_DEVICE)
    assert(0);
#else
    destroy();
#endif
  }
}

template <typename T>
GPUhd() VectorHIP<T>::VectorHIP(VectorHIP<T>&& other)
  : mArrayPointer{other.mArrayPointer},
    mDeviceSize{other.mDeviceSize},
    mCapacity{other.mCapacity},
    mIsWeak{other.mIsWeak}
{
  other.mArrayPointer = nullptr;
  other.mDeviceSize = nullptr;
}

template <typename T>
VectorHIP<T>& VectorHIP<T>::operator=(VectorHIP<T>&& other)
{
  destroy();

  mArrayPointer = other.mArrayPointer;
  mDeviceSize = other.mDeviceSize;
  mCapacity = other.mCapacity;
  mIsWeak = other.mIsWeak;

  other.mArrayPointer = nullptr;
  other.mDeviceSize = nullptr;

  return *this;
}

template <typename T>
int VectorHIP<T>::getSizeFromDevice() const
{
  int size;
  utils::host_hip::gpuMemcpyDeviceToHost(&size, mDeviceSize, sizeof(int));

  return size;
}

template <typename T>
void VectorHIP<T>::resize(const int size)
{
  utils::host_hip::gpuMemcpyHostToDevice(mDeviceSize, &size, sizeof(int));
}

template <typename T>
void VectorHIP<T>::reset(const int capacity, const int initialSize)
{
  reset(nullptr, capacity, initialSize);
}

template <typename T>
void VectorHIP<T>::reset(const T* const source, const int size, const int initialSize)
{
  if (size > mCapacity) {
    if (mArrayPointer != nullptr) {
      utils::host_hip::gpuFree(mArrayPointer);
    }

    utils::host_hip::gpuMalloc(reinterpret_cast<void**>(&mArrayPointer), size * sizeof(T));
    mCapacity = size;
  }

  if (source != nullptr) {

    utils::host_hip::gpuMemcpyHostToDevice(mArrayPointer, source, size * sizeof(T));
    utils::host_hip::gpuMemcpyHostToDevice(mDeviceSize, &size, sizeof(int));

  } else {
    utils::host_hip::gpuMemcpyHostToDevice(mDeviceSize, &initialSize, sizeof(int));
  }
}

template <typename T>
void VectorHIP<T>::copyIntoVector(std::vector<T>& destinationVector, const int size)
{

  T* hostPrimitivePointer = nullptr;

  try {

    hostPrimitivePointer = static_cast<T*>(malloc(size * sizeof(T)));
    utils::host_hip::gpuMemcpyDeviceToHost(hostPrimitivePointer, mArrayPointer, size * sizeof(T));

    destinationVector = std::move(std::vector<T>(hostPrimitivePointer, hostPrimitivePointer + size));

  } catch (...) {

    if (hostPrimitivePointer != nullptr) {

      free(hostPrimitivePointer);
    }

    throw;
  }
}

template <typename T>
void VectorHIP<T>::copyIntoSizedVector(std::vector<T>& destinationVector)
{
  utils::host_hip::gpuMemcpyDeviceToHost(destinationVector.data(), mArrayPointer, destinationVector.size() * sizeof(T));
}

template <typename T>
inline void VectorHIP<T>::destroy()
{
  if (mArrayPointer != nullptr) {

    utils::host_hip::gpuFree(mArrayPointer);
  }

  if (mDeviceSize != nullptr) {

    utils::host_hip::gpuFree(mDeviceSize);
  }
}

template <typename T>
GPUhd() T* VectorHIP<T>::get() const
{
  return mArrayPointer;
}

template <typename T>
GPUhd() int VectorHIP<T>::capacity() const
{
  return mCapacity;
}

template <typename T>
GPUhd() VectorHIP<T> VectorHIP<T>::getWeakCopy() const
{
  return VectorHIP{*this, true};
}

template <typename T>
GPUhd() T& VectorHIP<T>::operator[](const int index) const
{
  return mArrayPointer[index];
}

template <typename T>
T VectorHIP<T>::getElementFromDevice(const int index) const
{
  T element;
  utils::host_hip::gpuMemcpyDeviceToHost(&element, mArrayPointer + index, sizeof(T));

  return element;
}

template <typename T>
GPUhd() int VectorHIP<T>::size() const
{
  return *mDeviceSize;
}

template <typename T>
GPUd() int VectorHIP<T>::extend(const int sizeIncrement) const
{
  const int startIndex = utils::device_hip::gpuAtomicAdd(mDeviceSize, sizeIncrement);
  assert(size() <= mCapacity);

  return startIndex;
}

template <typename T>
template <typename... Args>
GPUd() void VectorHIP<T>::emplace(const int index, Args&&... arguments)
{
  new (mArrayPointer + index) T(std::forward<Args>(arguments)...);
}

template <typename T>
GPUhd() void VectorHIP<T>::dump()
{
  printf("mArrayPointer = %p\nmDeviceSize   = %p\nmCapacity     = %d\nmIsWeak       = %s\n",
         mArrayPointer, mDeviceSize, mCapacity, mIsWeak ? "true" : "false");
}
} // namespace gpu
} // namespace its
} // namespace o2

#endif /* O2_ITS_TRACKING_INCLUDE_VECTOR_HIP_H_ */
