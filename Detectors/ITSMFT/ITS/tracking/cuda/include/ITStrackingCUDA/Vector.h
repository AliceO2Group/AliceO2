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
/// \file Vector.h
/// \brief
///

#ifndef TRAKINGITSU_INCLUDE_GPU_VECTOR_H_
#define TRAKINGITSU_INCLUDE_GPU_VECTOR_H_

#include <assert.h>
#include <new>
#include <type_traits>
#include <vector>

#include "ITStracking/Definitions.h"
#include "ITStrackingCUDA/Stream.h"
#include "ITStrackingCUDA/Utils.h"

namespace o2
{
namespace its
{
namespace GPU
{

template <typename T>
class Vector final
{
  static_assert(std::is_trivially_destructible<T>::value, "Vector only supports trivially destructible objects.");

 public:
  Vector();
  explicit Vector(const int, const int = 0);
  Vector(const T* const, const int, const int = 0);
  GPU_HOST_DEVICE ~Vector();

  Vector(const Vector&) = delete;
  Vector& operator=(const Vector&) = delete;

  GPU_HOST_DEVICE Vector(Vector&&);
  Vector& operator=(Vector&&);

  int getSizeFromDevice() const;
  void resize(const int);
  void reset(const int, const int = 0);
  void reset(const T* const, const int, const int = 0);
  void copyIntoVector(std::vector<T>&, const int);

  GPU_HOST_DEVICE T* get() const;
  GPU_HOST_DEVICE int capacity() const;
  GPU_HOST_DEVICE Vector<T> getWeakCopy() const;
  GPU_DEVICE T& operator[](const int) const;
  GPU_DEVICE int size() const;
  GPU_DEVICE int extend(const int) const;
  GPU_HOST_DEVICE void dump();

  template <typename... Args>
  GPU_DEVICE void emplace(const int, Args&&...);

 protected:
  void destroy();

 private:
  GPU_HOST_DEVICE Vector(const Vector&, const bool);

  T* mArrayPointer = nullptr;
  int* mDeviceSize = nullptr;
  int mCapacity;
  bool mIsWeak;
};

template <typename T>
Vector<T>::Vector() : Vector{nullptr, 0}
{
  // Nothing to do
}

template <typename T>
Vector<T>::Vector(const int capacity, const int initialSize) : Vector{nullptr, capacity, initialSize}
{
  // Nothing to do
}

template <typename T>
Vector<T>::Vector(const T* const source, const int size, const int initialSize) : mCapacity{size}, mIsWeak{false}
{
  if (size > 0) {
    try {

      Utils::Host::gpuMalloc(reinterpret_cast<void**>(&mArrayPointer), size * sizeof(T));
      Utils::Host::gpuMalloc(reinterpret_cast<void**>(&mDeviceSize), sizeof(int));

      if (source != nullptr) {

        Utils::Host::gpuMemcpyHostToDevice(mArrayPointer, source, size * sizeof(T));
        Utils::Host::gpuMemcpyHostToDevice(mDeviceSize, &size, sizeof(int));

      } else {

        Utils::Host::gpuMemcpyHostToDevice(mDeviceSize, &initialSize, sizeof(int));
      }

    } catch (...) {

      destroy();

      throw;
    }
  }
}

template <typename T>
Vector<T>::Vector(const Vector& other, const bool isWeak)
  : mArrayPointer{other.mArrayPointer},
    mDeviceSize{other.mDeviceSize},
    mCapacity{other.mCapacity},
    mIsWeak{isWeak}
{
  // Nothing to do
}

template <typename T>
GPU_HOST_DEVICE Vector<T>::~Vector()
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
GPU_HOST_DEVICE Vector<T>::Vector(Vector<T>&& other)
  : mArrayPointer{other.mArrayPointer},
    mDeviceSize{other.mDeviceSize},
    mCapacity{other.mCapacity},
    mIsWeak{other.mIsWeak}
{
  other.mArrayPointer = nullptr;
  other.mDeviceSize = nullptr;
}

template <typename T>
Vector<T>& Vector<T>::operator=(Vector<T>&& other)
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
int Vector<T>::getSizeFromDevice() const
{
  int size;
  Utils::Host::gpuMemcpyDeviceToHost(&size, mDeviceSize, sizeof(int));

  return size;
}

template <typename T>
void Vector<T>::resize(const int size)
{
  Utils::Host::gpuMemcpyHostToDevice(mDeviceSize, &size, sizeof(int));
}

template <typename T>
void Vector<T>::reset(const int capacity, const int initialSize)
{
  reset(nullptr, capacity, initialSize);
}

template <typename T>
void Vector<T>::reset(const T* const source, const int size, const int initialSize)
{
  if (size > mCapacity) {
    if (mArrayPointer != nullptr) {
      Utils::Host::gpuFree(mArrayPointer);
    }

    Utils::Host::gpuMalloc(reinterpret_cast<void**>(&mArrayPointer), size * sizeof(T));
    mCapacity = size;
  }

  if (source != nullptr) {

    Utils::Host::gpuMemcpyHostToDevice(mArrayPointer, source, size * sizeof(T));
    Utils::Host::gpuMemcpyHostToDevice(mDeviceSize, &size, sizeof(int));

  } else {
    Utils::Host::gpuMemcpyHostToDevice(mDeviceSize, &initialSize, sizeof(int));
  }
}

template <typename T>
void Vector<T>::copyIntoVector(std::vector<T>& destinationVector, const int size)
{

  T* hostPrimitivePointer = nullptr;

  try {

    hostPrimitivePointer = static_cast<T*>(malloc(size * sizeof(T)));
    Utils::Host::gpuMemcpyDeviceToHost(hostPrimitivePointer, mArrayPointer, size * sizeof(T));

    destinationVector = std::move(std::vector<T>(hostPrimitivePointer, hostPrimitivePointer + size));

  } catch (...) {

    if (hostPrimitivePointer != nullptr) {

      free(hostPrimitivePointer);
    }

    throw;
  }
}

template <typename T>
inline void Vector<T>::destroy()
{
  if (mArrayPointer != nullptr) {

    Utils::Host::gpuFree(mArrayPointer);
  }

  if (mDeviceSize != nullptr) {

    Utils::Host::gpuFree(mDeviceSize);
  }
}

template <typename T>
GPU_HOST_DEVICE inline T* Vector<T>::get() const
{
  return mArrayPointer;
}

template <typename T>
GPU_HOST_DEVICE inline int Vector<T>::capacity() const
{
  return mCapacity;
}

template <typename T>
GPU_HOST_DEVICE inline Vector<T> Vector<T>::getWeakCopy() const
{
  return Vector{*this, true};
}

template <typename T>
GPU_DEVICE inline T& Vector<T>::operator[](const int index) const
{
  return mArrayPointer[index];
}

template <typename T>
GPU_DEVICE inline int Vector<T>::size() const
{
  return *mDeviceSize;
}

template <typename T>
GPU_DEVICE int Vector<T>::extend(const int sizeIncrement) const
{
  const int startIndex = Utils::Device::gpuAtomicAdd(mDeviceSize, sizeIncrement);
  assert(size() <= mCapacity);

  return startIndex;
}

template <typename T>
template <typename... Args>
GPU_DEVICE void Vector<T>::emplace(const int index, Args&&... arguments)
{
  new (mArrayPointer + index) T(std::forward<Args>(arguments)...);
}

template <typename T>
GPU_HOST_DEVICE void Vector<T>::dump()
{
  printf("mArrayPointer = %p\nmDeviceSize   = %p\nmCapacity     = %d\nmIsWeak       = %s\n",
         mArrayPointer, mDeviceSize, mCapacity, mIsWeak ? "true" : "false");
}
} // namespace GPU
} // namespace its
} // namespace o2

#endif /* TRAKINGITSU_INCLUDE_GPU_VECTOR_H_ */
