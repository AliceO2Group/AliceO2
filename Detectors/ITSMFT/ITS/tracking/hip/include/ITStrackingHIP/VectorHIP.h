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

#include <assert.h>
#include <new>
#include <type_traits>
#include <vector>

#include "GPUCommonDef.h"
#include "ITStracking/Definitions.h"
#include "ITStrackingHIP/StreamHIP.h"
#include "ITStrackingHIP/UtilsHIP.h"

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
  GPUhd() ~Vector();

  Vector(const Vector&) = delete;
  Vector& operator=(const Vector&) = delete;

  GPUhd() Vector(Vector&&);
  Vector& operator=(Vector&&);

  int getSizeFromDevice() const;

  T getElementFromDevice(const int) const;

  void resize(const int);
  void reset(const int, const int = 0);
  void reset(const T* const, const int, const int = 0);
  void copyIntoVector(std::vector<T>&, const int);
  void copyIntoSizedVector(std::vector<T>&);

  GPUhd() T* get() const;
  GPUhd() int capacity() const;
  GPUhd() Vector<T> getWeakCopy() const;
  GPUd() T& operator[](const int) const;

  GPUd() int size() const;
  GPUd() int extend(const int) const;
  GPUhd() void dump();

  template <typename... Args>
  GPUd() void emplace(const int, Args&&...);

 protected:
  void destroy();

 private:
  GPUhd() Vector(const Vector&, const bool);

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
GPUhd() Vector<T>::~Vector()
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
GPUhd() Vector<T>::Vector(Vector<T>&& other)
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
void Vector<T>::copyIntoSizedVector(std::vector<T>& destinationVector)
{
  Utils::Host::gpuMemcpyDeviceToHost(destinationVector.data(), mArrayPointer, destinationVector.size() * sizeof(T));
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
GPUhd() T* Vector<T>::get() const
{
  return mArrayPointer;
}

template <typename T>
GPUhd() int Vector<T>::capacity() const
{
  return mCapacity;
}

template <typename T>
GPUhd() Vector<T> Vector<T>::getWeakCopy() const
{
  return Vector{*this, true};
}

template <typename T>
GPUd() T& Vector<T>::operator[](const int index) const
{
  return mArrayPointer[index];
}

template <typename T>
GPUh() T Vector<T>::getElementFromDevice(const int index) const
{
  T element;
  Utils::Host::gpuMemcpyDeviceToHost(&element, mArrayPointer + index, sizeof(T));

  return element;
}

template <typename T>
GPUd() int Vector<T>::size() const
{
  return *mDeviceSize;
}

template <typename T>
GPUd() int Vector<T>::extend(const int sizeIncrement) const
{
  const int startIndex = Utils::Device::gpuAtomicAdd(mDeviceSize, sizeIncrement);
  assert(size() <= mCapacity);

  return startIndex;
}

template <typename T>
template <typename... Args>
GPUd() void Vector<T>::emplace(const int index, Args&&... arguments)
{
  new (mArrayPointer + index) T(std::forward<Args>(arguments)...);
}

template <typename T>
GPUhd() void Vector<T>::dump()
{
  printf("mArrayPointer = %p\nmDeviceSize   = %p\nmCapacity     = %d\nmIsWeak       = %s\n",
         mArrayPointer, mDeviceSize, mCapacity, mIsWeak ? "true" : "false");
}
} // namespace GPU
} // namespace its
} // namespace o2

#endif /* O2_ITS_TRACKING_INCLUDE_VECTOR_HIP_H_ */
