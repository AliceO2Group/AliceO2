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
/// \file Vector.h
/// \brief
///

#ifndef ITSTRACKINGGPU_VECTOR_H_
#define ITSTRACKINGGPU_VECTOR_H_

#include <cassert>
#include <new>
#include <type_traits>
#include <vector>

#include "Stream.h"
#include "Utils.h"

namespace o2
{
namespace its
{
namespace gpu
{

template <typename T>
class Vector final
{
  static_assert(std::is_trivially_destructible<T>::value, "Vector only supports trivially destructible objects.");

 public:
  Vector();
  explicit Vector(const size_t, const size_t = 0);
  Vector(const T* const, const size_t, const size_t = 0);
  GPUhd() ~Vector();

  Vector(const Vector&) = delete;
  Vector& operator=(const Vector&) = delete;

  GPUhd() Vector(Vector&&);
  Vector& operator=(Vector&&);

  size_t getSizeFromDevice() const;

  T getElementFromDevice(const size_t) const;

  void resize(const size_t);
  void reset(const size_t, const size_t = 0);
  void reset(const T* const, const size_t, const size_t = 0);
  void copyIntoSizedVector(std::vector<T>&);

  GPUhd() T* get() const;
  GPUhd() size_t capacity() const;
  GPUhd() Vector<T> getWeakCopy() const;
  GPUd() T& operator[](const size_t) const;

  GPUd() size_t size() const;
  GPUhd() void dump();

  template <typename... Args>
  GPUd() void emplace(const size_t, Args&&...);

 protected:
  void destroy();

 private:
  GPUhd() Vector(const Vector&, const bool);

  T* mArrayPointer = nullptr;
  size_t* mDeviceSize = nullptr;
  size_t mCapacity;
  bool mIsWeak;
};

template <typename T>
Vector<T>::Vector() : Vector{nullptr, 0}
{
  // Nothing to do
}

template <typename T>
Vector<T>::Vector(const size_t capacity, const size_t initialSize) : Vector{nullptr, capacity, initialSize}
{
  // Nothing to do
}

template <typename T>
Vector<T>::Vector(const T* const source, const size_t size, const size_t initialSize) : mCapacity{size}, mIsWeak{false}
{
  if (size > 0) {
    try {

      utils::host::gpuMalloc(reinterpret_cast<void**>(&mArrayPointer), size * sizeof(T));
      utils::host::gpuMalloc(reinterpret_cast<void**>(&mDeviceSize), sizeof(size_t));

      if (source != nullptr) {

        utils::host::gpuMemcpyHostToDevice(mArrayPointer, source, size * sizeof(T));
        utils::host::gpuMemcpyHostToDevice(mDeviceSize, &size, sizeof(size_t));

      } else {

        utils::host::gpuMemcpyHostToDevice(mDeviceSize, &initialSize, sizeof(size_t));
      }

    } catch (...) {

      destroy();

      throw;
    }
  }
}

template <typename T>
GPUhd() Vector<T>::Vector(const Vector& other, const bool isWeak)
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
size_t Vector<T>::getSizeFromDevice() const
{
  size_t size;
  utils::host::gpuMemcpyDeviceToHost(&size, mDeviceSize, sizeof(size_t));

  return size;
}

template <typename T>
void Vector<T>::resize(const size_t size)
{
  utils::host::gpuMemcpyHostToDevice(mDeviceSize, &size, sizeof(size_t));
}

template <typename T>
void Vector<T>::reset(const size_t capacity, const size_t initialSize)
{
  reset(nullptr, capacity, initialSize);
}

template <typename T>
void Vector<T>::reset(const T* const source, const size_t size, const size_t initialSize)
{
  if (size > mCapacity) {
    if (mArrayPointer != nullptr) {
      utils::host::gpuFree(mArrayPointer);
    }
    utils::host::gpuMalloc(reinterpret_cast<void**>(&mArrayPointer), size * sizeof(T));
    mCapacity = size;
  }

  if (source != nullptr) {
    utils::host::gpuMemcpyHostToDevice(mArrayPointer, source, size * sizeof(T));
    utils::host::gpuMemcpyHostToDevice(mDeviceSize, &size, sizeof(size_t));

  } else {
    if (mDeviceSize == nullptr) {
      utils::host::gpuMalloc(reinterpret_cast<void**>(&mDeviceSize), sizeof(size_t));
    }
    utils::host::gpuMemcpyHostToDevice(mDeviceSize, &initialSize, sizeof(size_t));
  }
}

template <typename T>
void Vector<T>::copyIntoSizedVector(std::vector<T>& destinationVector)
{
  utils::host::gpuMemcpyDeviceToHost(destinationVector.data(), mArrayPointer, destinationVector.size() * sizeof(T));
}

template <typename T>
inline void Vector<T>::destroy()
{
  if (mArrayPointer != nullptr) {

    utils::host::gpuFree(mArrayPointer);
  }

  if (mDeviceSize != nullptr) {

    utils::host::gpuFree(mDeviceSize);
  }
}

template <typename T>
GPUhd() T* Vector<T>::get() const
{
  return mArrayPointer;
}

template <typename T>
GPUhd() size_t Vector<T>::capacity() const
{
  return mCapacity;
}

template <typename T>
GPUhd() Vector<T> Vector<T>::getWeakCopy() const
{
  return Vector{*this, true};
}

template <typename T>
GPUd() T& Vector<T>::operator[](const size_t index) const
{
  return mArrayPointer[index];
}

template <typename T>
GPUh() T Vector<T>::getElementFromDevice(const size_t index) const
{
  T element;
  utils::host::gpuMemcpyDeviceToHost(&element, mArrayPointer + index, sizeof(T));

  return element;
}

template <typename T>
GPUd() size_t Vector<T>::size() const
{
  return *mDeviceSize;
}

template <typename T>
template <typename... Args>
GPUd() void Vector<T>::emplace(const size_t index, Args&&... arguments)
{
  new (mArrayPointer + index) T(std::forward<Args>(arguments)...);
}

template <typename T>
GPUhd() void Vector<T>::dump()
{
  printf("mArrayPointer = %p\nmDeviceSize   = %p\nmCapacity     = %d\nmIsWeak       = %s\n",
         mArrayPointer, mDeviceSize, mCapacity, mIsWeak ? "true" : "false");
}
} // namespace gpu
} // namespace its
} // namespace o2

#endif
