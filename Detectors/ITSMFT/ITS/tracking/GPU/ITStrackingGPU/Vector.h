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

  void resetWithInt(const size_t, const int value = 0);
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

  T* mArrayPtr = nullptr;
  size_t* mDeviceSizePtr = nullptr;
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

      utils::gpuMalloc(reinterpret_cast<void**>(&mArrayPtr), size * sizeof(T));
      utils::gpuMalloc(reinterpret_cast<void**>(&mDeviceSizePtr), sizeof(size_t));

      if (source != nullptr) {

        utils::gpuMemcpyHostToDevice(mArrayPtr, source, size * sizeof(T));
        utils::gpuMemcpyHostToDevice(mDeviceSizePtr, &size, sizeof(size_t));

      } else {

        utils::gpuMemcpyHostToDevice(mDeviceSizePtr, &initialSize, sizeof(size_t));
      }

    } catch (...) {

      destroy();

      throw;
    }
  }
}

template <typename T>
GPUhd() Vector<T>::Vector(const Vector& other, const bool isWeak)
  : mArrayPtr{other.mArrayPtr},
    mDeviceSizePtr{other.mDeviceSizePtr},
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
  : mArrayPtr{other.mArrayPtr},
    mDeviceSizePtr{other.mDeviceSizePtr},
    mCapacity{other.mCapacity},
    mIsWeak{other.mIsWeak}
{
  other.mArrayPtr = nullptr;
  other.mDeviceSizePtr = nullptr;
}

template <typename T>
Vector<T>& Vector<T>::operator=(Vector<T>&& other)
{
  destroy();

  mArrayPtr = other.mArrayPtr;
  mDeviceSizePtr = other.mDeviceSizePtr;
  mCapacity = other.mCapacity;
  mIsWeak = other.mIsWeak;

  other.mArrayPtr = nullptr;
  other.mDeviceSizePtr = nullptr;

  return *this;
}

template <typename T>
size_t Vector<T>::getSizeFromDevice() const
{
  size_t size;
  utils::gpuMemcpyDeviceToHost(&size, mDeviceSizePtr, sizeof(size_t));

  return size;
}

template <typename T>
void Vector<T>::resize(const size_t size)
{
  utils::gpuMemcpyHostToDevice(mDeviceSizePtr, &size, sizeof(size_t));
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
    if (mArrayPtr != nullptr) {
      utils::gpuFree(mArrayPtr);
    }
    utils::gpuMalloc(reinterpret_cast<void**>(&mArrayPtr), size * sizeof(T));
    mCapacity = size;
  }
  if (mDeviceSizePtr == nullptr) {
    utils::gpuMalloc(reinterpret_cast<void**>(&mDeviceSizePtr), sizeof(size_t));
  }

  if (source != nullptr) {
    utils::gpuMemcpyHostToDevice(mArrayPtr, source, size * sizeof(T));
    utils::gpuMemcpyHostToDevice(mDeviceSizePtr, &size, sizeof(size_t));
  } else {
    utils::gpuMemcpyHostToDevice(mDeviceSizePtr, &initialSize, sizeof(size_t));
  }
}

template <typename T>
void Vector<T>::resetWithInt(const size_t size, const int value)
{
  if (size > mCapacity) {
    if (mArrayPtr != nullptr) {
      utils::gpuFree(mArrayPtr);
    }
    utils::gpuMalloc(reinterpret_cast<void**>(&mArrayPtr), size * sizeof(int));
    mCapacity = size;
  }
  if (mDeviceSizePtr == nullptr) {
    utils::gpuMalloc(reinterpret_cast<void**>(&mDeviceSizePtr), sizeof(int));
  }

  utils::gpuMemset(mArrayPtr, value, size * sizeof(int));
  utils::gpuMemcpyHostToDevice(mDeviceSizePtr, &size, sizeof(int));
}

template <typename T>
void Vector<T>::copyIntoSizedVector(std::vector<T>& destinationVector)
{
  utils::gpuMemcpyDeviceToHost(destinationVector.data(), mArrayPtr, destinationVector.size() * sizeof(T));
}

template <typename T>
inline void Vector<T>::destroy()
{
  if (mArrayPtr != nullptr) {
    utils::gpuFree(mArrayPtr);
  }
  if (mDeviceSizePtr != nullptr) {
    utils::gpuFree(mDeviceSizePtr);
  }
}

template <typename T>
GPUhd() T* Vector<T>::get() const
{
  return mArrayPtr;
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
  return mArrayPtr[index];
}

template <typename T>
GPUh() T Vector<T>::getElementFromDevice(const size_t index) const
{
  T element;
  utils::gpuMemcpyDeviceToHost(&element, mArrayPtr + index, sizeof(T));

  return element;
}

template <typename T>
GPUd() size_t Vector<T>::size() const
{
  return *mDeviceSizePtr;
}

template <typename T>
template <typename... Args>
GPUd() void Vector<T>::emplace(const size_t index, Args&&... arguments)
{
  new (mArrayPtr + index) T(std::forward<Args>(arguments)...);
}

template <typename T>
GPUhd() void Vector<T>::dump()
{
  printf("mArrayPtr = %p\nmDeviceSize   = %p\nmCapacity     = %d\nmIsWeak       = %s\n",
         mArrayPtr, mDeviceSizePtr, mCapacity, mIsWeak ? "true" : "false");
}
} // namespace gpu
} // namespace its
} // namespace o2

#endif
