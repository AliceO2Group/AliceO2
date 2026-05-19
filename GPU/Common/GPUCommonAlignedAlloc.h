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

/// \file GPUCommonAlignedAlloc.h
/// \author David Rohr

#ifndef GPUCOMMONAKUGBEDALLOC_H
#define GPUCOMMONAKUGBEDALLOC_H

#include <memory>

namespace o2::gpu
{

template <typename T, std::size_t MIN_ALIGN = 0>
struct alignedDeleter {
  void operator()(void* ptr) { ::operator delete(ptr, std::align_val_t(std::max(MIN_ALIGN, alignof(T)))); }; // TODO: Make this static once we go to C++ 23
};

template <typename T, std::size_t MIN_ALIGN = 0>
struct alignedAllocator {
  using value_type = T;
  static T* allocate(std::size_t n)
  {
    return (T*)::operator new(n, std::align_val_t(std::max(MIN_ALIGN, alignof(T))));
  }
  static void deallocate(T* ptr, std::size_t)
  {
    alignedDeleter<T, MIN_ALIGN>()(ptr);
  }
};

template <typename T>
struct aligned_unique_buffer_ptr : public std::unique_ptr<char[], alignedDeleter<T>> {
  aligned_unique_buffer_ptr() = default;
  aligned_unique_buffer_ptr(size_t n) { alloc(n); }
  aligned_unique_buffer_ptr(T* ptr) { std::unique_ptr<char[], alignedDeleter<T>>::reset((char*)ptr); }
  char* getraw() { return std::unique_ptr<char[], alignedDeleter<T>>::get(); }
  const char* getraw() const { return std::unique_ptr<char[], alignedDeleter<T>>::get(); }
  T* get() { return (T*)std::unique_ptr<char[], alignedDeleter<T>>::get(); }
  const T* get() const { return (T*)std::unique_ptr<char[], alignedDeleter<T>>::get(); }
  T* operator->() { return get(); }
  const T* operator->() const { return get(); }
  T* alloc(std::size_t n)
  {
    std::unique_ptr<char[], alignedDeleter<T>>::reset((char*)alignedAllocator<T>().allocate(n));
    return get();
  }
};

} // namespace o2::gpu

#endif // GPUCOMMONAKUGBEDALLOC_H
