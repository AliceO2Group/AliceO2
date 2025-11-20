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
/// \file ExternalAllocator.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_EXTERNALALLOCATOR_H_
#define TRACKINGITSU_INCLUDE_EXTERNALALLOCATOR_H_

#include <memory_resource>
#include "GPUO2ExternalUser.h"
#include "Base/GPUMemoryResource.h"

namespace o2::its
{

class ExternalAllocator
{
  using Type = std::underlying_type_t<o2::gpu::GPUMemoryResource::MemoryType>;

 public:
  virtual void deallocate(char*, size_t) = 0;
  virtual void* allocate(size_t) = 0;
  void* allocate(size_t s, Type type)
  {
    auto old = mType;
    mType = type;
    void* p = allocate(s);
    mType = old;
    return p;
  }
  void* allocateStack(size_t s)
  {
    return allocate(s, (o2::gpu::GPUMemoryResource::MEMORY_GPU | o2::gpu::GPUMemoryResource::MEMORY_STACK));
  }
  virtual void pushTagOnStack(uint64_t) = 0;
  virtual void popTagOffStack(uint64_t) = 0;

  void setType(Type t) noexcept { mType = t; }
  Type getType() const noexcept { return mType; }

 protected:
  Type mType;
};

class ExternalAllocatorAdaptor final : public std::pmr::memory_resource
{
 public:
  explicit ExternalAllocatorAdaptor(ExternalAllocator* alloc) : mAlloc(alloc) {}

 protected:
  void* do_allocate(size_t bytes, size_t alignment) override
  {
    void* p = mAlloc->allocate(bytes, o2::gpu::GPUMemoryResource::MemoryType::MEMORY_HOST);
    if (!p) {
      throw std::bad_alloc();
    }
    return p;
  }

  void do_deallocate(void* p, size_t bytes, size_t) override
  {
    mAlloc->deallocate(static_cast<char*>(p), bytes);
  }

  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override
  {
    return this == &other;
  }

 private:
  ExternalAllocator* mAlloc;
};

} // namespace o2::its

#endif
