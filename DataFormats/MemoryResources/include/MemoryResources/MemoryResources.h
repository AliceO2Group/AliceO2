// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @copyright
/// © Copyright 2014 Copyright Holders of the ALICE O2 collaboration.
/// See https://aliceinfo.cern.ch/AliceO2 for details on the Copyright holders.
/// This software is distributed under the terms of the
/// GNU General Public License version 3 (GPL Version 3).
///
/// License text in a separate file.
///
/// In applying this license, CERN does not waive the privileges and immunities
/// granted to it by virtue of its status as an Intergovernmental Organization
/// or submit itself to any jurisdiction.

/// @brief O2 memory allocators and interfaces related to managing memory via the trasport layer
///
/// @author Mikolaj Krzewicki, mkrzewic@cern.ch

#ifndef ALICEO2_MEMORY_RESOURCES_
#define ALICEO2_MEMORY_RESOURCES_

#include <boost/container/flat_map.hpp>
#include <boost/container/pmr/memory_resource.hpp>
#include <boost/container/pmr/monotonic_buffer_resource.hpp>
#include <boost/container/pmr/polymorphic_allocator.hpp>
#include <cstring>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <unordered_map>
#include <FairMQMessage.h>
#include <FairMQTransportFactory.h>
#include <fairmq/MemoryResources.h>
#include <fairmq/MemoryResourceTools.h>
#include "Types.h"

namespace o2
{

namespace pmr
{

using FairMQMemoryResource = fair::mq::FairMQMemoryResource;
using ChannelResource = fair::mq::ChannelResource;
using namespace fair::mq::pmr;

template <typename ContainerT>
FairMQMessagePtr getMessage(ContainerT&& container, FairMQMemoryResource* targetResource = nullptr)
{
  return fair::mq::getMessage(std::forward<ContainerT>(container), targetResource);
}

//__________________________________________________________________________________________________
/// This memory resource only watches, does not allocate/deallocate anything.
/// Ownership of hte message is taken. Meant to be used for transparent data adoption in containers.
/// In combination with the SpectatorAllocator this is an alternative to using span, as raw memory
/// (e.g. an existing buffer message) will be accessible with appropriate container.
class MessageResource : public FairMQMemoryResource
{

 public:
  MessageResource() noexcept = delete;
  MessageResource(const MessageResource&) noexcept = default;
  MessageResource(MessageResource&&) noexcept = default;
  MessageResource& operator=(const MessageResource&) = default;
  MessageResource& operator=(MessageResource&&) = default;
  MessageResource(FairMQMessagePtr message)
    : mUpstream{message->GetTransport()->GetMemoryResource()},
      mMessageSize{message->GetSize()},
      mMessageData{mUpstream ? mUpstream->setMessage(std::move(message))
                             : throw std::runtime_error("MessageResource::MessageResource upstream is nullptr")}
  {
  }
  FairMQMessagePtr getMessage(void* p) override { return mUpstream->getMessage(p); }
  void* setMessage(FairMQMessagePtr message) override { return mUpstream->setMessage(std::move(message)); }
  FairMQTransportFactory* getTransportFactory() noexcept override { return nullptr; }
  size_t getNumberOfMessages() const noexcept override { return mMessageData ? 1 : 0; }

 protected:
  FairMQMemoryResource* mUpstream{nullptr};
  size_t mMessageSize{0};
  void* mMessageData{nullptr};
  bool initialImport{true};

  void* do_allocate(std::size_t bytes, std::size_t alignment) override
  {
    if (initialImport) {
      if (bytes > mMessageSize) {
        throw std::bad_alloc();
      }
      initialImport = false;
      return mMessageData;
    } else {
      return mUpstream->allocate(bytes, alignment);
    }
  }
  void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override
  {
    mUpstream->deallocate(p, bytes, alignment);
    return;
  }
  bool do_is_equal(const memory_resource& other) const noexcept override
  {
    // since this uniquely owns the message it can never be equal to anybody else
    return false;
  }
};

//__________________________________________________________________________________________________
// This in general (as in STL) is a bad idea, but here it is safe to inherit from an allocator since we
// have no additional data and only override some methods so we don't get into slicing and other problems.
template <typename T>
class SpectatorAllocator : public boost::container::pmr::polymorphic_allocator<T>
{
 public:
  using boost::container::pmr::polymorphic_allocator<T>::polymorphic_allocator;

  // skip default construction of empty elements
  // this is important for two reasons: one: it allows us to adopt an existing buffer (e.g. incoming message) and
  // quickly construct large vectors while skipping the element initialization.
  template <class U>
  void construct(U*)
  {
  }

  // dont try to call destructors, makes no sense since resource is managed externally AND allowed
  // types cannot have side effects
  template <typename U>
  void destroy(U*)
  {
  }

  T* allocate(size_t size) { return reinterpret_cast<T*>(this->resource()->allocate(size * sizeof(T), 0)); }
  void deallocate(T* ptr, size_t size)
  {
    this->resource()->deallocate(const_cast<typename std::remove_cv<T>::type*>(ptr), size);
  }
};

//__________________________________________________________________________________________________
/// This allocator has a pmr-like interface, but keeps the unique MessageResource as internal state,
/// allowing full resource (associated message) management internally without any global state.
template <typename T>
class OwningMessageSpectatorAllocator
{
 public:
  using value_type = T;

  MessageResource mResource;

  OwningMessageSpectatorAllocator() noexcept = default;
  OwningMessageSpectatorAllocator(const OwningMessageSpectatorAllocator&) noexcept = default;
  OwningMessageSpectatorAllocator(OwningMessageSpectatorAllocator&&) noexcept = default;
  OwningMessageSpectatorAllocator(MessageResource&& resource) noexcept : mResource{resource} {}

  template <class U>
  OwningMessageSpectatorAllocator(const OwningMessageSpectatorAllocator<U>& other) noexcept : mResource(other.mResource)
  {
  }

  OwningMessageSpectatorAllocator& operator=(const OwningMessageSpectatorAllocator& other)
  {
    mResource = other.mResource;
    return *this;
  }

  OwningMessageSpectatorAllocator select_on_container_copy_construction() const
  {
    return OwningMessageSpectatorAllocator();
  }

  boost::container::pmr::memory_resource* resource() { return &mResource; }

  // skip default construction of empty elements
  // this is important for two reasons: one: it allows us to adopt an existing buffer (e.g. incoming message) and
  // quickly construct large vectors while skipping the element initialization.
  template <class U>
  void construct(U*)
  {
  }

  // dont try to call destructors, makes no sense since resource is managed externally AND allowed
  // types cannot have side effects
  template <typename U>
  void destroy(U*)
  {
  }

  T* allocate(size_t size) { return reinterpret_cast<T*>(mResource.allocate(size * sizeof(T), 0)); }
  void deallocate(T* ptr, size_t size)
  {
    mResource.deallocate(const_cast<typename std::remove_cv<T>::type*>(ptr), size);
  }
};

//__________________________________________________________________________________________________
//__________________________________________________________________________________________________
//__________________________________________________________________________________________________
//__________________________________________________________________________________________________

using ByteSpectatorAllocator = SpectatorAllocator<o2::byte>;
using BytePmrAllocator = boost::container::pmr::polymorphic_allocator<o2::byte>;
template <class T>
using vector = std::vector<T, o2::pmr::polymorphic_allocator<T>>;

//__________________________________________________________________________________________________
/// Return a std::vector spanned over the contents of the message, takes ownership of the message
template <typename ElemT>
auto adoptVector(size_t nelem, FairMQMessagePtr message)
{
  static_assert(std::is_trivially_destructible<ElemT>::value);
  return std::vector<ElemT, OwningMessageSpectatorAllocator<ElemT>>(
    nelem, OwningMessageSpectatorAllocator<ElemT>(MessageResource{std::move(message)}));
};

//__________________________________________________________________________________________________
/// Get the allocator associated to a transport factory
inline static FairMQMemoryResource* getTransportAllocator(FairMQTransportFactory* factory)
{
  return *factory;
}

}; //namespace pmr

template <class T>
using vector = std::vector<T, o2::pmr::polymorphic_allocator<T>>;

}; //namespace o2

#endif
