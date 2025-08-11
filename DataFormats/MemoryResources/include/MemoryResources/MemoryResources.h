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

#include <cstring>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <unordered_map>
#include <fairmq/Message.h>
#include <fairmq/TransportFactory.h>
#include <fairmq/MemoryResources.h>
#include <fairmq/MemoryResourceTools.h>

namespace o2::pmr
{

template <typename ContainerT>
fair::mq::MessagePtr getMessage(ContainerT&& container, fair::mq::MemoryResource* targetResource = nullptr)
{
  return fair::mq::getMessage(std::forward<ContainerT>(container), targetResource);
}

//__________________________________________________________________________________________________
/// This memory resource only watches, does not allocate/deallocate anything.
/// Ownership of hte message is taken. Meant to be used for transparent data adoption in containers.
/// In combination with the SpectatorAllocator this is an alternative to using span, as raw memory
/// (e.g. an existing buffer message) will be accessible with appropriate container.
class MessageResource : public fair::mq::MemoryResource
{

 public:
  MessageResource() noexcept = delete;
  MessageResource(const MessageResource&) noexcept = default;
  MessageResource(MessageResource&&) noexcept = default;
  MessageResource& operator=(const MessageResource&) = default;
  MessageResource& operator=(MessageResource&&) = default;
  MessageResource(fair::mq::MessagePtr message)
    : mUpstream{message->GetTransport()->GetMemoryResource()},
      mMessageSize{message->GetSize()},
      mMessageData{mUpstream ? mUpstream->setMessage(std::move(message))
                             : throw std::runtime_error("MessageResource::MessageResource upstream is nullptr")}
  {
  }
  fair::mq::MessagePtr getMessage(void* p) override { return mUpstream->getMessage(p); }
  void* setMessage(fair::mq::MessagePtr message) override { return mUpstream->setMessage(std::move(message)); }
  fair::mq::TransportFactory* getTransportFactory() noexcept override { return nullptr; }
  size_t getNumberOfMessages() const noexcept override { return mMessageData ? 1 : 0; }

 protected:
  fair::mq::MemoryResource* mUpstream{nullptr};
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
      return mUpstream->allocate(bytes, alignment < 64 ? 64 : alignment);
    }
  }
  void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override
  {
    mUpstream->deallocate(p, bytes, alignment < 64 ? 64 : alignment);
    return;
  }
  bool do_is_equal(const memory_resource& /*other*/) const noexcept override
  {
    // since this uniquely owns the message it can never be equal to anybody else
    return false;
  }
};

// The NoConstructAllocator behaves like the normal pmr vector but does not call constructors / destructors
template <typename T>
class NoConstructAllocator : public std::pmr::polymorphic_allocator<T>
{
 public:
  using std::pmr::polymorphic_allocator<T>::polymorphic_allocator;
  using propagate_on_container_move_assignment = std::true_type;

  template <typename... Args>
  NoConstructAllocator(Args&&... args) : std::pmr::polymorphic_allocator<T>(std::forward<Args>(args)...)
  {
  }

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
};

//__________________________________________________________________________________________________
//__________________________________________________________________________________________________
//__________________________________________________________________________________________________
//__________________________________________________________________________________________________

using BytePmrAllocator = std::pmr::polymorphic_allocator<std::byte>;
template <class T>
using vector = std::vector<T, std::pmr::polymorphic_allocator<T>>;

//__________________________________________________________________________________________________
/// Get the allocator associated to a transport factory
inline static fair::mq::MemoryResource* getTransportAllocator(fair::mq::TransportFactory* factory)
{
  return *factory;
}

} // namespace o2::pmr

#endif
