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
#ifndef O2_HEADERS_STACK_H
#define O2_HEADERS_STACK_H

#include "MemoryResources/MemoryResources.h"
#include "Headers/DataHeader.h"

namespace o2
{

namespace header
{
//__________________________________________________________________________________________________
/// @struct Stack
/// @brief a move-only header stack with serialized headers
/// This is the flat buffer where all the headers in a multi-header go.
/// This guy knows how to move the serialized content to FairMQ
/// and inform it how to release when all is sent.
/// methods to construct a multi-header
/// intended use:
///   - as a variadic intializer list (as an argument to a function)
///
///   One can also use the ctor directly:
//    Stack::Stack(const T& header1, const T& header2, ...)
//    - arguments can be headers, or stacks, all will be concatenated in a new Stack
///   - returns a Stack ready to be shipped.
struct Stack {

  using memory_resource = o2::pmr::memory_resource;

 private:
  struct freeobj {
    freeobj(memory_resource* mr) : resource(mr) {}
    memory_resource* resource{nullptr};
    void operator()(std::byte* ptr) { resource->deallocate(ptr, 0, 0); }
  };

 public:
  using allocator_type = boost::container::pmr::polymorphic_allocator<std::byte>;
  using value_type = std::byte;
  using BufferType = std::unique_ptr<value_type[], freeobj>; //this gives us proper default move semantics for free

  Stack() = default;
  Stack(Stack&&) = default;
  Stack(Stack&) = delete;
  Stack& operator=(Stack&) = delete;
  Stack& operator=(Stack&&) = default;

  value_type* data() const { return buffer.get(); }
  size_t size() const { return bufferSize; }
  allocator_type get_allocator() const { return allocator; }
  const BaseHeader* first() const { return reinterpret_cast<const BaseHeader*>(this->data()); }
  static const BaseHeader* firstHeader(std::byte const* buf) { return BaseHeader::get(buf); }
  static const BaseHeader* lastHeader(std::byte const* buf)
  {
    const BaseHeader* last{firstHeader(buf)};
    while (last && last->flagsNextHeader) {
      last = last->next();
    }
    return last;
  }
  static size_t headerStackSize(std::byte const* buf)
  {
    size_t result = 0;
    const BaseHeader* last{firstHeader(buf)};
    if (last) {
      while (last->flagsNextHeader) {
        result += last->size();
        last = last->next();
      }
      result += last->size();
    }
    return result;
  }

  //______________________________________________________________________________________________
  /// The magic constructors: take arbitrary number of headers and serialize them
  /// into the buffer allocated by the specified polymorphic allocator. By default
  /// allocation is done using new_delete_resource.
  /// In the final stack the first header must be DataHeader.
  /// all headers must derive from BaseHeader, in addition also other stacks can be passed to ctor.
  template <typename FirstArgType, typename... Headers,
            typename std::enable_if_t<
              !std::is_convertible<FirstArgType, boost::container::pmr::polymorphic_allocator<std::byte>>::value, int> = 0>
  Stack(FirstArgType&& firstHeader, Headers&&... headers)
    : Stack(boost::container::pmr::new_delete_resource(), std::forward<FirstArgType>(firstHeader),
            std::forward<Headers>(headers)...)
  {
  }

  //______________________________________________________________________________________________
  template <typename... Headers>
  Stack(const allocator_type allocatorArg, Headers&&... headers)
    : allocator{allocatorArg},
      bufferSize{calculateSize(std::forward<Headers>(headers)...)},
      buffer{static_cast<std::byte*>(allocator.resource()->allocate(bufferSize, alignof(std::max_align_t))), freeobj{allocator.resource()}}
  {
    if constexpr (sizeof...(headers) > 1) {
      injectAll(buffer.get(), std::forward<Headers>(headers)...);
    } else if (sizeof...(headers) == 1) {
      injectBool(buffer.get(), std::forward<Headers>(headers)..., false);
    }
  }

  //______________________________________________________________________________________________
  template <typename T, typename... Args>
  constexpr static size_t calculateSize(T&& h, Args&&... args) noexcept
  {
    return calculateSize(std::forward<T>(h)) + calculateSize(std::forward<Args>(args)...);
  }

  //______________________________________________________________________________________________
  template <typename T>
  constexpr static size_t calculateSize(T&& h) noexcept
  {
    //if it's a pointer (to a stack) traverse it
    if constexpr (std::is_convertible_v<T, std::byte*>) {
      const BaseHeader* next = BaseHeader::get(std::forward<T>(h));
      if (!next) {
        return 0;
      }
      size_t size = next->size();
      while ((next = next->next())) {
        size += next->size();
      }
      return size;
      //otherwise get the size directly
    } else {
      return h.size();
    }
  }

  //recursion terminator
  constexpr static size_t calculateSize() { return 0; }

 private:
  allocator_type allocator{boost::container::pmr::new_delete_resource()};
  size_t bufferSize{0};
  BufferType buffer{nullptr, freeobj{allocator.resource()}};

  //______________________________________________________________________________________________
  template <typename T>
  static std::byte* injectBool(std::byte* here, T&& h, bool more) noexcept
  {
    using headerType = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
    if (here == nullptr) {
      return nullptr;
    }
    if constexpr (std::is_same_v<headerType, Stack>) {
      if (h.data() == nullptr) {
        return here;
      }
      std::copy(h.data(), h.data() + h.size(), here);
      BaseHeader* last = const_cast<BaseHeader*>(lastHeader(here));
      if (!last) {
        return here;
      }
      last->flagsNextHeader = more;
      return here + h.size();
    } else if constexpr (std::is_same_v<BaseHeader, headerType>) {
      std::copy(h.data(), h.data() + h.size(), here);
      reinterpret_cast<BaseHeader*>(here)->flagsNextHeader = more;
      return here + h.size();
    } else if constexpr (std::is_base_of_v<BaseHeader, headerType>) {
      ::new (static_cast<void*>(here)) headerType(std::forward<T>(h));
      reinterpret_cast<BaseHeader*>(here)->flagsNextHeader = more;
      return here + h.size();
    } else if constexpr (std::is_same_v<headerType, std::byte*>) {
      BaseHeader* from{BaseHeader::get(h)};
      BaseHeader* last{nullptr};
      while (from) {
        last = reinterpret_cast<BaseHeader*>(here);
        std::copy(from->data(), from->data() + from->size(), here);
        here += from->size();
        from = from->next();
      };
      if (last) {
        last->flagsNextHeader = more;
      }
      return here;
    } else {
      static_assert(true, "Stack can only be constructed from other stacks and BaseHeader derived classes");
    }
  }

  //______________________________________________________________________________________________
  template <typename T, typename... Args>
  static std::byte* injectAll(std::byte* here, T&& h, Args&&... args) noexcept
  {
    bool more = hasNonEmptyArg(args...);
    auto alsohere = injectBool(here, h, more);
    if constexpr (sizeof...(args) > 1) {
      return injectAll(alsohere, args...);
    } else {
      return injectBool(alsohere, args..., false);
    }
  }

  //______________________________________________________________________________________________
  // helper function to check if there is at least one non-empty header/stack in the argument pack
  template <typename T, typename... Args>
  static bool hasNonEmptyArg(const T& h, const Args&... args) noexcept
  {
    if (h.size() > 0) {
      return true;
    }
    return hasNonEmptyArg(args...);
  }

  //______________________________________________________________________________________________
  template <typename T>
  static bool hasNonEmptyArg(const T& h) noexcept
  {
    if constexpr (std::is_convertible_v<T, std::byte*>) {
      return get<BaseHeader*>(h);
    } else {
      if (h.size() > 0) {
        return true;
      }
      return false;
    };
  }
};

} // namespace header
} // namespace o2

#endif // HEADERS_STACK_H
