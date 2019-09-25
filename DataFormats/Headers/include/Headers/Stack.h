// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
    void operator()(o2::byte* ptr) { resource->deallocate(ptr, 0, 0); }
  };

 public:
  using allocator_type = boost::container::pmr::polymorphic_allocator<o2::byte>;
  using value_type = o2::byte;
  using BufferType = std::unique_ptr<value_type[], freeobj>; //this gives us proper default move semantics for free

  Stack() = default;
  Stack(Stack&&) = default;
  Stack(Stack&) = delete;
  Stack& operator=(Stack&) = delete;
  Stack& operator=(Stack&&) = default;

  value_type* data() const { return buffer.get(); }
  size_t size() const { return bufferSize; }
  allocator_type get_allocator() const { return allocator; }

  //

  /// The magic constructors: take arbitrary number of headers and serialize them
  /// into the buffer buffer allocated by the specified polymorphic allocator. By default
  /// allocation is done using new_delete_resource.
  /// In the final stack the first header must be DataHeader.
  /// all headers must derive from BaseHeader, in addition also other stacks can be passed to ctor.
  template <typename FirstArgType, typename... Headers,
            typename std::enable_if_t<
              !std::is_convertible<FirstArgType, boost::container::pmr::polymorphic_allocator<o2::byte>>::value, int> = 0>
  Stack(FirstArgType&& firstHeader, Headers&&... headers)
    : Stack(boost::container::pmr::new_delete_resource(), std::forward<FirstArgType>(firstHeader),
            std::forward<Headers>(headers)...)
  {
  }

  template <typename... Headers>
  Stack(const allocator_type allocatorArg, Headers&&... headers)
    : allocator{allocatorArg},
      bufferSize{calculateSize(std::forward<Headers>(headers)...)},
      buffer{static_cast<o2::byte*>(allocator.resource()->allocate(bufferSize, alignof(std::max_align_t))),
             freeobj(allocator.resource())}
  {
    inject(buffer.get(), std::forward<Headers>(headers)...);
  }

 private:
  allocator_type allocator{boost::container::pmr::new_delete_resource()};
  size_t bufferSize{0};
  BufferType buffer{nullptr, freeobj{allocator.resource()}};

  template <typename T, typename... Args>
  static size_t calculateSize(T&& h, Args&&... args) noexcept
  {
    return calculateSize(std::forward<T>(h)) + calculateSize(std::forward<Args>(args)...);
  }

  template <typename T>
  static size_t calculateSize(T&& h) noexcept
  {
    return h.size();
  }

  //recursion terminator
  constexpr static size_t calculateSize() { return 0; }

  template <typename T>
  static o2::byte* inject(o2::byte* here, T&& h) noexcept
  {
    using headerType = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
    static_assert(
      std::is_base_of<BaseHeader, headerType>::value == true || std::is_same<Stack, headerType>::value == true,
      "header stack parameters are restricted to stacks and headers derived from BaseHeader");
    std::copy(h.data(), h.data() + h.size(), here);
    return here + h.size();
    // somehow could not trigger copy elision for placed construction, TODO: check out if this is possible here
    // headerType* placed = new (here) headerType(std::forward<T>(h));
    // return here + placed->size();
  }

  template <typename T, typename... Args>
  static o2::byte* inject(o2::byte* here, T&& h, Args&&... args) noexcept
  {
    auto alsohere = inject(here, h);
    // the type might be a stack itself, loop through headers and set the flag in the last one
    if (h.size() > 0) {
      BaseHeader* next = BaseHeader::get(here);
      while (next->flagsNextHeader) {
        next = next->next();
      }
      next->flagsNextHeader = hasNonEmptyArg(args...);
    }
    return inject(alsohere, args...);
  }

  // helper function to check if there is at least one non-empty header/stack in the argument pack
  template <typename T, typename... Args>
  static bool hasNonEmptyArg(const T& h, const Args&... args) noexcept
  {
    if (h.size() > 0) {
      return true;
    }
    return hasNonEmptyArg(args...);
  }

  template <typename T>
  static bool hasNonEmptyArg(const T& h) noexcept
  {
    if (h.size() > 0) {
      return true;
    }
    return false;
  }
};

} // namespace header
} // namespace o2

#endif // HEADERS_STACK_H
