// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/*
 * ShmAllocator.h
 *
 *  Created on: Jun 18, 2018
 *      Author: swenzel
 */

#ifndef COMMON_UTILS_INCLUDE_COMMONUTILS_SHMALLOCATOR_H_
#define COMMON_UTILS_INCLUDE_COMMONUTILS_SHMALLOCATOR_H_

#include "CommonUtils/ShmManager.h"
#include <cassert>

namespace o2
{
namespace utils
{

// An allocator placing objects in shared memory as organized by ShmManager.
// Allocator used in a few std::vector so that they create stuff in shared mem
// The only non-trivial things are in methods allocate/deallocate
template <typename T>
class ShmAllocator
{
 public:
  typedef T value_type;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;

  typedef T* pointer;
  typedef const T* const_pointer;

  typedef T& reference;
  typedef const T& const_reference;

 public:
  inline ShmAllocator() throw() {}

  template <typename T2>
  inline ShmAllocator(const ShmAllocator<T2>&) throw()
  {
  }

  inline ~ShmAllocator() throw() {}

  inline pointer adress(reference r) { return &r; }

  inline const_pointer adress(const_reference r) const { return &r; }

  // the actually important functions:
  inline pointer allocate(size_type n)
  {
    auto& instance = ShmManager::Instance();
    if (instance.readyToAllocate()) {
      return (pointer)ShmManager::Instance().getmemblock(sizeof(value_type) * n);
    }
    return (pointer)malloc(sizeof(value_type) * n);
  }
  inline void deallocate(pointer p, size_type s)
  {
    auto& instance = ShmManager::Instance();
    if (instance.readyToAllocate()) {
      ShmManager::Instance().freememblock(p, s);
    } else {
      free(p);
    }
  }

  inline void construct(pointer p, const value_type& value)
  {
    new (p) value_type(value);
  }

  template <class U, class... Args>
  void construct(U* p, Args&&... args)
  {
    ::new ((void*)p) U(std::forward<Args>(args)...);
  }

  inline void destroy(pointer p) { p->~value_type(); }

  inline size_type max_size() const throw() { return size_type(-1) / sizeof(value_type); }

  template <typename T2>
  struct rebind {
    typedef ShmAllocator<T2> other;
  };

  bool operator!=(const ShmAllocator<T>& other) const { return !(*this == other); }

  // Returns true if and only if storage allocated from *this
  // can be deallocated from other, and vice versa.
  // Always returns true for stateless allocators.
  bool operator==(const ShmAllocator<T>& /*other*/) const { return true; }
};

template <typename T>
std::vector<T>* createSimVector()
{
  using vector_t = std::vector<T>;
  auto& instance = o2::utils::ShmManager::Instance();
  if (instance.isOperational() && instance.readyToAllocate()) {
    auto placement = instance.hasSegment() ? instance.getmemblock(sizeof(vector_t)) : malloc(sizeof(vector_t));
    return new (placement) vector_t;
  } else {
    return new vector_t;
  }
}

template <typename T>
void freeSimVector(std::vector<T>* ptr)
{
  return;
  using vector_t = std::vector<T>;
#ifdef USESHM
  auto& instance = o2::utils::ShmManager::Instance();
  ptr->clear();
  ptr->shrink_to_fit();
  if (instance.hasSegment() && instance.isPointerOk(ptr)) {
    instance.freememblock(ptr);
  } else {
    free(ptr);
  }
// at this moment we have to trust that std::
#else
  delete ptr;
#endif
}

} // end namespace utils
} // end namespace o2
#endif /* COMMON_UTILS_INCLUDE_COMMONUTILS_SHMALLOCATOR_H_ */
