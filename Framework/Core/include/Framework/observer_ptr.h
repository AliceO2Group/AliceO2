// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef FRAMEWORK_OBSERVER_PTR_H
#define FRAMEWORK_OBSERVER_PTR_H

#include <cstddef>
#include <type_traits>

namespace o2
{

template <typename W>
class observer_ptr
{
 public:
  using element_type = W;

  constexpr observer_ptr() noexcept = default;
  constexpr observer_ptr(std::nullptr_t) noexcept {}
  explicit observer_ptr(element_type* ptr) noexcept : mptr{ ptr } {}
  template <typename W2, typename std::enable_if<std::is_convertible<element_type*, W2*>::value, int>::type = 1>
  observer_ptr(observer_ptr<W2> other) noexcept : mptr{ other.get() }
  {
  }
  observer_ptr(const observer_ptr& other) = default;
  observer_ptr(observer_ptr&& other) = default;

  constexpr element_type* release() noexcept
  {
    auto tmp = *this;
    this->reset();
    return tmp.get();
  }
  constexpr void reset(element_type* p = nullptr) noexcept { *this = p; }
  constexpr void swap(observer_ptr& other) noexcept
  {
    observer_ptr<element_type> tmp(*this);
    *this = other;
    other = tmp;
  };
  constexpr void swap(std::nullptr_t) noexcept
  {
    *this = nullptr;
  };
  constexpr element_type* get() const noexcept { return mptr; }
  constexpr std::add_lvalue_reference_t<element_type> operator*() const { return *get(); }
  constexpr element_type* operator->() const noexcept { return get(); }
  constexpr std::add_lvalue_reference_t<observer_ptr<element_type>> operator=(const std::add_lvalue_reference_t<observer_ptr<element_type>> other) { mptr = other.mptr; }
  constexpr std::add_lvalue_reference_t<observer_ptr<element_type>> operator=(element_type* const other) { mptr = other; }

  constexpr explicit operator element_type*() const noexcept { return get(); }
  constexpr explicit operator bool() const noexcept { return get() != nullptr; }

 private:
  element_type* mptr{ nullptr };
};

} //namespace o2

#endif
