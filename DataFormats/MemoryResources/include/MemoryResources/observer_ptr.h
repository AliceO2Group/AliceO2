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
  explicit observer_ptr(element_type* ptr) noexcept : mptr{ptr} {}
  template <typename W2, typename std::enable_if<std::is_convertible<W2*, element_type*>::value, int>::type = 1>
  observer_ptr(observer_ptr<W2> other) noexcept : mptr{other.get()}
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
  constexpr std::add_lvalue_reference_t<observer_ptr<element_type>> operator=(const std::add_lvalue_reference_t<observer_ptr<element_type>> other)
  {
    mptr = other.mptr;
    return *this;
  }
  constexpr std::add_lvalue_reference_t<observer_ptr<element_type>> operator=(element_type* const other)
  {
    mptr = other;
    return *this;
  }

  constexpr explicit operator element_type*() const noexcept { return get(); }
  constexpr explicit operator bool() const noexcept { return get() != nullptr; }

 private:
  element_type* mptr{nullptr};
};

template <typename W>
observer_ptr<W> make_observer(W* p) noexcept
{
  return observer_ptr(p);
}

template <class W1, class W2>
bool operator==(const observer_ptr<W1>& p1, const observer_ptr<W2>& p2)
{
  return p1.get() == p2.get();
}

template <class W1, class W2>
bool operator!=(const observer_ptr<W1>& p1, const observer_ptr<W2>& p2)
{
  return !(p1 == p2);
}

template <class W>
bool operator==(const observer_ptr<W>& p, std::nullptr_t) noexcept
{
  return !p;
}

template <class W>
bool operator==(std::nullptr_t, const observer_ptr<W>& p) noexcept
{
  return !p;
}

template <class W>
bool operator!=(const observer_ptr<W>& p, std::nullptr_t) noexcept
{
  return static_cast<bool>(p);
}

template <class W>
bool operator!=(std::nullptr_t, const observer_ptr<W>& p) noexcept
{
  return static_cast<bool>(p);
}

template <class W1, class W2>
bool operator<(const observer_ptr<W1>& p1, const observer_ptr<W2>& p2)
{
  return p1.get() < p2.get();
}

template <class W1, class W2>
bool operator>(const observer_ptr<W1>& p1, const observer_ptr<W2>& p2)
{
  return p1.get() > p2.get();
}

template <class W1, class W2>
bool operator<=(const observer_ptr<W1>& p1, const observer_ptr<W2>& p2)
{
  return p1.get() <= p2.get();
}

template <class W1, class W2>
bool operator>=(const observer_ptr<W1>& p1, const observer_ptr<W2>& p2)
{
  return p1.get() >= p2.get();
}

} //namespace o2

namespace std
{
template <class W>
void swap(o2::observer_ptr<W>& lhs, o2::observer_ptr<W>& rhs) noexcept
{
  lhs.swap(rhs);
}
template <class T>
struct hash<o2::observer_ptr<T>> {
  std::size_t operator()(const o2::observer_ptr<T>& in) const noexcept
  {
    return std::hash<T*>()(in.get());
  };
};
} //namespace std

#endif
