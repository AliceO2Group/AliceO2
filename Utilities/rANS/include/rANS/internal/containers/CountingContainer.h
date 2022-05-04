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

/// @file   CountingContainer.h
/// @author Michael Lettrich
/// @brief  container that counts number of samples passed with optimizations for small source alphabets (8, 16 Bits)

#ifndef RANS_INTERNAL_CONTAINERS_COUNTINGCONTAINER_H_
#define RANS_INTERNAL_CONTAINERS_COUNTINGCONTAINER_H_

#include <cstdint>
#include <string>

#include "rANS/internal/containers/Container.h"
#include "rANS/internal/common/utils.h"

#include "fairlogger/Logger.h"

namespace o2
{
namespace rans
{
namespace internal
{

template <class source_T>
class CountingContainerBase : public Container<source_T, uint32_t, CountingContainerBase<source_T>>
{
  using base_type = Container<source_T, uint32_t, CountingContainerBase<source_T>>;
  friend base_type;

 public:
  using source_type = typename base_type::source_type;
  using value_type = typename base_type::value_type;
  using container_type = typename base_type::container_type;
  using size_type = typename base_type::size_type;
  using difference_type = typename base_type::difference_type;
  using reference = typename base_type::reference;
  using const_reference = typename base_type::const_reference;
  using pointer = typename base_type::pointer;
  using const_pointer = typename base_type::const_pointer;
  using const_iterator = typename base_type::const_iterator;
  using iterator = typename base_type::iterator;
  using const_reverse_iterator = typename base_type::const_reverse_iterator;
  using reverse_iterator = typename base_type::reverse_iterator;

  [[nodiscard]] inline bool empty() const noexcept { return mNSamples == 0; };

  [[nodiscard]] inline size_type getNumSamples() const noexcept { return mNSamples; };

  friend void swap(CountingContainerBase& a, CountingContainerBase& b) noexcept
  {
    using std::swap;
    swap(a.mNSamples, b.mNSamples);
    swap(static_cast<typename CountingContainerBase::base_type&>(a),
         static_cast<typename CountingContainerBase::base_type&>(b));
  };

 protected:
  inline bool isValidSymbol(const value_type& value) const noexcept
  {
    return value > 0;
  };

  CountingContainerBase() = default;
  CountingContainerBase(size_type size, source_type offset) : base_type(size, offset){};

  size_type mNSamples{};
};

template <typename source_T, class = void>
class CountingContainer;

template <typename source_T>
class CountingContainer<source_T, std::enable_if_t<(sizeof(source_T) <= 2)>> : public CountingContainerBase<source_T>
{
  using base_type = CountingContainerBase<source_T>;

 public:
  using source_type = typename base_type::source_type;
  using value_type = typename base_type::value_type;
  using container_type = typename base_type::container_type;
  using size_type = typename base_type::size_type;
  using difference_type = typename base_type::difference_type;
  using reference = typename base_type::reference;
  using const_reference = typename base_type::const_reference;
  using pointer = typename base_type::pointer;
  using const_pointer = typename base_type::const_pointer;
  using const_iterator = typename base_type::const_iterator;
  using iterator = typename base_type::iterator;
  using const_reverse_iterator = typename base_type::const_reverse_iterator;
  using reverse_iterator = typename base_type::reverse_iterator;

  [[nodiscard]] inline constexpr size_type size() const noexcept { return internal::pow2(internal::toBits<source_type>()); };

  friend void swap(CountingContainer& a, CountingContainer& b) noexcept
  {
    using std::swap;
    swap(static_cast<typename CountingContainer::base_type&>(a),
         static_cast<typename CountingContainer::base_type&>(b));
  }

 protected:
  CountingContainer() : base_type{this->size(), std::numeric_limits<source_type>::min()} {};
};

template <typename source_T>
class CountingContainer<source_T, std::enable_if_t<(sizeof(source_T) == 4)>> : public CountingContainerBase<source_T>
{
  using base_type = CountingContainerBase<source_T>;

 public:
  using source_type = typename base_type::source_type;
  using value_type = typename base_type::value_type;
  using container_type = typename base_type::container_type;
  using size_type = typename base_type::size_type;
  using difference_type = typename base_type::difference_type;
  using reference = typename base_type::reference;
  using const_reference = typename base_type::const_reference;
  using pointer = typename base_type::pointer;
  using const_pointer = typename base_type::const_pointer;
  using const_iterator = typename base_type::const_iterator;
  using iterator = typename base_type::iterator;
  using const_reverse_iterator = typename base_type::const_reverse_iterator;
  using reverse_iterator = typename base_type::reverse_iterator;

  friend void swap(CountingContainer& a, CountingContainer& b) noexcept
  {
    using std::swap;
    swap(static_cast<typename CountingContainer::base_type&>(a),
         static_cast<typename CountingContainer::base_type&>(b));
  };

 protected:
  CountingContainer() = default;
};

} // namespace internal
} // namespace rans
} // namespace o2

#endif /* RANS_INTERNAL_CONTAINERS_COUNTINGCONTAINER_H_ */
