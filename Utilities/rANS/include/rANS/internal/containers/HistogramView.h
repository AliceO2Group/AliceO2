// Copyright 2019-2023 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   HistogramView.h
/// @author michael.lettrich@cern.ch
/// @brief  Non-owning, lightweight structure for histogram manipulation

#ifndef RANS_INTERNAL_CONTAINERS_HISTOGRAMVIEW_H_
#define RANS_INTERNAL_CONTAINERS_HISTOGRAMVIEW_H_

#include <iterator>
#include <algorithm>
#include <ostream>
#include <iterator>

#include <fairlogger/Logger.h>

#include "rANS/internal/common/utils.h"
#include "rANS/internal/transform/algorithm.h"

namespace o2::rans
{

template <typename Hist_IT>
class HistogramView
{
 public:
  using size_type = size_t;
  using value_type = typename std::iterator_traits<Hist_IT>::value_type;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using difference_type = std::ptrdiff_t;
  using iterator = Hist_IT;
  using reverse_iterator = std::reverse_iterator<iterator>;

  HistogramView() = default;
  HistogramView(const HistogramView&) = default;
  HistogramView(HistogramView&&) = default;
  HistogramView& operator=(const HistogramView&) = default;
  HistogramView& operator=(HistogramView&&) = default;
  ~HistogramView() = default;

  HistogramView(iterator begin, iterator end, difference_type offset = 0) : mBegin{begin}, mEnd{end}, mOffset{offset} {};

  [[nodiscard]] inline size_t size() const { return std::distance(mBegin, mEnd); };

  [[nodiscard]] inline bool empty() const { return mBegin == mEnd; };

  [[nodiscard]] inline difference_type getOffset() const noexcept { return mOffset; };

  [[nodiscard]] inline difference_type getMin() const noexcept { return this->getOffset(); };

  [[nodiscard]] inline difference_type getMax() const { return this->getOffset() + static_cast<difference_type>(this->size() - !this->empty()); };

  [[nodiscard]] inline iterator begin() const { return mBegin; };

  [[nodiscard]] inline iterator end() const { return mEnd; };

  [[nodiscard]] inline reverse_iterator rbegin() const { return std::make_reverse_iterator(this->end()); };

  [[nodiscard]] inline reverse_iterator rend() const { return std::make_reverse_iterator(this->begin()); };

  friend std::ostream& operator<<(std::ostream& os, const HistogramView& view)
  {
    os << fmt::format("HistogramView: size {}, offset {}", view.size(), view.getOffset());
    return os;
  };

  [[nodiscard]] inline const value_type& operator[](difference_type idx) const
  {
    auto iter = utils::advanceIter(mBegin, idx - this->getOffset());

    assert(iter >= this->begin());
    assert(iter < this->end());
    return *iter;
  }

  [[nodiscard]] inline const_pointer data() const
  {
    assert(mBegin != mEnd);
    return &(*mBegin);
  }

 private:
  iterator mBegin{};
  iterator mEnd{};
  difference_type mOffset{};

  static_assert(std::is_same_v<typename std::iterator_traits<Hist_IT>::iterator_category, std::random_access_iterator_tag>, "This template is defined only for random access iterators");
};

template <typename Hist_IT>
[[nodiscard]] HistogramView<Hist_IT> trim(const HistogramView<Hist_IT>& buffer)
{

  using value_type = typename HistogramView<Hist_IT>::value_type;

  auto isZero = [](const value_type& i) { return i == value_type{}; };
  auto nonZeroBegin = std::find_if_not(buffer.begin(), buffer.end(), isZero);
  auto nonZeroEnd = nonZeroBegin == buffer.end() ? buffer.end() : std::find_if_not(std::make_reverse_iterator(buffer.end()), std::make_reverse_iterator(buffer.begin()), isZero).base();

  std::ptrdiff_t newOffset;
  if (nonZeroBegin == nonZeroEnd) {
    newOffset = buffer.getOffset();
  } else {
    newOffset = buffer.getMin() + std::distance(buffer.begin(), nonZeroBegin);
  }

  return {nonZeroBegin, nonZeroEnd, newOffset};
};

template <typename HistA_IT, typename HistB_IT>
[[nodiscard]] inline std::ptrdiff_t getLeftOffset(const HistogramView<HistA_IT>& histA, const HistogramView<HistB_IT>& histB) noexcept
{
  return histB.getOffset() - histA.getOffset();
}

template <typename HistA_IT, typename HistB_IT>
[[nodiscard]] inline std::ptrdiff_t getRightOffset(const HistogramView<HistA_IT>& histA, const HistogramView<HistB_IT>& histB)
{
  return histB.getMax() - histA.getMax();
}

template <typename HistA_IT, typename HistB_IT>
[[nodiscard]] HistogramView<HistA_IT> getIntersection(const HistogramView<HistA_IT>& histA, const HistogramView<HistB_IT>& histB)
{
  auto mkEmpty = [&histA]() { return HistogramView(histA.end(), histA.end(), 0); };

  if (histA.empty() || histB.empty()) {
    // one is empty
    return mkEmpty();
  } else if (histA.getMin() > histB.getMax() || histA.getMax() < histB.getMin()) {
    // disjoint
    return mkEmpty();
  } else {
    // intersecting
    const std::ptrdiff_t leftOffset = getLeftOffset(histA, histB);
    const std::ptrdiff_t rightOffset = getRightOffset(histA, histB);
    HistA_IT begin = leftOffset > 0 ? utils::advanceIter(histA.begin(), leftOffset) : histA.begin();
    HistA_IT end = rightOffset < 0 ? utils::advanceIter(histA.end(), rightOffset) : histA.end();
    std::ptrdiff_t offset = leftOffset > 0 ? histA.getOffset() + leftOffset : histA.getOffset();
    return {begin, end, offset};
  }
} // namespace utils

template <typename HistA_IT, typename HistB_IT>
[[nodiscard]] HistogramView<HistA_IT> getLeftTail(const HistogramView<HistA_IT>& histA, const HistogramView<HistB_IT>& histB)
{
  if (histA.empty() || histB.empty()) {
    return histA;
  }

  const std::ptrdiff_t leftOffset = getLeftOffset(histA, histB);
  if (leftOffset <= 0) {
    // case 1 no left difference
    return {histA.end(), histA.end(), 0};
  } else if (histA.getMin() > histB.getMax()) {
    // case 2 disjoint
    return histA;
  } else {
    // case 3 0 < leftOffset <= histA.size()
    return {histA.begin(), utils::advanceIter(histA.begin(), leftOffset), histA.getOffset()};
  }
};

template <typename HistA_IT, typename HistB_IT>
[[nodiscard]] HistogramView<HistA_IT> getRightTail(const HistogramView<HistA_IT>& histA, const HistogramView<HistB_IT>& histB)
{
  if (histA.empty() || histB.empty()) {
    return histA;
  }

  const std::ptrdiff_t rightOffset = getRightOffset(histA, histB);

  if (rightOffset > 0) {
    // case 1 no right tail
    return {histA.end(), histA.end(), 0};
  } else if (histA.getMax() < histB.getMin()) {
    // case 2 disjoint
    return histA;
  } else {
    // case 3 0 < -rightOffset <= histA.size()
    auto newBegin = utils::advanceIter(histA.end(), rightOffset);
    return {newBegin, histA.end(), *newBegin};
  }
};

template <typename container_T>
inline auto makeHistogramView(container_T& container, std::ptrdiff_t offset) noexcept -> HistogramView<decltype(std::begin(container))>
{
  return {std::begin(container), std::end(container), offset};
}

template <typename container_T>
inline auto makeHistogramView(const container_T& container, std::ptrdiff_t offset) noexcept -> HistogramView<decltype(std::cbegin(container))>
{
  return {std::cbegin(container), std::cend(container), offset};
}

namespace histogramview_impl
{

template <class, class = void>
struct has_getOffset : std::false_type {
};

template <class T>
struct has_getOffset<T, std::void_t<decltype(std::declval<T>().getOffset())>> : std::true_type {
};

template <typename T>
inline constexpr bool has_getOffset_v = has_getOffset<T>::value;

} // namespace histogramview_impl

template <typename container_T, std::enable_if_t<histogramview_impl::has_getOffset_v<container_T>, bool> = true>
inline auto makeHistogramView(const container_T& container) noexcept -> HistogramView<decltype(std::cbegin(container))>
{
  return {std::cbegin(container), std::cend(container), container.getOffset()};
}

template <typename container_T, std::enable_if_t<histogramview_impl::has_getOffset_v<container_T>, bool> = true>
inline auto makeHistogramView(container_T& container) noexcept -> HistogramView<decltype(std::begin(container))>
{
  return {std::begin(container), std::end(container), container.getOffset()};
}

} // namespace o2::rans

#endif /* RANS_INTERNAL_CONTAINERS_HISTOGRAMVIEW_H_ */