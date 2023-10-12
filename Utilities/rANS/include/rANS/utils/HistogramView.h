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

/// \file HistogramView.h
/// \brief
/// \author michael.lettrich@cern.ch

#ifndef INCLUDE_RANS_UTILS_HISTOGRAMVIEW_H_
#define INCLUDE_RANS_UTILS_HISTOGRAMVIEW_H_

#include <iterator>
#include <algorithm>

#include <fairlogger/Logger.h>

#include "rANS/definitions.h"
#include "rANS/internal/helper.h"

namespace o2
{
namespace rans
{
namespace utils
{

template <typename Hist_IT>
class HistogramView
{
 public:
  HistogramView(Hist_IT begin, Hist_IT end, std::ptrdiff_t offset = 0) : mBegin{begin}, mEnd{end}, mOffset{offset} {};

  inline size_t size() const { return std::distance(mBegin, mEnd); };

  inline bool empty() const { return mBegin == mEnd; };

  inline std::ptrdiff_t getOffset() const noexcept { return mOffset; };

  inline std::ptrdiff_t getMin() const noexcept { return this->getOffset(); };

  inline std::ptrdiff_t getMax() const { return this->getOffset() + std::max(0, static_cast<int32_t>(this->size()) - 1); };

  inline Hist_IT begin() const { return mBegin; };

  inline Hist_IT end() const { return mEnd; };

  inline auto rbegin() const { return std::make_reverse_iterator(this->end()); };

  inline auto rend() const { return std::make_reverse_iterator(this->begin()); };

 private:
  Hist_IT mBegin;
  Hist_IT mEnd;
  std::ptrdiff_t mOffset{};
};

template <typename Hist_IT>
HistogramView<Hist_IT> trim(const HistogramView<Hist_IT>& buffer)
{
  auto isZero = [](count_t i) { return i == 0; };
  auto nonZeroBegin = std::find_if_not(buffer.begin(), buffer.end(), isZero);
  auto nonZeroEnd = std::find_if_not(buffer.rbegin(), buffer.rend(), isZero).base();

  if (std::distance(nonZeroBegin, nonZeroEnd) < 0) {
    return {buffer.end(), buffer.end(), buffer.getOffset()};
  } else {
    const ptrdiff_t newOffset = buffer.getMin() + std::distance(buffer.begin(), nonZeroBegin);
    return {nonZeroBegin, nonZeroEnd, newOffset};
  }
};

template <typename HistA_IT, typename HistB_IT>
inline std::ptrdiff_t leftOffset(const HistogramView<HistA_IT>& histA, const HistogramView<HistB_IT>& histB) noexcept
{
  return histB.getOffset() - histA.getOffset();
}

template <typename HistA_IT, typename HistB_IT>
inline std::ptrdiff_t rightOffset(const HistogramView<HistA_IT>& histA, const HistogramView<HistB_IT>& histB)
{
  return histB.getMax() - histA.getMax();
}

template <typename HistA_IT, typename HistB_IT>
HistogramView<HistA_IT> intersection(const HistogramView<HistA_IT>& histA, const HistogramView<HistB_IT>& histB)
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
    const ptrdiff_t leftOffset = utils::leftOffset(histA, histB);
    const ptrdiff_t rightOffset = utils::rightOffset(histA, histB);
    HistA_IT begin = leftOffset > 0 ? internal::advanceIter(histA.begin(), leftOffset) : histA.begin();
    HistA_IT end = rightOffset < 0 ? internal::advanceIter(histA.end(), rightOffset) : histA.end();
    std::ptrdiff_t offset = leftOffset > 0 ? histA.getOffset() + leftOffset : histA.getOffset();
    return {begin, end, offset};
  }
} // namespace utils

template <typename HistA_IT, typename HistB_IT>
HistogramView<HistA_IT> leftTail(const HistogramView<HistA_IT>& histA, const HistogramView<HistB_IT>& histB)
{
  if (histA.empty() || histB.empty()) {
    return histA;
  }

  const ptrdiff_t leftOffset = utils::leftOffset(histA, histB);
  if (leftOffset <= 0) {
    // case 1 no left difference
    return {histA.end(), histA.end(), 0};
  } else if (histA.getMin() > histB.getMax()) {
    // case 2 disjoint
    return histA;
  } else {
    // case 3 0 < leftOffset <= histA.size()
    return {histA.begin(), internal::advanceIter(histA.begin(), leftOffset), histA.getOffset()};
  }
};

template <typename HistA_IT, typename HistB_IT>
HistogramView<HistA_IT> rightTail(const HistogramView<HistA_IT>& histA, const HistogramView<HistB_IT>& histB)
{
  if (histA.empty() || histB.empty()) {
    return histA;
  }

  const ptrdiff_t rightOffset = utils::rightOffset(histA, histB);

  if (rightOffset > 0) {
    // case 1 no right tail
    return {histA.end(), histA.end(), 0};
  } else if (histA.getMax() < histB.getMin()) {
    // case 2 disjoint
    return histA;
  } else {
    // case 3 0 < -rightOffset <= histA.size()
    auto newBegin = internal::advanceIter(histA.end(), rightOffset);
    return {newBegin, histA.end(), *newBegin};
  }
};

} // namespace utils
} // namespace rans
} // namespace o2

#endif /* INCLUDE_RANS_UTILS_HISTOGRAMVIEW_H_ */