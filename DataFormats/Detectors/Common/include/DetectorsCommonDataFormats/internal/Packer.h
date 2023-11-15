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

/// \file Packer.h
/// \author michael.lettrich@cern.ch
/// \brief Interfaces for BitPacking using librans

#ifndef ALICEO2_PACKER_H_
#define ALICEO2_PACKER_H_

#ifndef __CLING__
#include "rANS/pack.h"
#include "rANS/metrics.h"
#endif

namespace o2::ctf::internal
{

template <typename source_T>
class Packer
{
 public:
  using source_type = source_T;

  Packer() = default;
#ifndef __CLING__
  explicit Packer(rans::Metrics<source_type>& metrics) : mOffset{metrics.getDatasetProperties().min},
                                                         mPackingWidth{metrics.getDatasetProperties().alphabetRangeBits} {};
#endif

  template <typename source_IT>
  Packer(source_IT srcBegin, source_IT srcEnd);

  [[nodiscard]] inline source_type getOffset() const noexcept { return mOffset; };

  [[nodiscard]] inline size_t getPackingWidth() const noexcept { return mPackingWidth; };

  template <typename buffer_T>
  [[nodiscard]] size_t getPackingBufferSize(size_t messageLength) const noexcept;

  template <typename source_IT, typename dst_T>
  [[nodiscard]] dst_T* pack(source_IT srcBegin, source_IT srcEnd, dst_T* dstBegin, dst_T* dstEnd) const;

  template <typename dst_T>
  [[nodiscard]] dst_T* pack(const source_T* __restrict srcBegin, size_t extent, dst_T* dstBegin, dst_T* dstEnd) const;

 private:
  source_type mOffset{};
  size_t mPackingWidth{};
};

template <typename source_T>
template <typename source_IT>
Packer<source_T>::Packer(source_IT srcBegin, source_IT srcEnd)
{
#ifndef __CLING__
  static_assert(rans::utils::isCompatibleIter_v<source_type, source_IT>);
  if (srcBegin != srcEnd) {

    const auto [min, max] = [&]() {
      if constexpr (std::is_pointer_v<source_IT>) {
        return rans::utils::minmax(gsl::span<const source_type>(srcBegin, srcEnd));
      } else {
        const auto [minIter, maxIter] = std::minmax_element(srcBegin, srcEnd);
        return std::make_pair<source_type>(*minIter, *maxIter);
      }
    }();

    mOffset = min;
    mPackingWidth = rans::utils::getRangeBits(min, max);
  }
#endif
};

template <typename source_T>
template <typename buffer_T>
[[nodiscard]] inline size_t Packer<source_T>::getPackingBufferSize(size_t messageLength) const noexcept
{
#ifndef __CLING__
  return rans::computePackingBufferSize<buffer_T>(messageLength, mPackingWidth);
#else
  return 0;
#endif
};

template <typename source_T>
template <typename dst_T>
[[nodiscard]] inline dst_T* Packer<source_T>::pack(const source_T* __restrict srcBegin, size_t extent, dst_T* dstBegin, dst_T* dstEnd) const
{
  return pack(srcBegin, srcBegin + extent, dstBegin, dstEnd);
}

template <typename source_T>
template <typename source_IT, typename dst_T>
[[nodiscard]] dst_T* Packer<source_T>::pack(source_IT srcBegin, source_IT srcEnd, dst_T* dstBegin, dst_T* dstEnd) const
{
  static_assert(std::is_same_v<source_T, typename std::iterator_traits<source_IT>::value_type>);
  size_t extent = std::distance(srcBegin, srcEnd);

  if (extent == 0) {
    return dstBegin;
  }
#ifndef __CLING__
  rans::BitPtr packEnd = rans::pack(srcBegin, extent, dstBegin, mPackingWidth, mOffset);
  auto* end = packEnd.toPtr<dst_T>();
  ++end; // one past end iterator;
  rans::utils::checkBounds(end, dstEnd);
  return end;
#else
  return nullptr;
#endif
};

} // namespace o2::ctf::internal

#endif /* ALICEO2_PACKER_H_ */