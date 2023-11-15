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

/// @file   dictCompression.h
/// @author michael.lettrich@cern.ch
/// @brief

#ifndef RANS_INTERNAL_PACK_DICTCOMPRESSION_H_
#define RANS_INTERNAL_PACK_DICTCOMPRESSION_H_

#include <type_traits>
#include <cstdint>
#include <stdexcept>

#include "rANS/internal/common/utils.h"
#include "rANS/internal/pack/eliasDelta.h"
#include "rANS/internal/common/exceptions.h"

namespace o2::rans::internal
{

template <typename buffer_IT>
[[nodiscard]] inline constexpr BitPtr seekEliasDeltaEnd(buffer_IT begin, buffer_IT end)
{
  using value_type = uint64_t;
  assert(end >= begin);

  for (buffer_IT iter = end; iter-- != begin;) {
    auto value = static_cast<value_type>(*iter);
    if (value > 0) {
      const intptr_t offset = utils::toBits<value_type>() - __builtin_clzl(value);
      return {iter, offset};
    }
  }

  return {};
};

[[nodiscard]] inline constexpr intptr_t getEliasDeltaOffset(BitPtr begin, BitPtr iter)
{
  assert(iter >= begin);
  intptr_t delta = (iter - begin);
  assert(delta > 0);
  return std::min<intptr_t>(delta, EliasDeltaDecodeMaxBits);
}

template <typename source_T>
class DictionaryStreamParser
{
 public:
  using source_type = source_T;
  using count_type = count_t;
  using value_type = std::pair<source_type, count_type>;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;

  template <typename buffer_IT>
  DictionaryStreamParser(buffer_IT begin, buffer_IT end, source_type max);

  [[nodiscard]] count_type getIncompressibleSymbolFrequency() const;
  [[nodiscard]] bool hasNext() const;
  [[nodiscard]] value_type getNext();
  [[nodiscard]] inline source_type getIndex() const noexcept { return mIndex; };

 private:
  template <typename buffer_IT>
  [[nodiscard]] BitPtr seekStreamEnd(buffer_IT begin, buffer_IT end) const;
  [[nodiscard]] count_type decodeNext();

  bool mIsFirst{true};
  BitPtr mPos{};
  BitPtr mEnd{};
  source_type mIndex{};
  count_type mIncompressibleSymbolFrequency{};
};

template <typename source_T>
template <typename buffer_IT>
DictionaryStreamParser<source_T>::DictionaryStreamParser(buffer_IT begin, buffer_IT end, source_type max) : mPos{end}, mEnd{begin}, mIndex(max)
{
  static_assert(std::is_pointer_v<buffer_IT>, "can only deserialize from raw pointers");

  mPos = seekEliasDeltaEnd(begin, end);
  if (mPos == BitPtr{}) {
    throw ParsingError{"failed to read renormed dictionary: could not find end of data stream"};
  }
  if (decodeNext() != 1) {
    throw ParsingError{"failed to read renormed dictionary: could not find end of stream delimiter"};
  }
  mIncompressibleSymbolFrequency = decodeNext() - 1;
}

template <typename source_T>
template <typename buffer_IT>
[[nodiscard]] BitPtr DictionaryStreamParser<source_T>::seekStreamEnd(buffer_IT begin, buffer_IT end) const
{
  using value_type = uint64_t;
  assert(end >= begin);

  for (buffer_IT iter = end; iter-- != begin;) {
    auto value = static_cast<value_type>(*iter);
    if (value > 0) {
      const intptr_t offset = utils::toBits<value_type>() - __builtin_clzl(value);
      return {iter, offset};
    }
  }

  return {};
};

template <typename source_T>
[[nodiscard]] inline auto DictionaryStreamParser<source_T>::decodeNext() -> count_type
{
  assert(mPos >= mEnd);
  intptr_t delta = getEliasDeltaOffset(mEnd, mPos);
  return eliasDeltaDecode<count_type>(mPos, delta);
};

template <typename source_T>
[[nodiscard]] inline auto DictionaryStreamParser<source_T>::getIncompressibleSymbolFrequency() const -> count_type
{
  return mIncompressibleSymbolFrequency;
}

template <typename source_T>
[[nodiscard]] inline bool DictionaryStreamParser<source_T>::hasNext() const
{
  assert(mPos >= mEnd);
  return mPos != mEnd;
}

template <typename source_T>
[[nodiscard]] auto DictionaryStreamParser<source_T>::getNext() -> value_type
{
  assert(hasNext());
  count_type frequency{};

  if (mIsFirst) {
    frequency = decodeNext();
    mIsFirst = false;
  } else {
    const auto offset = decodeNext();
    frequency = decodeNext();
    mIndex -= offset;
  }
  return {mIndex, frequency};
}

} // namespace o2::rans::internal

#endif /* RANS_INTERNAL_PACK_DICTCOMPRESSION_H_ */