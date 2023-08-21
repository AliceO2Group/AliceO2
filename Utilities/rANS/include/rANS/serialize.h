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

/// @file   serialize.h
/// @author michael.lettrich@cern.ch
/// @brief  public interface for serializing histograms (dictionaries) to JSON or compressed binary.

#ifndef RANS_SERIALIZE_H_
#define RANS_SERIALIZE_H_

#include <type_traits>
#include <cstdint>
#include <stdexcept>

#ifdef RANS_ENABLE_JSON
#include <rapidjson/writer.h>
#endif
#include "rANS/internal/common/utils.h"
#include "rANS/internal/common/typetraits.h"
#include "rANS/internal/containers/HistogramView.h"
#include "rANS/internal/pack/pack.h"
#include "rANS/internal/pack/eliasDelta.h"
#include "rANS/internal/common/exceptions.h"
#include "rANS/internal/transform/algorithm.h"

namespace o2::rans
{

namespace internal
{

template <typename container_T>
inline constexpr count_t getFrequency(const container_T& container, typename container_T::const_reference symbol)
{
  if constexpr (isSymbolTable_v<container_T>) {
    return container.isEscapeSymbol(symbol) ? 0 : symbol.getFrequency();
  } else {
    return symbol;
  }
};

template <typename container_T, std::enable_if_t<isSparseContainer_v<container_T>, bool> = true>
inline constexpr count_t getFrequency(const container_T& container, typename container_T::const_iterator::value_type symbolPair)
{
  return getFrequency(container, symbolPair.second);
};

template <typename container_T, std::enable_if_t<isHashContainer_v<container_T>, bool> = true>
inline constexpr count_t getFrequency(const container_T& container, const typename container_T::const_iterator::value_type& symbolPair)
{
  const auto& symbol = symbolPair.second;
  return getFrequency(container, symbol);
};

template <typename container_T>
inline constexpr count_t getIncompressibleFrequency(const container_T& container) noexcept
{
  if constexpr (isSymbolTable_v<container_T>) {
    return container.getEscapeSymbol().getFrequency();
  } else if constexpr (isRenormedHistogram_v<container_T>) {
    return container.getIncompressibleSymbolFrequency();
  } else {
    return 0;
  }
};

template <typename container_T>
auto getNullElement(const container_T& container) -> typename container_T::value_type
{
  if constexpr (isSymbolTable_v<container_T>) {
    return container.getEscapeSymbol();
  } else {
    return {};
  }
}

template <typename T>
[[nodiscard]] inline constexpr size_t getDictExtent(T min, T max, size_t renormingPrecision) noexcept
{
  assert(max >= min);
  // special case - empty dictionary
  if (renormingPrecision == 0) {
    return 0;
  } else {
    return static_cast<size_t>(max - min) + 1;
  }
};

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

}; // namespace internal

#ifdef RANS_ENABLE_JSON
template <typename container_T, typename jsonBuffer_T>
void toJSON(const container_T& container, rapidjson::Writer<jsonBuffer_T>& writer)
{
  using namespace utils;

  writer.StartObject();
  writer.Key("Offset");
  writer.Int64(container.getOffset());
  writer.Key("Index");
  uint32_t index = 0;
  std::vector<count_t> nonzeroFrequencies;
  writer.StartArray();
  for (auto iter = container.begin(); iter != container.end(); ++iter) {
    auto frequency = getFrequency(container, iter);
    if (frequency > 0) {
      nonzeroFrequencies.push_back(frequency);
      writer.Uint(index);
    }
    ++index;
  }
  writer.EndArray();

  writer.Key("Value");
  writer.StartArray();
  for (auto freq : nonzeroFrequencies) {
    writer.Uint(freq);
  }
  writer.EndArray();

  writer.Key("Incompressible");
  writer.Int64(getIncompressibleFrequency(container));
  writer.EndObject();
};
#endif /* RANS_ENABLE_JSON */

template <typename container_T, typename dest_IT>
dest_IT compressRenormedDictionary(const container_T& container, dest_IT dstBufferBegin)
{
  using namespace internal;
  static_assert(std::is_pointer_v<dest_IT>, "only raw pointers are permited as a target for serialization");
  static_assert((isSymbolTable_v<container_T> || isRenormedHistogram_v<container_T>), "only renormed Histograms and symbol tables are accepted. Non-renormed histograms might not compress well");

  using source_type = typename container_T::source_type;
  using const_iterator = typename container_T::const_iterator;

  BitPtr dstIter{dstBufferBegin};
  const auto [trimmedBegin, trimmedEnd] = trim(container, getNullElement(container));
  std::optional<source_type> lastValidIndex{};
  forEachIndexValue(container, trimmedBegin, trimmedEnd, [&](const source_type& index, const auto& symbol) {
    auto frequency = getFrequency(container, symbol);
    if (lastValidIndex.has_value()) {
      if (frequency > 0) {
        assert(index > *lastValidIndex);
        uint32_t offset = index - *lastValidIndex;
        lastValidIndex = index;
        dstIter = eliasDeltaEncode(dstIter, offset);
        dstIter = eliasDeltaEncode(dstIter, frequency);
      }
    } else {
      if (frequency > 0) {
        dstIter = eliasDeltaEncode(dstIter, frequency);
        lastValidIndex = index;
      }
    }
  });
  // write out incompressibleFrequency
  dstIter = eliasDeltaEncode(dstIter, getIncompressibleFrequency(container) + 1);
  // finish off by a 1 to identify start of the sequence.
  dstIter = eliasDeltaEncode(dstIter, 1);

  // extract raw Pointer from BitPtr
  const dest_IT iterEnd = [dstIter]() {
    using buffer_type = typename std::iterator_traits<dest_IT>::value_type;
    dest_IT iterEnd = dstIter.toPtr<buffer_type>();
    // one past the end
    return ++iterEnd;
  }();

  return iterEnd;
} // namespace o2::rans

template <typename source_T, typename buffer_IT>
RenormedHistogram<source_T> readRenormedDictionary(buffer_IT begin, buffer_IT end, source_T min, source_T max, size_t renormingPrecision)
{
  static_assert(std::is_pointer_v<buffer_IT>, "can only deserialize from raw pointers");

  using namespace internal;
  using container_type = typename RenormedHistogram<source_T>::container_type;
  using value_type = typename container_type::value_type;

  const size_t dictExtent = getDictExtent(min, max, renormingPrecision);

  container_type container(dictExtent, min);

  BitPtr iter = seekEliasDeltaEnd(begin, end);
  BitPtr beginPos{begin};

  auto deltaDecode = [beginPos](BitPtr& iter) -> value_type {
    intptr_t delta = getEliasDeltaOffset(beginPos, iter);
    return eliasDeltaDecode<value_type>(iter, delta);
  };

  if (iter == BitPtr{}) {
    throw ParsingError{"failed to read renormed dictionary: could not find end of data stream"};
  }

  if (deltaDecode(iter) != 1) {
    throw ParsingError{"failed to read renormed dictionary: could not find end of stream delimiter"};
  }

  const value_type incompressibleSymbolFrequency = deltaDecode(iter) - 1;

  source_T idx = max;
  if (iter != beginPos) {
    // first value at max, without index offset
    container[idx] = deltaDecode(iter);
  }

  while (iter != beginPos) {
    const auto offset = deltaDecode(iter);
    const auto frequency = deltaDecode(iter);
    idx -= offset;
    container[idx] = frequency;
  }

  if (idx != min) {
    throw ParsingError{fmt::format("failed to read renormed dictionary: reached EOS at index {} before parsing min {} ", idx, min)};
  }
  return {std::move(container), renormingPrecision, incompressibleSymbolFrequency};
};
} // namespace o2::rans

#endif /* RANS_SERIALIZE_H_ */