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

/// \file CTFEntropyCoder.h
/// \author michael.lettrich@cern.ch
/// \brief ANS Entropy Coding and packing specialization for CTF Coders

#ifndef ALICEO2_INPLACEENTROPYCODER_H_
#define ALICEO2_INPLACEENTROPYCODER_H_

#include <optional>
#include <variant>
#include <type_traits>

#include "DetectorsCommonDataFormats/internal/Packer.h"

#include "rANS/encode.h"
#include "rANS/factory.h"
#include "rANS/histogram.h"
#include "rANS/metrics.h"
#include "rANS/serialize.h"

namespace o2::ctf::internal
{

template <typename source_T>
class InplaceEntropyCoder
{
  using dense_histogram_type = rans::DenseHistogram<source_T>;
  using adaptive_histogram_type = rans::AdaptiveHistogram<source_T>;
  using sparse_histogram_type = rans::SparseHistogram<source_T>;

  using dense_encoder_type = rans::denseEncoder_type<source_T>;
  using adaptive_encoder_type = rans::adaptiveEncoder_type<source_T>;
  using sparse_encoder_type = rans::sparseEncoder_type<source_T>;

  using dict_buffer_type = std::vector<uint8_t>;

 public:
  using source_type = source_T;
  using metrics_type = rans::Metrics<source_type>;
  using packer_type = Packer<source_type>;
  using histogram_type = std::variant<dense_histogram_type, adaptive_histogram_type, sparse_histogram_type>;
  using encoder_type = std::variant<dense_encoder_type, adaptive_encoder_type, sparse_encoder_type>;
  using incompressible_buffer_type = std::vector<source_type>;

  InplaceEntropyCoder() = default;

  template <typename source_IT>
  InplaceEntropyCoder(source_IT srcBegin, source_IT srcEnd);

  template <typename source_IT>
  InplaceEntropyCoder(source_IT srcBegin, source_IT srcEnd, source_type min, source_type max);

  void makeEncoder();

  // getters

  [[nodiscard]] inline const metrics_type& getMetrics() const noexcept { return mMetrics; };

  [[nodiscard]] inline size_t getNIncompressibleSamples() const noexcept { return mIncompressibleBuffer.size(); };

  [[nodiscard]] size_t getNStreams() const;

  [[nodiscard]] size_t getSymbolTablePrecision() const;

  template <typename dst_T = uint8_t>
  [[nodiscard]] size_t getPackedIncompressibleSize() const noexcept;

  // operations
  template <typename src_IT, typename dst_IT>
  [[nodiscard]] dst_IT encode(src_IT srcBegin, src_IT srcEnd, dst_IT dstBegin, dst_IT dstEnd);

  template <typename dst_IT>
  [[nodiscard]] dst_IT writeDictionary(dst_IT dstBegin, dst_IT dstEnd);

  template <typename dst_T>
  [[nodiscard]] dst_T* writeIncompressible(dst_T* dstBegin, dst_T* dstEnd);

 private:
  template <typename source_IT, std::enable_if_t<(sizeof(typename std::iterator_traits<source_IT>::value_type) < 4), bool> = true>
  void init(source_IT srcBegin, source_IT srcEnd, source_type min, source_type max);

  template <typename source_IT, std::enable_if_t<(sizeof(typename std::iterator_traits<source_IT>::value_type) == 4), bool> = true>
  void init(source_IT srcBegin, source_IT srcEnd, source_type min, source_type max);

  template <typename source_IT, std::enable_if_t<(sizeof(typename std::iterator_traits<source_IT>::value_type) < 4), bool> = true>
  void init(source_IT srcBegin, source_IT srcEnd);

  template <typename source_IT, std::enable_if_t<(sizeof(typename std::iterator_traits<source_IT>::value_type) == 4), bool> = true>
  void init(source_IT srcBegin, source_IT srcEnd);

  template <typename container_T>
  void serializeDictionary(const container_T&);

  std::optional<histogram_type> mHistogram{};
  metrics_type mMetrics{};
  std::optional<encoder_type> mEncoder{};
  incompressible_buffer_type mIncompressibleBuffer{};
  dict_buffer_type mDictBuffer{};
  packer_type mIncompressiblePacker{};
};

template <typename source_T>
template <typename src_IT>
InplaceEntropyCoder<source_T>::InplaceEntropyCoder(src_IT srcBegin, src_IT srcEnd)
{
  static_assert(std::is_same_v<source_T, typename std::iterator_traits<src_IT>::value_type>);

  const size_t nSamples = std::distance(srcBegin, srcEnd);
  if constexpr (std::is_pointer_v<src_IT>) {
    if (sizeof(source_type) > 2 && nSamples > 0) {
      const auto [min, max] = rans::internal::minmax(gsl::span<const source_type>(srcBegin, srcEnd));
      init(srcBegin, srcEnd, min, max);
    } else {
      init(srcBegin, srcEnd);
    }
  } else {
    init(srcBegin, srcEnd);
  }

  mIncompressiblePacker = Packer(mMetrics);
};

template <typename source_T>
template <typename source_IT>
InplaceEntropyCoder<source_T>::InplaceEntropyCoder(source_IT srcBegin, source_IT srcEnd, source_type min, source_type max)
{
  static_assert(std::is_same_v<source_T, typename std::iterator_traits<source_IT>::value_type>);
  init(srcBegin, srcEnd, min, max);
  mIncompressiblePacker = Packer(mMetrics);
};

template <typename source_T>
[[nodiscard]] inline size_t InplaceEntropyCoder<source_T>::getNStreams() const
{
  size_t nStreams{};
  std::visit([&, this](auto&& encoder) { nStreams = encoder.getNStreams(); }, *mEncoder);
  return nStreams;
}

template <typename source_T>
[[nodiscard]] inline size_t InplaceEntropyCoder<source_T>::getSymbolTablePrecision() const
{
  size_t precision{};
  std::visit([&, this](auto&& encoder) { precision = encoder.getSymbolTable().getPrecision(); }, *mEncoder);
  return precision;
}

template <typename source_T>
void InplaceEntropyCoder<source_T>::makeEncoder()
{
  std::visit([this](auto&& histogram) {
    auto renormed = rans::renorm(std::move(histogram), mMetrics);

    if (std::holds_alternative<sparse_histogram_type>(*mHistogram)) {
      serializeDictionary(renormed);
    }

    const size_t rangeBits = rans::utils::getRangeBits(*mMetrics.getCoderProperties().min, *mMetrics.getCoderProperties().max);
    const size_t nUsedAlphabetSymbols = mMetrics.getDatasetProperties().nUsedAlphabetSymbols;

    if (rangeBits <= 18) {
      // dense symbol tables if they fit into cache, or source data covers the range of the alphabet well
      mEncoder = encoder_type{std::in_place_type<dense_encoder_type>, renormed};
    } else if (nUsedAlphabetSymbols < rans::utils::pow2(14)) {
      // sparse symbol table makes sense if it fits into L3 Cache
      mEncoder = encoder_type{std::in_place_type<sparse_encoder_type>, renormed};
    } else {
      // adaptive symbol table otherwise
      mEncoder = encoder_type{std::in_place_type<adaptive_encoder_type>, renormed};
    }
  },
             *mHistogram);
};

template <typename source_T>
template <typename src_IT, typename dst_IT>
[[nodiscard]] dst_IT InplaceEntropyCoder<source_T>::encode(src_IT srcBegin, src_IT srcEnd, dst_IT dstBegin, dst_IT dstEnd)
{
  static_assert(std::is_same_v<source_T, typename std::iterator_traits<src_IT>::value_type>);

  dst_IT messageEnd = dstBegin;

  std::visit([&, this](auto&& encoder) {
    if (encoder.getSymbolTable().hasEscapeSymbol()) {
      mIncompressibleBuffer.reserve(*mMetrics.getCoderProperties().nIncompressibleSamples);
      auto [encodedMessageEnd, literalsEnd] = encoder.process(srcBegin, srcEnd, dstBegin, std::back_inserter(mIncompressibleBuffer));
      messageEnd = encodedMessageEnd;
    } else {
      messageEnd = encoder.process(srcBegin, srcEnd, dstBegin);
    }
    rans::utils::checkBounds(messageEnd, dstEnd);
  },
             *mEncoder);

  return messageEnd;
};

template <typename source_T>
template <typename dst_IT>
[[nodiscard]] inline dst_IT InplaceEntropyCoder<source_T>::writeDictionary(dst_IT dstBegin, dst_IT dstEnd)
{
  static_assert(std::is_pointer_v<dst_IT>);

  using dst_type = std::remove_pointer_t<dst_IT>;

  dst_IT ret{};
  if (mDictBuffer.empty()) {
    std::visit([&, this](auto&& encoder) { ret = rans::compressRenormedDictionary(encoder.getSymbolTable(), dstBegin); }, *mEncoder);
  } else {
    // copy
    std::memcpy(dstBegin, mDictBuffer.data(), mDictBuffer.size());

    // determine location of end
    auto end = reinterpret_cast<uint8_t*>(dstBegin) + mDictBuffer.size();
    // realign pointer
    constexpr size_t alignment = std::alignment_of_v<dst_type>;
    end += (alignment - reinterpret_cast<uintptr_t>(end) % alignment) % alignment;
    // and convert it back to ret
    ret = reinterpret_cast<dst_IT>(end);
  }

  rans::utils::checkBounds(ret, dstEnd);
  return ret;
};

template <typename source_T>
template <typename dst_T>
inline dst_T* InplaceEntropyCoder<source_T>::writeIncompressible(dst_T* dstBegin, dst_T* dstEnd)
{
  return mIncompressiblePacker.pack(mIncompressibleBuffer.data(), mIncompressibleBuffer.size(), dstBegin, dstEnd);
};

template <typename source_T>
template <typename dst_T>
[[nodiscard]] inline size_t InplaceEntropyCoder<source_T>::getPackedIncompressibleSize() const noexcept
{
  return mIncompressiblePacker.template getPackingBufferSize<dst_T>(getNIncompressibleSamples());
}

template <typename source_T>
template <typename source_IT, std::enable_if_t<(sizeof(typename std::iterator_traits<source_IT>::value_type) < 4), bool>>
void InplaceEntropyCoder<source_T>::init(source_IT srcBegin, source_IT srcEnd, source_type min, source_type max)
{
  mHistogram.emplace(histogram_type{rans::makeDenseHistogram::fromSamples(srcBegin, srcEnd)});
  mMetrics = metrics_type{std::get<dense_histogram_type>(*mHistogram), min, max};
};

template <typename source_T>
template <typename source_IT, std::enable_if_t<(sizeof(typename std::iterator_traits<source_IT>::value_type) == 4), bool>>
void InplaceEntropyCoder<source_T>::init(source_IT srcBegin, source_IT srcEnd, source_type min, source_type max)
{
  const size_t nSamples = std::distance(srcBegin, srcEnd);
  const size_t rangeBits = rans::utils::getRangeBits(min, max);

  if ((rangeBits <= 18) || ((nSamples / rans::utils::pow2(rangeBits)) >= 0.80)) {
    // either the range of source symbols is distrubuted such that it fits into L3 Cache
    // Or it is possible for the data to cover a very significant fraction of the total [min,max] range
    mHistogram = histogram_type{std::in_place_type<dense_histogram_type>, rans::makeDenseHistogram::fromSamples(srcBegin, srcEnd, min, max)};
    mMetrics = metrics_type{std::get<dense_histogram_type>(*mHistogram), min, max};
  } else if (nSamples / rans::utils::pow2(rangeBits) <= 0.3) {
    // or the range of source symbols is spread very thinly accross a large range
    mHistogram = histogram_type{std::in_place_type<sparse_histogram_type>, rans::makeSparseHistogram::fromSamples(srcBegin, srcEnd)};
    mMetrics = metrics_type{std::get<sparse_histogram_type>(*mHistogram), min, max};
  } else {
    // no strong evidence of either extreme case
    mHistogram = histogram_type{std::in_place_type<adaptive_histogram_type>, rans::makeAdaptiveHistogram::fromSamples(srcBegin, srcEnd)};
    mMetrics = metrics_type{std::get<adaptive_histogram_type>(*mHistogram), min, max};
  }
};

template <typename source_T>
template <typename source_IT, std::enable_if_t<(sizeof(typename std::iterator_traits<source_IT>::value_type) < 4), bool>>
void InplaceEntropyCoder<source_T>::init(source_IT srcBegin, source_IT srcEnd)
{
  mHistogram = histogram_type{std::in_place_type<dense_histogram_type>, rans::makeDenseHistogram::fromSamples(srcBegin, srcEnd)};
  mMetrics = metrics_type{std::get<dense_histogram_type>(*mHistogram)};
};

template <typename source_T>
template <typename source_IT, std::enable_if_t<(sizeof(typename std::iterator_traits<source_IT>::value_type) == 4), bool>>
void InplaceEntropyCoder<source_T>::init(source_IT srcBegin, source_IT srcEnd)
{
  mHistogram = histogram_type{std::in_place_type<sparse_histogram_type>, rans::makeSparseHistogram::fromSamples(srcBegin, srcEnd)};
  mMetrics = metrics_type{std::get<sparse_histogram_type>(*mHistogram)};
};

template <typename source_T>
template <typename container_T>
void InplaceEntropyCoder<source_T>::serializeDictionary(const container_T& renormedHistogram)
{

  mDictBuffer.resize(mMetrics.getSizeEstimate().getCompressedDictionarySize(), 0);
  auto end = rans::compressRenormedDictionary(renormedHistogram, mDictBuffer.data());
  rans::utils::checkBounds(end, mDictBuffer.data() + mDictBuffer.size());
  mDictBuffer.resize(std::distance(mDictBuffer.data(), end));

  assert(mDictBuffer.size() > 0);
};

} // namespace o2::ctf::internal

#endif /* ALICEO2_INPLACEENTROPYCODER_H_ */