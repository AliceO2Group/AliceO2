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

#ifndef ALICEO2_CTFENTROPYCODER_H_
#define ALICEO2_CTFENTROPYCODER_H_

#include <optional>
#include <type_traits>

#include "rANS/encode.h"
#include "rANS/factory.h"
#include "rANS/histogram.h"
#include "rANS/internal/containers/BitPtr.h"
#include "rANS/metrics.h"
#include "rANS/serialize.h"

namespace o2::ctf
{
template <typename source_T>
class Packer
{
 public:
  using source_type = source_T;

  Packer() = default;

  explicit Packer(rans::Metrics<source_type>& metrics) : mOffset{metrics.getDatasetProperties().min},
                                                         mPackingWidth{metrics.getDatasetProperties().alphabetRangeBits} {};

  template <typename source_IT>
  Packer(source_IT srcBegin, source_IT srcEnd)
  {
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
  };

  [[nodiscard]] inline source_type getOffset() const noexcept { return mOffset; };

  [[nodiscard]] inline size_t getPackingWidth() const noexcept { return mPackingWidth; };

  template <typename buffer_T>
  [[nodiscard]] inline size_t getPackingBufferSize(size_t messageLength) const noexcept
  {
    return rans::computePackingBufferSize<buffer_T>(messageLength, mPackingWidth);
  };

  template <typename source_IT, typename dst_T>
  inline dst_T* pack(source_IT srcBegin, source_IT srcEnd, dst_T* dstBegin, dst_T* dstEnd) const
  {
    static_assert(std::is_same_v<source_T, typename std::iterator_traits<source_IT>::value_type>);
    size_t extent = std::distance(srcBegin, srcEnd);

    if (extent == 0) {
      return dstBegin;
    }

    rans::BitPtr packEnd = rans::pack(srcBegin, extent, dstBegin, mPackingWidth, mOffset);
    auto* end = packEnd.toPtr<dst_T>();
    ++end; // one past end iterator;
    rans::utils::checkBounds(end, dstEnd);
    return end;
  };

  template <typename dst_T>
  inline dst_T* pack(const source_T* __restrict srcBegin, size_t extent, dst_T* dstBegin, dst_T* dstEnd) const
  {
    return pack(srcBegin, srcBegin + extent, dstBegin, dstEnd);
  }

 private:
  source_type mOffset{};
  size_t mPackingWidth{};
};

template <typename source_T>
class InplaceEntropyCoder
{
 public:
  using source_type = source_T;
  using histogram_type = rans::Histogram<source_type>;
  using renormedHistogram_type = rans::RenormedHistogram<source_type>;
  using metrics_type = rans::Metrics<source_type>;
  using encoder_type = typename rans::defaultEncoder_type<source_type>;

  InplaceEntropyCoder() = default;

  template <typename source_IT>
  InplaceEntropyCoder(source_IT srcBegin, source_IT srcEnd);

  void makeEncoder();

  // getters

  inline const metrics_type& getMetrics() const noexcept { return mMetrics; };

  inline const encoder_type& getEncoder() const { return const_cast<const encoder_type&>(const_cast<InplaceEntropyCoder&>(*this).getEncoderImpl()); };

  inline size_t getNIncompressibleSamples() const noexcept { return mIncompressibleBuffer.size(); };

  template <typename dst_T = uint8_t>
  inline size_t getPackedIncompressibleSize() const noexcept;

  // operations
  template <typename src_IT, typename dst_IT>
  dst_IT encode(src_IT srcBegin, src_IT srcEnd, dst_IT dstBegin, dst_IT dstEnd);

  template <typename dst_IT>
  inline dst_IT writeDictionary(dst_IT dstBegin, dst_IT dstEnd);

  template <typename dst_T>
  inline dst_T* writeIncompressible(dst_T* dstBegin, dst_T* dstEnd);

 private:
  inline histogram_type& getHistogram()
  {
    if (mHistogram.has_value()) {
      return *mHistogram;
    } else {
      throw std::runtime_error("uninitialized histogram");
    };
  };

  inline void setHistogram(histogram_type&& hist)
  {
    mHistogram = std::move(hist);
  };

  inline encoder_type& getEncoderImpl()
  {
    if (mEncoder.has_value()) {
      return *mEncoder;
    } else {
      throw std::runtime_error("uninitialized encoder");
    }
  };
  inline void setEncoder(encoder_type&& encoder) { mEncoder = std::move(encoder); };

  std::optional<histogram_type> mHistogram{};
  metrics_type mMetrics{};

  std::optional<encoder_type> mEncoder{};
  std::vector<source_T> mIncompressibleBuffer{};
  Packer<source_type> mIncompressiblePacker{};
};

template <typename source_T>
template <typename src_IT>
InplaceEntropyCoder<source_T>::InplaceEntropyCoder(src_IT srcBegin, src_IT srcEnd)
{
  static_assert(std::is_same_v<source_T, typename std::iterator_traits<src_IT>::value_type>);

  setHistogram(rans::makeHistogram::fromSamples(srcBegin, srcEnd));
  mMetrics = metrics_type{getHistogram()};
  mIncompressiblePacker = Packer(mMetrics);
};

template <typename source_T>
void InplaceEntropyCoder<source_T>::makeEncoder()
{
  auto& hist = getHistogram();
  auto renormed = rans::renorm(std::move(hist), mMetrics);
  mEncoder = rans::makeEncoder<>::fromRenormed(renormed);
  mIncompressiblePacker = Packer(mMetrics);
};

template <typename source_T>
template <typename src_IT, typename dst_IT>
dst_IT InplaceEntropyCoder<source_T>::encode(src_IT srcBegin, src_IT srcEnd, dst_IT dstBegin, dst_IT dstEnd)
{
  static_assert(std::is_same_v<source_T, typename std::iterator_traits<src_IT>::value_type>);

  dst_IT messageEnd = dstBegin;
  auto& encoder = getEncoderImpl();

  if (encoder.getSymbolTable().hasEscapeSymbol()) {
    mIncompressibleBuffer.reserve(*mMetrics.getCoderProperties().nIncompressibleSamples);
    auto [encodedMessageEnd, literalsEnd] = encoder.process(srcBegin, srcEnd, dstBegin, std::back_inserter(mIncompressibleBuffer));
    messageEnd = encodedMessageEnd;
  } else {
    messageEnd = encoder.process(srcBegin, srcEnd, dstBegin);
  }
  rans::utils::checkBounds(messageEnd, dstEnd);
  return messageEnd;
};

template <typename source_T>
template <typename dst_IT>
inline dst_IT InplaceEntropyCoder<source_T>::writeDictionary(dst_IT dstBegin, dst_IT dstEnd)
{
  dst_IT end = rans::compressRenormedDictionary(mEncoder->getSymbolTable(), dstBegin);
  rans::utils::checkBounds(end, dstEnd);
  return end;
};

template <typename source_T>
template <typename dst_T>
inline dst_T* InplaceEntropyCoder<source_T>::writeIncompressible(dst_T* dstBegin, dst_T* dstEnd)
{
  return mIncompressiblePacker.pack(mIncompressibleBuffer.data(), mIncompressibleBuffer.size(), dstBegin, dstEnd);
};

template <typename source_T>
template <typename dst_T>
inline size_t InplaceEntropyCoder<source_T>::getPackedIncompressibleSize() const noexcept
{
  return mIncompressiblePacker.template getPackingBufferSize<dst_T>(getNIncompressibleSamples());
}

template <typename source_T>
class ExternalEntropyCoder
{
 public:
  using source_type = source_T;
  using encoder_type = typename rans::defaultEncoder_type<source_type>;
  using metrics_type = rans::Metrics<source_type>;

  ExternalEntropyCoder(const encoder_type& encoder) : mEncoder{&encoder}
  {
    if (!getEncoder().getSymbolTable().hasEscapeSymbol()) {
      throw std::runtime_error("External entropy encoder must be able to handle incompressible symbols.");
    }
  };

  inline const encoder_type& getEncoder() const noexcept { return *mEncoder; };

  template <typename dst_T = uint8_t>
  inline size_t computePayloadSizeEstimate(size_t nElements, double_t safetyFactor = 1)
  {
    constexpr size_t Overhead = 10 * rans::utils::pow2(10); // 10KB overhead safety margin
    const double_t RelativeSafetyFactor = 2.0 * safetyFactor;
    const size_t messageSizeB = nElements * sizeof(source_type);
    return rans::utils::nBytesTo<dst_T>(std::ceil(safetyFactor * messageSizeB) + Overhead);
  }

  template <typename src_IT, typename dst_IT>
  dst_IT encode(src_IT srcBegin, src_IT srcEnd, dst_IT dstBegin, dst_IT dstEnd)
  {
    const size_t incompressibleSymbolFrequency = [&]() {
      const auto& symbolTable = mEncoder->getSymbolTable();
      const double_t incompressibleSymbolProbability = static_cast<double_t>(symbolTable.getEscapeSymbol().getFrequency()) / rans::utils::pow2(symbolTable.getPrecision());
      return std::ceil(std::distance(srcBegin, srcEnd) * incompressibleSymbolProbability);
    }();

    mIncompressibleBuffer.reserve(incompressibleSymbolFrequency);
    auto [encodedMessageEnd, literalsEnd] = mEncoder->process(srcBegin, srcEnd, dstBegin, std::back_inserter(mIncompressibleBuffer));
    rans::utils::checkBounds(encodedMessageEnd, dstEnd);
    mIncompressiblePacker = Packer<source_type>{mIncompressibleBuffer.data(), mIncompressibleBuffer.data() + mIncompressibleBuffer.size()};

    return encodedMessageEnd;
  };

  inline size_t getNIncompressibleSamples() const noexcept { return mIncompressibleBuffer.size(); };

  inline source_type getIncompressibleSymbolOffset() const noexcept { return mIncompressiblePacker.getOffset(); };

  inline size_t getIncompressibleSymbolPackingBits() const noexcept { return mIncompressiblePacker.getPackingWidth(); };

  template <typename dst_T = uint8_t>
  inline size_t computePackedIncompressibleSize() const noexcept
  {
    return mIncompressiblePacker.template getPackingBufferSize<dst_T>(mIncompressibleBuffer.size());
  }

  template <typename dst_T>
  inline dst_T* writeIncompressible(dst_T* dstBegin, dst_T* dstEnd)
  {
    return mIncompressiblePacker.pack(mIncompressibleBuffer.data(), mIncompressibleBuffer.size(), dstBegin, dstEnd);
  };

 private:
  const encoder_type* mEncoder{};
  std::vector<source_type> mIncompressibleBuffer{};
  Packer<source_type> mIncompressiblePacker{};
};

} // namespace o2::ctf

#endif /* ALICEO2_CTFENTROPYCODER_H_ */