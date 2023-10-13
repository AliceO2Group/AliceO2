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
/// \brief Interface for externally provided rANS entropy coders

#ifndef ALICEO2_EXTERNALENTROPYCODER_H_
#define ALICEO2_EXTERNALENTROPYCODER_H_

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
class ExternalEntropyCoder
{
 public:
  using source_type = source_T;
  using encoder_type = typename rans::denseEncoder_type<source_type>;

  ExternalEntropyCoder(const encoder_type& encoder);

  [[nodiscard]] inline const encoder_type& getEncoder() const noexcept { return *mEncoder; };

  template <typename dst_T = uint8_t>
  [[nodiscard]] inline size_t computePayloadSizeEstimate(size_t nElements, double_t safetyFactor = 1);

  template <typename src_IT, typename dst_IT>
  [[nodiscard]] dst_IT encode(src_IT srcBegin, src_IT srcEnd, dst_IT dstBegin, dst_IT dstEnd);

  [[nodiscard]] inline size_t getNIncompressibleSamples() const noexcept { return mIncompressibleBuffer.size(); };

  [[nodiscard]] inline source_type getIncompressibleSymbolOffset() const noexcept { return mIncompressiblePacker.getOffset(); };

  [[nodiscard]] inline size_t getIncompressibleSymbolPackingBits() const noexcept { return mIncompressiblePacker.getPackingWidth(); };

  template <typename dst_T = uint8_t>
  [[nodiscard]] size_t computePackedIncompressibleSize() const noexcept;

  template <typename dst_T>
  [[nodiscard]] dst_T* writeIncompressible(dst_T* dstBegin, dst_T* dstEnd) const;

 private:
  const encoder_type* mEncoder{};
  std::vector<source_type> mIncompressibleBuffer{};
  Packer<source_type> mIncompressiblePacker{};
};

template <typename source_T>
ExternalEntropyCoder<source_T>::ExternalEntropyCoder(const encoder_type& encoder) : mEncoder{&encoder}
{
  if (!getEncoder().getSymbolTable().hasEscapeSymbol()) {
    throw std::runtime_error("External entropy encoder must be able to handle incompressible symbols.");
  }
};

template <typename source_T>
template <typename dst_T>
[[nodiscard]] inline size_t ExternalEntropyCoder<source_T>::computePayloadSizeEstimate(size_t nElements, double_t safetyFactor)
{
  constexpr size_t Overhead = 10 * rans::utils::pow2(10); // 10KB overhead safety margin
  const double_t RelativeSafetyFactor = 2.0 * safetyFactor;
  const size_t messageSizeB = nElements * sizeof(source_type);
  return rans::utils::nBytesTo<dst_T>(std::ceil(RelativeSafetyFactor * messageSizeB) + Overhead);
}

template <typename source_T>
template <typename src_IT, typename dst_IT>
[[nodiscard]] dst_IT ExternalEntropyCoder<source_T>::encode(src_IT srcBegin, src_IT srcEnd, dst_IT dstBegin, dst_IT dstEnd)
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

template <typename source_T>
template <typename dst_T>
[[nodiscard]] inline size_t ExternalEntropyCoder<source_T>::computePackedIncompressibleSize() const noexcept
{
  return mIncompressiblePacker.template getPackingBufferSize<dst_T>(mIncompressibleBuffer.size());
};

template <typename source_T>
template <typename dst_T>
[[nodiscard]] inline dst_T* ExternalEntropyCoder<source_T>::writeIncompressible(dst_T* dstBegin, dst_T* dstEnd) const
{
  return mIncompressiblePacker.pack(mIncompressibleBuffer.data(), mIncompressibleBuffer.size(), dstBegin, dstEnd);
};

} // namespace o2::ctf::internal

#endif /* ALICEO2_EXTERNALENTROPYCODER_H_ */