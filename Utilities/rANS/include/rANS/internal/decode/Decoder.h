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

/// @file   Decoder.h
/// @author Michael Lettrich
/// @brief  Decoder - User facing class to decode a rANS encoded stream back into the source data based on the same statistical distribution used for encoding.

#ifndef RANS_INTERNAL_DECODE_DECODER_H_
#define RANS_INTERNAL_DECODE_DECODER_H_

#include <rANS/internal/containers/RenormedHistogram.h>
#include <rANS/internal/containers/LowRangeDecoderTable.h>
#include <rANS/internal/containers/HighRangeDecoderTable.h>
#include <rANS/internal/decode/DecoderConcept.h>

#include <fairlogger/Logger.h>
#include <gsl/span>

#include "rANS/internal/common/utils.h"

namespace o2::rans
{

template <typename source_T, class decoder_T>
class Decoder
{

 public:
  using source_type = source_T;
  using coder_type = decoder_T;

 private:
  using lowRangeTable_type = LowRangeDecoderTable<source_type>;
  using highRangeTable_type = HighRangeDecoderTable<source_type>;

  static_assert(std::is_same_v<typename lowRangeTable_type::symbol_type, typename highRangeTable_type::symbol_type>);
  using symbol_type = typename lowRangeTable_type::symbol_type;

  using lowRangeDecoder_type = DecoderConcept<coder_type, lowRangeTable_type>;
  using highRangDecoder_type = DecoderConcept<coder_type, highRangeTable_type>;

  using decoder_type = std::variant<lowRangeDecoder_type, highRangDecoder_type>;

 public:
  using stream_type = typename coder_type::stream_type;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  Decoder() noexcept = default;

  template <typename container_T>
  explicit Decoder(const RenormedHistogramConcept<container_T>& renormedHistogram)
  {

    const auto [min, max] = internal::getMinMax(renormedHistogram);
    const size_t alphabetRangeBits = utils::getRangeBits(min, max);

    if (renormedHistogram.getRenormingBits() > alphabetRangeBits) {
      mImpl.template emplace<lowRangeDecoder_type>(renormedHistogram);
    } else {
      mImpl.template emplace<highRangDecoder_type>(renormedHistogram);
    }
  };

  [[nodiscard]] inline size_t getSymbolTablePrecision() const noexcept
  {
    size_t precision{};
    std::visit([&precision](auto&& decoder) { precision = decoder.getSymbolTable().getPrecision(); }, mImpl);
    return precision;
  };

  template <typename stream_IT, typename source_IT, typename literals_IT = std::nullptr_t>
  void process(stream_IT inputEnd, source_IT outputBegin, size_t messageLength, size_t nStreams, literals_IT literalsEnd = nullptr) const
  {
    static_assert(utils::isCompatibleIter_v<source_type, source_IT>);
    std::visit([&](auto&& decoder) { decoder.process(inputEnd, outputBegin, messageLength, nStreams, literalsEnd); }, mImpl);
  }

  template <typename literals_IT = std::nullptr_t>
  inline void process(gsl::span<const stream_type> inputStream, gsl::span<source_type> outputStream, size_t messageLength, size_t nStreams, literals_IT literalsEnd = nullptr) const
  {
    process(inputStream.data() + inputStream.size(), outputStream.data(), nStreams, literalsEnd);
  };

 protected:
  decoder_type mImpl{};

  static_assert(coder_type::getNstreams() == 1, "implementation supports only single stream encoders");
};

} // namespace o2::rans

#endif /* RANS_INTERNAL_DECODE_DECODER_H_ */
