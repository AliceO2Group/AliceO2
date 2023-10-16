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

/// @file   Encoder.h
/// @author michael.lettrich@cern.ch
/// @brief  Encoder - User facing class to perform rANS entropy coding of source symbols onto a rANS state based on the statistical distribution in the symbol table.

#ifndef RANS_INTERNAL_ENCODE_ENCODER_H_
#define RANS_INTERNAL_ENCODE_ENCODER_H_

#include <algorithm>
#include <iomanip>
#include <memory>

#include <fairlogger/Logger.h>
#include <stdexcept>

#include <gsl/span>

#include "rANS/internal/common/utils.h"
#include "rANS/internal/encode/EncoderSymbolMapper.h"

namespace o2::rans
{

namespace encoderImpl
{

template <typename stream_IT, typename literals_IT = std::nullptr_t>
[[nodiscard]] inline constexpr decltype(auto) makeReturn(stream_IT streamEnd, literals_IT literalsEnd = nullptr) noexcept
{
  if constexpr (std::is_null_pointer_v<literals_IT>) {
    return streamEnd;
  } else {
    return std::make_pair(streamEnd, literalsEnd);
  }
};

} // namespace encoderImpl

template <class encoder_T, class symbolTable_T, std::size_t nStreams_V>
class Encoder
{
 public:
  using symbolTable_type = symbolTable_T;
  using symbol_type = typename symbolTable_T::value_type;
  using coder_type = encoder_T;
  using source_type = typename symbolTable_type::source_type;
  using stream_type = typename coder_type::stream_type;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  static constexpr size_type NStreams = nStreams_V;

  Encoder() = default;

  template <typename renormedSymbolTable_T>
  Encoder(const renormedSymbolTable_T& renormedFrequencyTable) : mSymbolTable{renormedFrequencyTable}
  {
    const size_t symbolTablePrecission = mSymbolTable.getPrecision();
    const size_t encoderLowerBound = coder_type::getStreamingLowerBound();
    if (symbolTablePrecission > encoderLowerBound) {
      throw EncodingError(fmt::format(
        "Renorming precision of symbol table ({} Bits) exceeds renorming lower bound of encoder ({} Bits).\
      This can cause overflows during encoding.",
        symbolTablePrecission, encoderLowerBound));
    }
  };

  [[nodiscard]] inline const symbolTable_type& getSymbolTable() const noexcept { return mSymbolTable; };

  [[nodiscard]] inline static constexpr size_type getNStreams() noexcept { return NStreams; };

  template <typename stream_IT, typename source_IT, typename literals_IT = std::nullptr_t, std::enable_if_t<utils::isCompatibleIter_v<typename symbolTable_T::source_type, source_IT>, bool> = true>
  decltype(auto) process(source_IT inputBegin, source_IT inputEnd, stream_IT outputBegin, literals_IT literalsBegin = nullptr) const;

  template <typename literals_IT = std::nullptr_t>
  inline decltype(auto) process(gsl::span<const source_type> inputStream, gsl::span<stream_type> outputStream, literals_IT literalsBegin = nullptr) const
  {
    return process(inputStream.data(), inputStream.data() + inputStream.size(), outputStream.data(), literalsBegin);
  };

 protected:
  symbolTable_type mSymbolTable{};

  static constexpr size_type NCoderStreams = coder_type::getNstreams();
  static constexpr size_type NCoders = NStreams / NCoderStreams;

  // compile time preconditions:
  static_assert(internal::isPow2(nStreams_V), "the number of streams must be a power of 2");
  static_assert(coder_type::getNstreams() <= Encoder::getNStreams(), "The specified coder type has more streams than your encoder");
  static_assert(NCoders % 2 == 0, "At least 2 encoders must run in parallel");
};

template <class encoder_T, class symbolTable_T, std::size_t nStreams_V>
template <typename stream_IT, typename source_IT, typename literals_IT, std::enable_if_t<utils::isCompatibleIter_v<typename symbolTable_T::source_type, source_IT>, bool>>
decltype(auto) Encoder<encoder_T, symbolTable_T, nStreams_V>::process(source_IT inputBegin, source_IT inputEnd, stream_IT outputBegin, literals_IT literalsBegin) const
{

  using namespace internal;
  using namespace utils;
  using namespace encoderImpl;

  if (inputBegin == inputEnd) {
    LOG(warning) << "passed empty message to encoder, skip encoding";
    return makeReturn(outputBegin, literalsBegin);
  }

  if (std::is_null_pointer_v<literals_IT> && mSymbolTable.hasEscapeSymbol()) {
    throw HistogramError("The Symbol table used requires you to pass a literals iterator");
  }

  std::array<coder_type, NCoders> coders;

  for (auto& coder : coders) {
    coder = coder_type{mSymbolTable.getPrecision()};
  }

  // calculate sizes and numbers of iterations:
  const auto inputBufferSize = std::distance(inputBegin, inputEnd); // size of the input buffer
  const size_t nRemainderSymbols = inputBufferSize % NStreams;
  const size_t nPartialCoderIterations = nRemainderSymbols / NCoderStreams;
  const size_t nFractionalEncodes = nRemainderSymbols % NCoderStreams;

  // from here on, everything runs backwards!
  // We are encoding symbols from the end of the message to the beginning of the message.
  // Since rANS works like a stack, this allows the decoder to work in forward direction.
  // To guarante consistency betweenn different coder implementations, all interleaved
  // streams have to run backwards as well, i.e. stream m of coder n+1 runs before stream p of coder n.

  stream_IT outputIter = outputBegin;
  source_IT inputIter = advanceIter(inputEnd, -1);
  source_IT inputREnd = advanceIter(inputBegin, -1);

  EncoderSymbolMapper<symbolTable_type, coder_type, literals_IT> symbolMapper{this->mSymbolTable, literalsBegin};
  typename coder_type::symbol_type encoderSymbols[2]{};

  // one past the end. Will not be dereferenced, thus safe.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
  coder_type* codersREnd = advanceIter(coders.data(), -1);
#pragma GCC diagnostic pop
  coder_type* activeCoder = codersREnd + nPartialCoderIterations;

  // we are encoding backwards!
  if (nFractionalEncodes) {
    // one more encoding step than nPartialCoderIterations for masked encoding
    // will not cause out of range for coders
    ++activeCoder;
    inputIter = symbolMapper.unpackSymbols(inputIter, encoderSymbols[0], nFractionalEncodes);
    outputIter = activeCoder->putSymbols(outputIter, encoderSymbols[0], nFractionalEncodes);
    --activeCoder;
  }

  // align encoders
  while (activeCoder != codersREnd) {
    if (inputIter != inputREnd) {
      inputIter = symbolMapper.unpackSymbols(inputIter, encoderSymbols[0]);
      outputIter = (activeCoder--)->putSymbols(outputIter, encoderSymbols[0]);
    }
  }

  constexpr size_t lastCoderIdx = NCoders - 1;
  while (inputIter != inputREnd) {
    // iterate over coders
    for (size_t i = 0; i < coders.size(); i += 2) {
      inputIter = symbolMapper.unpackSymbols(inputIter, encoderSymbols[0]);
      outputIter = coders[lastCoderIdx - i].putSymbols(outputIter, encoderSymbols[0]);
      inputIter = symbolMapper.unpackSymbols(inputIter, encoderSymbols[1]);
      outputIter = coders[lastCoderIdx - (i + 1)].putSymbols(outputIter, encoderSymbols[1]);
    }
  }

  for (size_t i = coders.size(); i-- > 0;) {
    outputIter = coders[i].flush(outputIter);
  }

  return makeReturn(outputIter, symbolMapper.getIncompressibleIterator());
}

}; // namespace o2::rans

#endif /* RANS_INTERNAL_ENCODE_ENCODER_H_ */
