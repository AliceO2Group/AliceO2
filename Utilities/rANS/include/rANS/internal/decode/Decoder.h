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

#include <fairlogger/Logger.h>
#include <gsl/span>
#include <stdexcept>

#include "rANS/internal/containers/ReverseSymbolLookupTable.h"

namespace o2::rans
{

template <class decoder_T, class symbolTable_T>
class Decoder
{
 public:
  using symbolTable_type = symbolTable_T;
  using symbol_type = typename symbolTable_T::value_type;
  using coder_type = decoder_T;
  using source_type = typename symbolTable_type::source_type;
  using stream_type = typename coder_type::stream_type;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  Decoder() = default;
  template <typename renormedSymbolTable_T>
  Decoder(const renormedSymbolTable_T& renormedFrequencyTable) : mSymbolTable{renormedFrequencyTable}, mReverseLUT{renormedFrequencyTable} {};

  [[nodiscard]] inline const symbolTable_type& getSymbolTable() const noexcept { return mSymbolTable; };

  template <typename stream_IT, typename source_IT, typename literals_IT = std::nullptr_t, std::enable_if_t<utils::isCompatibleIter_v<typename symbolTable_T::source_type, source_IT>, bool> = true>
  void process(stream_IT inputEnd, source_IT outputBegin, size_t messageLength, size_t nStreams, literals_IT literalsEnd = nullptr) const
  {
    {

      if (messageLength == 0) {
        LOG(warning) << "Empty message passed to decoder, skipping decode process";
        return;
      }

      if (!(nStreams > 1 && internal::isPow2(nStreams))) {
        throw DecodingError(fmt::format("Invalid number of decoder streams {}", nStreams));
      }

      stream_IT inputIter = inputEnd;
      --inputIter;
      source_IT outputIter = outputBegin;
      literals_IT literalsIter = literalsEnd;

      auto decode = [&, this](coder_type& decoder) {
        const auto cumul = decoder.get();
        source_type sourceSymbol{};
        typename symbolTable_type::const_pointer decoderSymbol{};

        if constexpr (!std::is_null_pointer_v<literals_IT>) {
          if (this->mReverseLUT.isIncompressible(cumul)) {
            sourceSymbol = *(--literalsIter);
            decoderSymbol = &(this->mSymbolTable.getEscapeSymbol());
          } else {
            sourceSymbol = (this->mReverseLUT)[cumul];
            decoderSymbol = this->mSymbolTable.lookupUnsafe(sourceSymbol);
          }
        } else {
          sourceSymbol = (this->mReverseLUT)[cumul];
          decoderSymbol = this->mSymbolTable.lookupUnsafe(sourceSymbol);
        }

#ifdef O2_RANS_PRINT_PROCESSED_DATA
        arrayLogger << sourceSymbol;
#endif
        return std::make_tuple(sourceSymbol, decoder.advanceSymbol(inputIter, *decoderSymbol));
      };

      std::vector<coder_type> decoders{nStreams, coder_type{this->mSymbolTable.getPrecision()}};
      for (auto& decoder : decoders) {
        inputIter = decoder.init(inputIter);
      }

      const size_t nLoops = messageLength / nStreams;
      const size_t nLoopRemainder = messageLength % nStreams;

      for (size_t i = 0; i < nLoops; ++i) {
        for (auto& decoder : decoders) {
          std::tie(*outputIter++, inputIter) = decode(decoder);
        }
      }

      for (size_t i = 0; i < nLoopRemainder; ++i) {
        std::tie(*outputIter++, inputIter) = decode(decoders[i]);
      }

#ifdef O2_RANS_PRINT_PROCESSED_DATA
      LOG(info) << "decoderOutput:" << arrayLogger;
#endif
    }
  }

  template <typename literals_IT = std::nullptr_t>
  inline void process(gsl::span<const stream_type> inputStream, gsl::span<source_type> outputStream, size_t messageLength, size_t nStreams, literals_IT literalsEnd = nullptr) const
  {
    process(inputStream.data() + inputStream.size(), outputStream.data(), nStreams, literalsEnd);
  };

 protected:
  symbolTable_type mSymbolTable{};
  internal::ReverseSymbolLookupTable<source_type> mReverseLUT{};

  static_assert(coder_type::getNstreams() == 1, "implementation supports only single stream encoders");
};

} // namespace o2::rans

#endif /* RANS_INTERNAL_DECODE_DECODER_H_ */
