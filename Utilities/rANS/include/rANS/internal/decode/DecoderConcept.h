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

/// @file   DecoderConcept.h
/// @author Michael Lettrich
/// @brief  DecoderConcept - User facing class to decode a rANS encoded stream back into the source data based on the same statistical distribution used for encoding.

#ifndef RANS_INTERNAL_DECODE_DECODER_CONCEPT_H_
#define RANS_INTERNAL_DECODE_DECODER_CONCEPT_H_

#include <fairlogger/Logger.h>
#include <gsl/span>
#include <stdexcept>

#include "rANS/internal/common/utils.h"
#include "rANS/internal/containers/RenormedHistogram.h"

namespace o2::rans
{

template <class decoder_T, class symbolTable_T>
class DecoderConcept
{
 public:
  using symbolTable_type = symbolTable_T;
  using symbol_type = typename symbolTable_type::symbol_type;
  using coder_type = decoder_T;
  using source_type = typename symbolTable_type::source_type;
  using stream_type = typename coder_type::stream_type;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

 private:
  using value_type = typename symbolTable_type::value_type;

 public:
  DecoderConcept() = default;

  template <typename container_T>
  explicit DecoderConcept(const RenormedHistogramConcept<container_T>& renormedHistogram) : mSymbolTable{renormedHistogram} {};

  [[nodiscard]] inline const symbolTable_type& getSymbolTable() const noexcept { return this->mSymbolTable; };

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

      auto lookupSymbol = [&literalsIter, this](uint32_t cumulativeFrequency) -> value_type {
        if constexpr (!std::is_null_pointer_v<literals_IT>) {
          if (this->mSymbolTable.isEscapeSymbol(cumulativeFrequency)) {
            return value_type{*(--literalsIter), this->mSymbolTable.getEscapeSymbol()};
          } else {
            return this->mSymbolTable[cumulativeFrequency];
          }
        } else {
          return this->mSymbolTable[cumulativeFrequency];
        }
      };

      auto decode = [&, this](coder_type& decoder) {
        const auto cumul = decoder.get();
        const value_type symbol = lookupSymbol(cumul);
#ifdef RANS_LOG_PROCESSED_DATA
        arrayLogger << symbol.first;
#endif
        return std::make_tuple(symbol.first, decoder.advanceSymbol(inputIter, symbol.second));
      };

      std::vector<coder_type> decoders{nStreams, coder_type{this->mSymbolTable.getPrecision()}};
      for (auto& decoder : decoders) {
        inputIter = decoder.init(inputIter);
      }

      const size_t nLoops = messageLength / nStreams;
      const size_t nLoopRemainder = messageLength % nStreams;

      for (size_t i = 0; i < nLoops; ++i) {
#if defined(RANS_OPENMP)
#pragma omp unroll partial(2)
#endif
        for (auto& decoder : decoders) {
          std::tie(*outputIter++, inputIter) = decode(decoder);
        }
      }

      for (size_t i = 0; i < nLoopRemainder; ++i) {
        std::tie(*outputIter++, inputIter) = decode(decoders[i]);
      }

#ifdef RANS_LOG_PROCESSED_DATA
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

  static_assert(coder_type::getNstreams() == 1, "implementation supports only single stream encoders");
};

} // namespace o2::rans

#endif /* RANS_INTERNAL_DECODE_DECODER_CONCEPT_H_ */
