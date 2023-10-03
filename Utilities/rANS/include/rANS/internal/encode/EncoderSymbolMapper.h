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

/// @file   EncoderSymbolMapper.h
/// @author michael.lettrich@cern.ch
/// @brief  Looks up statistical properties in symbol table for each source symbol. Specialed for each coder type

#ifndef RANS_INTERNAL_ENCODE_ENCODERSYMBOLMAPPER_H_
#define RANS_INTERNAL_ENCODE_ENCODERSYMBOLMAPPER_H_

#include <array>

#include <fairlogger/Logger.h>

#include "rANS/internal/common/defines.h"

#include "rANS/internal/common/utils.h"

#ifdef RANS_SIMD
#include "rANS/internal/encode/SIMDEncoderImpl.h"
#include "rANS/internal/encode/simdKernel.h"
#endif /* RANS_SIMD */
#include "rANS/internal/encode/SingleStreamEncoderImpl.h"
#include "rANS/internal/containers/DenseSymbolTable.h"

namespace o2::rans::internal
{

template <typename symbolTable_T, typename coder_T, typename incompressible_IT, typename derived_T>
class EncoderSymbolMapperInterface
{
 public:
  using symbolTable_type = symbolTable_T;
  using coder_type = coder_T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using source_type = typename symbolTable_type::source_type;
  using symbol_type = typename symbolTable_type::symbol_type;
  using coderSymbol_type = typename coder_type::symbol_type;
  using incompressible_iterator = incompressible_IT;

  template <typename source_IT>
  [[nodiscard]] source_IT unpackSymbols(source_IT sourceIter, coderSymbol_type& symbol, size_type nStreams)
  {
    return static_cast<derived_T*>(this)->unpackSymbols(sourceIter, symbol, nStreams);
  };

  template <typename source_IT>
  [[nodiscard]] source_IT unpackSymbols(source_IT sourceIter, coderSymbol_type& symbol)
  {
    return static_cast<derived_T*>(this)->unpackSymbols(sourceIter, symbol);
  };

  incompressible_iterator getIncompressibleIterator() const { return mIncompressibleIter; };

 protected:
  template <typename source_IT>
  [[nodiscard]] inline const symbol_type& lookupSymbol(source_IT sourceIter)
  {
    // LOGP(info, "unpacking {}", fmt::ptr(sourceIter));
    if constexpr (!std::is_null_pointer_v<incompressible_iterator>) {
      const symbol_type& symbol = (*mSymbolTable)[*sourceIter];
      if (mSymbolTable->isEscapeSymbol(symbol)) {
        *mIncompressibleIter++ = *sourceIter;
      }
      return symbol;
    } else {
      return *mSymbolTable->lookupUnsafe(*sourceIter);
    }
  };

  EncoderSymbolMapperInterface() = default;
  EncoderSymbolMapperInterface(const symbolTable_type& symbolTable,
                               incompressible_IT incompressibleIter = nullptr) : mSymbolTable{&symbolTable},
                                                                                 mIncompressibleIter{incompressibleIter} {};

  const symbolTable_type* mSymbolTable{};
  incompressible_iterator mIncompressibleIter{};
};

template <typename symbolTable_T, typename coder_T, typename incompressible_IT = std::nullptr_t>
class EncoderSymbolMapper : public EncoderSymbolMapperInterface<symbolTable_T,
                                                                coder_T,
                                                                incompressible_IT,
                                                                EncoderSymbolMapper<symbolTable_T, coder_T, incompressible_IT>>
{
  using base_type = EncoderSymbolMapperInterface<symbolTable_T, coder_T, incompressible_IT, EncoderSymbolMapper<symbolTable_T, coder_T, incompressible_IT>>;

 public:
  using symbolTable_type = typename base_type::symbolTable_type;
  using coder_type = typename base_type::coder_type;
  using size_type = typename base_type::size_type;
  using difference_type = typename base_type::difference_type;
  using source_type = typename base_type::source_type;
  using symbol_type = typename base_type::symbol_type;
  using coderSymbol_type = typename base_type::coderSymbol_type;
  using incompressible_iterator = typename base_type::incompressible_iterator;

  static_assert(coder_type::getNstreams() == 1);

  EncoderSymbolMapper() = default;

  EncoderSymbolMapper(const symbolTable_type& symbolTable, incompressible_IT incompressibleIter = nullptr) : base_type{symbolTable, incompressibleIter} {};

  template <typename source_IT>
  [[nodiscard]] inline source_IT unpackSymbols(source_IT sourceIter, coderSymbol_type& unpacked)
  {
    unpacked = this->lookupSymbol(sourceIter);
    return --sourceIter;
  };

  template <typename source_IT>
  [[nodiscard]] inline source_IT unpackSymbols(source_IT sourceIter, coderSymbol_type& unpacked, size_type nStreams)
  {
    return this->unpackSymbols(sourceIter, unpacked);
  };
};

template <size_t streamingLowerBound_V, typename symbolTable_T, typename incompressible_IT>
class EncoderSymbolMapper<symbolTable_T,
                          CompatEncoderImpl<streamingLowerBound_V>,
                          incompressible_IT> : public EncoderSymbolMapperInterface<symbolTable_T,
                                                                                   CompatEncoderImpl<streamingLowerBound_V>,
                                                                                   incompressible_IT,
                                                                                   EncoderSymbolMapper<symbolTable_T, incompressible_IT>>
{
  using base_type = EncoderSymbolMapperInterface<symbolTable_T, CompatEncoderImpl<streamingLowerBound_V>, incompressible_IT, EncoderSymbolMapper<symbolTable_T, incompressible_IT>>;

 public:
  using symbolTable_type = typename base_type::symbolTable_type;
  using coder_type = typename base_type::coder_type;
  using size_type = typename base_type::size_type;
  using difference_type = typename base_type::difference_type;
  using source_type = typename base_type::source_type;
  using symbol_type = typename base_type::symbol_type;
  using coderSymbol_type = typename base_type::coderSymbol_type;
  using incompressible_iterator = typename base_type::incompressible_iterator;

  static_assert(coder_type::getNstreams() == 1);

  EncoderSymbolMapper() = default;

  EncoderSymbolMapper(const symbolTable_type& symbolTable, incompressible_IT incompressibleIter = nullptr) : base_type{symbolTable, incompressibleIter} {};

  template <typename source_IT>
  [[nodiscard]] inline source_IT unpackSymbols(source_IT sourceIter, coderSymbol_type& unpacked)
  {
    unpacked = &this->lookupSymbol(sourceIter);
    return --sourceIter;
  };

  template <typename source_IT>
  [[nodiscard]] inline source_IT unpackSymbols(source_IT sourceIter, coderSymbol_type& unpacked, size_type nStreams)
  {
    return this->unpackSymbols(sourceIter, unpacked);
  };
};

#ifdef RANS_SINGLE_STREAM
template <size_t streamingLowerBound_V, typename symbolTable_T, typename incompressible_IT>
class EncoderSymbolMapper<symbolTable_T,
                          SingleStreamEncoderImpl<streamingLowerBound_V>,
                          incompressible_IT> : public EncoderSymbolMapperInterface<symbolTable_T,
                                                                                   SingleStreamEncoderImpl<streamingLowerBound_V>,
                                                                                   incompressible_IT,
                                                                                   EncoderSymbolMapper<symbolTable_T, incompressible_IT>>
{
  using base_type = EncoderSymbolMapperInterface<symbolTable_T, SingleStreamEncoderImpl<streamingLowerBound_V>, incompressible_IT, EncoderSymbolMapper<symbolTable_T, incompressible_IT>>;

 public:
  using symbolTable_type = typename base_type::symbolTable_type;
  using coder_type = typename base_type::coder_type;
  using size_type = typename base_type::size_type;
  using difference_type = typename base_type::difference_type;
  using source_type = typename base_type::source_type;
  using symbol_type = typename base_type::symbol_type;
  using coderSymbol_type = typename base_type::coderSymbol_type;
  using incompressible_iterator = typename base_type::incompressible_iterator;

  static_assert(coder_type::getNstreams() == 1);

  EncoderSymbolMapper() = default;

  EncoderSymbolMapper(const symbolTable_type& symbolTable, incompressible_IT incompressibleIter = nullptr) : base_type{symbolTable, incompressibleIter} {};

  template <typename source_IT>
  [[nodiscard]] inline source_IT unpackSymbols(source_IT sourceIter, coderSymbol_type& unpacked)
  {
    unpacked = &this->lookupSymbol(sourceIter);
    return --sourceIter;
  };

  template <typename source_IT>
  [[nodiscard]] inline source_IT unpackSymbols(source_IT sourceIter, coderSymbol_type& unpacked, size_type nStreams)
  {
    return this->unpackSymbols(sourceIter, unpacked);
  };
};
#endif /* RANS_SINGLE_STREAM */

#ifdef RANS_SSE
template <size_t streamingLowerBound_V, typename symbolTable_T, typename incompressible_IT>
class EncoderSymbolMapper<symbolTable_T,
                          SSEEncoderImpl<streamingLowerBound_V>,
                          incompressible_IT> : public EncoderSymbolMapperInterface<symbolTable_T,
                                                                                   SSEEncoderImpl<streamingLowerBound_V>,
                                                                                   incompressible_IT,
                                                                                   EncoderSymbolMapper<symbolTable_T, incompressible_IT>>
{
  using base_type = EncoderSymbolMapperInterface<symbolTable_T, SSEEncoderImpl<streamingLowerBound_V>, incompressible_IT, EncoderSymbolMapper<symbolTable_T, incompressible_IT>>;

 public:
  using symbolTable_type = typename base_type::symbolTable_type;
  using coder_type = typename base_type::coder_type;
  using size_type = typename base_type::size_type;
  using difference_type = typename base_type::difference_type;
  using source_type = typename base_type::source_type;
  using symbol_type = typename base_type::symbol_type;
  using coderSymbol_type = typename base_type::coderSymbol_type;
  using incompressible_iterator = typename base_type::incompressible_iterator;

  static_assert(coder_type::getNstreams() == 4);

  EncoderSymbolMapper() = default;

  EncoderSymbolMapper(const symbolTable_type& symbolTable, incompressible_IT incompressibleIter = nullptr) : base_type{symbolTable, incompressibleIter} {};

  template <typename source_IT>
  [[nodiscard]] inline source_IT unpackSymbols(source_IT sourceIter, coderSymbol_type& unpacked)
  {
    using namespace simd;
    std::array<const symbol_type*, 4> ret;
    ret[3] = &this->lookupSymbol(sourceIter - 0);
    ret[2] = &this->lookupSymbol(sourceIter - 1);
    ret[1] = &this->lookupSymbol(sourceIter - 2);
    ret[0] = &this->lookupSymbol(sourceIter - 3);

    aosToSoa(gsl::make_span(ret).template subspan<0, 2>(), &unpacked.frequencies[0], &unpacked.cumulativeFrequencies[0]);
    aosToSoa(gsl::make_span(ret).template subspan<2, 2>(), &unpacked.frequencies[1], &unpacked.cumulativeFrequencies[1]);

    return utils::advanceIter(sourceIter, -coder_type::getNstreams());
  };

  template <typename source_IT>
  [[nodiscard]] inline source_IT unpackSymbols(source_IT sourceIter, coderSymbol_type& unpacked, size_type nActiveStreams)
  {
    using namespace internal::simd;

    difference_type currentStream = nActiveStreams;

    epi32_t<SIMDWidth::SSE, 2> frequencies;
    epi32_t<SIMDWidth::SSE, 2> cumulativeFrequencies;

    while (currentStream-- > 0) {
      const auto& symbol = this->lookupSymbol(sourceIter--);
      frequencies(currentStream) = symbol.getFrequency();
      cumulativeFrequencies(currentStream) = symbol.getCumulative();
    }

    unpacked.frequencies[0] = load(frequencies[0]);
    unpacked.frequencies[1] = load(frequencies[1]);

    unpacked.cumulativeFrequencies[0] = load(cumulativeFrequencies[0]);
    unpacked.cumulativeFrequencies[1] = load(cumulativeFrequencies[1]);

    return sourceIter;
  };
};
#endif /* RANS_SSE */

#ifdef RANS_AVX2
template <size_t streamingLowerBound_V, typename symbolTable_T, typename incompressible_IT>
class EncoderSymbolMapper<symbolTable_T,
                          AVXEncoderImpl<streamingLowerBound_V>,
                          incompressible_IT> : public EncoderSymbolMapperInterface<symbolTable_T,
                                                                                   AVXEncoderImpl<streamingLowerBound_V>,
                                                                                   incompressible_IT,
                                                                                   EncoderSymbolMapper<symbolTable_T, incompressible_IT>>
{
  using base_type = EncoderSymbolMapperInterface<symbolTable_T, AVXEncoderImpl<streamingLowerBound_V>, incompressible_IT, EncoderSymbolMapper<symbolTable_T, incompressible_IT>>;

 public:
  using symbolTable_type = typename base_type::symbolTable_type;
  using coder_type = typename base_type::coder_type;
  using size_type = typename base_type::size_type;
  using difference_type = typename base_type::difference_type;
  using source_type = typename base_type::source_type;
  using symbol_type = typename base_type::symbol_type;
  using coderSymbol_type = typename base_type::coderSymbol_type;
  using incompressible_iterator = typename base_type::incompressible_iterator;

  static_assert(coder_type::getNstreams() == 8);

  EncoderSymbolMapper() = default;

  EncoderSymbolMapper(const symbolTable_type& symbolTable, incompressible_IT incompressibleIter = nullptr) : base_type{symbolTable, incompressibleIter} {};

  template <typename source_IT>
  [[nodiscard]] inline source_IT unpackSymbols(source_IT sourceIter, coderSymbol_type& unpacked)
  {
    using namespace simd;
    std::array<const Symbol*, 8> ret;
    ret[7] = &this->lookupSymbol(sourceIter - 0);
    ret[6] = &this->lookupSymbol(sourceIter - 1);
    ret[5] = &this->lookupSymbol(sourceIter - 2);
    ret[4] = &this->lookupSymbol(sourceIter - 3);
    ret[3] = &this->lookupSymbol(sourceIter - 4);
    ret[2] = &this->lookupSymbol(sourceIter - 5);
    ret[1] = &this->lookupSymbol(sourceIter - 6);
    ret[0] = &this->lookupSymbol(sourceIter - 7);

    aosToSoa(gsl::make_span(ret).template subspan<0, 4>(), &unpacked.frequencies[0], &unpacked.cumulativeFrequencies[0]);
    aosToSoa(gsl::make_span(ret).template subspan<4, 4>(), &unpacked.frequencies[1], &unpacked.cumulativeFrequencies[1]);

    return utils::advanceIter(sourceIter, -coder_type::getNstreams());
  };

  template <typename source_IT>
  [[nodiscard]] inline source_IT unpackSymbols(source_IT sourceIter, coderSymbol_type& unpacked, size_type nActiveStreams)
  {
    using namespace internal::simd;

    difference_type currentStream = nActiveStreams;

    epi32_t<SIMDWidth::SSE, 2> frequencies;
    epi32_t<SIMDWidth::SSE, 2> cumulativeFrequencies;

    while (currentStream-- > 0) {
      const auto& symbol = this->lookupSymbol(sourceIter--);
      frequencies(currentStream) = symbol.getFrequency();
      cumulativeFrequencies(currentStream) = symbol.getCumulative();
    }

    unpacked.frequencies[0] = load(frequencies[0]);
    unpacked.frequencies[1] = load(frequencies[1]);

    unpacked.cumulativeFrequencies[0] = load(cumulativeFrequencies[0]);
    unpacked.cumulativeFrequencies[1] = load(cumulativeFrequencies[1]);

    return sourceIter;
  };
};
#endif /* RANS_AVX2 */

} // namespace o2::rans::internal

#endif /* RANS_INTERNAL_ENCODE_ENCODERSYMBOLMAPPER_H_ */