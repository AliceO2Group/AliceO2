// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   SymbolTable.h
/// @author Michael Lettrich
/// @since  2019-06-21
/// @brief  Container for information needed to encode/decode a symbol of the alphabet

#ifndef RANS_INTERNAL_SIMD_SYMBOLMAPPER_H
#define RANS_INTERNAL_SIMD_SYMBOLMAPPER_H

#include "rANS/internal/backend/simd/Symbol.h"
#include "rANS/internal/backend/simd/SymbolTable.h"
#include "rANS/internal/backend/simd/types.h"
#include "rANS/internal/backend/simd/kernel.h"
#include "rANS/internal/backend/simd/utils.h"
#include "rANS/internal/helper.h"

namespace o2
{
namespace rans
{
namespace internal
{
namespace simd
{

// struct UnrolledSymbols {
//   epi32_t<SIMDWidth::SSE, 2> frequencies;
//   epi32_t<SIMDWidth::SSE, 2> cumulativeFrequencies;
// };

// template <typename source_IT>
// inline const internal::simd::Symbol* lookupSymbol(source_IT iter, const simd::SymbolTable& symbolTable, std::vector<typename std::iterator_traits<source_IT>::value_type>& literals) noexcept
// {
//   const auto symbol = *iter;
//   const auto* encoderSymbol = &(symbolTable[symbol]);
//   if (symbolTable.isEscapeSymbol(*encoderSymbol)) {
//     literals.push_back(symbol);
//   }
//   return encoderSymbol;
// };

// template <typename source_IT, SIMDWidth width_V>
// std::tuple<source_IT, UnrolledSymbols> getSymbols(source_IT symbolIter, const o2::rans::internal::simd::SymbolTable& symbolTable, std::vector<typename std::iterator_traits<source_IT>::value_type>& literals)
// {
//   UnrolledSymbols unrolledSymbols;

//   if constexpr (width_V == SIMDWidth::SSE) {
//     AlignedArray<const Symbol*, simd::SIMDWidth::SSE, 4> ret;
//     ret[3] = lookupSymbol(symbolIter - 1, symbolTable, literals);
//     ret[2] = lookupSymbol(symbolIter - 2, symbolTable, literals);
//     ret[1] = lookupSymbol(symbolIter - 3, symbolTable, literals);
//     ret[0] = lookupSymbol(symbolIter - 4, symbolTable, literals);

//     AlignedArray<symbol_t, SIMDWidth::SSE, 4> symbols{
//       static_cast<symbol_t>(*(symbolIter - 4)),
//       static_cast<symbol_t>(*(symbolIter - 3)),
//       static_cast<symbol_t>(*(symbolIter - 2)),
//       static_cast<symbol_t>(*(symbolIter - 1)),
//     };

//     const __m128i symbolsVec = load(toConstSIMDView(symbols));
//     const __m128i minVec = _mm_set1_epi32(symbolTable.getMinSymbol());
//     const __m128i maxVec = _mm_set1_epi32(symbolTable.getMaxSymbol());
//     const __m128i escapeIdxVec = _mm_set1_epi32(symbolTable.size() - 1);

//     // mask
//     const __m128i isGT = _mm_cmpgt_epi32(symbolsVec, maxVec);
//     const __m128i isLT = _mm_cmplt_epi32(symbolsVec, minVec);
//     const __m128i isOutOfRange = _mm_or_si128(isGT, isLT);

//     const __m128i offsets = _mm_blendv_epi8(_mm_sub_epi32(symbolsVec, minVec), escapeIdxVec, isOutOfRange);

//     LOG(info) << offsets << store<uint32_t>(offsets);

//     aosToSoa(ArrayView{ret}.template subView<0, 2>(),
//              toSIMDView(unrolledSymbols.frequencies).template subView<0, 1>(),
//              toSIMDView(unrolledSymbols.cumulativeFrequencies).template subView<0, 1>());
//     aosToSoa(ArrayView{ret}.template subView<2, 2>(),
//              toSIMDView(unrolledSymbols.frequencies).template subView<1, 1>(),
//              toSIMDView(unrolledSymbols.cumulativeFrequencies).template subView<1, 1>());
//     //aosToSoa(ret, toSIMDView(unrolledSymbols.frequencies), toSIMDView(unrolledSymbols.cumulativeFrequencies));
//     return {symbolIter - 4, unrolledSymbols};
//   } else {
//     AlignedArray<const Symbol*, simd::SIMDWidth::SSE, 8> ret;
//     ret[7] = lookupSymbol(symbolIter - 1, symbolTable, literals);
//     ret[6] = lookupSymbol(symbolIter - 2, symbolTable, literals);
//     ret[5] = lookupSymbol(symbolIter - 3, symbolTable, literals);
//     ret[4] = lookupSymbol(symbolIter - 4, symbolTable, literals);
//     ret[3] = lookupSymbol(symbolIter - 5, symbolTable, literals);
//     ret[2] = lookupSymbol(symbolIter - 6, symbolTable, literals);
//     ret[1] = lookupSymbol(symbolIter - 7, symbolTable, literals);
//     ret[0] = lookupSymbol(symbolIter - 8, symbolTable, literals);

//     aosToSoa(ArrayView{ret}.template subView<0, 4>(),
//              toSIMDView(unrolledSymbols.frequencies).template subView<0, 1>(),
//              toSIMDView(unrolledSymbols.cumulativeFrequencies).template subView<0, 1>());
//     aosToSoa(ArrayView{ret}.template subView<4, 4>(),
//              toSIMDView(unrolledSymbols.frequencies).template subView<1, 1>(),
//              toSIMDView(unrolledSymbols.cumulativeFrequencies).template subView<1, 1>());
//     return {symbolIter - 8, unrolledSymbols};
//   }
// };

inline constexpr std::array<epi8_t<SIMDWidth::SSE>, 16>
  SSEIncompressibleMapping{{
    {0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8}, //0b0000   0xFFFFu
    {0x00_u8, 0x01_u8, 0x02_u8, 0x03_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8}, //0b0001   0x0FFFu
    {0x04_u8, 0x05_u8, 0x06_u8, 0x07_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8}, //0b0010   0x1FFFu
    {0x04_u8, 0x05_u8, 0x06_u8, 0x07_u8, 0x00_u8, 0x01_u8, 0x02_u8, 0x03_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8}, //0b0011   0x10FFu
    {0x08_u8, 0x09_u8, 0x0A_u8, 0x0B_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8}, //0b0100   0x2FFFu
    {0x08_u8, 0x09_u8, 0x0A_u8, 0x0B_u8, 0x00_u8, 0x01_u8, 0x02_u8, 0x03_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8}, //0b0101   0x20FFu
    {0x08_u8, 0x09_u8, 0x0A_u8, 0x0B_u8, 0x04_u8, 0x05_u8, 0x06_u8, 0x07_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8}, //0b0110   0x21FFu
    {0x08_u8, 0x09_u8, 0x0A_u8, 0x0B_u8, 0x04_u8, 0x05_u8, 0x06_u8, 0x07_u8, 0x00_u8, 0x01_u8, 0x02_u8, 0x03_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8}, //0b0111   0x210Fu
    {0x0C_u8, 0x0D_u8, 0x0E_u8, 0x0F_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8}, //0b1000   0x3FFFu
    {0x0C_u8, 0x0D_u8, 0x0E_u8, 0x0F_u8, 0x00_u8, 0x01_u8, 0x02_u8, 0x03_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8}, //0b1001   0x30FFu
    {0x0C_u8, 0x0D_u8, 0x0E_u8, 0x0F_u8, 0x04_u8, 0x05_u8, 0x06_u8, 0x07_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8}, //0b1010   0x31FFu
    {0x0C_u8, 0x0D_u8, 0x0E_u8, 0x0F_u8, 0x04_u8, 0x05_u8, 0x06_u8, 0x07_u8, 0x00_u8, 0x01_u8, 0x02_u8, 0x03_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8}, //0b1011   0x310Fu
    {0x0C_u8, 0x0D_u8, 0x0E_u8, 0x0F_u8, 0x08_u8, 0x09_u8, 0x0A_u8, 0x0B_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8}, //0b1100   0x32FFu
    {0x0C_u8, 0x0D_u8, 0x0E_u8, 0x0F_u8, 0x08_u8, 0x09_u8, 0x0A_u8, 0x0B_u8, 0x00_u8, 0x01_u8, 0x02_u8, 0x03_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8}, //0b1101   0x320Fu
    {0x0C_u8, 0x0D_u8, 0x0E_u8, 0x0F_u8, 0x08_u8, 0x09_u8, 0x0A_u8, 0x0B_u8, 0x04_u8, 0x05_u8, 0x06_u8, 0x07_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8, 0xFF_u8}, //0b1110   0x321Fu
    {0x0C_u8, 0x0D_u8, 0x0E_u8, 0x0F_u8, 0x08_u8, 0x09_u8, 0x0A_u8, 0x0B_u8, 0x04_u8, 0x05_u8, 0x06_u8, 0x07_u8, 0x00_u8, 0x01_u8, 0x02_u8, 0x03_u8}  //0b1111   0x3210u
  }};

template <SIMDWidth simdWidth_V>
class SymbolMapper;

template <>
class SymbolMapper<SIMDWidth::SSE>
{

 public:
  SymbolMapper(const SymbolTable& symbolTable) : mSymbolTable(&symbolTable),
                                                 mMinVec(_mm_set1_epi32(mSymbolTable->getMinSymbol())),
                                                 mMaxVec(_mm_set1_epi32(mSymbolTable->getMaxSymbol())),
                                                 mEscapeIdxVec(_mm_set1_epi32(mSymbolTable->size() - 1)),
                                                 mIncompressibleCumulatedFrequencies(_mm_set1_epi32(mSymbolTable->getEscapeSymbol().getCumulative())){};

  //   template <typename source_IT>
  //   std::tuple<source_IT, UnrolledSymbols> readSymbols(source_IT symbolIter);
  template <typename source_IT, typename literal_IT>
  std::tuple<source_IT, literal_IT, UnrolledSymbols> readSymbols(source_IT symbolIter, literal_IT literalIter);

 private:
  const SymbolTable* mSymbolTable{};
  __m128i mMinVec;
  __m128i mMaxVec;
  __m128i mEscapeIdxVec;
  __m128i mIncompressibleCumulatedFrequencies;
};

template <typename source_IT, typename literal_IT>
inline auto SymbolMapper<SIMDWidth::SSE>::readSymbols(source_IT symbolIter, literal_IT literalIter) -> std::tuple<source_IT, literal_IT, UnrolledSymbols>
{
  using source_t = typename std::iterator_traits<source_IT>::value_type;

  UnrolledSymbols unrolledSymbols;

  __m128i symbolsVec;

  if constexpr (std::is_pointer_v<source_IT>) {

    symbolsVec = _mm_loadu_si128(reinterpret_cast<const __m128i_u*>(symbolIter - 4));
    if constexpr (std::is_same_v<source_t, uint8_t>) {
      symbolsVec = _mm_cvtepu8_epi32(symbolsVec);
    } else if constexpr (std::is_same_v<source_t, int8_t>) {
      symbolsVec = _mm_cvtepi8_epi32(symbolsVec);
    } else if constexpr (std::is_same_v<source_t, uint16_t>) {
      symbolsVec = _mm_cvtepu16_epi32(symbolsVec);
    } else if constexpr (std::is_same_v<source_t, int16_t>) {
      symbolsVec = _mm_cvtepi16_epi32(symbolsVec);
    }
    //no conversion needed for int32
  } else {
    AlignedArray<symbol_t, SIMDWidth::SSE, 4> symbols{
      static_cast<symbol_t>(*(symbolIter - 4)),
      static_cast<symbol_t>(*(symbolIter - 3)),
      static_cast<symbol_t>(*(symbolIter - 2)),
      static_cast<symbol_t>(*(symbolIter - 1)),
    };
    symbolsVec = load(toConstSIMDView(symbols));
  }

  // range check
  const __m128i isGT = _mm_cmpgt_epi32(symbolsVec, mMaxVec);
  const __m128i isLT = _mm_cmplt_epi32(symbolsVec, mMinVec);
  const __m128i isOutOfRange = _mm_or_si128(isGT, isLT);
  // make sure we're in the right range
  const __m128i offsetsVec = _mm_blendv_epi8(_mm_sub_epi32(symbolsVec, mMinVec), mEscapeIdxVec, isOutOfRange);

  auto offset = store<uint32_t>(offsetsVec);

  __m128i symbol0 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(&mSymbolTable->at(offset[0])));
  __m128i symbol1 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(&mSymbolTable->at(offset[1])));
  __m128i symbol2 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(&mSymbolTable->at(offset[2])));
  __m128i symbol3 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(&mSymbolTable->at(offset[3])));

  // Unpack Symbols
  __m128i merged0 = _mm_unpacklo_epi32(symbol0, symbol1);
  __m128i merged1 = _mm_unpacklo_epi32(symbol2, symbol3);
  __m128i frequencies = _mm_unpacklo_epi64(merged0, merged1);
  __m128i cumulativeFrequencies = _mm_unpackhi_epi64(merged0, merged1);

  // find all incompressibleSymbols and pass them to a literals vector
  const __m128i isIncompressible = _mm_cmpeq_epi32(cumulativeFrequencies, mIncompressibleCumulatedFrequencies);

  const uint32_t id = _mm_movemask_ps(_mm_castsi128_ps(isIncompressible));

  if (id > 0) {
    const uint32_t nIncompressible = _mm_popcnt_u32(id);
    __m128i shuffleMaskVec = load(toConstSIMDView(SSEIncompressibleMapping[id]));
    symbolsVec = _mm_shuffle_epi8(symbolsVec, shuffleMaskVec);

    if constexpr (std::is_pointer_v<source_IT>) {
      // store;
      if constexpr (std::is_same_v<source_t, uint8_t>) {
        symbolsVec = _mm_packus_epi32(symbolsVec, symbolsVec);
        symbolsVec = _mm_packus_epi16(symbolsVec, symbolsVec);
      } else if constexpr (std::is_same_v<source_t, int8_t>) {
        symbolsVec = _mm_packs_epi32(symbolsVec, symbolsVec);
        symbolsVec = _mm_packs_epi16(symbolsVec, symbolsVec);
      } else if constexpr (std::is_same_v<source_t, uint16_t>) {
        symbolsVec = _mm_packus_epi32(symbolsVec, symbolsVec);
      } else if constexpr (std::is_same_v<source_t, int16_t>) {
        symbolsVec = _mm_packs_epi32(symbolsVec, symbolsVec);
      }
      _mm_storeu_si128(reinterpret_cast<__m128i_u*>(literalIter), symbolsVec);
      literalIter += nIncompressible;
    } else {
      auto incompressibleSymbols = store<uint32_t>(symbolsVec);
      for (size_t i = 0; i < nIncompressible; ++i) {
        *literalIter = static_cast<typename std::iterator_traits<source_IT>::value_type>(incompressibleSymbols[i]);
        ++literalIter;
      }
    }
  };

  // if (nIncompressible > 0) {
  //   LOG(info) << "Cumul " << store<uint32_t>(mIncompressibleCumulatedFrequencies) << " vs " << store<uint32_t>(cumulativeFrequencies);
  //   LOG(info) << "Incompressible: " << asHex(store<uint32_t>(isIncompressible)) << "at " << store<uint32_t>(isIncompressible);
  // }

  UnrolledSymbols unrolledSymbols2;

  store(frequencies, toSIMDView(unrolledSymbols2.frequencies).template subView<0, 1>());
  store(_mm_bsrli_si128(frequencies, 8), toSIMDView(unrolledSymbols2.frequencies).template subView<1, 1>());

  store(cumulativeFrequencies, toSIMDView(unrolledSymbols2.cumulativeFrequencies).template subView<0, 1>());
  store(_mm_bsrli_si128(cumulativeFrequencies, 8), toSIMDView(unrolledSymbols2.cumulativeFrequencies).template subView<1, 1>());

  //auto incompressibleSymbols = store<uint32_t>(symbolsVec);

  // std::vector<typename std::iterator_traits<source_IT>::value_type> fakeLiterals;
  // fakeLiterals.reserve(4);

  // AlignedArray<const Symbol*, simd::SIMDWidth::SSE, 4>
  //   ret;
  // ret[3] = lookupSymbol(symbolIter - 1, *mSymbolTable, fakeLiterals);
  // ret[2] = lookupSymbol(symbolIter - 2, *mSymbolTable, fakeLiterals);
  // ret[1] = lookupSymbol(symbolIter - 3, *mSymbolTable, fakeLiterals);
  // ret[0] = lookupSymbol(symbolIter - 4, *mSymbolTable, fakeLiterals);

  // aosToSoa(ArrayView{ret}.template subView<0, 2>(),
  //          toSIMDView(unrolledSymbols.frequencies).template subView<0, 1>(),
  //          toSIMDView(unrolledSymbols.cumulativeFrequencies).template subView<0, 1>());
  // aosToSoa(ArrayView{ret}.template subView<2, 2>(),
  //          toSIMDView(unrolledSymbols.frequencies).template subView<1, 1>(),
  //          toSIMDView(unrolledSymbols.cumulativeFrequencies).template subView<1, 1>());
  // //aosToSoa(ret, toSIMDView(unrolledSymbols.frequencies), toSIMDView(unrolledSymbols.cumulativeFrequencies));

  // auto checkEqual = [](auto a, auto b) {
  //   if (a != b) {
  //     LOGP(warning, "{}!={}", a, b);
  //   }
  // };

  // // checks
  // //LOG(info) << "frequency check";
  // checkEqual(unrolledSymbols2.frequencies[0], unrolledSymbols.frequencies[0]);
  // checkEqual(unrolledSymbols2.frequencies[1], unrolledSymbols.frequencies[1]);
  // checkEqual(unrolledSymbols2.frequencies[4], unrolledSymbols.frequencies[4]);
  // checkEqual(unrolledSymbols2.frequencies[5], unrolledSymbols.frequencies[5]);

  // //LOG(info) << "cumulativeFrequencies check";
  // checkEqual(unrolledSymbols2.cumulativeFrequencies[0], unrolledSymbols.cumulativeFrequencies[0]);
  // checkEqual(unrolledSymbols2.cumulativeFrequencies[1], unrolledSymbols.cumulativeFrequencies[1]);
  // checkEqual(unrolledSymbols2.cumulativeFrequencies[4], unrolledSymbols.cumulativeFrequencies[4]);
  // checkEqual(unrolledSymbols2.cumulativeFrequencies[5], unrolledSymbols.cumulativeFrequencies[5]);

  // // LOG(info) << "checking incompressible sizes";
  // checkEqual(_mm_popcnt_u32(id), fakeLiterals.size());

  // // LOG(info) << "incompressible symbols check";
  // for (size_t i = 0; i < fakeLiterals.size(); ++i) {
  //   checkEqual(incompressibleSymbols[i], fakeLiterals[i]);
  // }

  // for (size_t i = 0; i < nIncompressible; ++i) {
  //   *literalIter = static_cast<typename std::iterator_traits<source_IT>::value_type>(incompressibleSymbols[i]);
  //   ++literalIter;
  // }

  // for (auto& s : fakeLiterals) {
  //   literals.push_back(s);
  // }

  return {symbolIter - 4, literalIter, unrolledSymbols2};
};

#ifdef __AVX2__

template <>
class SymbolMapper<SIMDWidth::AVX>
{

 public:
  SymbolMapper(const SymbolTable& symbolTable) : mMinVec(_mm256_set1_epi32(symbolTable.getMinSymbol())),
                                                 mMaxVec(_mm256_set1_epi32(symbolTable.getMaxSymbol())),
                                                 mEscapeIdxVec(_mm256_set1_epi32(symbolTable.size() - 1)),
                                                 mIncompressibleCumulatedFrequencies(_mm_set1_epi32(symbolTable.getEscapeSymbol().getCumulative())),
                                                 mSymbolTable(&symbolTable){};

  //   template <typename source_IT>
  //   std::tuple<source_IT, UnrolledSymbols> readSymbols(source_IT symbolIter);
  template <typename source_IT, typename literal_IT>
  std::tuple<source_IT, literal_IT, UnrolledSymbols> readSymbols(source_IT symbolIter, literal_IT literalIter);

 private:
  __m256i mMinVec;
  __m256i mMaxVec;
  __m256i mEscapeIdxVec;
  __m128i mIncompressibleCumulatedFrequencies;
  const SymbolTable* mSymbolTable{};
};

template <typename source_IT, typename literal_IT>
inline auto SymbolMapper<SIMDWidth::AVX>::readSymbols(source_IT symbolIter, literal_IT literalIter) -> std::tuple<source_IT, literal_IT, UnrolledSymbols>
{
  using source_t = typename std::iterator_traits<source_IT>::value_type;

  UnrolledSymbols unrolledSymbols;

  __m256i symbolsVec;

  if constexpr (std::is_pointer_v<source_IT>) {
    if constexpr (std::is_same_v<source_t, uint8_t>) {
      auto inVec = _mm_loadu_si128(reinterpret_cast<const __m128i_u*>(symbolIter - 8));
      symbolsVec = _mm256_cvtepu8_epi32(inVec);
    } else if constexpr (std::is_same_v<source_t, int8_t>) {
      auto inVec = _mm_loadu_si128(reinterpret_cast<const __m128i_u*>(symbolIter - 8));
      symbolsVec = _mm256_cvtepi8_epi32(inVec);
    } else if constexpr (std::is_same_v<source_t, uint16_t>) {
      auto inVec = _mm_loadu_si128(reinterpret_cast<const __m128i_u*>(symbolIter - 8));
      symbolsVec = _mm256_cvtepu16_epi32(inVec);
    } else if constexpr (std::is_same_v<source_t, int16_t>) {
      auto inVec = _mm_loadu_si128(reinterpret_cast<const __m128i_u*>(symbolIter - 8));
      symbolsVec = _mm256_cvtepi16_epi32(inVec);
    } else {
      symbolsVec = _mm256_loadu_si256(reinterpret_cast<const __m256i_u*>(symbolIter - 8));
    }
    //no conversion needed for int32
  } else {
    AlignedArray<symbol_t, SIMDWidth::AVX, 8> symbols{
      static_cast<symbol_t>(*(symbolIter - 8)),
      static_cast<symbol_t>(*(symbolIter - 7)),
      static_cast<symbol_t>(*(symbolIter - 6)),
      static_cast<symbol_t>(*(symbolIter - 5)),
      static_cast<symbol_t>(*(symbolIter - 4)),
      static_cast<symbol_t>(*(symbolIter - 3)),
      static_cast<symbol_t>(*(symbolIter - 2)),
      static_cast<symbol_t>(*(symbolIter - 1)),
    };
    symbolsVec = load(toConstSIMDView(symbols));
  }

  auto symbolsVecCopy = symbolsVec;

  // range check
  __m256i offsetsVec = _mm256_sub_epi32(symbolsVec, mMinVec);
  const __m256i isGT = _mm256_cmpgt_epi32(offsetsVec, mEscapeIdxVec);
  const __m256i isLT = _mm256_cmpgt_epi32(_mm256_setzero_si256(), offsetsVec);
  const __m256i isOutOfRange = _mm256_or_si256(isGT, isLT);
  // make sure we're in the right range
  offsetsVec = _mm256_blendv_epi8(offsetsVec, mEscapeIdxVec, isOutOfRange);

  auto offset = store<uint32_t>(offsetsVec);

  __m128i symbols[8];
  const auto tableBegin = mSymbolTable->data();

  symbols[0] = _mm_loadu_si128(reinterpret_cast<__m128i const*>(tableBegin + offset[0]));
  symbols[1] = _mm_loadu_si128(reinterpret_cast<__m128i const*>(tableBegin + offset[1]));
  symbols[2] = _mm_loadu_si128(reinterpret_cast<__m128i const*>(tableBegin + offset[2]));
  symbols[3] = _mm_loadu_si128(reinterpret_cast<__m128i const*>(tableBegin + offset[3]));
  symbols[4] = _mm_loadu_si128(reinterpret_cast<__m128i const*>(tableBegin + offset[4]));
  symbols[5] = _mm_loadu_si128(reinterpret_cast<__m128i const*>(tableBegin + offset[5]));
  symbols[6] = _mm_loadu_si128(reinterpret_cast<__m128i const*>(tableBegin + offset[6]));
  symbols[7] = _mm_loadu_si128(reinterpret_cast<__m128i const*>(tableBegin + offset[7]));

  __m128i frequencies[2];
  __m128i cumulativeFrequencies[2];
  __m128i merged[2];

  // Unpack Symbols
  merged[0] = _mm_unpacklo_epi32(symbols[0], symbols[1]);
  merged[1] = _mm_unpacklo_epi32(symbols[2], symbols[3]);
  frequencies[0] = _mm_unpacklo_epi64(merged[0], merged[1]);
  cumulativeFrequencies[0] = _mm_unpackhi_epi64(merged[0], merged[1]);

  merged[0] = _mm_unpacklo_epi32(symbols[4], symbols[5]);
  merged[1] = _mm_unpacklo_epi32(symbols[6], symbols[7]);
  frequencies[1] = _mm_unpacklo_epi64(merged[0], merged[1]);
  cumulativeFrequencies[1] = _mm_unpackhi_epi64(merged[0], merged[1]);

  // find all incompressibleSymbols and pass them to a literals vector
  auto encodeIncompressible = [this](__m128i cumulativeFrequencies, __m128i symbolsVec, auto literalIter) {
    const __m128i isIncompressible = _mm_cmpeq_epi32(cumulativeFrequencies, this->mIncompressibleCumulatedFrequencies);

    const uint32_t id = _mm_movemask_ps(_mm_castsi128_ps(isIncompressible));

    if (id > 0) {
      const uint32_t nIncompressible = _mm_popcnt_u32(id);
      __m128i shuffleMaskVec = load(toConstSIMDView(SSEIncompressibleMapping[id]));

      auto symbolsVecOriginal = symbolsVec;
      symbolsVec = _mm_shuffle_epi8(symbolsVec, shuffleMaskVec);

      if constexpr (std::is_pointer_v<source_IT>) {
        // store;
        if constexpr (std::is_same_v<source_t, uint8_t>) {
          symbolsVec = _mm_packus_epi32(symbolsVec, symbolsVec);
          symbolsVec = _mm_packus_epi16(symbolsVec, symbolsVec);
        } else if constexpr (std::is_same_v<source_t, int8_t>) {
          symbolsVec = _mm_packs_epi32(symbolsVec, symbolsVec);
          symbolsVec = _mm_packs_epi16(symbolsVec, symbolsVec);
        } else if constexpr (std::is_same_v<source_t, uint16_t>) {
          symbolsVec = _mm_packus_epi32(symbolsVec, symbolsVec);
        } else if constexpr (std::is_same_v<source_t, int16_t>) {
          symbolsVec = _mm_packs_epi32(symbolsVec, symbolsVec);
        }
        _mm_storeu_si128(reinterpret_cast<__m128i_u*>(literalIter), symbolsVec);
        literalIter += nIncompressible;
      } else {
        auto incompressibleSymbols = store<uint32_t>(symbolsVec);
        for (size_t i = 0; i < nIncompressible; ++i) {
          *literalIter = static_cast<typename std::iterator_traits<source_IT>::value_type>(incompressibleSymbols[i]);
          ++literalIter;
        }
      }
      // LOG(info) << "Cumul " << store<uint32_t>(mIncompressibleCumulatedFrequencies) << " vs " << store<uint32_t>(cumulativeFrequencies);
      // LOG(info) << "Incompressible: " << asHex(store<uint32_t>(isIncompressible)) << "at " << store<uint32_t>(symbolsVecOriginal);
      // LOG(info) << "Permutation mask" << asHex(store<uint8_t>(shuffleMaskVec)) << " from " << store<uint32_t>(symbolsVecOriginal) << " to " << store<source_t>(symbolsVec);
    }
    return literalIter;
  };
  auto oldIter = literalIter;

  literalIter = encodeIncompressible(cumulativeFrequencies[1], _mm256_extracti128_si256(symbolsVec, 1), literalIter);
  literalIter = encodeIncompressible(cumulativeFrequencies[0], _mm256_extracti128_si256(symbolsVec, 0), literalIter);

  UnrolledSymbols unrolledSymbols2;

  store(frequencies[0], toSIMDView(unrolledSymbols2.frequencies).template subView<0, 1>());
  store(frequencies[1], toSIMDView(unrolledSymbols2.frequencies).template subView<1, 1>());

  store(cumulativeFrequencies[0], toSIMDView(unrolledSymbols2.cumulativeFrequencies).template subView<0, 1>());
  store(cumulativeFrequencies[1], toSIMDView(unrolledSymbols2.cumulativeFrequencies).template subView<1, 1>());

  //auto incompressibleSymbols = store<uint32_t>(symbolsVec);

  // std::vector<typename std::iterator_traits<source_IT>::value_type> fakeLiterals;
  // fakeLiterals.reserve(8);
  // for (size_t i = 0; i < 8; ++i) {
  //   fakeLiterals[i] = 0;
  // }

  // AlignedArray<const Symbol*, simd::SIMDWidth::AVX, 8>
  //   ret;
  // ret[7] = lookupSymbol(symbolIter - 1, *mSymbolTable, fakeLiterals);
  // ret[6] = lookupSymbol(symbolIter - 2, *mSymbolTable, fakeLiterals);
  // ret[5] = lookupSymbol(symbolIter - 3, *mSymbolTable, fakeLiterals);
  // ret[4] = lookupSymbol(symbolIter - 4, *mSymbolTable, fakeLiterals);
  // ret[3] = lookupSymbol(symbolIter - 5, *mSymbolTable, fakeLiterals);
  // ret[2] = lookupSymbol(symbolIter - 6, *mSymbolTable, fakeLiterals);
  // ret[1] = lookupSymbol(symbolIter - 7, *mSymbolTable, fakeLiterals);
  // ret[0] = lookupSymbol(symbolIter - 8, *mSymbolTable, fakeLiterals);

  // aosToSoa(ArrayView{ret}.template subView<0, 4>(),
  //          toSIMDView(unrolledSymbols.frequencies).template subView<0, 1>(),
  //          toSIMDView(unrolledSymbols.cumulativeFrequencies).template subView<0, 1>());
  // aosToSoa(ArrayView{ret}.template subView<4, 4>(),
  //          toSIMDView(unrolledSymbols.frequencies).template subView<1, 1>(),
  //          toSIMDView(unrolledSymbols.cumulativeFrequencies).template subView<1, 1>());

  // auto checkEqual = [](auto a, auto b) {
  //   if (a != b) {
  //     LOGP(warning, "{}!={}", a, b);
  //     return false;
  //   } else {
  //     return true;
  //   }
  // };

  // // checks
  // LOG(info) << "frequency check";
  // checkEqual(unrolledSymbols2.frequencies[0], unrolledSymbols.frequencies[0]);
  // checkEqual(unrolledSymbols2.frequencies[1], unrolledSymbols.frequencies[1]);
  // checkEqual(unrolledSymbols2.frequencies[2], unrolledSymbols.frequencies[2]);
  // checkEqual(unrolledSymbols2.frequencies[3], unrolledSymbols.frequencies[3]);
  // checkEqual(unrolledSymbols2.frequencies[4], unrolledSymbols.frequencies[4]);
  // checkEqual(unrolledSymbols2.frequencies[5], unrolledSymbols.frequencies[5]);
  // checkEqual(unrolledSymbols2.frequencies[6], unrolledSymbols.frequencies[6]);
  // checkEqual(unrolledSymbols2.frequencies[7], unrolledSymbols.frequencies[7]);

  // LOG(info) << "cumulativeFrequencies check";
  // checkEqual(unrolledSymbols2.cumulativeFrequencies[0], unrolledSymbols.cumulativeFrequencies[0]);
  // checkEqual(unrolledSymbols2.cumulativeFrequencies[1], unrolledSymbols.cumulativeFrequencies[1]);
  // checkEqual(unrolledSymbols2.cumulativeFrequencies[2], unrolledSymbols.cumulativeFrequencies[2]);
  // checkEqual(unrolledSymbols2.cumulativeFrequencies[3], unrolledSymbols.cumulativeFrequencies[3]);
  // checkEqual(unrolledSymbols2.cumulativeFrequencies[4], unrolledSymbols.cumulativeFrequencies[4]);
  // checkEqual(unrolledSymbols2.cumulativeFrequencies[5], unrolledSymbols.cumulativeFrequencies[5]);
  // checkEqual(unrolledSymbols2.cumulativeFrequencies[6], unrolledSymbols.cumulativeFrequencies[6]);
  // checkEqual(unrolledSymbols2.cumulativeFrequencies[7], unrolledSymbols.cumulativeFrequencies[7]);

  // LOG_IF(info, fakeLiterals.size() > 0) << "checking incompressible sizes";
  // size_t nIncompressible = std::distance(oldIter, literalIter);
  // checkEqual(nIncompressible, fakeLiterals.size());

  // LOG_IF(info, fakeLiterals.size() > 0) << "incompressible symbols check";
  // bool isCorrect = true;
  // for (size_t i = 0; i < nIncompressible; ++i) {
  //   isCorrect = isCorrect && checkEqual(static_cast<uint32_t>(*(oldIter + i)), static_cast<uint32_t>(fakeLiterals[i]));
  // }

  // if (!isCorrect) {
  //   LOG(info) << "nIncompressible: " << nIncompressible;
  //   // LOG(info) << "Cumul " << store<uint32_t>(mIncompressibleCumulatedFrequencies) << " vs " << store<uint32_t>(_mm256_set_m128i(cumulativeFrequencies0, cumulativeFrequencies1));
  //   // LOG(info) << "Incompressible: " << asHex(store<uint32_t>(isIncompressible)) << "at " << store<uint32_t>(symbolsVecCopy);
  //   // LOG(info) << "Permutation mask" << asHex(store<uint32_t>(permutationMask)) << " from " << store<uint32_t>(symbolsVecCopy) << " to " << store<uint32_t>(StreamOutVec);
  //   LOG(info) << "ActualSaved: [" << static_cast<typename std::iterator_traits<source_IT>::value_type>(oldIter[0]) << ", " << static_cast<typename std::iterator_traits<source_IT>::value_type>(oldIter[1]) << ", " << static_cast<typename std::iterator_traits<source_IT>::value_type>(oldIter[2]) << ", " << static_cast<typename std::iterator_traits<source_IT>::value_type>(oldIter[3]) << ", " << static_cast<typename std::iterator_traits<source_IT>::value_type>(oldIter[4]) << ", " << static_cast<typename std::iterator_traits<source_IT>::value_type>(oldIter[5]) << ", " << static_cast<typename std::iterator_traits<source_IT>::value_type>(oldIter[6]) << ", " << static_cast<typename std::iterator_traits<source_IT>::value_type>(oldIter[7]) << "]";
  //   LOG(info) << "Actual: [" << static_cast<typename std::iterator_traits<source_IT>::value_type>(fakeLiterals[0]) << ", " << static_cast<typename std::iterator_traits<source_IT>::value_type>(fakeLiterals[1]) << ", " << static_cast<typename std::iterator_traits<source_IT>::value_type>(fakeLiterals[2]) << ", " << static_cast<typename std::iterator_traits<source_IT>::value_type>(fakeLiterals[3]) << ", " << static_cast<typename std::iterator_traits<source_IT>::value_type>(fakeLiterals[4]) << ", " << static_cast<typename std::iterator_traits<source_IT>::value_type>(fakeLiterals[5]) << ", " << static_cast<typename std::iterator_traits<source_IT>::value_type>(fakeLiterals[6]) << ", " << static_cast<typename std::iterator_traits<source_IT>::value_type>(fakeLiterals[7]) << "]";
  // };

  // for (size_t i = 0; i < nIncompressible; ++i) {
  //   *literalIter = static_cast<typename std::iterator_traits<source_IT>::value_type>(fakeLiterals2[i]);
  //   ++literalIter;
  // }

  // for (auto& s : fakeLiterals) {
  //   *literalIter = static_cast<typename std::iterator_traits<source_IT>::value_type>(s);
  //   ++literalIter;
  // }

  return {symbolIter - 8, literalIter, unrolledSymbols2};
};

#endif /* __AVX2__ */

} // namespace simd
} // namespace internal
} // namespace rans
} // namespace o2
#endif /* RANS_INTERNAL_SIMD_SYMBOLMAPPER_H */
