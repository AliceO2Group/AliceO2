// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   Decoder.h
/// @author Michael Lettrich
/// @since  2020-04-06
/// @brief  Decoder - decode a rANS encoded state back into source symbols

#ifndef RANS_DECODER_H
#define RANS_DECODER_H

#include <type_traits>
#include <iostream>

#include "SymbolTable.h"
#include "DecoderSymbol.h"
#include "ReverseSymbolLookupTable.h"
#include "Coder.h"

namespace o2
{
namespace rans
{
template <typename coder_t, typename source_T>
class Decoder
{
 private:
  using decoderSymbol_t = SymbolTable<DecoderSymbol>;
  using reverseSymbolLookupTable_t = ReverseSymbolLookupTable<source_T>;

 public:
  Decoder(const Decoder& d);
  Decoder(Decoder&& d) = default;
  Decoder<coder_t, source_T>& operator=(const Decoder& d);
  Decoder<coder_t, source_T>& operator=(Decoder&& d) = default;
  ~Decoder() = default;
  Decoder(const SymbolStatistics& stats, size_t probabilityBits, size_t probabilityScale);

  template <typename stream_IT, typename source_IT>
  void process(const source_IT outputBegin, const stream_IT inputBegin, size_t numSymbols) const;

 private:
  std::unique_ptr<decoderSymbol_t> mSymbolTable;
  std::unique_ptr<reverseSymbolLookupTable_t> mReverseLUT;
  size_t mProbabilityBits;
};

template <typename coder_t, typename source_T>
Decoder<coder_t, source_T>::Decoder(const Decoder& d) : mSymbolTable(nullptr), mReverseLUT(nullptr), mProbabilityBits(d.mProbabilityBits)
{
  mSymbolTable = std::make_unique<decoderSymbol_t>(*d.mSymbolTable);
  mReverseLUT = std::make_unique<reverseSymbolLookupTable_t>(*d.mReverseLUT);
}

template <typename coder_t, typename source_T>
Decoder<coder_t, source_T>& Decoder<coder_t, source_T>::operator=(const Decoder& d)
{
  mSymbolTable = std::make_unique<decoderSymbol_t>(*d.mSymbolTable);
  mReverseLUT = std::make_unique<reverseSymbolLookupTable_t>(*d.mReverseLUT);
  mProbabilityBits = d.mProbabilityBits;
  return *this;
}

template <typename coder_t, typename source_T>
Decoder<coder_t, source_T>::Decoder(const SymbolStatistics& stats, size_t probabilityBits, size_t probabilityScale) : mSymbolTable(nullptr), mReverseLUT(nullptr), mProbabilityBits(probabilityBits)
{
  mSymbolTable = std::make_unique<decoderSymbol_t>(stats, probabilityBits);
  mReverseLUT = std::make_unique<reverseSymbolLookupTable_t>(probabilityScale, stats);
};

template <typename coder_t, typename source_T>
template <typename stream_IT, typename source_IT>
void Decoder<coder_t, source_T>::process(const source_IT outputBegin, const stream_IT inputBegin, size_t numSymbols) const
{
  static_assert(std::is_same<typename std::iterator_traits<source_IT>::value_type, source_T>::value);

  typedef typename std::iterator_traits<stream_IT>::value_type stream_t;
  using ransDecoder = Coder<coder_t, stream_t>;

  State<coder_t> rans0, rans1;
  stream_t* ptr = &(*inputBegin);
  source_IT it = outputBegin;
  ransDecoder::decInit(&rans0, &ptr);
  ransDecoder::decInit(&rans1, &ptr);

  for (size_t i = 0; i < (numSymbols & ~1); i += 2) {
    const stream_t s0 =
      (*mReverseLUT)[ransDecoder::decGet(&rans0, mProbabilityBits)];
    const stream_t s1 =
      (*mReverseLUT)[ransDecoder::decGet(&rans1, mProbabilityBits)];
    *it++ = s0;
    *it++ = s1;
    ransDecoder::decAdvanceSymbolStep(&rans0, &(*mSymbolTable)[s0],
                                      mProbabilityBits);
    ransDecoder::decAdvanceSymbolStep(&rans1, &(*mSymbolTable)[s1],
                                      mProbabilityBits);
    ransDecoder::decRenorm(&rans0, &ptr);
    ransDecoder::decRenorm(&rans1, &ptr);
  }

  // last byte, if number of bytes was odd
  if (numSymbols & 1) {
    const stream_t s0 =
      (*mReverseLUT)[ransDecoder::decGet(&rans0, mProbabilityBits)];
    *it = s0;
    ransDecoder::decAdvanceSymbol(&rans0, &ptr, &(*mSymbolTable)[s0],
                                  mProbabilityBits);
  }
}
} // namespace rans
} // namespace o2

#endif /* RANS_DECODER_H */
