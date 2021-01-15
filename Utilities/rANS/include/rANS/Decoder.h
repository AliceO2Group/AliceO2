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

#include "internal/Decoder.h"

#include <cstddef>
#include <type_traits>
#include <iostream>
#include <memory>

#include <fairlogger/Logger.h>

#include "FrequencyTable.h"
#include "internal/DecoderSymbol.h"
#include "internal/ReverseSymbolLookupTable.h"
#include "internal/SymbolTable.h"
#include "internal/Decoder.h"
#include "internal/SymbolStatistics.h"
#include "internal/helper.h"

namespace o2
{
namespace rans
{

template <typename coder_T, typename stream_T, typename source_T>
class Decoder
{

 protected:
  using decoderSymbolTable_t = internal::SymbolTable<internal::DecoderSymbol>;
  using reverseSymbolLookupTable_t = internal::ReverseSymbolLookupTable;
  using ransDecoder = internal::Decoder<coder_T, stream_T>;

 public:
  Decoder(const Decoder& d);
  Decoder(Decoder&& d) = default;
  Decoder<coder_T, stream_T, source_T>& operator=(const Decoder& d);
  Decoder<coder_T, stream_T, source_T>& operator=(Decoder&& d) = default;
  ~Decoder() = default;
  Decoder(const FrequencyTable& stats, size_t probabilityBits);

  template <typename stream_IT, typename source_IT, std::enable_if_t<internal::isCompatibleIter_v<stream_T, stream_IT> && internal::isCompatibleIter_v<source_T, source_IT>, bool> = true>
  void process(const source_IT outputBegin, const stream_IT inputEnd, size_t messageLength) const;

  size_t getAlphabetRangeBits() const { return mSymbolTable->getAlphabetRangeBits(); }
  int getMinSymbol() const { return mSymbolTable->getMinSymbol(); }
  int getMaxSymbol() const { return mSymbolTable->getMaxSymbol(); }

  using coder_t = coder_T;
  using stream_t = stream_T;
  using source_t = source_T;

 protected:
  std::unique_ptr<decoderSymbolTable_t> mSymbolTable;
  std::unique_ptr<reverseSymbolLookupTable_t> mReverseLUT;
  size_t mProbabilityBits;
};

template <typename coder_T, typename stream_T, typename source_T>
Decoder<coder_T, stream_T, source_T>::Decoder(const Decoder& d) : mSymbolTable(nullptr), mReverseLUT(nullptr), mProbabilityBits(d.mProbabilityBits)
{
  mSymbolTable = std::make_unique<decoderSymbolTable_t>(*d.mSymbolTable);
  mReverseLUT = std::make_unique<reverseSymbolLookupTable_t>(*d.mReverseLUT);
}

template <typename coder_T, typename stream_T, typename source_T>
Decoder<coder_T, stream_T, source_T>& Decoder<coder_T, stream_T, source_T>::operator=(const Decoder& d)
{
  mSymbolTable = std::make_unique<decoderSymbolTable_t>(*d.mSymbolTable);
  mReverseLUT = std::make_unique<reverseSymbolLookupTable_t>(*d.mReverseLUT);
  mProbabilityBits = d.mProbabilityBits;
  return *this;
}

template <typename coder_T, typename stream_T, typename source_T>
Decoder<coder_T, stream_T, source_T>::Decoder(const FrequencyTable& frequencies, size_t probabilityBits) : mSymbolTable(nullptr), mReverseLUT(nullptr), mProbabilityBits(probabilityBits)
{
  using namespace internal;

  SymbolStatistics stats(frequencies, mProbabilityBits);
  mProbabilityBits = stats.getSymbolTablePrecision();

  RANSTimer t;
  t.start();
  mSymbolTable = std::make_unique<decoderSymbolTable_t>(stats);
  t.stop();
  LOG(debug1) << "Decoder SymbolTable inclusive time (ms): " << t.getDurationMS();
  t.start();
  mReverseLUT = std::make_unique<reverseSymbolLookupTable_t>(mProbabilityBits, stats);
  t.stop();
  LOG(debug1) << "ReverseSymbolLookupTable inclusive time (ms): " << t.getDurationMS();
};

template <typename coder_T, typename stream_T, typename source_T>
template <typename stream_IT, typename source_IT, std::enable_if_t<internal::isCompatibleIter_v<stream_T, stream_IT> && internal::isCompatibleIter_v<source_T, source_IT>, bool>>
void Decoder<coder_T, stream_T, source_T>::process(const source_IT outputBegin, const stream_IT inputEnd, size_t messageLength) const
{
  using namespace internal;
  LOG(trace) << "start decoding";
  RANSTimer t;
  t.start();

  if (messageLength == 0) {
    LOG(warning) << "Empty message passed to decoder, skipping decode process";
    return;
  }

  stream_IT inputIter = inputEnd;
  source_IT it = outputBegin;

  // make Iter point to the last last element
  --inputIter;

  ransDecoder rans0, rans1;
  inputIter = rans0.init(inputIter);
  inputIter = rans1.init(inputIter);

  for (size_t i = 0; i < (messageLength & ~1); i += 2) {
    const int64_t s0 = (*mReverseLUT)[rans0.get(mProbabilityBits)];
    const int64_t s1 = (*mReverseLUT)[rans1.get(mProbabilityBits)];
    *it++ = s0;
    *it++ = s1;
    inputIter = rans0.advanceSymbol(inputIter, (*mSymbolTable)[s0], mProbabilityBits);
    inputIter = rans1.advanceSymbol(inputIter, (*mSymbolTable)[s1], mProbabilityBits);
  }

  // last byte, if message length was odd
  if (messageLength & 1) {
    const int64_t s0 = (*mReverseLUT)[rans0.get(mProbabilityBits)];
    *it = s0;
    inputIter = rans0.advanceSymbol(inputIter, (*mSymbolTable)[s0], mProbabilityBits);
  }
  t.stop();
  LOG(debug1) << "Decoder::" << __func__ << " { DecodedSymbols: " << messageLength << ","
              << "processedBytes: " << messageLength * sizeof(source_T) << ","
              << " inclusiveTimeMS: " << t.getDurationMS() << ","
              << " BandwidthMiBPS: " << std::fixed << std::setprecision(2) << (messageLength * sizeof(source_T) * 1.0) / (t.getDurationS() * 1.0 * (1 << 20)) << "}";

  LOG(trace) << "done decoding";
}
} // namespace rans
} // namespace o2

#endif /* RANS_DECODER_H */
