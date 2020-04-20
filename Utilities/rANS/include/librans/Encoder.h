// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   Encoder.h
/// @author Michael Lettrich
/// @since  2020-04-06
/// @brief  Encoder - code symbol into a rANS encoded state

#ifndef RANS_ENCODER_H
#define RANS_ENCODER_H

#include <memory>
#include <algorithm>

#include "SymbolTable.h"
#include "EncoderSymbol.h"
#include "Coder.h"

namespace o2
{
namespace rans
{
template <typename coder_t>
class Encoder
{
 private:
  using encoderSymbolTable_t = SymbolTable<EncoderSymbol<coder_t>>;

 public:
  Encoder() = delete;
  ~Encoder() = default;
  Encoder(Encoder&& e) = default;
  Encoder(const Encoder& e);
  Encoder& operator=(const Encoder& e);
  Encoder& operator=(Encoder&& e) = default;

  Encoder(const encoderSymbolTable_t& e, size_t probabilityBits);
  Encoder(encoderSymbolTable_t&& e, size_t probabilityBits);
  Encoder(const SymbolStatistics& stats, size_t probabilityBits);
  template <typename source_IT>
  Encoder(source_IT begin, source_IT end, size_t range, size_t probabilityBits);

  template <typename stream_IT, typename source_IT>
  const stream_IT process(const stream_IT outputBegin, const stream_IT outputEnd, const source_IT inputBegin, const source_IT inputEnd) const;

 private:
  std::unique_ptr<encoderSymbolTable_t> mSymbolTable;
  size_t mProbabilityBits;
};

template <typename coder_t>
Encoder<coder_t>::Encoder(const Encoder& e) : mSymbolTable(nullptr), mProbabilityBits(e.mProbabilityBits)
{
  mSymbolTable = std::make_unique<encoderSymbolTable_t>(*e.mSymbolTable);
};

template <typename coder_t>
Encoder<coder_t>& Encoder<coder_t>::operator=(const Encoder& e)
{
  mProbabilityBits = e.mProbabilityBits;
  mSymbolTable = std::make_unique<encoderSymbolTable_t>(*e.mSymbolTable);
  return *this;
};

template <typename coder_t>
Encoder<coder_t>::Encoder(const encoderSymbolTable_t& e, size_t probabilityBits) : mSymbolTable(nullptr), mProbabilityBits(probabilityBits)
{
  mSymbolTable = std::make_unique<encoderSymbolTable_t>(e);
};

template <typename coder_t>
Encoder<coder_t>::Encoder(encoderSymbolTable_t&& e, size_t probabilityBits) : mSymbolTable(std::move(e.mSymbolTable)), mProbabilityBits(probabilityBits){};

template <typename coder_t>
template <typename source_IT>
Encoder<coder_t>::Encoder(source_IT begin, source_IT end, size_t range, size_t probabilityBits) : mSymbolTable(nullptr), mProbabilityBits(probabilityBits)
{
  SymbolStatistics stats(begin, end, range);
  stats.rescaleFrequencyTable(mProbabilityBits);
  mSymbolTable = std::make_unique<encoderSymbolTable_t>(stats, probabilityBits);
}

template <typename coder_t>
Encoder<coder_t>::Encoder(const SymbolStatistics& stats,
                          size_t probabilityBits) : mSymbolTable(nullptr), mProbabilityBits(probabilityBits)
{
  mSymbolTable = std::make_unique<encoderSymbolTable_t>(stats, probabilityBits);
}

template <typename coder_t>
template <typename stream_IT, typename source_IT>
const stream_IT Encoder<coder_t>::Encoder::process(const stream_IT outputBegin, const stream_IT outputEnd, const source_IT inputBegin, const source_IT inputEnd) const
{

  typedef typename std::iterator_traits<stream_IT>::value_type stream_t;
  using ransCoder = Coder<coder_t, stream_t>;

  State<coder_t> rans0, rans1;
  ransCoder::encInit(&rans0);
  ransCoder::encInit(&rans1);

  stream_t* ptr = &(*outputEnd);
  source_IT inputIT = inputEnd;

  const auto inputBufferSize = inputEnd - inputBegin;

  // odd number of bytes?
  if (inputBufferSize & 1) {
    const coder_t s = *(--inputIT);
    ransCoder::encPutSymbol(&rans0, &ptr, &(*mSymbolTable)[s],
                            mProbabilityBits);
  }

  while (inputIT > inputBegin) { // NB: working in reverse!
    const coder_t s1 = *(--inputIT);
    const coder_t s0 = *(--inputIT);
    ransCoder::encPutSymbol(&rans1, &ptr, &(*mSymbolTable)[s1],
                            mProbabilityBits);
    ransCoder::encPutSymbol(&rans0, &ptr, &(*mSymbolTable)[s0],
                            mProbabilityBits);
  }
  ransCoder::encFlush(&rans1, &ptr);
  ransCoder::encFlush(&rans0, &ptr);

  return outputBegin + std::distance(&(*outputBegin), ptr);
};

} // namespace rans
} // namespace o2

#endif /* RANS_ENCODER_H */
