// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_PAGEPARSER_H
#define O2_MCH_RAW_PAGEPARSER_H

#include <gsl/span>
#include "MCHRawDecoder/RawDataHeaderHandler.h"
#include "PayloadDecoder.h"

namespace
{
bool hasOrbitJump(uint32_t orb1, uint32_t orb2)
{
  return std::abs(static_cast<long int>(orb1 - orb2)) > 1;
}
} // namespace

namespace o2::mch::raw
{

template <typename RDH, typename PAYLOADDECODER>
class PageParser
{
 public:
  PageParser(RawDataHeaderHandler<RDH> rdhHandler, PAYLOADDECODER decoder);

  DecoderStat parse(gsl::span<uint8_t> buffer);

 private:
  RawDataHeaderHandler<RDH> mRdhHandler;
  PAYLOADDECODER mDecoder;
  uint32_t mOrbit{0};
  DecoderStat mStats;
};

template <typename RDH, typename PAYLOADDECODER>
PageParser<RDH, PAYLOADDECODER>::PageParser(RawDataHeaderHandler<RDH> rdhHandler, PAYLOADDECODER decoder)
  : mRdhHandler(rdhHandler), mDecoder(decoder), mStats{}
{
}

template <typename RDH, typename PAYLOADDECODER>
DecoderStat PageParser<RDH, PAYLOADDECODER>::parse(gsl::span<uint8_t> buffer)
{
  RDH originalRDH;
  const size_t nofRDHWords = sizeof(originalRDH);
  size_t index{0};
  uint64_t nbytes{0};

  while ((index + nofRDHWords) < buffer.size()) {
    originalRDH = createRDH<RDH>(buffer.subspan(index, nofRDHWords));
    if (!isValid(originalRDH)) {
      std::cout << "Got an invalid RDH\n";
      impl::dumpBuffer(buffer.subspan(index, nofRDHWords));
      return mStats;
    }
    if (hasOrbitJump(rdhOrbit(originalRDH), mOrbit)) {
      ++mStats.nofOrbitJumps;
      mDecoder.reset();
    } else if (rdhOrbit(originalRDH) != mOrbit) {
      ++mStats.nofOrbitSeen;
    }
    mOrbit = rdhOrbit(originalRDH);
    auto rdhOpt = mRdhHandler(originalRDH);
    if (!rdhOpt.has_value()) {
      break;
    }
    auto rdh = rdhOpt.value();
    int payloadSize = rdhPayloadSize(rdh);
    size_t n = static_cast<size_t>(payloadSize);
    if (n) {
      size_t pos = static_cast<size_t>(index + nofRDHWords);
      mDecoder.process(rdh, buffer.subspan(pos, n));
      nbytes += n + nofRDHWords;
    }
    index += rdh.offsetToNext;
  }
  mStats.nofBytesUsed += nbytes;
  return mStats;
}

} // namespace o2::mch::raw

#endif
