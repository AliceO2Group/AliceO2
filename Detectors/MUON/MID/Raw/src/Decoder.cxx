// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Raw/src/Decoder.cxx
/// \brief  MID raw data decoder
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   30 September 2019

#include "MIDRaw/Decoder.h"

#include "Headers/RDHAny.h"
#include "DPLUtils/RawParser.h"

namespace o2
{
namespace mid
{

template <typename GBTDECODER>
Decoder<GBTDECODER>::Decoder() : mData(), mROFRecords(), mGBTDecoders(), mFEEIdConfig(), mMasks()
{
  /// Default constructor
  init();
}

template <typename GBTDECODER>
void Decoder<GBTDECODER>::clear()
{
  /// Clears the decoded data
  mData.clear();
  mROFRecords.clear();
}

template <typename GBTDECODER>
void Decoder<GBTDECODER>::init(bool isDebugMode)
{
  /// Initializes the decoder
  for (uint16_t igbt = 0; igbt < crateparams::sNGBTs; ++igbt) {
    if constexpr (std::is_same_v<GBTDECODER, GBTBareDecoder>) {
      mGBTDecoders[igbt].init(igbt, mMasks.getMask(igbt), isDebugMode);
    } else {
      mGBTDecoders[igbt].init(igbt, isDebugMode);
    }
    mGBTDecoders[igbt].setElectronicsDelay(mElectronicsDelay);
  }
}

template <typename GBTDECODER>
void Decoder<GBTDECODER>::process(gsl::span<const uint8_t> bytes)
{
  /// Decodes the buffer
  clear();
  o2::framework::RawParser parser(bytes.data(), bytes.size());
  for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
    if (it.size() == 0) {
      continue;
    }
    gsl::span<const uint8_t> payload(it.data(), it.size());
    auto const* rdhPtr = reinterpret_cast<const o2::header::RDHAny*>(it.raw());
    process(payload, *rdhPtr);
  }
  flush();
}

template <typename GBTDECODER>
void Decoder<GBTDECODER>::flush()
{
  /// Flushes the GBT data
  for (auto& gbtDec : mGBTDecoders) {
    if (!gbtDec.getData().empty()) {
      size_t firstEntry = mData.size();
      mData.insert(mData.end(), gbtDec.getData().begin(), gbtDec.getData().end());
      size_t lastRof = mROFRecords.size();
      mROFRecords.insert(mROFRecords.end(), gbtDec.getROFRecords().begin(), gbtDec.getROFRecords().end());
      for (auto rofIt = mROFRecords.begin() + lastRof; rofIt != mROFRecords.end(); ++rofIt) {
        rofIt->firstEntry += firstEntry;
      }
      gbtDec.clear();
    }
  }
}

template <typename GBTDECODER>
bool Decoder<GBTDECODER>::isComplete() const
{
  /// Checks that all links have finished reading
  for (auto& decoder : mGBTDecoders) {
    if (!decoder.isComplete()) {
      return false;
    }
  }
  return true;
}

template class Decoder<GBTBareDecoder>;
template class Decoder<GBTUserLogicDecoder>;

} // namespace mid
} // namespace o2
