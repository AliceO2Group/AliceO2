// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_PAYLOAD_DECODER_H
#define O2_MCH_RAW_PAYLOAD_DECODER_H

#include "BareGBTDecoder.h"
#include "DumpBuffer.h"
#include "Headers/RAWDataHeader.h"
#include "MCHRawDecoder/Decoder.h"
#include "MakeArray.h"
#include "PayloadDecoder.h"
#include "UserLogicGBTDecoder.h"
#include <cstdlib>
#include <fmt/format.h>
#include <gsl/span>
#include <iostream>

namespace o2
{
namespace mch
{
namespace raw
{
/// @brief Decoder for MCH  Raw Data Format.

template <typename RDH, typename GBTDECODER>
class PayloadDecoder
{
 public:
  /// Constructs a decoder
  /// \param channelHandler the handler that will be called for each
  /// piece of sampa data (a SampaCluster, i.e. a part of a time window)
  PayloadDecoder(SampaChannelHandler channelHandler);

  /// decode the buffer
  /// \return the number of bytes used from the buffer
  size_t process(const RDH& rdh, gsl::span<uint8_t> buffer);

  void reset();

 private:
  std::map<uint16_t, GBTDECODER> mDecoders; //< helper decoders
  SampaChannelHandler mChannelHandler;
};

template <typename RDH, typename GBTDECODER>
PayloadDecoder<RDH, GBTDECODER>::PayloadDecoder(SampaChannelHandler channelHandler)
  : mChannelHandler(channelHandler)
{
}

template <typename RDH, typename GBTDECODER>
size_t PayloadDecoder<RDH, GBTDECODER>::process(const RDH& rdh, gsl::span<uint8_t> buffer)
{
  auto solarId = rdh.feeId;
  auto c = mDecoders.find(solarId);
  if (c == mDecoders.end()) {
    mDecoders.emplace(solarId, GBTDECODER(solarId, mChannelHandler));
    c = mDecoders.find(solarId);
  }
  return c->second.append(buffer);
}

template <typename RDH, typename GBTDECODER>
void PayloadDecoder<RDH, GBTDECODER>::reset()
{
  for (auto& c : mDecoders) {
    c.second.reset();
  }
}

} // namespace raw
} // namespace mch
} // namespace o2

#endif
