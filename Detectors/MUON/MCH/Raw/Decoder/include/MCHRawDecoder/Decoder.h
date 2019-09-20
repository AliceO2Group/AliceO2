// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_DECODER_H
#define O2_MCH_RAW_DECODER_H

#include <cstdlib>
#include <gsl/span>
#include "MCHRawDecoder/SampaChannelHandler.h"
#include "MCHRawDecoder/RawDataHeaderHandler.h"
#include <iostream>

namespace o2
{
namespace mch
{
/// Classes and functions to deal with MCH Raw Data Formats.
namespace raw
{

struct DecoderStat {
  uint64_t nofOrbitJumps{0};
  uint64_t nofOrbitSeen{0};
  uint64_t nofBytesUsed{0};
};

std::ostream& operator<<(std::ostream& out, const DecoderStat& decStat);

using Decoder = std::function<DecoderStat(gsl::span<uint8_t> buffer)>;

template <typename FORMAT, typename CHARGESUM, typename RDH>
Decoder createDecoder(RawDataHeaderHandler<RDH> rdhHandler,
                      SampaChannelHandler channelHandler);

} // namespace raw
} // namespace mch
} // namespace o2
#endif
