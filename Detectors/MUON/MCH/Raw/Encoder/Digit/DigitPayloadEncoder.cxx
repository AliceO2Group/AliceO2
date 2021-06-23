// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHRawEncoderDigit/DigitPayloadEncoder.h"

#include "DataFormatsMCH/Digit.h"
#include "DetectorsRaw/HBFUtils.h"
#include "MCHRawEncoderDigit/Digit2ElecMapper.h"
#include "Framework/Logger.h"
#include "MCHRawCommon/DataFormats.h"
#include "MCHRawCommon/SampaBunchCrossingCounter.h"
#include "MCHRawElecMap/DsDetId.h"
#include "MCHRawEncoderPayload/PayloadEncoder.h"
#include <cstdint>
#include <fmt/printf.h>
#include <functional>
#include <gsl/span>
#include <memory>
#include "Framework/Logger.h"

namespace o2::mch::raw
{
std::string asString(o2::mch::Digit d)
{
  return fmt::format("DetID {:4d} PadId {:10d} ADC {:10d} TFtime {:10d} NofSamples {:5d} {}",
                     d.getDetID(), d.getPadID(), d.getADC(), d.getTime(), d.getNofSamples(),
                     d.isSaturated() ? "(S)" : "");
}

DigitPayloadEncoder::DigitPayloadEncoder(Digit2ElecMapper digit2elec,
                                         PayloadEncoder& encoder)
  : mDigit2ElecMapper{digit2elec}, mEncoder{encoder}
{
}

void DigitPayloadEncoder::encodeDigits(gsl::span<o2::mch::Digit> digits,
                                       uint32_t orbit,
                                       uint16_t bc,
                                       std::vector<std::byte>& buffer)
{
  mEncoder.startHeartbeatFrame(orbit, bc);
  for (auto d : digits) {
    auto optElecId = mDigit2ElecMapper(d);
    if (!optElecId.has_value()) {
      LOGP(warning, "could not get elecId for digit {}", asString(d));
      continue;
    }
    auto elecId = optElecId.value().first;
    int dualSampaChannelId = optElecId.value().second;
    // FIXME : what to put as rel time ?
    uint10_t ts = 0;
    auto firstIR = o2::raw::HBFUtils::Instance().getFirstIR();
    uint20_t bxCount = sampaBunchCrossingCounter(orbit, bc, firstIR.orbit);
    auto clusters = {raw::SampaCluster(ts, bxCount, d.getADC(), d.getNofSamples())};
    mEncoder.addChannelData(elecId, dualSampaChannelId, clusters);
  }
  mEncoder.moveToBuffer(buffer);
}
} // namespace o2::mch::raw
