// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHBase/Digit.h"
#include "MCHMappingInterface/Segmentation.h"
#include "MCHRawCommon/DataFormats.h"
#include "MCHRawElecMap/DsDetId.h"
#include "DigitEncoder.h"
#include "MCHRawEncoderPayload/PayloadEncoder.h"
#include <cstdint>
#include <fmt/printf.h>
#include <functional>
#include <gsl/span>
#include <memory>

namespace o2::mch::raw
{

template <typename CHARGESUM>
std::vector<SampaCluster> createSampaClusters(uint16_t ts, float adc);

template <>
std::vector<SampaCluster> createSampaClusters<raw::ChargeSumMode>(uint16_t ts, float adc)
{
  return {raw::SampaCluster(ts, 0, static_cast<uint32_t>(adc))};
}

template <>
std::vector<SampaCluster> createSampaClusters<raw::SampleMode>(uint16_t ts, float adc)
{
  uint32_t a = static_cast<uint32_t>(adc);
  std::vector<uint16_t> samples;

  samples.emplace_back(static_cast<uint16_t>((a & 0xFFC00) >> 10));
  samples.emplace_back(static_cast<uint16_t>(a & 0x3FF));
  return {raw::SampaCluster(ts, 0, samples)};
}

// create a function that return the (DsElecId,chid) of a digit
Digit2ElecMapper createDigit2ElecMapper(Det2ElecMapper det2elec)
{
  return [det2elec](const o2::mch::Digit& digit) -> std::optional<std::pair<DsElecId, int>> {
    int deid = digit.getDetID();
    int dsid = mapping::segmentation(deid).padDualSampaId(digit.getPadID());
    DsDetId detId{deid, dsid};
    auto dselocopt = det2elec(DsDetId(deid, dsid));
    if (!dselocopt.has_value()) {
      //std::cout << fmt::format("WARNING : got no location for (de,ds)=({},{})\n", deid, dsid);
      return std::nullopt;
    }
    DsElecId elecId = dselocopt.value();
    int dschid = mapping::segmentation(deid).padDualSampaChannel(digit.getPadID());
    return std::make_pair(dselocopt.value(), dschid);
  };
}

template <typename FORMAT, typename CHARGESUM>
DigitEncoder createDigitEncoder(Det2ElecMapper det2elec)
{
  auto encoder = createPayloadEncoder<FORMAT, CHARGESUM>().release();
  auto digit2elec = createDigit2ElecMapper(det2elec);

  return [encoder, digit2elec](gsl::span<o2::mch::Digit> digits,
                               std::vector<std::byte>& buffer,
                               uint32_t orbit,
                               uint16_t bc) {
    static int nadd{0};
    encoder->startHeartbeatFrame(orbit, bc);

    for (auto d : digits) {
      auto optElecId = digit2elec(d);
      if (!optElecId.has_value()) {
        continue;
      }
      auto elecId = optElecId.value().first;
      int dschid = optElecId.value().second;
      // FIXME: add a warning if timestamp is not in range
      uint16_t ts(static_cast<int>(d.getTime().sampaTime) & 0x3FF);
      int sampachid = dschid % 32;
      // FIXME: add a warning if ADC is not in range
      auto clusters = createSampaClusters<CHARGESUM>(ts, d.getADC() & 0xFFFFF);
      encoder->addChannelData(elecId, sampachid, clusters);
    }
    encoder->moveToBuffer(buffer);
  };
}

DigitEncoder createDigitEncoder(bool userLogic, Det2ElecMapper det2elec)
{
  if (userLogic) {
    return createDigitEncoder<UserLogicFormat, ChargeSumMode>(det2elec);
  }
  return createDigitEncoder<BareFormat, ChargeSumMode>(det2elec);
}
} // namespace o2::mch::raw
